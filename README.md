# neutral, happiness, surprise, sadness, anger, disgust, fear, contempt
# Do Unity: neutral, boredom, happiness, surprise, anger, fear

import asyncio, os, time, json, sqlite3, threading, math
from typing import Dict, Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------- Konfiguracja -----------------
DB_PATH     = os.path.join("data", "telemetry.db")
USER_ID     = os.environ.get("FER_USER", "ola")
FPS_TARGET  = int(os.environ.get("FER_FPS", "10"))
MODEL_PATH  = os.path.join("models", os.environ.get("FER_MODEL", "ferplus.onnx"))
FER_MODE    = os.environ.get("FER_MODE", "auto")          # auto | mock | onnx

CONF_THRESH = float(os.environ.get("FER_CONF_THRESH", "0.30"))  # confidence
BUF_LEN     = int(os.environ.get("FER_BUFFER", "10"))           # bufor
MIN_FACE    = int(os.environ.get("FER_MIN_FACE", "100"))        
PAD_RATIO   = float(os.environ.get("FER_PAD_RATIO", "0.15"))   

SAD_TO_BORED_THRESH = float(os.environ.get("FER_SAD_TO_BORED_THRESH", "0.025"))
NEU_MIN_FOR_BORED   = float(os.environ.get("FER_NEU_MIN_FOR_BORED", "0.55"))

SADNESS_AS_BOREDOM = int(os.environ.get("FER_SADNESS_AS_BOREDOM", "1")) == 1

# jeśli brak modelu -> mock; jeśli model jest -> onnx
USE_MOCK = FER_MODE == "mock" or (FER_MODE == "auto" and not os.path.exists(MODEL_PATH))

# ----------------- Aplikacja -----------------
app = FastAPI(title="FER Microservice (MOCK/ONNX)", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# przygotowanie bazy danych
os.makedirs("data", exist_ok=True)
_db = sqlite3.connect(DB_PATH, check_same_thread=False)
_db.execute("""
CREATE TABLE IF NOT EXISTS emotions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  user TEXT NOT NULL,
  emotion TEXT NOT NULL,
  arousal REAL NOT NULL,
  confidence REAL NOT NULL
) 
""")
_db.commit() 

latest_event_lock = threading.Lock()
latest_event: Dict = {}
ws_clients: List[WebSocket] = []

# ----------------- FER+ (8) -----------------
FERPLUS_LABELS = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]
IDX = {name:i for i, name in enumerate(FERPLUS_LABELS)}

# Co wysyłane do Unity:
TARGET_LABELS = ["neutral","boredom","happiness","surprise","anger","fear","sadness"]  # sadness na końcu (opcjonalnie)

EMO_TO_AROUSAL = {
    "neutral":0.40, "boredom":0.20, "happiness":0.60, "surprise":0.75,
    "sadness":0.30, "anger":0.85, "fear":0.90
}


ALLOWED8 = ["neutral", "happiness", "surprise", "anger", "fear", "sadness"] # jedyne główne wyjścia, pomocnicze inne

# nadanie ts (timestamp)
def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

# zapisywanie do bazy danych
def save_evt(evt: Dict):
    _db.execute(
        "INSERT INTO emotions(ts,user,emotion,arousal,confidence) VALUES (?,?,?,?,?)",
        (evt["ts"], evt["user"], evt["emotion"], evt["arousal"], evt["confidence"])
    )
    _db.commit()
    # blokada! 
    with latest_event_lock:
        latest_event.clear()
        latest_event.update(evt)

    print(f"{evt['emotion']}  conf={evt['confidence']}")
# wysyłka JSON do klientów, obsługa klientów 
async def broadcast(evt: Dict):
    dead=[]
    for ws in ws_clients:
        try:
            await ws.send_text(json.dumps(evt))
        except Exception:
            dead.append(ws)
    for ws in dead:
        try: ws_clients.remove(ws)
        except ValueError: pass

# ----------------- ONNX utils -----------------
def onnx_available() -> bool:
    return os.path.exists(MODEL_PATH)

def build_onnx_runtime():
    import onnxruntime as ort
    import numpy as np
    import cv2
    sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return sess, in_name, out_name, face_cascade, np, cv2

def create_yunet(cv2, model_path=os.path.join("models", "face_detection_yunet_2023mar.onnx"),
                 score_thresh=0.6, nms_thresh=0.3, top_k=5000):
    try:
        if not hasattr(cv2, "FaceDetectorYN_create"):
            print("ℹ️ OpenCV bez FaceDetectorYN (opencv-contrib-python). Fallback na Haar.")
            return None
        if os.path.exists(model_path):
            det = cv2.FaceDetectorYN_create(model_path, "", (320, 320), score_thresh, nms_thresh, top_k)
            print("✅ YuNet model found:", model_path)
            return det
        print("ℹ️ YuNet model not found, using Haar.")
        return None
    except Exception as e:
        print("⚠️ YuNet load failed:", e)
        return None

def softmax_np(np, x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-8)
# dla pewnosci modelu, 2 najwieksze, porownanie 
def top2_allowed(mean8):
    pairs = [(name, float(mean8[IDX[name]])) for name in ALLOWED8]
    pairs.sort(key=lambda x: x[1], reverse=True)
    (n1,p1),(n2,p2) = pairs[0], pairs[1]
    return n1,p1,n2,p2

def boredom_condition(mean8, std8, p8_inst=None):
    """
    Boredom: nie z samego neutral!
    Ma wejść dopiero gdy:
    - neutral jest dość wysoki
    - ALE jest też “wycofanie”: sadness/contempt
    - i sygnał jest stabilny
    """
    neu = float(mean8[IDX["neutral"]])
    sad = float(mean8[IDX["sadness"]])
    con = float(mean8[IDX["contempt"]])

    # neutral arę wysoki
    if neu < NEU_MIN_FOR_BORED:
        return False

    #  trochę smutku / pogardy
    if (sad + 0.5*con) < 0.10:
        return False
    if sad >= SAD_TO_BORED_THRESH:
        return True

    # blokery: jak jest coś “mocnego”, to to nie boredom
    blockers = ["anger", "fear", "surprise", "happiness"]
    if max(float(mean8[IDX[b]]) for b in blockers) > 0.22:
        return False

    # brak pików na bieżącej klatce
    if p8_inst is not None:
        if float(p8_inst[IDX["surprise"]]) > 0.18:
            return False
        if float(p8_inst[IDX["fear"]]) > 0.16:
            return False

    # stabilność, blokowanie boredom
    low_var = float(std8.max()) < 0.030 if std8 is not None else True
    if not low_var:
        return False

    return True

# ----------------- Pętla -----------------
async def capture_loop():
    frame_interval = 1.0 / FPS_TARGET

    use_mock = USE_MOCK
    if not use_mock and not onnx_available():
        use_mock = True

    if use_mock:
        print("🟡 Start w trybie MOCK (scenariusz emocji co 5s).")

        emotions_cycle = [
            ("neutral",   0.40),
            ("boredom",   0.20),
            ("happiness", 0.65),
            ("anger",     0.85),
            ("fear",      0.90),
            ("surprise",  0.75),
        ]

        idx = 0
        last_switch = time.time()

        while True:
            now = time.time()

            # zmiana emocji co 5 sekund
            if now - last_switch >= 5.0:
                idx = (idx + 1) % len(emotions_cycle)
                last_switch = now

            emotion, base_arousal = emotions_cycle[idx]


            arousal = base_arousal + 0.05 * math.sin(now * 0.8)
            arousal = max(0.0, min(1.0, arousal))

            evt = {
                "ts": now_iso(),
                "user": USER_ID,
                "emotion": emotion,
                "arousal": round(arousal, 3),
                "confidence": 0.95   
            }

            save_evt(evt)
            await broadcast(evt)
            await asyncio.sleep(frame_interval)

    print("🟢 Start w trybie ONNX (kamera + model) – 8-class + boredom override.")
    sess, in_name, out_name, face_cascade, np, cv2 = build_onnx_runtime()

    from collections import deque
    probs8_long = deque(maxlen=BUF_LEN)

    def infer_probs8(face_gray_u8):
        f = cv2.resize(face_gray_u8, (48, 48), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        f = np.expand_dims(f, axis=-1)  # HxW -> HxWx1
        f = np.expand_dims(f, axis=0)   # -> 1x48x48x1
        out = sess.run([out_name], {in_name: f})[0][0]
        return softmax_np(np, out.astype(np.float64))

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("⚠️ Kamera niedostępna – przełączam na tryb awaryjny.")
        while True:
            evt = {"ts": now_iso(), "user": USER_ID, "emotion": "neutral", "arousal": 0.2, "confidence": 1.0}
            save_evt(evt)
            await broadcast(evt)
            await asyncio.sleep(frame_interval)

    app._yunet = create_yunet(cv2)
    if app._yunet is not None:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        app._yunet.setInputSize((fw, fh))
        print(f"✅ YuNet enabled at input size {fw}x{fh}")
    else:
        print("ℹ️ Falling back to Haar cascade.")

    app._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok or frame is None:
            await asyncio.sleep(0.05)
            continue
        top3_8 = [("n/a", 0.0), ("n/a", 0.0), ("n/a", 0.0)]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = []
        if getattr(app, "_yunet", None) is not None:
            _, dets = app._yunet.detect(frame)
            if dets is not None and len(dets) > 0:
                faces = [[int(d[0]), int(d[1]), int(d[2]), int(d[3])] for d in dets]
        else:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 80))

        # brak twarzy / za mała twarz -> neutral 
        if len(faces) == 0:
            label, conf = "neutral", 0.0
        else:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            if w < MIN_FACE or h < MIN_FACE:
                label, conf = "neutral", 0.0
            else:
                pad = int(PAD_RATIO * max(w, h))
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                face_gray = gray[y0:y1, x0:x1]

                eq = app._clahe.apply(face_gray)

                p8_inst = infer_probs8(eq)
                probs8_long.append(p8_inst)

                mean8 = np.mean(np.stack(list(probs8_long)), axis=0)
                std8  = np.std (np.stack(list(probs8_long)), axis=0) # odchylenie

                # wybór etykiety (z allowed)
                n1, p1, n2, p2 = top2_allowed(mean8)
                label8, conf8 = n1, p1
                margin = p1 - p2

                # mała przewaga -> neutral
                if margin < 0.06:
                    label8, conf8 = "neutral", float(mean8[IDX["neutral"]])

                label, conf = label8, float(conf8)

                # jeśli w grze nie chcesz sadness — zamień sadness na boredom (opcjonalnie)
                if SADNESS_AS_BOREDOM and label == "sadness":
                    label = "boredom"
                    conf = float(mean8[IDX["sadness"]])

                # boredom override (tylko na neutral lub sadness)
                if label in ("neutral", "sadness") and boredom_condition(mean8, std8, p8_inst=p8_inst):
                    label = "boredom"
                    conf = max(float(mean8[IDX["neutral"]]), float(mean8[IDX["sadness"]]))

                # próg pewności
                conf_thresh = CONF_THRESH - (0.05 if label in ("fear","surprise") else 0.0)
                if conf < conf_thresh:
                    label = "neutral"
                    conf = float(mean8[IDX["neutral"]])

                ii8 = np.argsort(mean8)[-3:][::-1]
                top3_8 = [(FERPLUS_LABELS[i], float(mean8[i])) for i in ii8]
                #print(f"[FER dbg] top3(8 mean): {top3_8} | picked={label} ({conf:.2f}) margin={margin:.2f}")
                

        base = EMO_TO_AROUSAL.get(label, 0.5)
        arousal = float(max(0.0, min(1.0, 0.2*base + 0.8*(base*max(conf, 0.25)))))
        print(
                    f"[FER] picked={label:<9} conf={conf:.2f} | "
                    f"TOP3: "
                    f"{top3_8[0][0]}={top3_8[0][1]:.2f}, "
                    f"{top3_8[1][0]}={top3_8[1][1]:.2f}, "
                    f"{top3_8[2][0]}={top3_8[2][1]:.2f}"
                )

        evt = {"ts": now_iso(), "user": USER_ID, "emotion": label,
               "arousal": round(arousal, 3), "confidence": round(conf, 3)}

        save_evt(evt)
        await broadcast(evt)

        elapsed = time.perf_counter() - t0
        await asyncio.sleep(max(0.0, frame_interval - elapsed))

# ----------------- Lifecycle -----------------
@app.on_event("startup")
async def on_startup():
    asyncio.create_task(capture_loop())

@app.on_event("shutdown")
def on_shutdown():
    _db.close()

# ----------------- API -----------------
class LatestResponse(BaseModel):
    ts: Optional[str] = None
    user: Optional[str] = None
    emotion: Optional[str] = None
    arousal: Optional[float] = None
    confidence: Optional[float] = None

@app.get("/")
def root():
    return {"service": "FER microservice", "mode": ("MOCK" if USE_MOCK else "AUTO/ONNX"),
            "endpoints": ["/health", "/latest", "/ws"]}

@app.get("/health")
def health():
    return {"status": "ok", "mode": ("MOCK" if USE_MOCK else "AUTO/ONNX")}

@app.get("/latest", response_model=LatestResponse)
def get_latest(user: Optional[str] = None):
    with latest_event_lock:
        return latest_event or LatestResponse().dict()

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        try: ws_clients.remove(websocket)
        except ValueError: pass
