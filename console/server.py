"""Live drone detection page: one page, one WebSocket per viewer.

    python -m console                  # http://localhost:8080
    python -m console --host 0.0.0.0   # serve other devices (microphone needs HTTPS)

Acoustic (primary): the browser captures sound, from the microphone or an audio
file it plays out loud, and streams raw samples here. Once per second the audio
detector judges the last second. Nothing is recorded or stored.

Radio (secondary): a laptop cannot receive raw 2.4 GHz signals (its Wi-Fi chip
only hands over decoded Wi-Fi packets), so the radio panel replays recorded
drone and no-drone captures from the held-out test split, scored by the RF v2
detector. Sound and radio are judged independently: the replay has nothing to
do with what the microphone hears, so they are not fused here (the fusion
ablation is in reports/fusion.md).

Microphone model: the original detector by default; DRONA_MIC_MODEL=speaker
uses the one also trained on played-back sound (scripts/train_mic_detector.py).
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from math import gcd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from scipy.signal import resample_poly

STATIC = Path(__file__).parent / "static"
HOP_S = 1.0                     # one decision per second, the spacing the alarm rule was calibrated on
DRONE_STATES = {"drone", "unknown drone"}

_cal = Path(__file__).parent / "calibration.json"           # written by scripts/calibrate_live.py
CALIBRATION = ({k: v for k, v in json.loads(_cal.read_text()).items() if k in ("confirm", "hold")}
               if _cal.exists() else {})

_model: dict = {}
_lock = threading.Lock()


class Session:
    """Per-window detector states -> what the operator is told.

    - "drone": `confirm` drone windows in a row, held for `hold` more windows
      so the display does not flicker.
    - "maybe": fewer drone windows than that, or two unsure windows in a row.
    - "clear": otherwise.
    """

    def __init__(self, confirm: int = 2, hold: int = 3):
        self.confirm, self.hold = confirm, hold
        self.run = self.unsure = self.hold_left = 0
        self.unfamiliar = False

    def step(self, state: str) -> str:
        hit = state in DRONE_STATES
        self.run = self.run + 1 if hit else 0
        self.unsure = self.unsure + 1 if state == "low confidence" else 0
        if self.run >= self.confirm:
            self.hold_left = self.hold
            self.unfamiliar = state == "unknown drone"
            return "drone"
        if self.hold_left > 0:
            self.hold_left -= 1
            return "drone"
        if hit or self.unsure >= 2:
            return "maybe"
        return "clear"


def session() -> Session:
    return Session(**CALIBRATION)


def detector(mic: bool = False):
    """The deployed audio detector, or for the microphone with
    DRONA_MIC_MODEL=speaker the playback-trained one. Both share one encoder."""
    if "det" not in _model:
        from dronedet import live as L
        _model["det"] = L.LiveDetector("detector")
        path = L.MODELS / "detector_mic.joblib"
        if path.exists():
            m = copy.copy(_model["det"])
            m.det = joblib.load(path)
            m.model, m.threshold = m.det.head, m.det.threshold
            _model["mic"] = m
    if mic and os.environ.get("DRONA_MIC_MODEL") == "speaker":
        return _model.get("mic", _model["det"])
    return _model["det"]


def classify(window_16k: np.ndarray, mic: bool = False) -> str:
    """1 s of 16 kHz audio -> detector state."""
    det = detector(mic)
    with _lock:
        df, _ = det.analyse(window_16k, 1.0)
    return str(df.state.iloc[0])


def to_16k(x: np.ndarray, rate: int) -> np.ndarray:
    if rate == 16000:
        return x
    g = gcd(16000, rate)
    return resample_poly(x, 16000 // g, rate // g).astype(np.float32)


# --- radio: recorded captures -----------------------------------------------

def scene_name(label: str) -> tuple[str, str | None]:
    """RFFeed source label -> (display name, group); group None = not offered."""
    names = {"bebop": "Parrot Bebop", "ar": "Parrot AR.Drone", "phantom": "DJI Phantom"}
    if label.startswith("DroneRF drone: "):
        d = label.removeprefix("DroneRF drone: ")
        return f"{names.get(d, d)} controller link", "Drone"
    if label.startswith("Tampere drone"):
        d = " ".join(w if w.isupper() else w.capitalize()
                     for w in label.split(": ", 1)[1].split("_"))
        return f"{d}, never heard in training", "Drone"
    if label.startswith("DroneRF background"):
        return "Lab background, no drone", "No drone"
    if label == "Receiver noise, no drone: white (dronerf pipeline)":
        return "Empty airwaves, receiver noise only", "No drone"
    return label, None


def rf_feed():
    if "rf" not in _model:
        try:
            from dronedet import live as L
            feed = L.RFFeed()
            scenes = [(lab, *scene_name(lab)) for lab in sorted(feed.sources)]
            _model["rf_scenes"] = {f"s{i}": sc for i, sc in enumerate(scenes) if sc[2]}
            _model["rf"] = feed
        except FileNotFoundError:
            _model["rf"] = None
    return _model["rf"]


def rf_column(psd: np.ndarray, rows: int = 96) -> list[int]:
    """(2 bands, bins) dB -> one waterfall column over 2.400-2.480 GHz, 0-255."""
    x = np.concatenate([psd[0], psd[1]])
    x = x[: len(x) // rows * rows].reshape(rows, -1).max(1)
    lo = np.median(x)
    return np.clip((x - lo + 3) / 33 * 255, 0, 255).astype(int).tolist()


def rf_window(key: str, i: int) -> tuple[str, list[int]]:
    """The i-th window of a recorded scene -> (detector state, waterfall column)."""
    feed = rf_feed()
    X = feed.sources[_model["rf_scenes"][key][0]]
    psd = X[i % len(X)][None]
    state = feed.det.assess(feed.features(psd))["state"][0]
    return str(state), rf_column(psd[0])


# --- app ----------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(detector)
    await asyncio.to_thread(rf_feed)
    yield


app = FastAPI(title="Drona", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return JSONResponse({"ok": "det" in _model, "radio": _model.get("rf") is not None,
                         "mic_model": os.environ.get("DRONA_MIC_MODEL", "original")})


@app.get("/rf/scenes")
def rf_scenes():
    if rf_feed() is None:
        return JSONResponse([])
    return JSONResponse([{"key": k, "name": n, "group": g}
                         for k, (_, n, g) in _model["rf_scenes"].items()])


@app.websocket("/ws")
async def ws(sock: WebSocket):
    """Two independent channels on one socket: "audio" (primary) and "radio"."""
    await sock.accept()
    c = {"rate": 48000, "buf": np.zeros(0, np.float32), "pending": 0, "mic": False,
         "rf_key": None, "rf_i": 0, "audio": session(), "radio": session()}
    send_lock = asyncio.Lock()

    async def send(obj):
        async with send_lock:
            await sock.send_text(json.dumps(obj))

    async def radio_loop():
        while True:
            await asyncio.sleep(HOP_S)
            if c["rf_key"] is None:
                continue
            state, column = await asyncio.to_thread(rf_window, c["rf_key"], c["rf_i"])
            c["rf_i"] += 1
            v = c["radio"].step(state)
            await send({"type": "radio", "verdict": v, "column": column,
                        "unfamiliar": v == "drone" and c["radio"].unfamiliar})

    task = asyncio.create_task(radio_loop())
    try:
        while True:
            msg = await sock.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text"):
                m = json.loads(msg["text"])
                kind = m.get("type")
                if kind == "start":
                    c.update(rate=int(m.get("rate", 48000)), mic=m.get("input") == "mic",
                             buf=np.zeros(0, np.float32), pending=0, audio=session())
                elif kind == "stop":
                    c["audio"] = session()
                elif kind == "rf":
                    key = m.get("source")
                    c.update(rf_key=key if key in _model.get("rf_scenes", {}) else None,
                             rf_i=0, radio=session())
                continue
            x = np.frombuffer(msg["bytes"], dtype=np.float32)
            rate = c["rate"]
            c["buf"] = np.concatenate([c["buf"], x])[-rate:]
            c["pending"] += len(x)
            if len(c["buf"]) < rate or c["pending"] < HOP_S * rate:
                continue
            c["pending"] = 0
            state = await asyncio.to_thread(classify, to_16k(c["buf"], rate), c["mic"])
            v = c["audio"].step(state)
            await send({"type": "audio", "verdict": v,
                        "unfamiliar": v == "drone" and c["audio"].unfamiliar})
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
