"""Live drone detection page: one page, one WebSocket per viewer.

    python -m console                  # http://localhost:8080
    python -m console --host 0.0.0.0   # serve other devices (microphone needs HTTPS)

Acoustic (primary): the browser captures sound, from the microphone or an audio
file it plays out loud, and streams raw samples here. Once per second the audio
detector judges the last second.

Review and labels: the current source's sound is kept in memory (16 kHz, up to
SESSION_MAX_S, dropped when a new source starts or the page disconnects) so an
operator can replay any stretch and label it "drone" or "not drone": a missed
drone, a false alarm, a confirmed detection. Only labelled stretches are
written, to data/feedback/, where they can retrain the detector behind a
safety gate (src/dronedet/feedback.py; scripts/retrain_feedback.py, or POST
/retrain).

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
SR = 16000
SESSION_MAX_S = 900             # sound kept in memory for review: the last 15 minutes
LABEL_MIN_S, LABEL_MAX_S = 0.5, 60.0

_cal = Path(__file__).parent / "calibration.json"           # written by scripts/calibrate_live.py
CALIBRATION = ({k: v for k, v in json.loads(_cal.read_text()).items() if k in ("confirm", "hold")}
               if _cal.exists() else {})

_model: dict = {}
_lock = threading.Lock()


class SessionAudio:
    """The current source's sound at 16 kHz, in memory only, for replay and labels.

    Raw samples arrive at the browser's rate; they are resampled in whole
    one-second blocks so that a time in seconds maps to an exact sample. Times
    are seconds since the source started; the oldest sound beyond `max_s` is
    dropped."""

    def __init__(self, rate: int, max_s: float = SESSION_MAX_S):
        self.rate, self.max = rate, int(max_s * SR)
        self.pending = np.zeros(0, np.float32)
        self.chunks: list[np.ndarray] = []
        self.n = self.offset = 0

    def add(self, x: np.ndarray) -> None:
        self.pending = np.concatenate([self.pending, x])
        while len(self.pending) >= self.rate:
            block, self.pending = self.pending[:self.rate], self.pending[self.rate:]
            y = to_16k(block, self.rate)[:SR]
            self.chunks.append(np.pad(y, (0, SR - len(y))))
            self.n += SR
        while self.n > self.max:
            self.offset += len(self.chunks[0])
            self.n -= len(self.chunks.pop(0))

    @property
    def end_s(self) -> float:
        return (self.offset + self.n) / SR

    def slice(self, start_s: float, end_s: float) -> np.ndarray:
        a = max(int(round(start_s * SR)) - self.offset, 0)
        b = min(int(round(end_s * SR)) - self.offset, self.n)
        if b <= a:
            return np.zeros(0, np.float32)
        if len(self.chunks) > 1:
            self.chunks = [np.concatenate(self.chunks)]
        return self.chunks[0][a:b].copy()


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


def _variant(base, path: Path):
    """The live detector with another trained head (same encoder)."""
    m = copy.copy(base)
    m.det = joblib.load(path)
    m.model, m.threshold = m.det.head, m.det.threshold
    return m


def detector(mic: bool = False):
    """The audio detector in use (the latest promoted version, if feedback has
    produced one); for the microphone with DRONA_MIC_MODEL=speaker, the
    playback-trained one."""
    if "det" not in _model:
        from dronedet import feedback as FB
        from dronedet import live as L
        base = L.LiveDetector("detector")
        current = FB.current_model_path()
        _model["det"] = _variant(base, current) if current != FB.ORIGINAL else base
        if (L.MODELS / "detector_mic.joblib").exists():
            _model["mic"] = _variant(_model["det"], L.MODELS / "detector_mic.joblib")
    if mic and os.environ.get("DRONA_MIC_MODEL") == "speaker":
        return _model.get("mic", _model["det"])
    return _model["det"]


def classify(window_16k: np.ndarray, mic: bool = False) -> str:
    """1 s of 16 kHz audio -> detector state."""
    det = detector(mic)
    with _lock:
        emb, _ = det.enc([window_16k])
        return str(det.det.assess(emb)["state"][0])


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


@app.get("/config")
def config():
    return JSONResponse({"alarm_rule": CALIBRATION})


@app.get("/model")
def model_status():
    from dronedet import feedback as FB
    return JSONResponse(FB.status())


_retrain_lock = asyncio.Lock()


@app.post("/retrain")
async def retrain():
    from dronedet import feedback as FB
    if _retrain_lock.locked():
        return JSONResponse({"ok": False, "error": "already retraining"}, status_code=409)
    async with _retrain_lock:
        det = detector()
        res = await asyncio.to_thread(FB.retrain, det.enc, det.det)
        new = res.pop("model", None)
        if new is not None:
            with _lock:
                det.det, det.model, det.threshold = new, new.head, new.threshold
    return JSONResponse({"ok": True, "result": res, "model": FB.status()})


@app.post("/rollback")
async def rollback():
    from dronedet import feedback as FB
    FB.rollback()
    det = detector()
    m = joblib.load(FB.current_model_path())
    with _lock:
        det.det, det.model, det.threshold = m, m.head, m.threshold
    return JSONResponse({"ok": True, "model": FB.status()})


@app.get("/rf/scenes")
def rf_scenes():
    if rf_feed() is None:
        return JSONResponse([])
    return JSONResponse([{"key": k, "name": n, "group": g}
                         for k, (_, n, g) in _model["rf_scenes"].items()])


@app.websocket("/ws")
async def ws(sock: WebSocket):
    """One socket: "audio" (primary) and "radio" verdicts out; sound, control,
    review ("get") and label ("label") messages in."""
    await sock.accept()
    c = {"rate": 48000, "buf": np.zeros(0, np.float32), "pending": 0, "mic": False, "samples": 0,
         "sound": SessionAudio(48000), "rf_key": None, "rf_i": 0, "audio": session(), "radio": session()}
    send_lock = asyncio.Lock()

    async def send(obj, data: bytes | None = None):
        async with send_lock:
            await sock.send_text(json.dumps(obj))
            if data is not None:
                await sock.send_bytes(data)

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

    async def label(m):
        from dronedet import feedback as FB
        start, end = float(m.get("start", 0)), float(m.get("end", 0))
        if not LABEL_MIN_S <= end - start <= LABEL_MAX_S:
            return await send({"type": "labelled", "id": m.get("id"), "ok": False,
                               "error": f"choose between {LABEL_MIN_S:g} and {LABEL_MAX_S:g} seconds"})
        y = c["sound"].slice(start, end)
        try:
            await asyncio.to_thread(FB.save_clip, str(m.get("id", "")), str(m.get("label", "")),
                                    y, str(m.get("input", "")))
            await send({"type": "labelled", "id": m.get("id"), "ok": True, "label": m.get("label"),
                        "start": start, "end": end})
        except ValueError as e:
            await send({"type": "labelled", "id": m.get("id"), "ok": False, "error": str(e)})

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
                    rate = int(m.get("rate", 48000))
                    c.update(rate=rate, mic=m.get("input") == "mic", buf=np.zeros(0, np.float32),
                             pending=0, samples=0, sound=SessionAudio(rate), audio=session())
                elif kind == "stop":
                    c["audio"] = session()
                elif kind == "rf":
                    key = m.get("source")
                    c.update(rf_key=key if key in _model.get("rf_scenes", {}) else None,
                             rf_i=0, radio=session())
                elif kind == "get":                          # replay a stretch of this source
                    start, end = float(m.get("start", 0)), float(m.get("end", 0))
                    y = c["sound"].slice(start, min(end, start + LABEL_MAX_S))
                    await send({"type": "clip", "id": m.get("id"), "rate": SR, "n": len(y)}, y.tobytes())
                elif kind == "label":
                    await label(m)
                continue
            x = np.frombuffer(msg["bytes"], dtype=np.float32)
            rate = c["rate"]
            c["sound"].add(x)
            c["samples"] += len(x)
            c["buf"] = np.concatenate([c["buf"], x])[-rate:]
            c["pending"] += len(x)
            if len(c["buf"]) < rate or c["pending"] < HOP_S * rate:
                continue
            c["pending"] = 0
            state = await asyncio.to_thread(classify, to_16k(c["buf"], rate), c["mic"])
            v = c["audio"].step(state)
            await send({"type": "audio", "verdict": v, "t": round(c["samples"] / rate, 2),
                        "unfamiliar": v == "drone" and c["audio"].unfamiliar})
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
