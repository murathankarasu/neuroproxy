"""NeuroProxy state API (design doc section 10.1).

    POST /v1/sessions               create a session, get its id
    WS   /v1/sessions/{id}/stream   push frames, receive state at 1 Hz
    GET  /v1/sessions/{id}/summary  aggregate timeline for the session
    POST /v1/sessions/{id}/events   mark an event on the timeline
    GET  /v1/model                  engine and method versions

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not store video. Frames are decoded, consumed by the engine, and
dropped; only derived state and quality survive the request. Both source
documents make raw-video retention the default-off position (design doc
section 11), and the cheapest way to keep that promise is to have nowhere to
put it.

ARCHITECTURAL CAVEAT, STATED UP FRONT
-------------------------------------
This version accepts *frames* over the socket, which means raw video crosses
the network even though it is never written down. The target architecture in
design doc section 10.2 extracts features on the client and sends only those.
That is the right end state for both privacy and bandwidth; this is the
server-side version that makes the contract testable first. Anything built on
top should treat the frame-push path as provisional.
"""
from __future__ import annotations

import base64
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from fastapi.responses import FileResponse
from pathlib import Path as _Path

from neuroproxy.inference import StateEngine, StateSample
from neuroproxy.rppg.base import available_methods

ENGINE_VERSION = "0.1.0"

app = FastAPI(title="NeuroProxy State API", version=ENGINE_VERSION)


class SessionConfig(BaseModel):
    fps: float = Field(30.0, gt=0, le=120)
    method: str = "pos"
    window_s: float = Field(20.0, ge=5.0, le=60.0)
    emit_hz: float = Field(1.0, gt=0, le=5.0)
    calibration_s: float = Field(45.0, ge=0.0, le=180.0)
    # Free-form study metadata. Never used for inference; carried so a
    # researcher can join sessions to their own study records.
    metadata: Dict[str, str] = Field(default_factory=dict)


class SessionCreated(BaseModel):
    session_id: str
    config: SessionConfig
    engine_version: str


class EventMark(BaseModel):
    label: str
    t: Optional[float] = None


@dataclass
class Session:
    session_id: str
    config: SessionConfig
    engine: StateEngine
    created_at: float = field(default_factory=time.time)
    samples: List[StateSample] = field(default_factory=list)
    events: List[Dict[str, object]] = field(default_factory=list)
    closed: bool = False


SESSIONS: Dict[str, Session] = {}


DEMO_PAGE = _Path(__file__).resolve().parents[1] / "apps" / "web_demo" / "index.html"


@app.get("/")
def demo() -> FileResponse:
    """Single-page session demo. Localhost only -- it pushes lossless frames."""
    if not DEMO_PAGE.exists():
        raise HTTPException(404, "demo page not found")
    return FileResponse(str(DEMO_PAGE))


@app.post("/v1/sessions", response_model=SessionCreated)
def create_session(config: SessionConfig) -> SessionCreated:
    if config.method not in available_methods():
        raise HTTPException(400, "unknown method {!r}".format(config.method))
    session_id = uuid.uuid4().hex[:12]
    from neuroproxy.rppg.base import get_method

    SESSIONS[session_id] = Session(
        session_id=session_id,
        config=config,
        engine=StateEngine(
            session_id=session_id,
            fps=config.fps,
            method=get_method(config.method),
            window_s=config.window_s,
            emit_hz=config.emit_hz,
            calibration_s=config.calibration_s,
        ),
    )
    return SessionCreated(
        session_id=session_id, config=config, engine_version=ENGINE_VERSION
    )


@app.get("/v1/model")
def model_info() -> Dict[str, object]:
    return {
        "engine_version": ENGINE_VERSION,
        "methods": available_methods(),
        "default_method": "pos",
        "notes": (
            "State is reported as deviation from the subject's own baseline, "
            "not on an absolute scale. `state` is null with a reason whenever "
            "signal quality does not support an answer."
        ),
    }


@app.post("/v1/sessions/{session_id}/events")
def mark_event(session_id: str, event: EventMark) -> Dict[str, object]:
    session = _require(session_id)
    t = event.t if event.t is not None else session.engine._n_frames / session.config.fps
    record = {"label": event.label, "t": float(t)}
    session.events.append(record)
    return record


@app.get("/v1/sessions/{session_id}/summary")
def summary(session_id: str) -> Dict[str, object]:
    session = _require(session_id)
    samples = session.samples
    answered = [s for s in samples if s.state is not None]
    hrs = [
        s.physiology["heart_rate_bpm"] for s in samples
        if s.physiology.get("heart_rate_bpm") is not None
    ]
    return {
        "session_id": session_id,
        "engine_version": ENGINE_VERSION,
        "duration_s": samples[-1].t if samples else 0.0,
        "emissions": len(samples),
        # The share of emissions the engine stood behind. A researcher needs
        # this next to every aggregate: half of real sessions yield little.
        "answered": len(answered),
        "answered_ratio": (len(answered) / len(samples)) if samples else 0.0,
        "calibrated": any(s.calibrated for s in samples),
        "heart_rate_bpm": {
            "median": float(np.median(hrs)) if hrs else None,
            "min": float(np.min(hrs)) if hrs else None,
            "max": float(np.max(hrs)) if hrs else None,
        },
        "mean_confidence": float(np.mean([s.confidence for s in samples])) if samples else 0.0,
        "reasons": _count_reasons(samples),
        "events": session.events,
        "timeline": [s.as_dict() for s in samples],
    }


@app.websocket("/v1/sessions/{session_id}/stream")
async def stream(websocket: WebSocket, session_id: str) -> None:
    """Push frames, receive state.

    Client sends either binary JPEG frames, or a JSON message
    `{"frame": "<base64 jpeg>"}`. The server replies only on emission ticks,
    so a 30 FPS push produces roughly one message per second.
    """
    await websocket.accept()
    session = SESSIONS.get(session_id)
    if session is None:
        await websocket.close(code=4404)
        return

    try:
        while True:
            message = await websocket.receive()
            payload = message.get("bytes")
            if payload is None:
                text = message.get("text")
                if text is None:
                    break
                import json

                data = json.loads(text)
                if data.get("close"):
                    break
                payload = base64.b64decode(data["frame"])

            frame = _decode(payload)
            if frame is None:
                await websocket.send_json({"error": "undecodable frame"})
                continue

            sample = session.engine.push(frame)
            if sample is not None:
                session.samples.append(sample)
                await websocket.send_json(sample.as_dict())
    except WebSocketDisconnect:
        pass
    finally:
        session.closed = True


def _decode(payload: bytes) -> Optional[np.ndarray]:
    import cv2

    buf = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return None if img is None else img[:, :, ::-1].copy()


def _require(session_id: str) -> Session:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "unknown session {!r}".format(session_id))
    return session


def _count_reasons(samples: List[StateSample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in samples:
        if s.reason:
            counts[s.reason] = counts.get(s.reason, 0) + 1
    return counts
