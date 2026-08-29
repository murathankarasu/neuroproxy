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

TWO INGEST PATHS
----------------
`/stream` accepts frames. `/features` accepts per-frame features extracted on
the client. The second is the one to build on:

* Raw video never leaves the device, which is the privacy position both source
  documents take (design doc section 11).
* ~100 bytes per frame instead of a compressed image -- roughly 3 KB/s against
  15 MB/s for the lossless frames the frame path requires.
* No codec sits between the sensor and the signal. The frame path had to send
  lossless PNG because JPEG q75 transport drops answered windows from 57% to
  zero (docs/limitations.md 24); the feature path removes that trade entirely.

The frame path remains for offline replay and for clients that cannot run an
extractor, and is marked provisional.
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

from api.consent import NOTICE_VERSION, NOTICE_TEXT, notice, validate
from api.store import Store
from neuroproxy.inference import FramePacket, StateEngine, StateSample
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


# Live engines, keyed by session. Durable state lives in the store; this holds
# only the rolling buffers, which are meaningless after a restart anyway.
SESSIONS: Dict[str, Session] = {}
STORE = Store()


class StudyConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    session: SessionConfig = Field(default_factory=SessionConfig)
    # No unbounded option. Retention that is never reached is not retention.
    retention_days: int = Field(90, ge=1, le=730)


class ConsentGrant(BaseModel):
    scopes: List[str]
    notice_version: str


def _require_consent(session_id: str) -> Dict[str, object]:
    """Gate every ingest path. Without a recorded grant there is no session."""
    record = STORE.get_consent(session_id)
    if record is None or record.get("withdrawn_at"):
        raise HTTPException(403, "no active consent for this session")
    return record


@app.post("/v1/studies")
def create_study(cfg: StudyConfig) -> Dict[str, object]:
    study_id = STORE.create_study(
        cfg.name, cfg.session.model_dump(), retention_days=cfg.retention_days)
    return {"study_id": study_id, "name": cfg.name,
            "retention_days": cfg.retention_days,
            "participant_url": "/?study={}".format(study_id)}


@app.get("/v1/studies")
def list_studies() -> Dict[str, object]:
    return {"studies": STORE.list_studies()}


@app.get("/v1/studies/{study_id}")
def get_study(study_id: str) -> Dict[str, object]:
    study = STORE.get_study(study_id)
    if study is None:
        raise HTTPException(404, "unknown study")
    sessions = STORE.study_sessions(study_id)
    return {**study, "sessions": len(sessions)}


@app.get("/v1/studies/{study_id}/aggregate")
def aggregate(study_id: str) -> Dict[str, object]:
    """Study-level view across participants.

    Reports coverage alongside every aggregate. A mean heart rate over sessions
    that mostly declined to answer is not a finding, and the number that says
    so has to travel with it.
    """
    if STORE.get_study(study_id) is None:
        raise HTTPException(404, "unknown study")
    rows = []
    for sess in STORE.study_sessions(study_id):
        samples = STORE.get_samples(sess["session_id"])
        # Calibration windows are excluded from the denominator. Counting them
        # as unanswered penalises a session for the 45 s it spent working
        # normally, and made healthy sessions read as failures.
        scored = [s for s in samples if s.get("reason") != "calibrating"]
        answered = [s for s in scored if s.get("state")]
        hrs = [s["physiology"]["heart_rate_bpm"] for s in samples
               if (s.get("physiology") or {}).get("heart_rate_bpm") is not None]
        ratio = (len(answered) / len(scored)) if scored else 0.0
        rows.append({
            "session_id": sess["session_id"],
            "external_ref": sess["external_ref"],
            "emissions": len(samples),
            "calibrating": len(samples) - len(scored),
            "scored": len(scored),
            "answered": len(answered),
            "answered_ratio": ratio,
            "median_hr": float(np.median(hrs)) if hrs else None,
            "usable": bool(scored) and ratio >= 0.5,
        })
    usable = [r for r in rows if r["usable"]]
    return {
        "study_id": study_id,
        "participants": len(rows),
        # The product-level number: of everyone who took part, how many
        # produced anything worth analysing.
        "usable_sessions": len(usable),
        "usable_session_rate": (len(usable) / len(rows)) if rows else 0.0,
        "pooled_answered_ratio": (
            sum(r["answered"] for r in rows) / max(sum(r["scored"] for r in rows), 1)),
        "sessions": rows,
    }


@app.get("/v1/consent")
def consent_notice() -> Dict[str, object]:
    return notice()


@app.post("/v1/sessions/{session_id}/consent")
def grant_consent(session_id: str, grant: ConsentGrant) -> Dict[str, object]:
    if STORE.get_session(session_id) is None:
        raise HTTPException(404, "unknown session")
    if grant.notice_version != NOTICE_VERSION:
        # Consent to a superseded notice is consent to something else.
        raise HTTPException(
            409, "notice has changed; re-read and grant against version {}".format(
                NOTICE_VERSION))
    try:
        scopes = validate(grant.scopes)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return STORE.record_consent(session_id, NOTICE_VERSION, NOTICE_TEXT, scopes)


@app.post("/v1/sessions/{session_id}/withdraw")
def withdraw(session_id: str) -> Dict[str, object]:
    """Withdraw and erase. Available during or after a session."""
    if STORE.get_session(session_id) is None:
        raise HTTPException(404, "unknown session")
    SESSIONS.pop(session_id, None)
    return STORE.withdraw(session_id)


@app.get("/v1/sessions/{session_id}/export")
def export(session_id: str) -> Dict[str, object]:
    """Everything held about this session, for a participant access request."""
    if STORE.get_session(session_id) is None:
        raise HTTPException(404, "unknown session")
    return STORE.export_session(session_id)


@app.post("/v1/admin/purge")
def purge() -> Dict[str, int]:
    """Apply each study's retention window. Intended to run on a schedule."""
    return STORE.purge_expired()


DEMO_DIR = _Path(__file__).resolve().parents[1] / "apps" / "web_demo"
DEMO_PAGE = DEMO_DIR / "index.html"
DASHBOARD_PAGE = DEMO_DIR / "dashboard.html"


@app.get("/static/{path:path}")
def demo_asset(path: str) -> FileResponse:
    """Serve the demo's own assets (extractor, equivalence fixtures)."""
    target = (DEMO_DIR / path).resolve()
    if not str(target).startswith(str(DEMO_DIR.resolve())) or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(str(target))


@app.get("/studies")
def dashboard() -> FileResponse:
    """Researcher view: create studies, hand out links, watch sessions arrive.

    NO AUTHENTICATION. Anyone who can reach the server can read every study.
    That is acceptable for a laptop demo and nowhere else; the page says so
    itself rather than leaving it to be discovered.
    """
    if not DASHBOARD_PAGE.exists():
        raise HTTPException(404, "dashboard not found")
    return FileResponse(str(DASHBOARD_PAGE))


@app.get("/")
def demo() -> FileResponse:
    """Single-page session demo. Localhost only -- it pushes lossless frames."""
    if not DEMO_PAGE.exists():
        raise HTTPException(404, "demo page not found")
    return FileResponse(str(DEMO_PAGE))


@app.post("/v1/sessions", response_model=SessionCreated)
def create_session(
    config: SessionConfig, study_id: Optional[str] = None,
    external_ref: Optional[str] = None,
) -> SessionCreated:
    if config.method not in available_methods():
        raise HTTPException(400, "unknown method {!r}".format(config.method))
    if study_id is not None and STORE.get_study(study_id) is None:
        raise HTTPException(404, "unknown study")
    from neuroproxy.rppg.base import get_method

    session_id = STORE.create_session(study_id, config.model_dump(), external_ref)
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
    record = STORE.add_event(session_id, float(t), event.label)
    session.events.append(record)
    return record


@app.get("/v1/sessions/{session_id}/summary")
def summary(session_id: str) -> Dict[str, object]:
    """Session summary. Served from the store, so it survives a restart."""
    if STORE.get_session(session_id) is None:
        raise HTTPException(404, "unknown session {!r}".format(session_id))
    stored = STORE.get_samples(session_id)
    samples = [StateSample(**s) for s in stored]
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
        "events": STORE.get_events(session_id),
        "timeline": [s.as_dict() for s in samples],
    }


@app.websocket("/v1/sessions/{session_id}/features")
async def features(websocket: WebSocket, session_id: str) -> None:
    """Accept client-extracted per-frame features.

    Message is JSON, either one packet or a batch:
        {"rgb": [r, g, b], "valid": true, "face": 0.9, ...}
        {"frames": [ {...}, {...} ]}

    Batching matters: at 30 fps one WebSocket message per frame is mostly
    framing overhead, and the engine only emits once per second anyway.
    """
    await websocket.accept()
    session = SESSIONS.get(session_id)
    if session is None:
        await websocket.close(code=4404)
        return
    try:
        _require_consent(session_id)
    except HTTPException:
        await websocket.close(code=4403)
        return
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("close"):
                break
            packets = data.get("frames") or [data]
            emitted = []
            for raw in packets:
                sample = session.engine.push_features(_packet(raw))
                if sample is not None:
                    session.samples.append(sample)
                    emitted.append(sample.as_dict())
            if emitted:
                STORE.append_samples(session_id, emitted)
            for sample in emitted:
                await websocket.send_json(sample)
    except WebSocketDisconnect:
        pass
    finally:
        session.closed = True


def _packet(raw: Dict[str, object]) -> FramePacket:
    def num(key: str, default: float) -> float:
        try:
            return float(raw.get(key, default))
        except (TypeError, ValueError):
            return default

    rgb = raw.get("rgb")
    if isinstance(rgb, list) and len(rgb) == 3:
        try:
            rgb = [float(v) for v in rgb]
        except (TypeError, ValueError):
            rgb = None
    else:
        rgb = None
    return FramePacket(
        rgb=rgb,
        valid=bool(raw.get("valid")) and rgb is not None,
        face=num("face", 0.0),
        lighting=num("lighting", 0.0),
        sharpness=num("sharpness", 0.0),
        motion=num("motion", 1.0),
        skin_fraction=num("skin_fraction", 0.0),
        compression=num("compression", 1.0),
    )


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
        _require_consent(session_id)
    except HTTPException:
        await websocket.close(code=4403)
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
                STORE.append_samples(session_id, [sample.as_dict()])
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
