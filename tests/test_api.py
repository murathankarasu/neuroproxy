"""API contract: sessions, events, summary, and honest aggregates."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="API tests need the api extra")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def _session(**overrides):
    body = {"fps": 30.0, "method": "pos", "calibration_s": 45.0}
    body.update(overrides)
    return client.post("/v1/sessions", json=body)


def test_model_endpoint_states_the_output_contract():
    """A consumer must be told the scale is relative before reading a number."""
    info = client.get("/v1/model").json()
    assert "pos" in info["methods"]
    assert "baseline" in info["notes"]


def test_unknown_method_is_rejected_at_session_creation():
    assert _session(method="not_a_method").status_code == 400


def test_session_lifecycle():
    sid = _session().json()["session_id"]
    assert client.get("/v1/sessions/{}/summary".format(sid)).status_code == 200
    assert client.get("/v1/sessions/nonexistent/summary").status_code == 404


def test_events_land_on_the_timeline():
    sid = _session().json()["session_id"]
    marked = client.post("/v1/sessions/{}/events".format(sid),
                         json={"label": "stimulus", "t": 12.5}).json()
    assert marked == {"label": "stimulus", "t": 12.5}
    assert client.get("/v1/sessions/{}/summary".format(sid)).json()["events"] == [marked]


def test_summary_reports_the_answered_ratio():
    """Aggregates without a coverage figure are misleading: on real sessions
    the engine declines to answer roughly half the time."""
    sid = _session().json()["session_id"]
    summary = client.get("/v1/sessions/{}/summary".format(sid)).json()
    for key in ("answered", "answered_ratio", "reasons", "mean_confidence"):
        assert key in summary


def test_websocket_emits_state_after_one_window():
    """End-to-end over the socket, on a synthetic subject."""
    import cv2

    from training.datasets.synthetic import SyntheticConfig, generate

    rec = generate(SyntheticConfig(duration_s=22.0, hr_bpm=72.0))
    sid = _session(fps=rec.fps, calibration_s=0.0).json()["session_id"]
    received = []
    with client.websocket_connect("/v1/sessions/{}/stream".format(sid)) as ws:
        for i, frame in enumerate(rec.frames()):
            ok, buf = cv2.imencode(".png", frame[:, :, ::-1])
            ws.send_bytes(buf.tobytes())
            if (i + 1) == int(20 * rec.fps):
                received.append(ws.receive_json())
                break
    assert received
    sample = received[0]
    assert sample["session_id"] == sid
    # The synthetic generator deliberately carries +-6 bpm drift and +-3 bpm
    # respiratory arrhythmia -- a constant rate would let a broken estimator
    # look perfect -- so the true rate in any 20 s window spans roughly 63-81.
    assert sample["physiology"]["heart_rate_bpm"] == pytest.approx(72.0, abs=10.0)
    assert 0.0 <= sample["confidence"] <= 1.0


def test_undecodable_frame_is_reported_not_swallowed():
    sid = _session().json()["session_id"]
    with client.websocket_connect("/v1/sessions/{}/stream".format(sid)) as ws:
        ws.send_bytes(b"not an image")
        assert "error" in ws.receive_json()


def test_demo_page_is_served_and_states_its_own_caveat():
    """The page must carry the transport warning it is a demonstration of."""
    page = client.get("/")
    assert page.status_code == 200
    body = page.text
    assert "NeuroProxy session" in body
    # It must not silently imply an absolute score.
    assert "bpm_vs_baseline" in body or "vs baseline" in body
    # And it must distinguish calibrating from declining, not colour both as failure.
    assert "calibrating" in body and "declined" in body
