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


def _consented(**overrides):
    """A session that has passed the consent gate, as every real one must."""
    from api.consent import NOTICE_VERSION, SCOPES

    sid = _session(**overrides).json()["session_id"]
    r = client.post("/v1/sessions/{}/consent".format(sid),
                    json={"scopes": sorted(SCOPES), "notice_version": NOTICE_VERSION})
    assert r.status_code == 200, r.text
    return sid


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
    sid = _consented(fps=rec.fps, calibration_s=0.0)
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
    sid = _consented()
    with client.websocket_connect("/v1/sessions/{}/stream".format(sid)) as ws:
        ws.send_bytes(b"not an image")
        assert "error" in ws.receive_json()


def test_demo_page_carries_its_own_contract():
    """The page must state what the engine does, not just render numbers."""
    page = client.get("/")
    assert page.status_code == 200
    body = page.text
    assert "NeuroProxy" in body
    # Never implies an absolute score.
    assert "vs your baseline" in body
    # Distinguishes calibrating from declining, not both as failure.
    assert "calibrating" in body and "declined" in body
    # Consent comes before the camera, and withdrawal is offered.
    assert "/v1/consent" in body and "withdraw" in body
    # The visual refuses along with the engine.
    assert "stops rather than guess" in body



# --- consent, retention and withdrawal -----------------------------------

def test_ingest_is_refused_without_consent():
    """No recorded grant, no session. The gate is on the socket, not the UI."""
    from starlette.websockets import WebSocketDisconnect

    sid = _session().json()["session_id"]
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/v1/sessions/{}/features".format(sid)) as ws:
            ws.send_json({"rgb": [150, 110, 95], "valid": True})
            ws.receive_json()


def test_partial_consent_is_rejected():
    """There is no mode where the camera runs but nothing is recorded."""
    from api.consent import NOTICE_VERSION

    sid = _session().json()["session_id"]
    r = client.post("/v1/sessions/{}/consent".format(sid),
                    json={"scopes": ["camera"], "notice_version": NOTICE_VERSION})
    assert r.status_code == 400
    assert "missing" in r.text


def test_consent_to_a_superseded_notice_is_rejected():
    """Agreeing to an old notice is agreeing to something else."""
    sid = _session().json()["session_id"]
    r = client.post("/v1/sessions/{}/consent".format(sid),
                    json={"scopes": ["camera", "derived_state", "research_use"],
                          "notice_version": "1970-01-01.0"})
    assert r.status_code == 409


def test_the_notice_is_retrievable_and_versioned():
    n = client.get("/v1/consent").json()
    assert n["version"] and n["all_required"] is True
    assert "video does not" in n["text"].lower()
    assert {s["key"] for s in n["scopes"]} == {"camera", "derived_state", "research_use"}


def test_withdrawal_erases_rather_than_flags():
    """A withdrawal that leaves the data in place is not a withdrawal."""
    sid = _consented()
    client.post("/v1/sessions/{}/events".format(sid), json={"label": "x", "t": 1.0})
    assert client.get("/v1/sessions/{}/summary".format(sid)).status_code == 200

    assert client.post("/v1/sessions/{}/withdraw".format(sid)).json()["erased"] is True
    assert client.get("/v1/sessions/{}/summary".format(sid)).status_code == 404
    assert client.get("/v1/sessions/{}/export".format(sid)).status_code == 404


def test_export_states_that_no_video_is_held():
    sid = _consented()
    export = client.get("/v1/sessions/{}/export".format(sid)).json()
    assert set(("session", "consent", "samples", "events")) <= set(export)
    assert "no video" in export["note"].lower()
    assert export["consent"]["notice_text"], "the exact notice shown must be recoverable"


def test_study_aggregate_reports_coverage_next_to_every_number():
    """A mean over sessions that mostly declined is not a finding."""
    study = client.post("/v1/studies", json={"name": "checkout test",
                                             "retention_days": 30}).json()
    assert study["participant_url"].endswith(study["study_id"])
    sid = client.post("/v1/sessions?study_id={}&external_ref=P-1".format(study["study_id"]),
                      json={"fps": 30.0}).json()["session_id"]
    agg = client.get("/v1/studies/{}/aggregate".format(study["study_id"])).json()
    assert agg["participants"] == 1
    for key in ("usable_sessions", "usable_session_rate", "pooled_answered_ratio"):
        assert key in agg
    assert agg["sessions"][0]["external_ref"] == "P-1"


def test_retention_has_no_unbounded_option():
    assert client.post("/v1/studies", json={"name": "x", "retention_days": 0}).status_code == 422
    assert client.post("/v1/studies", json={"name": "x", "retention_days": 99999}).status_code == 422


def test_the_suite_is_not_writing_to_the_real_database():
    """A guard, because this was a real regression: the suite used to create
    studies in the database the dashboard reads from."""
    import os

    from api.main import STORE

    assert "neuroproxy-test-" in str(STORE.path), (
        "tests are pointed at {} -- conftest.py should have redirected "
        "NEUROPROXY_DB".format(STORE.path))
    assert os.environ.get("NEUROPROXY_DB")


def test_sessions_survive_a_restart():
    """Summaries come from the store, not from a process-local dict."""
    import api.main as m

    sid = _consented()
    client.post("/v1/sessions/{}/events".format(sid), json={"label": "e", "t": 2.0})
    m.SESSIONS.clear()                     # simulate a restart
    summary = client.get("/v1/sessions/{}/summary".format(sid)).json()
    assert summary["events"] == [{"t": 2.0, "label": "e"}]


# --- researcher dashboard and aggregate honesty ---------------------------

def test_dashboard_is_served_and_declares_its_own_gap():
    """It has no auth. The page must say so rather than let it be discovered."""
    page = client.get("/studies")
    assert page.status_code == 200
    body = page.text
    assert "No authentication" in body
    assert "Do not expose this host" in body
    # Coverage has to be visible next to the aggregates, not buried.
    assert "Usable sessions" in body and "Answered windows" in body


def test_calibration_does_not_count_against_a_session():
    """The regression this fixed: healthy sessions read as failures.

    Counting the 45 s calibration period as unanswered dragged a session with
    94% real coverage down to 55%, and a 72% one below the usable threshold
    entirely. Calibration is normal operation, not a refusal.
    """
    study = client.post("/v1/studies", json={"name": "calib", "retention_days": 7}).json()
    sid = client.post("/v1/sessions?study_id={}".format(study["study_id"]),
                      json={"fps": 30.0}).json()["session_id"]

    from api.main import STORE

    samples = []
    for i in range(100):
        calibrating = i < 45
        samples.append({
            "session_id": sid, "t": 20.0 + i,
            "physiology": {"heart_rate_bpm": 72.0},
            "state": None if calibrating else {"arousal_proxy": {"value": 1.0}},
            "quality": {"overall": 0.8}, "confidence": 0.5,
            "reason": "calibrating" if calibrating else None, "calibrated": not calibrating,
        })
    STORE.append_samples(sid, samples)

    row = client.get("/v1/studies/{}/aggregate".format(study["study_id"])).json()["sessions"][0]
    assert row["emissions"] == 100
    assert row["calibrating"] == 45
    assert row["scored"] == 55
    # Every scored window was answered, so this is a fully usable session.
    assert row["answered_ratio"] == pytest.approx(1.0)
    assert row["usable"] is True


def test_aggregate_marks_low_signal_sessions_rather_than_averaging_them_in():
    study = client.post("/v1/studies", json={"name": "mixed", "retention_days": 7}).json()
    from api.main import STORE

    for ref, answered in (("good", True), ("bad", False)):
        sid = client.post("/v1/sessions?study_id={}&external_ref={}".format(
            study["study_id"], ref), json={"fps": 30.0}).json()["session_id"]
        STORE.append_samples(sid, [{
            "session_id": sid, "t": 20.0 + i,
            "physiology": {"heart_rate_bpm": 70.0 if answered else None},
            "state": {"arousal_proxy": {"value": 0.0}} if answered else None,
            "quality": {"overall": 0.8}, "confidence": 0.5,
            "reason": None if answered else "low_confidence", "calibrated": True,
        } for i in range(20)])

    agg = client.get("/v1/studies/{}/aggregate".format(study["study_id"])).json()
    by_ref = {r["external_ref"]: r for r in agg["sessions"]}
    assert by_ref["good"]["usable"] is True
    assert by_ref["bad"]["usable"] is False
    assert agg["usable_session_rate"] == pytest.approx(0.5)
