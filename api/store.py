"""Persistence for studies, participants, sessions and consent.

WHY SQLITE
----------
Sessions previously lived in a module-level dict: a restart lost every session,
and a second worker process saw none of the first's. Neither is acceptable for
a study a researcher is actually running. SQLite keeps that fixed without
adding a service to deploy.

WHAT IS STORED, AND WHAT IS DELIBERATELY NOT
--------------------------------------------
Stored: derived state samples (heart rate, baseline deviation, quality,
confidence), event marks, and the consent record.

Never stored: video, images, or the per-frame colour traces they were derived
from. The frames a browser extracts from are discarded in the page; the
features are consumed by the engine and only the once-per-second state survives.
This is data minimisation implemented as an absence rather than a promise.

Participants are identified by an opaque per-session token. No name, no email,
no IP is recorded by this layer. A researcher who needs to join a session to
their own records supplies their own pseudonymous reference in
`external_ref`, and what that reference points to is their responsibility and
their lawful basis, not this system's.

LEGAL POSTURE -- READ BEFORE A PILOT
------------------------------------
Camera-derived heart rate is data concerning health, and in the EU/UK that is
likely a special category under GDPR Article 9, which needs a condition beyond
ordinary consent-as-lawful-basis. This module implements the *mechanisms* that
such a regime requires -- recorded affirmative consent, versioned consent text,
withdrawal, export, and hard deletion -- because those are engineering
problems. Whether a given study is lawful, under which condition, with which
notices and which controller/processor split, is a question for counsel and
is not answered here. See docs/privacy.md.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

DEFAULT_DB = Path("data/neuroproxy.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
    study_id     TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    config       TEXT NOT NULL,
    -- Retention in days for this study's derived data. Enforced by
    -- purge_expired(); there is no "keep forever" value on purpose.
    retention_days INTEGER NOT NULL DEFAULT 90,
    closed_at    REAL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    study_id     TEXT REFERENCES studies(study_id) ON DELETE CASCADE,
    created_at   REAL NOT NULL,
    -- Researcher's own pseudonymous reference, if they use one. Opaque here.
    external_ref TEXT,
    config       TEXT NOT NULL,
    closed_at    REAL,
    -- Set when a participant withdraws. Withdrawn sessions are excluded from
    -- every aggregate immediately, before the row is purged.
    withdrawn_at REAL
);

CREATE TABLE IF NOT EXISTS consents (
    session_id   TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    granted_at   REAL NOT NULL,
    -- Exact text and version the participant was shown. Consent to a notice
    -- nobody can reproduce is not a record of anything.
    notice_version TEXT NOT NULL,
    notice_text  TEXT NOT NULL,
    -- What they actually agreed to, itemised.
    scopes       TEXT NOT NULL,
    withdrawn_at REAL
);

CREATE TABLE IF NOT EXISTS samples (
    session_id   TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    t            REAL NOT NULL,
    payload      TEXT NOT NULL,
    PRIMARY KEY (session_id, t)
);

CREATE TABLE IF NOT EXISTS events (
    session_id   TEXT REFERENCES sessions(session_id) ON DELETE CASCADE,
    t            REAL NOT NULL,
    label        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_study ON sessions(study_id);
CREATE INDEX IF NOT EXISTS idx_samples_session ON samples(session_id);
"""


class Store:
    """Thin SQLite layer. One connection per call keeps it thread-safe enough."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- studies -----------------------------------------------------------

    def create_study(self, name: str, config: Dict, retention_days: int = 90) -> str:
        study_id = secrets.token_urlsafe(9)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO studies (study_id, name, created_at, config, retention_days)"
                " VALUES (?,?,?,?,?)",
                (study_id, name, time.time(), json.dumps(config), int(retention_days)),
            )
        return study_id

    def get_study(self, study_id: str) -> Optional[Dict]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM studies WHERE study_id = ?", (study_id,)).fetchone()
        return _study(row) if row else None

    def list_studies(self) -> List[Dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM studies ORDER BY created_at DESC").fetchall()
        return [_study(r) for r in rows]

    # -- sessions ----------------------------------------------------------

    def create_session(
        self, study_id: Optional[str], config: Dict, external_ref: Optional[str] = None
    ) -> str:
        session_id = secrets.token_urlsafe(9)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, study_id, created_at, external_ref, config)"
                " VALUES (?,?,?,?,?)",
                (session_id, study_id, time.time(), external_ref, json.dumps(config)),
            )
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            return None
        return {
            "session_id": row["session_id"], "study_id": row["study_id"],
            "created_at": row["created_at"], "external_ref": row["external_ref"],
            "config": json.loads(row["config"]),
            "closed_at": row["closed_at"], "withdrawn_at": row["withdrawn_at"],
        }

    def close_session(self, session_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE sessions SET closed_at = ? WHERE session_id = ?",
                         (time.time(), session_id))

    # -- consent -----------------------------------------------------------

    def record_consent(
        self, session_id: str, notice_version: str, notice_text: str, scopes: List[str]
    ) -> Dict:
        granted_at = time.time()
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO consents"
                " (session_id, granted_at, notice_version, notice_text, scopes)"
                " VALUES (?,?,?,?,?)",
                (session_id, granted_at, notice_version, notice_text, json.dumps(scopes)),
            )
        return {"session_id": session_id, "granted_at": granted_at,
                "notice_version": notice_version, "scopes": scopes}

    def get_consent(self, session_id: str) -> Optional[Dict]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM consents WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            return None
        return {"session_id": row["session_id"], "granted_at": row["granted_at"],
                "notice_version": row["notice_version"],
                "notice_text": row["notice_text"],
                "scopes": json.loads(row["scopes"]),
                "withdrawn_at": row["withdrawn_at"]}

    def withdraw(self, session_id: str, erase: bool = True) -> Dict:
        """Withdraw consent. By default this erases, it does not merely flag.

        A withdrawal that leaves the data in place and hopes downstream queries
        remember to filter it is not a withdrawal. `erase=False` exists only
        for the case where a researcher must retain an auditable record that a
        session existed; it still removes every sample.
        """
        now = time.time()
        with self.connect() as conn:
            conn.execute("UPDATE sessions SET withdrawn_at = ? WHERE session_id = ?",
                         (now, session_id))
            conn.execute("UPDATE consents SET withdrawn_at = ? WHERE session_id = ?",
                         (now, session_id))
            conn.execute("DELETE FROM samples WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            if erase:
                conn.execute("DELETE FROM consents WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        return {"session_id": session_id, "withdrawn_at": now, "erased": erase}

    # -- samples and events ------------------------------------------------

    def append_samples(self, session_id: str, samples: List[Dict]) -> None:
        if not samples:
            return
        with self.connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO samples (session_id, t, payload) VALUES (?,?,?)",
                [(session_id, float(s["t"]), json.dumps(s)) for s in samples],
            )

    def get_samples(self, session_id: str) -> List[Dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM samples WHERE session_id = ? ORDER BY t",
                (session_id,)).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def add_event(self, session_id: str, t: float, label: str) -> Dict:
        with self.connect() as conn:
            conn.execute("INSERT INTO events (session_id, t, label) VALUES (?,?,?)",
                         (session_id, float(t), label))
        return {"t": float(t), "label": label}

    def get_events(self, session_id: str) -> List[Dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT t, label FROM events WHERE session_id = ? ORDER BY t",
                (session_id,)).fetchall()
        return [{"t": r["t"], "label": r["label"]} for r in rows]

    def study_sessions(self, study_id: str, include_withdrawn: bool = False) -> List[Dict]:
        sql = "SELECT session_id FROM sessions WHERE study_id = ?"
        if not include_withdrawn:
            sql += " AND withdrawn_at IS NULL"
        with self.connect() as conn:
            rows = conn.execute(sql + " ORDER BY created_at", (study_id,)).fetchall()
        return [self.get_session(r["session_id"]) for r in rows]

    # -- retention ---------------------------------------------------------

    def purge_expired(self, now: Optional[float] = None) -> Dict[str, int]:
        """Delete sessions past their study's retention window.

        Retention that is written down but never executed is a policy, not a
        control. This is meant to be called on a schedule.
        """
        now = now if now is not None else time.time()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT s.session_id FROM sessions s JOIN studies st"
                " ON s.study_id = st.study_id"
                " WHERE s.created_at < ? - (st.retention_days * 86400)", (now,)
            ).fetchall()
            ids = [r["session_id"] for r in rows]
            for sid in ids:
                conn.execute("DELETE FROM samples WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM events WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM consents WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
        return {"purged_sessions": len(ids)}

    def export_session(self, session_id: str) -> Dict:
        """Everything held about one session, for a participant access request."""
        return {
            "session": self.get_session(session_id),
            "consent": self.get_consent(session_id),
            "samples": self.get_samples(session_id),
            "events": self.get_events(session_id),
            "note": (
                "This is the complete record. No video, image or per-frame data "
                "is held: features are computed in the participant's browser and "
                "discarded there, and only the once-per-second derived state "
                "reaches the server."
            ),
        }


def _study(row: sqlite3.Row) -> Dict:
    return {"study_id": row["study_id"], "name": row["name"],
            "created_at": row["created_at"], "config": json.loads(row["config"]),
            "retention_days": row["retention_days"], "closed_at": row["closed_at"]}
