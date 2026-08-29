"""Test isolation.

The API store defaults to `data/neuroproxy.db`. Without this, running the suite
wrote studies and sessions into the same database a researcher would be using,
and they showed up in the dashboard afterwards. Point every run at a temporary
file instead.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Set before `api.main` is imported anywhere, since it builds a Store at import.
_TMP = Path(tempfile.mkdtemp(prefix="neuroproxy-test-")) / "test.db"
os.environ["NEUROPROXY_DB"] = str(_TMP)
