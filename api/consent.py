"""The consent notice shown to participants, versioned.

A consent record that cannot reproduce what the person was shown records
nothing. The exact text is stored with each grant, along with its version, so
a later change to this file cannot rewrite what someone already agreed to.

The notice below describes what the system actually does. Every claim in it is
checkable against the code:

* "video never leaves your device" -- features are computed in the page
  (apps/web_demo/extractor.js) and only those are sent; the server has no
  code path that writes an image (api/store.py).
* "about one measurement per second" -- the engine emits at 1 Hz
  (neuroproxy/inference/engine.py).
* "it will often decline to answer" -- on real recordings roughly half of
  sessions yield little (docs/limitations.md 15-16). Telling participants that
  up front is more honest than a progress bar that implies success.

If the product changes, the notice changes and the version increments. Editing
the text without bumping the version would silently misattribute consent.
"""
from __future__ import annotations

from typing import Dict, List

NOTICE_VERSION = "2026-08-29.1"

SCOPES: Dict[str, str] = {
    "camera": "Use your camera for the length of this session.",
    "derived_state": (
        "Record the measurements derived from it -- heart rate, how far it sits "
        "from your own calm baseline, and a signal-quality score -- about once "
        "per second."
    ),
    "research_use": (
        "Let the researcher running this study analyse those measurements, "
        "together with those of other participants."
    ),
}

NOTICE_TEXT = """\
What this measures

Your camera is used to estimate your heart rate from very small colour changes
in the skin of your face. From that, the system reports how far your heart rate
sits from your own calm baseline, measured during the first 45 seconds.

It is not a medical device and gives no diagnosis. It reports a physiological
response, not a thought, mood or emotion.

What leaves your device

Video does not. The measurements are computed inside this page and only the
results are sent -- roughly one hundred bytes per frame, no images. The server
has no way to store video because it never receives any.

What is kept

Your heart rate, its deviation from your baseline, a signal-quality score, and
any events the researcher marks. Nothing that identifies you: no name, no email,
no address. Your session is labelled with a random code.

How long

For this study's retention period, after which it is deleted automatically. You
can withdraw at any time, during or after the session, and withdrawing erases
the measurements rather than hiding them.

What it will not do

It will often decline to answer. When the light is poor, when you move, or when
too little clear skin is visible, it reports that it cannot measure rather than
guessing. On typical recordings this happens for a substantial share of the
time, and that is the system working correctly.

Your choices

Taking part is voluntary and you may stop at any point. To withdraw afterwards,
use the session code shown to you at the end.
"""


def notice() -> Dict[str, object]:
    """The full notice, for display and for the consent record."""
    return {
        "version": NOTICE_VERSION,
        "text": NOTICE_TEXT,
        "scopes": [{"key": k, "description": v} for k, v in SCOPES.items()],
        # Every scope is required: there is no partial mode where the camera is
        # used but the measurements are not recorded. Offering a checkbox that
        # cannot actually be declined separately would be theatre.
        "all_required": True,
    }


def validate(granted: List[str]) -> List[str]:
    """Raise unless every scope was granted; return them normalised."""
    missing = [k for k in SCOPES if k not in set(granted or [])]
    if missing:
        raise ValueError(
            "consent requires all scopes; missing: {}".format(", ".join(missing))
        )
    return sorted(SCOPES)
