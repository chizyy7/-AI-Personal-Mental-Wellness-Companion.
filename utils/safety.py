"""
safety.py
---------
Phase 6 - Safety layer.

Detects language that suggests significant distress and, when it appears,
surfaces real crisis resources in a calm, non-alarming way.

Design notes:
    * This is a *support* tool, not a diagnostic or monitoring system. It errs
      on the side of offering help gently.
    * Detection is intentionally simple and transparent (curated phrases plus
      mood/emotion signals). It will miss things and can over-trigger — so the
      UI always leaves room for the user to dismiss the message.

Public API:
    detect_distress(text, mood_score=None, analysis=None) -> dict
    crisis_resources() -> list[dict]
    safety_message(level) -> str
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Signal phrases
# --------------------------------------------------------------------------- #
# Direct, high-concern language (self-harm / suicide / crisis).
CRISIS_PHRASES = [
    r"\bkill myself\b", r"\bkilling myself\b", r"\bend my life\b",
    r"\bend it all\b", r"\bwant to die\b", r"\bwanna die\b", r"\bwish i (was|were) dead\b",
    r"\bbetter off dead\b", r"\bsuicid", r"\bself[- ]?harm\b", r"\bcut(ting)? myself\b",
    r"\bhurt myself\b", r"\bharm myself\b", r"\bno reason to live\b",
    r"\bdon'?t want to (be here|live|exist)\b", r"\bcan'?t go on\b",
    r"\bdon'?t want to wake up\b", r"\boverdose\b",
]

# Strong-but-less-specific distress language (worth checking in on).
ELEVATED_PHRASES = [
    r"\bhopeless\b", r"\bno hope\b", r"\bworthless\b", r"\bcan'?t take (it|this) anymore\b",
    r"\bfalling apart\b", r"\bbreaking down\b", r"\bno ?one cares\b", r"\bnobody cares\b",
    r"\bfeel trapped\b", r"\btrapped\b", r"\bcompletely numb\b", r"\bfeel nothing\b",
    r"\bgive up\b", r"\bcan'?t cope\b", r"\bat my limit\b", r"\btoo much to bear\b",
]

_CRISIS_RE = [re.compile(p) for p in CRISIS_PHRASES]
_ELEVATED_RE = [re.compile(p) for p in ELEVATED_PHRASES]


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
def crisis_resources() -> List[Dict[str, str]]:
    """Return a short, practical list of crisis resources."""
    return [
        {
            "name": "988 Suicide & Crisis Lifeline (US)",
            "contact": "Call or text 988",
            "url": "https://988lifeline.org/",
            "note": "Free, confidential, 24/7 support.",
        },
        {
            "name": "Crisis Text Line",
            "contact": "Text HOME to 741741",
            "url": "https://www.crisistextline.org/",
            "note": "Text with a trained crisis counselor.",
        },
        {
            "name": "Emergency services",
            "contact": "Call your local emergency number (911 / 112 / 999)",
            "url": "",
            "note": "If you are in immediate danger, please call now.",
        },
        {
            "name": "International resources",
            "contact": "Find a helpline in your country",
            "url": "https://findahelpline.com/",
            "note": "Support is available worldwide.",
        },
    ]


def safety_message(level: str) -> str:
    """A warm, direct message for the given distress level."""
    if level == "high":
        return (
            "What you've shared sounds really heavy, and I'm glad you trusted me "
            "with it. You deserve support right now — more than I can give alone. "
            "Please consider reaching out to one of these free, confidential "
            "services, or someone you trust, today."
        )
    if level == "elevated":
        return (
            "It sounds like things are weighing on you a lot. You don't have to "
            "carry it on your own — talking to someone you trust, or a helpline, "
            "can really help. I'm still here with you."
        )
    return ""


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detect_distress(
    text: str,
    mood_score: Optional[int] = None,
    analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assess distress in a piece of text (plus optional mood/analysis signals).

    Returns:
        {
            "level": "none" | "elevated" | "high",
            "matched": [phrases/terms that triggered],
            "reasons": [human-readable reasons],
        }
    """
    text = (text or "").lower()
    matched: List[str] = []
    reasons: List[str] = []

    crisis_hits = [p.pattern for p in _CRISIS_RE if p.search(text)]
    elevated_hits = [p.pattern for p in _ELEVATED_RE if p.search(text)]

    if crisis_hits:
        matched.extend(crisis_hits)

    # Very low mood is, on its own, a gentle reason to check in.
    low_mood = False
    if mood_score is not None:
        try:
            low_mood = int(mood_score) <= 2
            if low_mood:
                reasons.append("a very low mood check-in")
        except (TypeError, ValueError):
            low_mood = False

    # High emotion intensity with a negative tone is another signal.
    if analysis:
        try:
            if (
                str(analysis.get("sentiment")) == "Negative"
                and float(analysis.get("intensity_score", 0)) >= 0.7
            ):
                reasons.append("very intense difficult emotion in the writing")
        except (TypeError, ValueError):
            pass

    # ---- Decide the level -------------------------------------------------- #
    if crisis_hits:
        level = "high"
    elif elevated_hits or len(reasons) >= 2 or low_mood:
        level = "elevated"
        matched.extend(elevated_hits)
    else:
        level = "none"

    return {"level": level, "matched": matched, "reasons": reasons}


def is_crisis(text: str) -> bool:
    """Quick check: does the text contain direct crisis language?"""
    return detect_distress(text)["level"] == "high"
