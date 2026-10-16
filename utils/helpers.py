"""
helpers.py
----------
Shared, low-level utilities for MindfulMe.

This module is intentionally free of any Streamlit imports so the model and
utility layers stay testable and reusable. Everything here is plain Python /
Pandas so it can be called from anywhere in the app.

Phase 1 responsibilities:
    * Resolve project paths (data/ directory, profile file, etc.)
    * Load and save the user profile (JSON persistence)
    * Small defensive helpers (safe JSON read/write, defaults)
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# BASE_DIR is the mindfulme/ project root (the folder that contains app.py).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.json")
MOOD_LOGS_PATH = os.path.join(DATA_DIR, "mood_logs.csv")
JOURNAL_PATH = os.path.join(DATA_DIR, "journal_entries.csv")


def ensure_data_dir() -> None:
    """Make sure the data directory exists before any read/write happens."""
    os.makedirs(DATA_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Generic JSON helpers
# --------------------------------------------------------------------------- #
def read_json(path: str, default: Any = None) -> Any:
    """Read a JSON file, returning `default` if it is missing or corrupt."""
    ensure_data_dir()
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # A corrupt file should never crash the app; fall back gracefully.
        return default


def write_json(path: str, payload: Any) -> bool:
    """Atomically-ish write JSON to disk. Returns True on success."""
    ensure_data_dir()
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# User profile
# --------------------------------------------------------------------------- #
# The canonical shape of a profile. Any missing keys get back-filled so the
# rest of the app can always rely on the same schema.
DEFAULT_PROFILE: Dict[str, Any] = {
    "name": "",
    "age": None,
    "challenges": [],
    "support_style": "Gentle",
    "onboarded": False,
    "theme": "Light",  # remembered UI preference (Phase 7)
    "created_at": None,
    "updated_at": None,
}


def load_profile() -> Dict[str, Any]:
    """
    Load the saved profile from disk and merge it over the defaults.

    Returns an empty (but complete) profile if nothing has been saved yet.
    """
    stored = read_json(PROFILE_PATH, default={}) or {}
    profile = {**DEFAULT_PROFILE, **stored}
    # Defensive: ensure list/None types are sane even if the file was tampered.
    if not isinstance(profile.get("challenges"), list):
        profile["challenges"] = []
    return profile


def save_profile(profile: Dict[str, Any]) -> bool:
    """
    Persist the profile, stamping created_at / updated_at timestamps.

    Returns True if the write succeeded.
    """
    now = datetime.now().isoformat(timespec="seconds")
    merged = {**DEFAULT_PROFILE, **profile}
    merged["updated_at"] = now
    if not merged.get("created_at"):
        merged["created_at"] = now
    merged["onboarded"] = True
    return write_json(PROFILE_PATH, merged)


def is_onboarded() -> bool:
    """True only when a profile exists on disk and is marked as onboarded."""
    profile = load_profile()
    return bool(profile.get("onboarded")) and bool(profile.get("name"))


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def greeting_for_now() -> str:
    """Return a time-appropriate greeting word (Morning / Afternoon / Evening)."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def today_str() -> str:
    """ISO date string for 'today' (YYYY-MM-DD)."""
    return datetime.now().strftime("%Y-%m-%d")


def now_iso() -> str:
    """Current local timestamp as an ISO string (seconds precision)."""
    return datetime.now().isoformat(timespec="seconds")


def _new_id() -> int:
    """A sortable, collision-resistant integer id (epoch milliseconds)."""
    return int(time.time() * 1000)


# --------------------------------------------------------------------------- #
# Mood scale
# --------------------------------------------------------------------------- #
# Maps each 1-10 mood score to an emoji and a short, plain-language label.
MOOD_SCALE: Dict[int, Dict[str, str]] = {
    1: {"emoji": "😢", "label": "Very low"},
    2: {"emoji": "😞", "label": "Very low"},
    3: {"emoji": "😔", "label": "Low"},
    4: {"emoji": "😕", "label": "Low"},
    5: {"emoji": "😐", "label": "Okay"},
    6: {"emoji": "🙂", "label": "Okay"},
    7: {"emoji": "😊", "label": "Good"},
    8: {"emoji": "😄", "label": "Good"},
    9: {"emoji": "😁", "label": "Great"},
    10: {"emoji": "🤩", "label": "Great"},
}


def mood_emoji(score: int) -> str:
    """Return the emoji for a 1-10 mood score (safe for out-of-range input)."""
    score = max(1, min(10, int(score)))
    return MOOD_SCALE[score]["emoji"]


def mood_label(score: int) -> str:
    """Return the short label ('Good', 'Low', ...) for a mood score."""
    score = max(1, min(10, int(score)))
    return MOOD_SCALE[score]["label"]


# --------------------------------------------------------------------------- #
# CSV persistence (mood logs + journal entries)
# --------------------------------------------------------------------------- #
MOOD_COLUMNS = ["id", "timestamp", "date", "mood_score", "mood_label", "note"]
# Journal entries carry optional analysis fields (populated in Phase 3).
JOURNAL_COLUMNS = [
    "id",
    "timestamp",
    "date",
    "title",
    "content",
    "sentiment",
    "emotion",
    "intensity",
    "patterns",
    "emotion_scores",
]


def _ensure_csv(path: str, columns: List[str]) -> None:
    """Create a CSV with just a header row if it doesn't exist yet."""
    ensure_data_dir()
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)


def _read_csv(path: str, columns: List[str]) -> pd.DataFrame:
    """Read a CSV safely, guaranteeing the expected columns and dtypes."""
    _ensure_csv(path, columns)
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        df = pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    return df[columns]


def _append_csv(path: str, columns: List[str], row: Dict[str, Any]) -> bool:
    """Append a single row to a CSV without rewriting the whole file."""
    _ensure_csv(path, columns)
    try:
        pd.DataFrame([row], columns=columns).to_csv(
            path, mode="a", header=False, index=False
        )
        return True
    except OSError:
        return False


# ---- Mood logs ------------------------------------------------------------ #
def load_mood_logs() -> pd.DataFrame:
    """Return all mood logs sorted oldest → newest."""
    df = _read_csv(MOOD_LOGS_PATH, MOOD_COLUMNS)
    if not df.empty:
        df["mood_score"] = pd.to_numeric(df["mood_score"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def save_mood_log(score: int, note: str = "") -> bool:
    """
    Save today's mood check-in.

    Only one check-in per day is kept: if today already has a log, it is
    updated in place (so the user can refine how they feel). Returns True on
    success.
    """
    ensure_data_dir()
    df = load_mood_logs()
    today = today_str()
    score = max(1, min(10, int(score)))
    if df.empty and not os.path.exists(MOOD_LOGS_PATH):
        df = pd.DataFrame(columns=MOOD_COLUMNS)
    for col in MOOD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[MOOD_COLUMNS]

    row = {
        "id": _new_id(),
        "timestamp": now_iso(),
        "date": today,
        "mood_score": score,
        "mood_label": mood_label(score),
        "note": (note or "").strip(),
    }

    existing = df.index[df["date"] == today].tolist()
    try:
        if existing:
            idx = existing[-1]
            row["id"] = df.at[idx, "id"]  # keep the original id stable
            for key, value in row.items():
                df.at[idx, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(MOOD_LOGS_PATH, index=False)
        return True
    except OSError:
        return False


def get_today_mood() -> Optional[Dict[str, Any]]:
    """Return today's mood log as a dict, or None if not checked in yet."""
    df = load_mood_logs()
    if df.empty:
        return None
    today_rows = df[df["date"] == today_str()]
    if today_rows.empty:
        return None
    return today_rows.iloc[-1].to_dict()


# ---- Journal entries ------------------------------------------------------ #
def load_journal_entries() -> pd.DataFrame:
    """Return all journal entries sorted newest → oldest."""
    df = _read_csv(JOURNAL_PATH, JOURNAL_COLUMNS)
    if not df.empty:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def save_journal_entry(
    title: str, content: str, analysis: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Persist a new journal entry with the current timestamp.

    `analysis` (Phase 3) is an optional dict of flattened analysis fields; if
    omitted the analysis columns are written blank and can be filled later with
    `update_journal_analysis`.
    """
    row = {
        "id": _new_id(),
        "timestamp": now_iso(),
        "date": today_str(),
        "title": (title or "").strip(),
        "content": (content or "").strip(),
        "sentiment": "",
        "emotion": "",
        "intensity": "",
        "patterns": "",
        "emotion_scores": "",
    }
    if analysis:
        for key in ("sentiment", "emotion", "intensity", "patterns", "emotion_scores"):
            if key in analysis:
                row[key] = analysis[key]
    return _append_csv(JOURNAL_PATH, JOURNAL_COLUMNS, row)


def update_journal_analysis(entry_id: int, analysis: Dict[str, Any]) -> bool:
    """Store/replace the analysis fields for an existing journal entry."""
    df = load_journal_entries()
    if df.empty:
        return False
    try:
        target = int(entry_id)
    except (TypeError, ValueError):
        return False

    mask = pd.to_numeric(df["id"], errors="coerce") == target
    if not mask.any():
        return False

    for key in ("sentiment", "emotion", "intensity", "patterns", "emotion_scores"):
        value = analysis.get(key, "")
        df.loc[mask, key] = json.dumps(value, ensure_ascii=False) if isinstance(
            value, (list, dict)
        ) else value
    # Update only the matching row but write the whole file back.
    return _write_csv_safe(df, JOURNAL_PATH)


def _write_csv_safe(df: pd.DataFrame, path: str) -> bool:
    """Write a DataFrame to CSV, returning False instead of raising on error."""
    try:
        df.to_csv(path, index=False)
        return True
    except OSError:
        return False


def delete_journal_entry(entry_id: int) -> bool:
    """Delete a journal entry by id. Returns True if a row was removed."""
    df = load_journal_entries()
    if df.empty:
        return False
    try:
        target = int(entry_id)
    except (TypeError, ValueError):
        return False
    before = len(df)
    df = df[pd.to_numeric(df["id"], errors="coerce") != target]
    try:
        df.to_csv(JOURNAL_PATH, index=False)
    except OSError:
        return False
    return len(df) < before


def clear_all_data() -> bool:
    """Reset every stored file: mood logs, journal, profile and the RAG index."""
    ensure_data_dir()
    try:
        pd.DataFrame(columns=MOOD_COLUMNS).to_csv(MOOD_LOGS_PATH, index=False)
        pd.DataFrame(columns=JOURNAL_COLUMNS).to_csv(JOURNAL_PATH, index=False)
        write_json(PROFILE_PATH, dict(DEFAULT_PROFILE))
        import shutil

        shutil.rmtree(os.path.join(DATA_DIR, "chroma"), ignore_errors=True)
        return True
    except OSError:
        return False


def pretty_timestamp(ts: str) -> str:
    """Turn a stored ISO timestamp into something friendly like 'Oct 16, 2026 · 4:10 PM'."""
    try:
        dt = datetime.fromisoformat(str(ts))
        return dt.strftime("%b %d, %Y · %I:%M %p").replace(" 0", " ")
    except (ValueError, TypeError):
        return str(ts)
