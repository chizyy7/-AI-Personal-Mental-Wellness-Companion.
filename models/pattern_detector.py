"""
pattern_detector.py
-------------------
Phase 4 - Emotional Pattern Detection.

Looks across the user's mood logs and journal entries for gentle, repeatable
patterns worth noticing, e.g.:

    * "Your Sundays tend to feel lower than the rest of the week."
    * "Your mood has been drifting upward over the last couple of weeks."
    * "Stress comes up often in your writing."
    * "Your entries seem to gather into a couple of moods: calm days and
       anxious days."          (unsupervised clustering with scikit-learn)

Every pattern is returned as a dict so the UI can present it plainly and
non-judgmentally:

    {
        "kind": "weekday" | "trend" | "emotion" | "thought" | "cluster",
        "title": short human sentence,
        "explanation": a warm sentence or two,
        "strength": 0.0-1.0  (how confident/notable),
        "emoji": a small emoji for the card,
    }
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.helpers import load_journal_entries, load_mood_logs

WEEKDAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]

EMOTION_DISPLAY = {
    "anxiety": "Anxiety", "stress": "Stress", "sadness": "Sadness",
    "anger": "Anger", "loneliness": "Loneliness", "guilt_shame": "Guilt / Shame",
    "gratitude": "Gratitude", "calmness": "Calmness", "happiness": "Happiness",
    "hope": "Hope",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_json(value, default):
    """Decode a JSON cell that may hold a list/dict; return default on failure."""
    if not value or isinstance(value, float):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _mood_with_weekday() -> pd.DataFrame:
    """Mood logs with parsed dates and a weekday column."""
    df = load_mood_logs()
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["mood_score"] = pd.to_numeric(df["mood_score"], errors="coerce")
    df = df.dropna(subset=["date", "mood_score"])
    df["weekday"] = df["date"].dt.day_name()
    df["dow"] = df["date"].dt.dayofweek
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Individual detectors
# --------------------------------------------------------------------------- #
def detect_weekday_patterns(mood: pd.DataFrame) -> List[Dict[str, object]]:
    """Find weekdays that reliably run higher or lower than the user's average."""
    patterns: List[Dict[str, object]] = []
    if len(mood) < 7 or mood["date"].dt.normalize().nunique() < 7:
        return patterns

    overall = float(mood["mood_score"].mean())
    grouped = mood.groupby("weekday")["mood_score"].agg(["mean", "count"])

    for day in WEEKDAYS:
        if day not in grouped.index:
            continue
        mean = float(grouped.loc[day, "mean"])
        count = int(grouped.loc[day, "count"])
        diff = mean - overall
        # Require at least 2 samples and a noticeable gap (±1.2 points).
        if count >= 2 and abs(diff) >= 1.2:
            direction = "lower" if diff < 0 else "lighter"
            strength = round(min(abs(diff) / 3.0, 1.0), 2)
            patterns.append(
                {
                    "kind": "weekday",
                    "title": f"Your {day}s tend to feel {direction}.",
                    "explanation": (
                        f"Across {count} {day} check-in(s), your mood averaged "
                        f"{mean:.1f}/10 versus {overall:.1f}/10 on other days. "
                        "It might be worth noticing what tends to happen around "
                        "then — and being extra gentle with yourself."
                    ),
                    "strength": strength,
                    "emoji": "📅",
                }
            )
    return patterns


def detect_trend(mood: pd.DataFrame) -> List[Dict[str, object]]:
    """Detect a meaningful upward/downward drift in mood over time."""
    patterns: List[Dict[str, object]] = []
    if len(mood) < 5:
        return patterns

    x = np.arange(len(mood))
    y = mood["mood_score"].to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])  # points per check-in

    if abs(slope) < 0.08:  # roughly flat over the window
        return patterns

    direction = "upward" if slope > 0 else "downward"
    strength = round(min(abs(slope) / 0.4, 1.0), 2)
    if slope > 0:
        explanation = (
            "Lately your check-ins have been climbing. Something seems to be "
            "helping — see if you can notice what it is and lean into it. 🌱"
        )
    else:
        explanation = (
            "Your mood has been drifting downward recently. That's worth "
            "listening to. You deserve support, and small caring steps count. 💙"
        )
    patterns.append(
        {
            "kind": "trend",
            "title": f"Your mood has been trending {direction}.",
            "explanation": explanation,
            "strength": strength,
            "emoji": "📈" if slope > 0 else "📉",
        }
    )
    return patterns


def detect_emotion_patterns(journal: pd.DataFrame) -> List[Dict[str, object]]:
    """Find emotions that recur across the user's journal writing."""
    patterns: List[Dict[str, object]] = []
    if journal.empty:
        return patterns

    counter: Counter = Counter()
    for _, row in journal.iterrows():
        scores = _safe_json(row.get("emotion_scores"), {})
        if isinstance(scores, dict) and scores:
            # Count the strongest emotion in each entry.
            top = max(scores, key=lambda k: scores[k])
            counter[top] += 1
        else:
            label = str(row.get("emotion") or "").strip()
            if label:
                counter[label] += 1

    total = sum(counter.values())
    if total < 3:
        return patterns

    for emotion, count in counter.most_common(3):
        if count < 2:
            continue
        share = count / total
        if share < 0.34:  # only call it a "pattern" if it clearly recurs
            continue
        name = EMOTION_DISPLAY.get(emotion, emotion.title())
        strength = round(min(share + 0.2, 1.0), 2)
        patterns.append(
            {
                "kind": "emotion",
                "title": f"{name} shows up often in your writing.",
                "explanation": (
                    f"{name} was the strongest feeling in {count} of your {total} "
                    "entries. Noticing a recurring thread is the first gentle "
                    "step toward understanding it."
                ),
                "strength": strength,
                "emoji": "🎭",
            }
        )
    return patterns


def detect_thought_patterns(journal: pd.DataFrame) -> List[Dict[str, object]]:
    """Surface thought patterns that keep returning in the journal."""
    patterns: List[Dict[str, object]] = []
    if journal.empty:
        return patterns

    counter: Counter = Counter()
    for _, row in journal.iterrows():
        found = _safe_json(row.get("patterns"), [])
        for item in found:
            if isinstance(item, dict):
                label = item.get("label")
            else:
                label = str(item)
            if label:
                counter[label] += 1

    for label, count in counter.most_common(2):
        if count < 2:
            continue
        patterns.append(
            {
                "kind": "thought",
                "title": f'"{label}" keeps appearing.',
                "explanation": (
                    f"This thinking pattern showed up {count} times. It's very "
                    "common and very human — and simply naming it can loosen "
                    "its grip a little. 🌿"
                ),
                "strength": round(min(0.4 + count * 0.15, 1.0), 2),
                "emoji": "🧩",
            }
        )
    return patterns


def detect_clusters(mood: pd.DataFrame, journal: pd.DataFrame) -> List[Dict[str, object]]:
    """
    Use scikit-learn KMeans to see whether the user's days fall into distinct
    mood 'clusters' — e.g. a bright cluster and a heavy cluster.
    """
    patterns: List[Dict[str, object]] = []
    if len(mood) < 6 or mood["date"].dt.normalize().nunique() < 6:
        return patterns

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except Exception:  # noqa: BLE001 - sklearn optional at runtime
        return patterns

    try:
        features = mood[["mood_score", "dow"]].to_numpy(dtype=float)
        if np.unique(features[:, 0]).size < 2:
            return patterns

        scaled = StandardScaler().fit_transform(features)
        k = 2 if len(mood) < 12 else 3
        k = min(k, len(mood))
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled)

        temp = mood.copy()
        temp["cluster"] = labels
        centers = temp.groupby("cluster")["mood_score"].mean().sort_values()

        if centers.empty:
            return patterns

        low = float(centers.iloc[0])
        high = float(centers.iloc[-1])
        if high - low < 1.5:
            return patterns

        low_share = float((temp["cluster"] == centers.index[0]).mean())
        high_share = float((temp["cluster"] == centers.index[-1]).mean())

        patterns.append(
            {
                "kind": "cluster",
                "title": "Your days group into a lower mood and a brighter mood.",
                "explanation": (
                    f"Roughly {low_share:.0%} of your check-ins sit around "
                    f"{low:.1f}/10 and {high_share:.0%} sit around {high:.1f}/10. "
                    "Two different 'weathers' — knowing which one you're in can "
                    "make it easier to respond with care. ☀️🌧️"
                ),
                "strength": round(min((high - low) / 5.0 + 0.3, 1.0), 2),
                "emoji": "🔵",
            }
        )
    except Exception:  # noqa: BLE001 - clustering must never break the app
        return patterns
    return patterns


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def detect_patterns() -> List[Dict[str, object]]:
    """
    Run every detector and return patterns sorted by strength (strongest first).
    """
    mood = _mood_with_weekday()
    journal = load_journal_entries()

    patterns: List[Dict[str, object]] = []
    patterns += detect_weekday_patterns(mood)
    patterns += detect_trend(mood)
    patterns += detect_emotion_patterns(journal)
    patterns += detect_thought_patterns(journal)
    patterns += detect_clusters(mood, journal)

    patterns.sort(key=lambda p: float(p.get("strength", 0.0)), reverse=True)
    return patterns


def has_enough_data() -> bool:
    """True when there's at least a little history to look for patterns in."""
    mood = load_mood_logs()
    journal = load_journal_entries()
    return len(mood) >= 3 or len(journal) >= 3
