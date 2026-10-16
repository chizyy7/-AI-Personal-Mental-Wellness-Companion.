"""
insights.py
-----------
Phase 6 - Weekly Emotional Report.

Gathers the last 7 days (configurable) of mood check-ins and journal entries and
turns them into a warm, plain-language report:

    * how many check-ins and entries, average mood, and change vs. the week before
    * the brightest and heaviest days
    * the emotions and thought patterns that showed up most
    * a few personalized coping suggestions
    * a short, gentle narrative summary

Public API:
    weekly_report(profile, days=7) -> dict
    report_to_markdown(report, profile) -> str
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.helpers import load_journal_entries, load_mood_logs, mood_label
from utils.recommendations import recommend

EMOTION_DISPLAY = {
    "anxiety": "Anxiety", "stress": "Stress", "sadness": "Sadness",
    "anger": "Anger", "loneliness": "Loneliness", "guilt_shame": "Guilt / Shame",
    "gratitude": "Gratitude", "calmness": "Calmness", "happiness": "Happiness",
    "hope": "Hope",
}


def _safe_json(value, default):
    """Decode a JSON cell (list/dict) or return the default."""
    if not value or isinstance(value, float):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _mood_frame() -> pd.DataFrame:
    """Mood logs with parsed dates and numeric scores."""
    df = load_mood_logs()
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["mood_score"] = pd.to_numeric(df["mood_score"], errors="coerce")
    return df.dropna(subset=["date", "mood_score"])


def _journal_frame() -> pd.DataFrame:
    """Journal entries with parsed dates."""
    df = load_journal_entries()
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


def _top_patterns_from_journal(journal: pd.DataFrame) -> List[str]:
    """Most common thought-pattern labels across the period's entries."""
    counter: Counter = Counter()
    for _, row in journal.iterrows():
        for item in _safe_json(row.get("patterns"), []):
            label = item.get("label") if isinstance(item, dict) else str(item)
            if label:
                counter[label] += 1
    return [label for label, _ in counter.most_common(3)]


def _top_emotions_from_journal(journal: pd.DataFrame) -> List[str]:
    """Most common dominant emotions across the period's entries."""
    counter: Counter = Counter()
    for _, row in journal.iterrows():
        scores = _safe_json(row.get("emotion_scores"), {})
        if isinstance(scores, dict) and scores:
            top = max(scores, key=lambda k: scores[k])
            counter[EMOTION_DISPLAY.get(top, top.title())] += 1
        else:
            label = str(row.get("emotion") or "").strip()
            if label:
                counter[label] += 1
    return [name for name, _ in counter.most_common(3)]


def weekly_report(profile: Dict[str, Any], days: int = 7) -> Dict[str, Any]:
    """
    Build the weekly emotional report for the last `days` days.

    Returns a dict with both raw figures and ready-to-display text. Never
    raises — missing data simply yields zeros and gentle wording.
    """
    days = max(1, int(days))
    today = datetime.now()
    start = today - timedelta(days=days)
    prev_start = start - timedelta(days=days)

    report: Dict[str, Any] = {
        "period_start": start.date().isoformat(),
        "period_end": today.date().isoformat(),
        "days": days,
        "checkins": 0,
        "journal_count": 0,
        "avg_mood": None,
        "prev_avg_mood": None,
        "mood_delta": None,
        "best_day": None,
        "worst_day": None,
        "mood_series": [],
        "sentiment_counts": {},
        "top_emotions": [],
        "top_patterns": [],
        "recommendations": [],
        "summary": "",
        "highlights": [],
    }

    # ---- Mood ------------------------------------------------------------- #
    mood = _mood_frame()
    if not mood.empty:
        period = mood[(mood["date"] >= start) & (mood["date"] <= today)]
        prev = mood[(mood["date"] >= prev_start) & (mood["date"] < start)]

        report["checkins"] = int(len(period))
        if not period.empty:
            report["avg_mood"] = round(float(period["mood_score"].mean()), 1)
            best = period.loc[period["mood_score"].idxmax()]
            worst = period.loc[period["mood_score"].idxmin()]
            report["best_day"] = {
                "date": best["date"].strftime("%b %d"),
                "score": int(best["mood_score"]),
                "label": str(best.get("mood_label") or mood_label(int(best["mood_score"]))),
            }
            report["worst_day"] = {
                "date": worst["date"].strftime("%b %d"),
                "score": int(worst["mood_score"]),
                "label": str(worst.get("mood_label") or mood_label(int(worst["mood_score"]))),
            }
            report["mood_series"] = [
                {"date": row["date"].strftime("%b %d"), "score": int(row["mood_score"])}
                for _, row in period.sort_values("date").iterrows()
            ]
        if not prev.empty:
            report["prev_avg_mood"] = round(float(prev["mood_score"].mean()), 1)
        if report["avg_mood"] is not None and report["prev_avg_mood"] is not None:
            report["mood_delta"] = round(report["avg_mood"] - report["prev_avg_mood"], 1)

    # ---- Journal ---------------------------------------------------------- #
    journal = _journal_frame()
    if not journal.empty:
        jperiod = journal[(journal["date"] >= start) & (journal["date"] <= today)]
        report["journal_count"] = int(len(jperiod))
        if not jperiod.empty:
            sentiments = Counter(
                str(s).title() for s in jperiod.get("sentiment", []) if str(s).strip()
            )
            report["sentiment_counts"] = dict(sentiments)
            report["top_emotions"] = _top_emotions_from_journal(jperiod)
            report["top_patterns"] = _top_patterns_from_journal(jperiod)
        combined_journal = jperiod
    else:
        combined_journal = journal

    # ---- Recommendations -------------------------------------------------- #
    # Use the most recent journal analysis (if any) to bias suggestions.
    latest_analysis = None
    if not combined_journal.empty:
        last = combined_journal.sort_values("date").iloc[-1]
        latest_analysis = {
            "emotions": _safe_json(last.get("emotion_scores"), {}),
            "sentiment": str(last.get("sentiment") or ""),
            "intensity_score": 0.0,
        }
    report["recommendations"] = recommend(profile, analysis=latest_analysis, limit=4)

    report["summary"] = _build_summary(report, profile)
    report["highlights"] = _build_highlights(report)
    return report


def _build_summary(report: Dict[str, Any], profile: Dict[str, Any]) -> str:
    """Compose the warm narrative summary of the week."""
    name = (profile or {}).get("name") or "friend"
    checkins = report["checkins"]
    entries = report["journal_count"]

    if checkins == 0 and entries == 0:
        return (
            f"There's nothing to report for the past {report['days']} days yet, "
            f"{name} — and that's okay. Whenever you're ready, a mood check-in "
            "or a few journal lines will let me reflect it back to you. 🌿"
        )

    parts: List[str] = []
    if checkins:
        parts.append(
            f"This week you checked in {checkins} time(s), with an average mood "
            f"of {report['avg_mood']}/10."
        )
        delta = report["mood_delta"]
        if delta is not None:
            if delta > 0.4:
                parts.append(f"That's up about {abs(delta)} points from the week before — something seems to be helping. 🌤️")
            elif delta < -0.4:
                parts.append(f"That's down about {abs(delta)} points from the week before — a gentle week, and worth being kind to yourself about. 💙")
            else:
                parts.append("That's fairly steady compared with the week before.")
    if entries:
        parts.append(f"You also wrote {entries} journal entr{'y' if entries == 1 else 'ies'}.")

    if report["top_emotions"]:
        emotions = ", ".join(report["top_emotions"])
        parts.append(f"The feelings that showed up most were: {emotions}.")
    if report["top_patterns"]:
        parts.append("A thought pattern to stay gently curious about: " + ", ".join(report["top_patterns"]) + ".")

    parts.append("Take only what feels useful, and let the rest go. 💚")
    return " ".join(parts)


def _build_highlights(report: Dict[str, Any]) -> List[str]:
    """A few short, skimmable highlights for the UI."""
    highlights: List[str] = []
    if report.get("best_day"):
        b = report["best_day"]
        highlights.append(f"Brightest day: {b['date']} ({b['score']}/10, {b['label']})")
    if report.get("worst_day"):
        w = report["worst_day"]
        highlights.append(f"Heaviest day: {w['date']} ({w['score']}/10, {w['label']})")
    if report.get("top_emotions"):
        highlights.append("Recurring emotions: " + ", ".join(report["top_emotions"]))
    if report.get("sentiment_counts"):
        counts = report["sentiment_counts"]
        highlights.append(
            "Journal tone: "
            + ", ".join(f"{k} {v}" for k, v in counts.items())
        )
    return highlights


def report_to_markdown(report: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> str:
    """Render the report as Markdown (for display or download)."""
    name = (profile or {}).get("name") or "friend"
    lines = [
        f"# MindfulMe Weekly Report — {name}",
        f"_{report['period_start']} → {report['period_end']} ({report['days']} days)_",
        "",
        report["summary"],
        "",
    ]
    if report["highlights"]:
        lines.append("## Highlights")
        lines += [f"- {h}" for h in report["highlights"]]
        lines.append("")
    if report["recommendations"]:
        lines.append("## Gentle suggestions for the week ahead")
        for item in report["recommendations"]:
            lines.append(f"- {item['emoji']} **{item['title']}** ({item['duration']}) — {item['how']}")
        lines.append("")
    lines.append("_MindfulMe is a supportive companion, not a medical service._")
    return "\n".join(lines)
