"""
recommendations.py
------------------
Phase 6 - Personalized coping suggestions.

Turns a user's challenges, current emotions and detected patterns into a short,
gentle list of coping ideas: breathing, grounding, reframing, small activities,
connection, sleep support, gratitude and self-compassion.

Everything is phrased as an *invitation*, never an instruction. Public API:

    recommend(profile, analysis=None, patterns=None, limit=6) -> list[dict]
    gratitude_prompts(n=3) -> list[str]
    reframing_prompts(keys=None) -> list[dict]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Coping library
# --------------------------------------------------------------------------- #
# Each strategy is tagged so it can be matched to what the user is experiencing.
COPING_LIBRARY: List[Dict[str, Any]] = [
    {
        "id": "box_breathing",
        "category": "Breathing",
        "title": "Box breathing (4-4-4-4)",
        "emoji": "🫁",
        "duration": "2 min",
        "why": "A slow, even breath can calm the body's alarm system.",
        "how": "Breathe in for 4, hold for 4, out for 4, hold for 4. Repeat four times.",
        "tags": ["anxiety", "stress", "panic", "overthinking", "anger"],
    },
    {
        "id": "physiological_sigh",
        "category": "Breathing",
        "title": "The physiological sigh",
        "emoji": "🌬️",
        "duration": "1 min",
        "why": "Two quick inhales and a long exhale helps release built-up tension.",
        "how": "Inhale through the nose, add a second short sniff, then a long slow exhale. Repeat 3–5 times.",
        "tags": ["anxiety", "stress", "anger", "panic", "overwhelm"],
    },
    {
        "id": "5_4_3_2_1",
        "category": "Grounding",
        "title": "5-4-3-2-1 grounding",
        "emoji": "🌳",
        "duration": "3 min",
        "why": "Bringing attention to your senses gently anchors you in the present.",
        "how": "Notice 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
        "tags": ["anxiety", "panic", "overthinking", "burnout", "dissociation"],
    },
    {
        "id": "body_scan",
        "category": "Grounding",
        "title": "A short body scan",
        "emoji": "🧘",
        "duration": "5 min",
        "why": "Noticing where tension lives helps it start to soften.",
        "how": "Slowly move attention from your head to your feet, relaxing each area as you go.",
        "tags": ["stress", "anxiety", "sleep", "burnout", "anger"],
    },
    {
        "id": "name_it",
        "category": "Reframing",
        "title": "Name it to tame it",
        "emoji": "🏷️",
        "duration": "2 min",
        "why": "Putting feelings into words reduces their intensity.",
        "how": "Complete: 'Right now I notice I'm feeling ___ because ___.' No judgment — just noticing.",
        "tags": ["anxiety", "sadness", "anger", "overthinking", "guilt_shame"],
    },
    {
        "id": "reframe_thought",
        "category": "Reframing",
        "title": "Gently question the thought",
        "emoji": "🔄",
        "duration": "5 min",
        "why": "Harsh thoughts are often not the whole truth.",
        "how": "Ask: Is this thought 100% true? What would I tell a friend? What's a kinder, truer version?",
        "tags": ["self_criticism", "overthinking", "anxiety", "guilt_shame", "sadness"],
    },
    {
        "id": "self_compassion",
        "category": "Self-compassion",
        "title": "A kind note to yourself",
        "emoji": "💗",
        "duration": "3 min",
        "why": "Self-criticism tightens; self-kindness creates room to breathe.",
        "how": "Write one sentence you'd say to a dear friend in your situation — then say it to yourself.",
        "tags": ["guilt_shame", "self_criticism", "sadness", "low_self_esteem"],
    },
    {
        "id": "small_step",
        "category": "Activity",
        "title": "One tiny next step",
        "emoji": "🐾",
        "duration": "5 min",
        "why": "Tiny actions break the loop of feeling stuck.",
        "how": "Choose the smallest possible helpful action (fill a glass of water, open a window). That counts.",
        "tags": ["sadness", "burnout", "stress", "focus", "low_mood"],
    },
    {
        "id": "move_body",
        "category": "Activity",
        "title": "A short, gentle walk",
        "emoji": "🚶",
        "duration": "10 min",
        "why": "Movement shifts stuck energy and lifts mood a little.",
        "how": "Even 5–10 minutes outside, at any pace, can change how your body feels.",
        "tags": ["sadness", "stress", "low_mood", "focus", "anger"],
    },
    {
        "id": "connect",
        "category": "Connection",
        "title": "Reach out to one person",
        "emoji": "📞",
        "duration": "5 min",
        "why": "Connection softens loneliness and lightens the load.",
        "how": "Send a simple message to someone safe — 'thinking of you' is enough.",
        "tags": ["loneliness", "sadness", "grief", "isolation"],
    },
    {
        "id": "wind_down",
        "category": "Sleep",
        "title": "A calm wind-down",
        "emoji": "🌙",
        "duration": "20 min",
        "why": "Giving your mind a gentle off-ramp helps sleep come.",
        "how": "Dim the lights, lower screens, and do something quiet and repetitive before bed.",
        "tags": ["sleep", "anxiety", "overthinking", "stress"],
    },
    {
        "id": "worry_window",
        "category": "Reframing",
        "title": "A 'worry window'",
        "emoji": "🕰️",
        "duration": "15 min",
        "why": "Giving worries a set time can keep them from taking over the day.",
        "how": "Set 10–15 minutes to write worries down. Outside that window, gently note 'later'.",
        "tags": ["overthinking", "anxiety", "stress", "sleep"],
    },
    {
        "id": "gratitude_three",
        "category": "Gratitude",
        "title": "Three small good things",
        "emoji": "🌻",
        "duration": "3 min",
        "why": "Noticing small positives gently widens attention.",
        "how": "Name three ordinary things that went okay or felt good today — however small.",
        "tags": ["sadness", "low_mood", "guilt_shame", "burnout"],
    },
    {
        "id": "gratitude_person",
        "category": "Gratitude",
        "title": "A quiet thank-you",
        "emoji": "🤝",
        "duration": "3 min",
        "why": "Gratitude toward others can warm how we feel inside.",
        "how": "Think of someone who helped you recently, and silently thank them — or tell them.",
        "tags": ["loneliness", "sadness", "gratitude"],
    },
]


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def _challenge_tags(profile: Dict[str, Any]) -> set:
    """Map the user's chosen challenges onto coping-library tags."""
    tags: set = set()
    for challenge in (profile or {}).get("challenges", []) or []:
        c = str(challenge).lower()
        if "anxiety" in c:
            tags.update({"anxiety", "overthinking"})
        if "stress" in c:
            tags.add("stress")
        if "overthink" in c:
            tags.update({"overthinking", "anxiety"})
        if "low mood" in c or "low" in c:
            tags.update({"sadness", "low_mood"})
        if "sleep" in c:
            tags.add("sleep")
        if "anger" in c or "irrit" in c:
            tags.add("anger")
        if "self-esteem" in c or "self esteem" in c:
            tags.update({"guilt_shame", "self_criticism", "low_self_esteem"})
        if "lonely" in c or "loneliness" in c:
            tags.add("loneliness")
        if "burnout" in c or "burn out" in c:
            tags.update({"stress", "burnout"})
        if "grief" in c:
            tags.update({"sadness", "grief"})
        if "focus" in c or "concentration" in c:
            tags.add("focus")
    return tags


def _analysis_tags(analysis: Optional[Dict[str, Any]]) -> set:
    """Use the emotions found in a piece of writing as matching tags."""
    if not analysis:
        return set()
    return set((analysis.get("emotions") or {}).keys())


def _pattern_tags(patterns: Optional[List[Dict[str, Any]]]) -> set:
    """Map detected pattern kinds onto coping tags."""
    tags: set = set()
    for p in patterns or []:
        kind = str(p.get("kind", ""))
        title = str(p.get("title", "")).lower()
        if kind == "thought" and "self" in title:
            tags.add("self_criticism")
        if kind == "thought":
            tags.add("overthinking")
        if kind == "trend" and "down" in title:
            tags.update({"sadness", "low_mood"})
    return tags


def recommend(
    profile: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    patterns: Optional[List[Dict[str, Any]]] = None,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    """
    Return up to `limit` coping suggestions, best-matched first.

    Matching blends the user's challenges, current emotions and detected
    patterns, and keeps category variety so the list never feels repetitive.
    """
    wanted = _challenge_tags(profile) | _analysis_tags(analysis) | _pattern_tags(patterns)

    scored: List[Dict[str, Any]] = []
    for item in COPING_LIBRARY:
        overlap = len(set(item["tags"]) & wanted)
        # A tiny base score keeps universally-helpful ideas in the mix.
        score = overlap * 2 + 0.5
        if overlap == 0:
            score = 0.4
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    # Enforce category variety (max 2 per category).
    per_category: Dict[str, int] = {}
    results: List[Dict[str, Any]] = []
    for score, item in scored:
        cat = item["category"]
        if per_category.get(cat, 0) >= 2:
            continue
        per_category[cat] = per_category.get(cat, 0) + 1
        results.append({**item, "score": round(score, 2)})
        if len(results) >= limit:
            break
    return results


# --------------------------------------------------------------------------- #
# Gratitude & reframing helpers
# --------------------------------------------------------------------------- #
GRATITUDE_PROMPTS = [
    "What's one small thing that made today feel a little lighter?",
    "Who or what are you quietly thankful for right now?",
    "What's something your body or mind did for you today?",
    "What's an ordinary moment you'd like to remember from this week?",
    "What's something you have now that you once hoped for?",
]


def gratitude_prompts(n: int = 3) -> List[str]:
    """Return up to `n` gentle gratitude prompts."""
    return GRATITUDE_PROMPTS[: max(1, n)]


# Reframes keyed by the thought-pattern keys used in emotion_analyzer.
REFRAMES: Dict[str, Dict[str, str]] = {
    "catastrophizing": {
        "label": "When your mind jumps to the worst case",
        "reframe": "Ask: what's the most *likely* outcome — and what's one thing that's still okay right now?",
    },
    "self_criticism": {
        "label": "When the inner critic gets loud",
        "reframe": "Swap 'I'm a failure' for 'This is hard, and I'm doing my best.' Speak to yourself as a friend.",
    },
    "overgeneralization": {
        "label": "When everything feels like 'always'",
        "reframe": "Try 'lately' or 'sometimes' instead of 'always' or 'never' — it's usually truer and kinder.",
    },
    "black_and_white": {
        "label": "When it feels all-or-nothing",
        "reframe": "Swap 'I must' for 'I'd like to', and allow for a messy middle.",
    },
    "mind_reading": {
        "label": "When you assume what others think",
        "reframe": "Separate what you *know* from what you're *assuming*, and give yourself the benefit of the doubt.",
    },
    "fortune_telling": {
        "label": "When the future feels already decided",
        "reframe": "The future isn't written. What's one small step you can take today, however it turns out?",
    },
}


def reframing_prompts(keys: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Return reframing prompts — either specific to the given thought-pattern
    keys, or a gentle default set.
    """
    if keys:
        chosen = [REFRAMES[k] for k in keys if k in REFRAMES]
        if chosen:
            return chosen
    # Default: a couple of universally helpful reframes.
    return [REFRAMES["self_criticism"], REFRAMES["catastrophizing"]]
