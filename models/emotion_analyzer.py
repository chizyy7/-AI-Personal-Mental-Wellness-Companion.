"""
emotion_analyzer.py
-------------------
Phase 3 - Emotion & Thought Pattern Analysis.

Given a piece of free text (a journal entry, or a mood note) this module returns
a gentle, structured reading of it:

    * sentiment            -> "Positive" / "Neutral" / "Negative"
    * emotions             -> {emotion: 0.0-1.0 intensity weight}
    * dominant_emotion     -> strongest emotion (or "calmness")
    * intensity            -> "Low" / "Moderate" / "High"
    * thought_patterns     -> list of possible unhelpful thinking patterns
    * summary / reflection -> warm, non-judgmental wording for the UI

Two engines are supported:

    1. "lexicon"      - a fast, fully offline, deterministic engine built on
                        curated word lists + negation/intensifier logic. This is
                        the DEFAULT and never needs a download, so the app is
                        always responsive and private.
    2. "transformer"  - an optional HuggingFace emotion model for deeper reads.
                        It is only used when explicitly requested and available;
                        if anything fails we silently fall back to the lexicon.

Design goals: robust, dependency-light by default, and emotionally safe. The
analysis is framed as *observations*, never diagnoses.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Lexicons
# --------------------------------------------------------------------------- #
# Each emotion has a set of trigger words. Weights let some words count more
# strongly than others (e.g. "panic" > "tense").
EMOTION_LEXICON: Dict[str, Dict[str, float]] = {
    "anxiety": {
        "anxious": 1.0, "anxiety": 1.0, "worried": 0.8, "worry": 0.8,
        "nervous": 0.8, "panic": 1.2, "panicking": 1.2, "scared": 0.9,
        "afraid": 0.9, "fear": 0.9, "fearful": 0.9, "uneasy": 0.7,
        "tense": 0.7, "dread": 0.9, "racing thoughts": 1.0, "on edge": 0.8,
        "overwhelmed": 0.8, "restless": 0.7, "apprehensive": 0.8,
    },
    "stress": {
        "stressed": 1.0, "stress": 1.0, "pressure": 0.8, "pressured": 0.8,
        "overloaded": 1.0, "burnout": 1.2, "burned out": 1.1, "deadline": 0.6,
        "too much": 0.8, "stretched": 0.7, "swamped": 0.9, "hectic": 0.7,
        "exhausted": 0.7, "drained": 0.7, "tired": 0.4, "workload": 0.5,
    },
    "sadness": {
        "sad": 1.0, "down": 0.7, "unhappy": 0.9, "low": 0.6, "empty": 0.9,
        "hopeless": 1.2, "crying": 1.0, "cried": 1.0, "tears": 0.9,
        "depressed": 1.2, "miserable": 1.1, "gloomy": 0.8, "lost": 0.6,
        "heartbroken": 1.2, "blue": 0.6, "grief": 1.1, "sorrow": 1.0,
        "disappointed": 0.8, "hurt": 0.7,
    },
    "anger": {
        "angry": 1.0, "anger": 1.0, "mad": 0.8, "furious": 1.2, "rage": 1.2,
        "irritated": 0.7, "annoyed": 0.6, "frustrated": 0.9, "resentful": 0.9,
        "upset": 0.6, "fuming": 1.1, "snapped": 0.8, "fed up": 0.8,
    },
    "loneliness": {
        "lonely": 1.0, "alone": 0.7, "isolated": 1.0, "disconnected": 0.9,
        "abandoned": 1.1, "unwanted": 0.9, "left out": 0.8, "no one": 0.7,
        "nobody": 0.6, "withdrawn": 0.8,
    },
    "guilt_shame": {
        "guilty": 1.0, "guilt": 1.0, "ashamed": 1.1, "shame": 1.1,
        "embarrassed": 0.8, "regret": 0.8, "failed": 0.8, "failure": 0.9,
        "my fault": 0.9, "worthless": 1.1, "useless": 0.9,
    },
    "gratitude": {
        "grateful": 1.0, "gratitude": 1.0, "thankful": 1.0, "blessed": 0.9,
        "appreciate": 0.8, "appreciation": 0.8, "thank you": 0.7,
    },
    "calmness": {
        "calm": 1.0, "peaceful": 1.0, "relaxed": 0.9, "content": 0.8,
        "serene": 0.9, "at ease": 0.9, "grounded": 0.8, "settled": 0.7,
        "okay": 0.5, "ok": 0.5, "fine": 0.4, "steady": 0.6, "breathe": 0.4,
    },
    "happiness": {
        "happy": 1.0, "joy": 1.0, "joyful": 1.0, "excited": 0.9,
        "glad": 0.8, "cheerful": 0.9, "delighted": 1.0, "great": 0.7,
        "wonderful": 0.9, "amazing": 0.9, "proud": 0.8, "smiled": 0.7,
        "laugh": 0.7, "laughing": 0.7, "enjoyed": 0.7, "love": 0.8,
    },
    "hope": {
        "hopeful": 1.0, "hope": 0.9, "optimistic": 1.0, "looking forward": 0.9,
        "better days": 0.8, "improving": 0.7, "progress": 0.6, "motivated": 0.8,
    },
}

# Positive vs negative words used for the overall sentiment score.
POSITIVE_WORDS = {
    "good", "great", "happy", "joy", "grateful", "thankful", "calm", "peaceful",
    "relaxed", "content", "proud", "excited", "love", "loved", "wonderful",
    "amazing", "better", "hopeful", "hope", "progress", "enjoyed", "enjoy",
    "beautiful", "kind", "supported", "safe", "warm", "smile", "smiled",
    "accomplished", "productive", "rested", "okay", "fine", "bright", "light",
}
NEGATIVE_WORDS = {
    "bad", "sad", "awful", "terrible", "horrible", "anxious", "worried",
    "stressed", "angry", "mad", "frustrated", "lonely", "empty", "hopeless",
    "tired", "exhausted", "overwhelmed", "scared", "afraid", "hurt", "cry",
    "crying", "depressed", "miserable", "guilty", "ashamed", "worthless",
    "useless", "failure", "hate", "hated", "pain", "painful", "struggle",
    "struggling", "hard", "difficult", "worst", "numb", "broken",
}

# Words that amplify the intensity of nearby emotion words.
INTENSIFIERS = {
    "very", "really", "so", "extremely", "incredibly", "super", "hugely",
    "deeply", "completely", "totally", "utterly", "absolutely", "beyond",
    "unbearably", "desperately",
}
NEGATIONS = {"not", "no", "never", "hardly", "barely", "isn't", "aren't",
             "don't", "doesn't", "didn't", "wasn't", "weren't", "can't",
             "cannot", "won't", "wouldn't", "couldn't", "shouldn't"}


# --------------------------------------------------------------------------- #
# Thought patterns
# --------------------------------------------------------------------------- #
# Each pattern: key -> (label, list of regex triggers, gentle explanation,
#                       a soft reframe to offer the user)
THOUGHT_PATTERNS: Dict[str, Dict[str, object]] = {
    "catastrophizing": {
        "label": "Catastrophizing",
        "regex": [
            r"\bwhat if\b", r"\bworst case\b", r"\bworst possible\b",
            r"\beverything (will|is going to) (fall|go|crash)\b",
            r"\bit'?s all over\b", r"\bcan'?t handle\b",
            r"\bfall apart\b", r"\bruin(ed)?\b", r"\bdisaster\b",
            r"\bend of the world\b",
        ],
        "explanation": (
            "It sounds like your mind may be leaping to the most frightening "
            "possible outcome. That's a very human way to try to protect "
            "yourself."
        ),
        "reframe": (
            "Gently ask: what's the most *likely* outcome, and what's one small "
            "thing that's still okay right now?"
        ),
    },
    "self_criticism": {
        "label": "Self-criticism",
        "regex": [
            r"\bi'?m (so |such a |a )?(stupid|dumb|an idiot|idiot)\b",
            r"\bi hate myself\b", r"\bi'?m worthless\b", r"\bi'?m useless\b",
            r"\bi'?m a failure\b", r"\bi'?m not (good enough|enough)\b",
            r"\bmy fault\b", r"\bi always mess (up|things up)\b",
            r"\bi'?m the problem\b", r"\bi can'?t do anything right\b",
        ],
        "explanation": (
            "There's a harsh inner voice in these words — one that's judging you "
            "much more than a good friend ever would."
        ),
        "reframe": (
            "Try speaking to yourself the way you'd speak to someone you love: "
            "'This is hard, and I'm doing my best.'"
        ),
    },
    "overgeneralization": {
        "label": "Overgeneralization",
        "regex": [
            r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bno ?one\b",
            r"\bnobody\b", r"\beverything\b", r"\bnothing ever\b",
            r"\ball the time\b", r"\bevery time\b",
        ],
        "explanation": (
            "Words like 'always', 'never', and 'everyone' can stretch one "
            "moment into a rule about your whole life."
        ),
        "reframe": (
            "See if you can soften it: 'lately', 'sometimes', or 'in this "
            "situation' often feels truer — and kinder."
        ),
    },
    "black_and_white": {
        "label": "All-or-nothing thinking",
        "regex": [
            r"\bmust\b", r"\bshould\b", r"\bhave to\b", r"\bhave got to\b",
            r"\bperfect\b", r"\ball or nothing\b", r"\beither\b.*\bor\b",
        ],
        "explanation": (
            "Some rigid 'must' and 'should' rules seem to be showing up. Real "
            "life is usually messier — and gentler — than perfect."
        ),
        "reframe": (
            "Swap 'I must' for 'I'd like to' and notice how much lighter it "
            "feels."
        ),
    },
    "mind_reading": {
        "label": "Mind-reading",
        "regex": [
            r"\bthey (think|must think|probably think)\b",
            r"\beveryone thinks\b", r"\bthey hate me\b", r"\bthey'?re judging\b",
            r"\bpeople (think|must think)\b",
        ],
        "explanation": (
            "It looks like you might be guessing what others are thinking. We "
            "rarely know for certain what's in someone else's mind."
        ),
        "reframe": (
            "Consider what you *know* versus what you're *assuming* — and give "
            "yourself the benefit of the doubt."
        ),
    },
    "fortune_telling": {
        "label": "Fortune-telling",
        "regex": [
            r"\bit (will|'ll) never\b", r"\bwon'?t work\b",
            r"\bit'?s going to fail\b", r"\bthere'?s no point\b",
            r"\bthings will never\b", r"\bnothing will\b",
        ],
        "explanation": (
            "Your mind seems to be predicting a negative future as if it were "
            "already decided."
        ),
        "reframe": (
            "The future isn't written yet. What's one tiny step you *can* take "
            "today, regardless of how it turns out?"
        ),
    },
}

# Emotions considered "positive" for summary purposes.
POSITIVE_EMOTIONS = {"happiness", "calmness", "gratitude", "hope"}

EMOTION_DISPLAY = {
    "anxiety": "Anxiety", "stress": "Stress", "sadness": "Sadness",
    "anger": "Anger", "loneliness": "Loneliness",
    "guilt_shame": "Guilt / Shame", "gratitude": "Gratitude",
    "calmness": "Calmness", "happiness": "Happiness", "hope": "Hope",
}


# --------------------------------------------------------------------------- #
# Lexicon engine
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-z']+")


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens used for lexicon matching."""
    return _WORD_RE.findall(text.lower())


def _has_negation_before(tokens: List[str], index: int, window: int = 3) -> bool:
    """True if a negation word appears within `window` tokens before `index`."""
    start = max(0, index - window)
    return any(tok in NEGATIONS for tok in tokens[start:index])


def _has_intensifier_before(tokens: List[str], index: int, window: int = 2) -> bool:
    """True if an intensifier appears within `window` tokens before `index`."""
    start = max(0, index - window)
    return any(tok in INTENSIFIERS for tok in tokens[start:index])


def _lexicon_emotions(text: str) -> Dict[str, float]:
    """Score each emotion using keyword hits, negations, and intensifiers."""
    low = text.lower()
    tokens = _tokenize(text)
    token_index = {tok: i for i, tok in enumerate(tokens)}
    scores: Dict[str, float] = {}

    for emotion, words in EMOTION_LEXICON.items():
        total = 0.0
        for phrase, weight in words.items():
            if " " in phrase:
                # Multi-word phrase: simple substring presence.
                if phrase in low:
                    total += weight
                continue
            # Single word: check for negation + intensifier context.
            if phrase in token_index:
                idx = token_index[phrase]
                value = weight
                if _has_negation_before(tokens, idx):
                    continue  # "not happy" shouldn't count as happiness
                if _has_intensifier_before(tokens, idx):
                    value *= 1.6
                total += value
        if total > 0:
            scores[emotion] = total
    return scores


def _lexicon_sentiment(text: str) -> float:
    """
    Return a sentiment polarity in [-1, 1] based on positive/negative words,
    honouring simple negation.
    """
    tokens = _tokenize(text)
    pos = neg = 0.0
    for i, tok in enumerate(tokens):
        negated = _has_negation_before(tokens, i)
        if tok in POSITIVE_WORDS:
            if negated:
                neg += 1
            else:
                pos += 1
        elif tok in NEGATIVE_WORDS:
            if negated:
                # "not bad" is mildly positive
                pos += 0.5
            else:
                neg += 1
    # Exclamation marks nudge intensity but not direction much.
    if pos == 0 and neg == 0:
        return 0.0
    total = pos + neg
    raw = (pos - neg) / total
    # Dampen: pure-word sentiment is noisy; keep it in a trustable band.
    return max(-1.0, min(1.0, raw * (min(total, 5) / 5)))


# --------------------------------------------------------------------------- #
# Transformer engine (optional)
# --------------------------------------------------------------------------- #
# Maps the 7-emotion model output onto our schema.
_TRANSFORMER_EMOTION_MAP = {
    "anger": "anger", "disgust": "anger", "fear": "anxiety", "joy": "happiness",
    "neutral": "calmness", "sadness": "sadness", "surprise": "anxiety",
}


@lru_cache(maxsize=2)
def _load_transformer(model_name: str):
    """
    Lazily load a HuggingFace text-classification pipeline.

    Returns the pipeline, or None if transformers/torch or the model are
    unavailable. We never let a model-loading problem break the app.
    """
    try:
        from transformers import pipeline  # imported lazily

        return pipeline("text-classification", model=model_name, top_k=None)
    except Exception:  # noqa: BLE001 - any failure -> graceful fallback
        return None


def _transformer_emotions(text: str, model_name: str) -> Optional[Dict[str, float]]:
    """Run the transformer pipeline and map results into our emotion schema."""
    pipe = _load_transformer(model_name)
    if pipe is None:
        return None
    try:
        results = pipe(text[:512])
    except Exception:  # noqa: BLE001
        return None
    # top_k=None returns a list of [{'label':..,'score':..}] (possibly nested).
    if results and isinstance(results[0], list):
        results = results[0]
    scores: Dict[str, float] = {}
    for item in results:
        label = str(item.get("label", "")).lower()
        mapped = _TRANSFORMER_EMOTION_MAP.get(label, label)
        scores[mapped] = scores.get(mapped, 0.0) + float(item.get("score", 0.0))
    return scores or None


# --------------------------------------------------------------------------- #
# Thought-pattern detection
# --------------------------------------------------------------------------- #
def detect_thought_patterns(text: str) -> List[Dict[str, str]]:
    """Return a list of possible unhelpful thought patterns found in `text`."""
    low = text.lower()
    found: List[Dict[str, str]] = []
    for key, spec in THOUGHT_PATTERNS.items():
        matched = False
        snippet = ""
        for pattern in spec["regex"]:  # type: ignore[union-attr]
            m = re.search(pattern, low)
            if m:
                matched = True
                # Capture a short, human-readable snippet around the match.
                start = max(0, m.start() - 28)
                end = min(len(text), m.end() + 28)
                snippet = ("…" if start > 0 else "") + text[start:end].strip() + (
                    "…" if end < len(text) else ""
                )
                break
        if matched:
            found.append(
                {
                    "key": key,
                    "label": str(spec["label"]),
                    "explanation": str(spec["explanation"]),
                    "reframe": str(spec["reframe"]),
                    "snippet": snippet,
                }
            )
    return found


# --------------------------------------------------------------------------- #
# Intensity + summary helpers
# --------------------------------------------------------------------------- #
def _compute_intensity(text: str, emotion_mass: float, sentiment: float) -> Dict[str, object]:
    """Estimate emotional intensity from words, punctuation, and emotion mass."""
    exclamations = text.count("!")
    caps_words = len(re.findall(r"\b[A-Z]{3,}\b", text))
    intensifier_hits = sum(1 for t in _tokenize(text) if t in INTENSIFIERS)
    raw = (
        min(emotion_mass / 3.0, 1.0) * 0.55
        + min(len(text) / 220.0, 1.0) * 0.12
        + min(exclamations / 3.0, 1.0) * 0.12
        + min(caps_words / 2.0, 1.0) * 0.08
        + min(intensifier_hits / 3.0, 1.0) * 0.08
        + abs(sentiment) * 0.15
    )
    score = round(max(0.0, min(1.0, raw)), 2)
    if score < 0.35:
        level = "Low"
    elif score < 0.65:
        level = "Moderate"
    else:
        level = "High"
    return {"level": level, "score": score}


def _sentiment_label(score: float) -> str:
    """Turn a polarity score into a human label."""
    if score > 0.12:
        return "Positive"
    if score < -0.12:
        return "Negative"
    return "Neutral"


def _build_summary(sentiment: str, emotions: Dict[str, float], intensity: str) -> str:
    """Compose one warm, plain-language sentence describing the reading."""
    if not emotions:
        return "This entry feels fairly even — I don't sense a strong emotion tugging at it right now."

    top = max(emotions, key=emotions.get)
    top_name = EMOTION_DISPLAY.get(top, top.title())
    if sentiment == "Positive":
        lead = f"There's a warm, mostly positive tone here, with {top_name.lower()} standing out."
    elif sentiment == "Negative":
        lead = f"This reads as heavy, with {top_name.lower()} standing out most."
    else:
        lead = f"The tone feels mixed, with a thread of {top_name.lower()} running through it."

    if sentiment == "Positive":
        tail = {
            "Low": "A soft bit of light — worth noticing.",
            "Moderate": "It comes through warmly, and it's lovely to see it here.",
            "High": "This is glowing — hold onto this feeling.",
        }[intensity]
    elif sentiment == "Negative":
        tail = {
            "Low": "It doesn't feel overwhelming — just worth noticing.",
            "Moderate": "There's a real weight to it, and it makes sense that you'd feel it.",
            "High": "This carries a lot of feeling. That's a lot to hold — be gentle with yourself.",
        }[intensity]
    else:
        tail = {
            "Low": "It feels fairly steady right now.",
            "Moderate": "There's a gentle pull in more than one direction.",
            "High": "There's a lot moving at once — no wonder it feels full.",
        }[intensity]
    return f"{lead} {tail}"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class EmotionAnalyzer:
    """
    Analyzes text for sentiment, emotions, intensity, and thought patterns.

    Usage:
        analyzer = EmotionAnalyzer()
        result = analyzer.analyze("I'm so stressed about everything...")
    """

    def __init__(self, engine: str = "lexicon", transformer_model: Optional[str] = None):
        # engine: "lexicon" (default) or "transformer"
        self.engine = engine
        self.transformer_model = (
            transformer_model or "j-hartmann/emotion-english-distilroberta-base"
        )

    def analyze(self, text: str) -> Dict[str, object]:
        """
        Analyze `text` and return a structured, gentle reading.

        Never raises: empty or odd input simply yields a neutral result.
        """
        text = (text or "").strip()
        if not text:
            return self._empty_result()

        emotions: Dict[str, float] = {}
        engine_used = "lexicon"

        # 1. Optional transformer read (falls back silently to lexicon).
        if self.engine == "transformer":
            tr = _transformer_emotions(text, self.transformer_model)
            if tr:
                emotions = tr
                engine_used = "transformer"

        # 2. Lexicon read (always computed; also used for intensity/sentiment).
        lex_emotions = _lexicon_emotions(text)
        # Raw (un-normalised) emotion mass drives the intensity estimate.
        raw_emotion_mass = sum(lex_emotions.values())
        if engine_used == "lexicon":
            emotions = lex_emotions
        else:
            # Blend so short/edge inputs still get sensible emotion coverage.
            for k, v in lex_emotions.items():
                emotions[k] = emotions.get(k, 0.0) + v * 0.5

        if not emotions:
            # Nothing matched: treat a clearly neutral note gracefully.
            emotions = {"calmness": 0.3}

        # Normalise weights into 0..1 for display friendliness.
        top_raw = max(emotions.values()) or 1.0
        emotions = {k: round(min(v / top_raw, 1.0), 2) for k, v in emotions.items()}
        # Keep only the meaningful ones, strongest first.
        emotions = dict(
            sorted(
                ((k, v) for k, v in emotions.items() if v >= 0.2),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ) or {"calmness": 0.3}

        sentiment_score = _lexicon_sentiment(text)

        # If the transformer saw strong joy/sadness we nudge sentiment to agree.
        if engine_used == "transformer":
            pos = sum(emotions.get(e, 0.0) for e in POSITIVE_EMOTIONS)
            neg = sum(
                emotions.get(e, 0.0)
                for e in ("sadness", "anxiety", "stress", "anger", "guilt_shame", "loneliness")
            )
            if pos > neg + 0.4:
                sentiment_score = max(sentiment_score, 0.4)
            elif neg > pos + 0.4:
                sentiment_score = min(sentiment_score, -0.4)

        sentiment_score = round(sentiment_score, 2)
        sentiment = _sentiment_label(sentiment_score)

        intensity = _compute_intensity(text, raw_emotion_mass, sentiment_score)
        patterns = detect_thought_patterns(text)
        dominant = max(emotions, key=emotions.get)

        return {
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "emotions": emotions,
            "dominant_emotion": dominant,
            "dominant_emotion_display": EMOTION_DISPLAY.get(dominant, dominant.title()),
            "intensity": intensity["level"],
            "intensity_score": intensity["score"],
            "thought_patterns": patterns,
            "summary": _build_summary(sentiment, emotions, intensity["level"]),
            "engine": engine_used,
        }

    @staticmethod
    def _empty_result() -> Dict[str, object]:
        """A neutral result for empty input."""
        return {
            "sentiment": "Neutral",
            "sentiment_score": 0.0,
            "emotions": {"calmness": 0.3},
            "dominant_emotion": "calmness",
            "dominant_emotion_display": "Calmness",
            "intensity": "Low",
            "intensity_score": 0.0,
            "thought_patterns": [],
            "summary": "Nothing to read yet — write a little and I'll listen. 🌿",
            "engine": "lexicon",
        }


# ---- Convenience functions ------------------------------------------------- #
@lru_cache(maxsize=4)
def get_analyzer(engine: str = "lexicon") -> EmotionAnalyzer:
    """Return a cached analyzer for the given engine."""
    return EmotionAnalyzer(engine=engine)


def analyze_text(text: str, engine: str = "lexicon") -> Dict[str, object]:
    """Module-level shortcut: analyze `text` with the chosen engine."""
    return get_analyzer(engine).analyze(text)


# ---- (De)serialisation helpers for storage -------------------------------- #
def analysis_to_storable(result: Dict[str, object]) -> Dict[str, str]:
    """Flatten an analysis result into CSV-friendly string fields."""
    return {
        "sentiment": str(result.get("sentiment", "Neutral")),
        "emotion": str(result.get("dominant_emotion_display", "Calmness")),
        "intensity": str(result.get("intensity", "Low")),
        # Full pattern objects so the UI can show explanations/reframes later.
        "patterns": json.dumps(result.get("thought_patterns", []), ensure_ascii=False),
        "emotion_scores": json.dumps(result.get("emotions", {}), ensure_ascii=False),
    }
