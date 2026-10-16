"""
rag_engine.py
-------------
Phase 5 - Supportive AI Companion (Retrieval-Augmented Generation + chat).

This module turns the user's own data into a warm, context-aware conversation:

    1. It builds a small knowledge base from journal entries, mood logs and
       detected patterns.
    2. It embeds those documents and stores them in a vector store
       (ChromaDB when available, otherwise an in-memory cosine index).
    3. When the user chats, it retrieves the most relevant memories and asks an
       LLM (Ollama or any OpenAI-compatible API) to respond supportively.
    4. If no LLM is available, a careful, rule-based companion still replies
       warmly using the retrieved context — so the feature always works offline.

Design principles: modular LLM, graceful degradation, and emotional safety.
This is a supportive companion, never a clinician.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from utils.helpers import (
    BASE_DIR,
    DATA_DIR,
    load_journal_entries,
    load_mood_logs,
    mood_label,
)

CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
COLLECTION_NAME = "mindfulme_memories"


# =========================================================================== #
# Embeddings
# =========================================================================== #
class _Embedder:
    """Base class: subclasses must implement embed() and expose `name`."""

    name = "base"

    def embed(self, texts: List[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class SentenceTransformerEmbedder(_Embedder):
    """Primary embedder: sentence-transformers (all-MiniLM-L6-v2)."""

    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class OllamaEmbedder(_Embedder):
    """Fallback embedder using a local Ollama embedding model."""

    name = "ollama/nomic-embed-text"

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434") -> None:
        import requests

        self._requests = requests
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: List[str]) -> np.ndarray:
        out = []
        for text in texts:
            resp = self._requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            out.append(resp.json()["embedding"])
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-8, None)


class TfidfEmbedder(_Embedder):
    """Last-resort offline embedder: TF-IDF hashing (no downloads needed)."""

    name = "sklearn/tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2")

    def embed(self, texts: List[str]) -> np.ndarray:
        matrix = self._vec.transform(texts)
        return np.asarray(matrix.toarray(), dtype=np.float32)


_EMBEDDER_CACHE: Dict[str, _Embedder] = {}


def get_embedder() -> _Embedder:
    """
    Return the best available embedder, trying in order of preference and
    caching the result for the session.
    """
    if "current" in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE["current"]

    for factory in (SentenceTransformerEmbedder, OllamaEmbedder, TfidfEmbedder):
        try:
            embedder = factory()
            _EMBEDDER_CACHE["current"] = embedder
            return embedder
        except Exception:  # noqa: BLE001 - try the next option
            continue

    # TfidfEmbedder should always work, but stay safe regardless.
    fallback = TfidfEmbedder()
    _EMBEDDER_CACHE["current"] = fallback
    return fallback


# =========================================================================== #
# Vector store
# =========================================================================== #
class _NumpyStore:
    """Simple in-memory cosine similarity store."""

    backend = "numpy"

    def __init__(self) -> None:
        self._vectors: Optional[np.ndarray] = None
        self._docs: List[Dict[str, Any]] = []

    def build(self, docs: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        self._docs = docs
        self._vectors = embeddings

    def query(self, embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        if self._vectors is None or not self._docs:
            return []
        q = np.asarray(embedding, dtype=np.float32).ravel()
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        sims = self._vectors @ q
        order = np.argsort(-sims)[:k]
        results = []
        for idx in order:
            doc = dict(self._docs[int(idx)])
            doc["score"] = round(float(sims[int(idx)]), 4)
            results.append(doc)
        return results


class _ChromaStore:
    """ChromaDB-backed persistent vector store."""

    backend = "chromadb"

    def __init__(self) -> None:
        import chromadb

        os.makedirs(CHROMA_DIR, exist_ok=True)
        self._client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._docs: List[Dict[str, Any]] = []

    def _collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def build(self, docs: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        # Use a fresh collection each build to avoid stale rows.
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass

        self._docs = docs
        col = self._collection(COLLECTION_NAME)
        if not docs:
            return
        col.add(
            ids=[str(d["id"]) for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[d["metadata"] for d in docs],
            embeddings=[v.tolist() for v in embeddings],
        )

    def query(self, embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        if not self._docs:
            return []
        col = self._collection(COLLECTION_NAME)
        res = col.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32).ravel().tolist()],
            n_results=min(k, len(self._docs)),
            include=["documents", "metadatas", "distances"],
        )
        out: List[Dict[str, Any]] = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "id": (meta or {}).get("id", ""),
                    "text": text,
                    "metadata": meta or {},
                    "score": round(1.0 - float(dist), 4),  # cosine distance -> sim
                }
            )
        return out


_STORE_CACHE: Dict[str, Any] = {}


def _get_store():
    """Return a vector store, preferring ChromaDB and falling back to numpy."""
    if "store" in _STORE_CACHE:
        return _STORE_CACHE["store"]
    try:
        store = _ChromaStore()
    except Exception:  # noqa: BLE001 - chroma missing / failed -> numpy
        store = _NumpyStore()
    _STORE_CACHE["store"] = store
    return store


# =========================================================================== #
# Knowledge base construction
# =========================================================================== #
def _build_documents() -> List[Dict[str, Any]]:
    """Gather journal entries, mood logs and patterns into retrievable docs."""
    docs: List[Dict[str, Any]] = []

    # --- Journal entries --------------------------------------------------- #
    journal = load_journal_entries()
    for _, row in journal.iterrows():
        title = str(row.get("title") or "").strip()
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        text = (f"{title}. " if title else "") + content
        docs.append(
            {
                "id": f"journal-{row.get('id')}",
                "text": text,
                "metadata": {
                    "id": str(row.get("id")),
                    "type": "journal",
                    "date": str(row.get("date") or ""),
                    "emotion": str(row.get("emotion") or ""),
                    "sentiment": str(row.get("sentiment") or ""),
                },
            }
        )

    # --- Mood logs --------------------------------------------------------- #
    mood = load_mood_logs()
    for _, row in mood.iterrows():
        try:
            score = int(float(row.get("mood_score")))
        except (TypeError, ValueError):
            continue
        note = str(row.get("note") or "").strip()
        label = str(row.get("mood_label") or mood_label(score))
        text = f"On {row.get('date')} I logged my mood as {score}/10 ({label})."
        if note:
            text += f" I noted: {note}"
        docs.append(
            {
                "id": f"mood-{row.get('id')}",
                "text": text,
                "metadata": {
                    "id": str(row.get("id")),
                    "type": "mood",
                    "date": str(row.get("date") or ""),
                    "score": score,
                },
            }
        )

    # --- Detected patterns (optional, computed lazily) --------------------- #
    try:
        from models.pattern_detector import detect_patterns

        for i, pattern in enumerate(detect_patterns()):
            docs.append(
                {
                    "id": f"pattern-{i}",
                    "text": f"{pattern.get('title')} {pattern.get('explanation')}",
                    "metadata": {"id": f"pattern-{i}", "type": "pattern", "date": ""},
                }
            )
    except Exception:  # noqa: BLE001 - patterns are a nice-to-have here
        pass

    return docs


def _signature(docs: List[Dict[str, Any]]) -> str:
    """A hash of the corpus so we only rebuild the index when data changes."""
    h = hashlib.md5()
    for doc in docs:
        h.update(doc["id"].encode("utf-8"))
        h.update(doc["text"].encode("utf-8"))
    return h.hexdigest()


# =========================================================================== #
# RAG engine
# =========================================================================== #
class RagEngine:
    """Builds and queries the personal memory index."""

    def __init__(self) -> None:
        self.embedder = get_embedder()
        self.store = _get_store()
        self._signature: Optional[str] = None
        self.ready = False

    def build_index(self, force: bool = False) -> int:
        """(Re)build the index if the underlying data changed. Returns doc count."""
        docs = _build_documents()
        signature = _signature(docs)
        if not force and signature == self._signature and self.ready:
            return len(docs)
        try:
            if docs:
                embeddings = self.embedder.embed([d["text"] for d in docs])
            else:
                embeddings = np.zeros((0, 1), dtype=np.float32)
            self.store.build(docs, embeddings)
            self._signature = signature
            self.ready = True
        except Exception:  # noqa: BLE001 - never let indexing break the chat
            self.ready = False
        return len(docs)

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Return the k most relevant memories for `query`."""
        if not self.ready:
            self.build_index()
        if not query.strip() or not self.ready:
            return []
        try:
            qvec = self.embedder.embed([query])[0]
            results = self.store.query(qvec, k)
            # Drop weak matches so we don't force irrelevant context.
            return [r for r in results if r.get("score", 0) > 0.15]
        except Exception:  # noqa: BLE001
            return []

    @property
    def backend(self) -> str:
        return getattr(self.store, "backend", "unknown")


_RAG_CACHE: Dict[str, RagEngine] = {}


def get_rag_engine() -> RagEngine:
    """Return a cached RAG engine for the session."""
    if "engine" not in _RAG_CACHE:
        _RAG_CACHE["engine"] = RagEngine()
    return _RAG_CACHE["engine"]


# =========================================================================== #
# LLM interface (modular)
# =========================================================================== #
@dataclass
class LLMConfig:
    """Which LLM to use. `provider='auto'` probes for a sensible default."""

    provider: str = "auto"  # 'auto' | 'ollama' | 'openai' | 'none'
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    temperature: float = 0.6


class LLMClient:
    """A tiny, provider-agnostic chat client with availability probing."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()
        self._provider = self._resolve_provider()

    # ---- provider resolution --------------------------------------------- #
    def _resolve_provider(self) -> str:
        provider = (self.config.provider or "auto").lower()
        if provider != "auto":
            return provider
        # auto: prefer a local Ollama, then an env-provided API key.
        if self._ollama_available():
            return "ollama"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "none"

    def _ollama_available(self) -> bool:
        try:
            import requests

            r = requests.get(f"{self.config.base_url.rstrip('/')}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def ollama_models(self) -> List[str]:
        """List locally available Ollama models (empty if unavailable)."""
        try:
            import requests

            r = requests.get(f"{self.config.base_url.rstrip('/')}/api/tags", timeout=3)
            if r.status_code == 200:
                return [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:  # noqa: BLE001
            pass
        return []

    @property
    def provider(self) -> str:
        return self._provider

    def is_available(self) -> bool:
        if self._provider == "ollama":
            return self._ollama_available()
        if self._provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY") or self.config.api_key)
        return False

    # ---- generation -------------------------------------------------------- #
    def chat(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Send chat messages; return the reply text or None if it failed."""
        try:
            if self._provider == "ollama":
                return self._chat_ollama(messages)
            if self._provider == "openai":
                return self._chat_openai(messages)
        except Exception:  # noqa: BLE001 - caller falls back to template reply
            return None
        return None

    def _chat_ollama(self, messages: List[Dict[str, str]]) -> Optional[str]:
        import requests

        resp = requests.post(
            f"{self.config.base_url.rstrip('/')}/api/chat",
            json={
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.config.temperature},
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message") or {}).get("content", "").strip() or None

    def _chat_openai(self, messages: List[Dict[str, str]]) -> Optional[str]:
        import requests

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY", "")
        base = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        resp = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip() or None


# =========================================================================== #
# The companion
# =========================================================================== #
def build_system_prompt(profile: Dict[str, Any]) -> str:
    """Compose the warm, safety-conscious system prompt for the companion."""
    name = profile.get("name") or "friend"
    challenges = ", ".join(profile.get("challenges", []) or ["general wellbeing"])
    style = profile.get("support_style", "Gentle")
    style_guide = {
        "Gentle": "Speak softly and validate feelings. Never rush or push.",
        "Motivational": "Offer warm encouragement and gentle momentum.",
        "Practical": "Suggest one or two concrete, doable steps alongside empathy.",
    }.get(style, "Be warm and validating.")

    return (
        f"You are MindfulMe, a warm and supportive mental-wellness companion "
        f"talking with {name}. They would like support with: {challenges}. "
        f"Support style: {style}. {style_guide}\n\n"
        "Guidelines:\n"
        "- Be kind, non-judgmental, and encouraging. Validate before advising.\n"
        f"- Address the user by name ({name}) naturally, not in every sentence.\n"
        "- Keep replies fairly short (2-5 sentences) and human, like a caring friend.\n"
        "- When relevant, gently reference the user's past entries, moods or "
        "patterns that are provided as context — but never invent memories.\n"
        "- Ask at most one gentle follow-up question.\n"
        "- You are not a therapist or doctor. Never diagnose or give medical advice.\n"
        "- If the user seems in real distress or danger, gently encourage them to "
        "reach out to a trusted person or a crisis line, and stay supportive.\n"
    )


class Companion:
    """Ties retrieval + LLM together into a supportive conversation."""

    def __init__(
        self,
        profile: Dict[str, Any],
        llm_config: Optional[LLMConfig] = None,
        engine: Optional[RagEngine] = None,
    ) -> None:
        self.profile = profile or {}
        self.name = self.profile.get("name") or "friend"
        self.engine = engine or get_rag_engine()
        self.llm = LLMClient(llm_config)

    # ---- helpers ---------------------------------------------------------- #
    def _context_block(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        lines = ["Relevant notes from the user's own history (use only if helpful):"]
        for m in memories:
            meta = m.get("metadata", {})
            kind = meta.get("type", "note")
            date = meta.get("date", "")
            lines.append(f"- ({kind}{', ' + date if date else ''}) {m.get('text', '')}")
        return "\n".join(lines)

    def respond(
        self, user_message: str, history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Produce a supportive reply.

        Returns: {reply, sources, used_llm, provider}
        """
        history = history or []
        memories = self.engine.retrieve(user_message, k=4)
        sources = [
            {"type": m.get("metadata", {}).get("type", "note"),
             "date": m.get("metadata", {}).get("date", ""),
             "text": m.get("text", ""),
             "score": m.get("score", 0.0)}
            for m in memories
        ]

        system = build_system_prompt(self.profile)
        context = self._context_block(memories)
        if context:
            system += f"\n\n{context}"

        messages = [{"role": "system", "content": system}]
        # Keep only the last few turns for context efficiency.
        for turn in history[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})

        reply = None
        used_llm = False
        if self.llm.is_available():
            reply = self.llm.chat(messages)
            used_llm = reply is not None

        if reply is None:
            reply = self._fallback_reply(user_message, memories)

        return {
            "reply": reply,
            "sources": sources,
            "used_llm": used_llm,
            "provider": self.llm.provider,
        }

    # ---- graceful, offline fallback --------------------------------------- #
    def _fallback_reply(
        self, user_message: str, memories: List[Dict[str, Any]]
    ) -> str:
        """
        A careful, template-based reply used when no LLM is available. It still
        reflects the user's words and gently references their history.
        """
        from models.emotion_analyzer import analyze_text

        analysis = analyze_text(user_message)
        emotion = analysis.get("dominant_emotion_display", "what you're feeling")
        sentiment = analysis.get("sentiment", "Neutral")

        opening = {
            "Negative": f"Thank you for trusting me with this, {self.name}. It sounds like {str(emotion).lower()} is sitting heavily with you right now.",
            "Positive": f"It's lovely to hear this, {self.name}. There's a warmth in your words — {str(emotion).lower()} really comes through.",
            "Neutral": f"Thanks for sharing that with me, {self.name}. I'm here with you.",
        }[sentiment]

        reflection = ""
        if memories:
            top = memories[0]
            date = top.get("metadata", {}).get("date", "")
            when = f"back on {date}" if date else "recently"
            reflection = (
                f" This reminds me of something you wrote {when} — "
                "it might be worth sitting with how far you've come since then."
            )

        # A gentle, non-clinical nudge.
        suggestion = (
            "If it feels okay, one slow breath — in for four, out for six — can "
            "make the moment a little more spacious. What feels most present for you right now?"
        )

        note = ""
        if not self.llm.is_available():
            note = "\n\n_(My deeper AI voice isn't connected right now, but I'm still here listening.)_"

        return f"{opening}{reflection} {suggestion}{note}"
