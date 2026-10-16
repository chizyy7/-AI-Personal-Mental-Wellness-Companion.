# 🌿 MindfulMe

**AI Personal Mental Wellness Companion** — a calm, supportive Streamlit app
that helps you notice how you feel, reflect gently, and find small moments of
calm.

MindfulMe combines daily mood tracking, in-app journaling, automatic emotion and
thought-pattern analysis, mood forecasting, a supportive AI companion, and
weekly insights — and it keeps all of your data **local on your device**.

> **A gentle companion, not a clinician.**
> MindfulMe offers supportive reflections and self-help ideas. It is **not** a
> medical device and **not** a substitute for professional mental-health care.
> If you are in crisis, please contact your local emergency number or a crisis
> line (see [Safety](#-safety)).

---

## ✨ Features

| Area | What it does |
|------|--------------|
| **Onboarding & profile** | Collects your name, age, focus areas and preferred support style (Gentle / Motivational / Practical). Editable anytime. |
| **Daily mood check-in** | A 1–10 emoji scale with an optional note, kept to one check-in per day (update anytime). |
| **In-app journal** | Write entries directly in the app, browse past ones, and delete as you like. |
| **Emotion & thought-pattern analysis** | Sentiment, detected emotions, intensity, and gentle detection of patterns like catastrophizing, self-criticism and overgeneralization. |
| **Mood history & forecast** | A **Prophet** forecast for the next 3–7 days with a confidence band, plotted with Plotly. |
| **Emotional patterns** | Weekday patterns, trends, recurring emotions/thoughts, and **scikit-learn** mood clustering — explained simply. |
| **Supportive AI companion** | **RAG** chat over your own entries and moods, powered by Ollama or any OpenAI-compatible API, with a warm offline fallback. |
| **Weekly insights** | A plain-language weekly emotional report with personalized coping suggestions, gratitude and reframing prompts, and Markdown export. |
| **Safety layer** | Detects signs of significant distress and surfaces real crisis resources, calmly and clearly. |
| **Polished UI** | Calm design, **Light/Dark mode** (remembered), responsive layout, loading states and friendly error handling. |

> _Add your own screenshots to `assets/` and reference them here if you like._

---

## 🧱 Tech stack

| Layer | Technology | Fallback if unavailable |
|-------|------------|-------------------------|
| Frontend | **Streamlit** | — |
| Data | **Pandas, NumPy** | — |
| Emotion analysis | **Transformers** (deep model) | Built-in lexicon engine (always on) |
| Forecasting | **Prophet** | Simple linear-trend estimate |
| Patterns | **scikit-learn** (KMeans) | Statistical weekday/trend detectors |
| RAG retrieval | **sentence-transformers + ChromaDB** | Ollama embeddings / TF-IDF / in-memory index |
| LLM | **Ollama** or any OpenAI-compatible API | Warm built-in companion replies |
| Visualization | **Plotly** | — |
| Storage | `st.session_state` + local JSON/CSV | — |

---

## 🏗️ How it works

```
 Mood check-ins ─┐
 Journal entries ┼─► CSV/JSON (local) ─► Emotion analyzer ─► Insights
 Pattern detector┘                          │
                                            ├─► Charts (Plotly) & forecast (Prophet)
                                            └─► RAG index (embeddings + Chroma)
                                                     │
                                        Companion (retrieval → LLM → reply)
```

Each module is independent and defensive: if a heavy dependency or a model is
missing, the app falls back to a lighter approach instead of failing.

---

## 📁 Project structure

```
mindfulme/
├── app.py                     # Streamlit entry point (all pages + UI)
├── requirements.txt
├── README.md
├── data/                      # your data (created/updated at runtime)
│   ├── mood_logs.csv
│   ├── journal_entries.csv
│   └── user_profile.json
├── models/
│   ├── emotion_analyzer.py    # sentiment, emotions, intensity, thought patterns
│   ├── mood_forecaster.py     # Prophet forecast + trend fallback
│   ├── pattern_detector.py    # weekday/trend/emotion/thought patterns + KMeans
│   └── rag_engine.py          # embeddings, vector store, modular LLM, companion
├── utils/
│   ├── helpers.py             # paths, JSON/CSV persistence, profile, utilities
│   ├── insights.py            # weekly emotional report
│   ├── recommendations.py     # personalized coping + gratitude/reframing
│   └── safety.py              # distress detection + crisis resources
└── assets/                    # optional images/icons
```

---

## 🚀 Getting started

### 1. Set up an environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run may download a small embedding model (all-MiniLM-L6-v2) the
> first time the companion builds its memory index. Everything else works
> offline.

### 3. Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

---

## 🤖 Configuring the AI companion

MindfulMe works immediately with a **warm built-in companion**, so no setup is
required. For richer, model-generated replies, connect a model in
**Settings → AI Companion**.

### Option A — Ollama (local, private, free)

```bash
# install Ollama from https://ollama.com, then:
ollama pull llama3.1        # a good general chat model
ollama pull nomic-embed-text  # optional: embeddings fallback
ollama serve                # usually starts automatically
```

Then choose **Ollama (local)** in Settings (default URL `http://localhost:11434`).

### Option B — OpenAI-compatible API

Choose **OpenAI-compatible API**, set the base URL and model, and paste an API
key. Any compatible provider works (OpenAI, Groq, Together, OpenRouter, …). You
can also set `OPENAI_API_KEY` in a local `.env` file.

---

## 🔒 Data & privacy

- All profile data, mood logs, and journal entries are stored **locally** in the
  `data/` folder. The RAG memory index lives in `data/chroma/`.
- Nothing is uploaded anywhere unless **you** connect a cloud LLM, in which case
  only the relevant chat/context is sent to your chosen provider.
- **Settings → Data & Privacy** lets you export your data (CSV) or delete
  everything in one step.

---

## 🩺 Safety

MindfulMe includes a safety layer that looks for signs of significant distress
in recent check-ins, journal entries, and chats, and — when needed — gently
surfaces real resources, including:

- **988 Suicide & Crisis Lifeline (US):** call or text **988**
- **Crisis Text Line:** text **HOME** to **741741**
- **Emergency services:** your local number (911 / 112 / 999)
- **International:** https://findahelpline.com/

If you are in immediate danger, please call your local emergency number now.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit: command not found` | Ensure your virtualenv is active and re-run `pip install -r requirements.txt`. |
| Forecast shows "trend estimate" | Prophet isn't installed/failed — `pip install prophet`. A trend estimate is used otherwise. |
| Companion uses built-in replies | No LLM detected. Start Ollama or configure an API key in Settings. |
| First chat is slow | The embedding model / LLM is loading for the first time. Subsequent replies are faster. |
| Emoji look wrong in the terminal | Cosmetic only; set a UTF-8 capable terminal. |

---

## 🗺️ Roadmap

All eight build phases are complete:

1. ✅ Project setup + user profile / onboarding
2. ✅ Daily mood check-in + in-app journaling
3. ✅ Emotion & thought-pattern analysis
4. ✅ Mood forecasting + pattern detection
5. ✅ Supportive AI companion (RAG + chat)
6. ✅ Insights, recommendations & safety
7. ✅ Full app UI (light/dark, responsive)
8. ✅ Final polish, docs & requirements

---

## 📄 License

Provided as a personal project for learning and wellbeing support. Use
responsibly and kindly.
