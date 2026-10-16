"""
MindfulMe - AI Personal Mental Wellness Companion
=================================================

Phases implemented so far:
    * Phase 1 - Project setup + User Profile / Onboarding
    * Phase 2 - Daily Mood Check-in + In-App Journaling

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.helpers import (
    MOOD_SCALE,
    clear_all_data,
    delete_journal_entry,
    get_today_mood,
    greeting_for_now,
    load_journal_entries,
    load_profile,
    load_mood_logs,
    mood_emoji,
    mood_label,
    pretty_timestamp,
    save_journal_entry,
    save_mood_log,
    save_profile,
    update_journal_analysis,
)
from models.emotion_analyzer import (
    EMOTION_DISPLAY,
    analysis_to_storable,
    analyze_text,
)
from models.mood_forecaster import forecast_mood, forecast_summary
from models.pattern_detector import detect_patterns, has_enough_data
from models.rag_engine import Companion, LLMClient, LLMConfig, get_rag_engine
from utils.insights import report_to_markdown, weekly_report
from utils.recommendations import gratitude_prompts, recommend, reframing_prompts
from utils.safety import crisis_resources, detect_distress, safety_message

# --------------------------------------------------------------------------- #
# Page configuration (must be the first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="MindfulMe",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CHALLENGE_OPTIONS = [
    "Anxiety",
    "Stress",
    "Overthinking",
    "Low Mood",
    "Sleep issues",
    "Anger / Irritability",
    "Low self-esteem",
    "Loneliness",
    "Burnout",
    "Grief",
    "Focus / Concentration",
]

SUPPORT_STYLES = ["Gentle", "Motivational", "Practical"]

SUPPORT_STYLE_BLURB = {
    "Gentle": "Soft, validating words that give you room to breathe.",
    "Motivational": "Warm encouragement that nudges you gently forward.",
    "Practical": "Concrete, down-to-earth strategies you can try today.",
}

# Sidebar navigation. Later phases extend this list.
NAV_SECTIONS = [
    "🏠 Home",
    "🌈 Daily Check-in",
    "📖 Journal",
    "📈 Mood & Forecast",
    "🧭 Patterns",
    "💬 Companion",
    "🌱 Insights",
    "⚙️ Settings",
]

APP_VERSION = "1.0"

# Emoji used to soften the analysis presentation.
SENTIMENT_EMOJI = {"Positive": "💚", "Neutral": "🌤️", "Negative": "💙"}
INTENSITY_EMOJI = {"Low": "🌱", "Moderate": "🌊", "High": "🌧️"}


# --------------------------------------------------------------------------- #
# Theming
# --------------------------------------------------------------------------- #
def _palette(theme: str) -> dict:
    """Return the colour palette for the given theme."""
    if theme == "Dark":
        return {
            "bg": "#0f141a",
            "bg2": "#141b23",
            "card": "#1a2129",
            "text": "#e9eef3",
            "subtext": "#9fb0bd",
            "accent": "#7fc8a9",
            "accent_soft": "rgba(127, 200, 169, 0.16)",
            "border": "#26303a",
        }
    return {
        "bg": "#f4f8f5",
        "bg2": "#eef5f0",
        "card": "#ffffff",
        "text": "#22303a",
        "subtext": "#6b7b86",
        "accent": "#4a9e7f",
        "accent_soft": "rgba(74, 158, 127, 0.12)",
        "border": "#e3ece7",
    }


def inject_css(theme: str) -> None:
    """
    Inject the app's calm stylesheet. Light is the default; dark flips the
    palette. Includes soft cards, roomy spacing, and light responsiveness for
    smaller screens. Deliberately gentle so it never fights Streamlit.
    """
    c = _palette(theme)
    st.markdown(
        f"""
        <style>
            :root {{
                --mm-bg: {c['bg']};
                --mm-card: {c['card']};
                --mm-text: {c['text']};
                --mm-subtext: {c['subtext']};
                --mm-accent: {c['accent']};
                --mm-accent-soft: {c['accent_soft']};
                --mm-border: {c['border']};
            }}

            .stApp {{ background: linear-gradient(180deg, {c['bg']} 0%, {c['bg2']} 100%);
                      color: {c['text']}; }}

            /* Softer, roomier content column */
            .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 940px; }}

            /* Sidebar */
            section[data-testid="stSidebar"] {{
                background-color: {c['card']};
                border-right: 1px solid {c['border']};
            }}

            /* Cards */
            .mm-card {{
                background-color: {c['card']};
                border: 1px solid {c['border']};
                border-radius: 20px;
                padding: 1.4rem 1.6rem;
                box-shadow: 0 6px 20px rgba(0,0,0,0.05);
                margin-bottom: 1rem;
            }}
            .mm-title {{ color: {c['accent']}; margin-bottom: 0.25rem; font-weight: 700; }}
            .mm-subtext {{ color: {c['subtext']}; font-size: 0.95rem; }}

            /* Hero */
            .mm-hero {{ text-align: center; padding: 0.8rem 0 1rem 0; }}
            .mm-hero h1 {{
                color: {c['text']};
                margin-bottom: 0.15rem;
                font-size: 2.1rem;
                letter-spacing: -0.02em;
            }}
            .mm-hero p {{ margin-top: 0; }}

            /* Pills / badges */
            .mm-pill {{
                display: inline-block;
                background-color: var(--mm-accent-soft);
                color: {c['accent']};
                border-radius: 999px;
                padding: 0.18rem 0.75rem;
                margin: 0.15rem 0.25rem 0.15rem 0;
                font-size: 0.82rem;
                font-weight: 600;
            }}

            /* Journal entry */
            .mm-entry {{
                background-color: {c['card']};
                border: 1px solid {c['border']};
                border-left: 4px solid {c['accent']};
                border-radius: 14px;
                padding: 1rem 1.15rem;
                margin-bottom: 0.7rem;
            }}
            .mm-entry h4 {{ margin: 0 0 0.25rem 0; color: {c['accent']}; }}
            .mm-meta {{ color: {c['subtext']}; font-size: 0.8rem; }}

            /* Dashboard stat tiles */
            .mm-stats {{ display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 0.4rem 0 1rem 0; }}
            .mm-stat {{
                flex: 1 1 140px;
                background-color: {c['card']};
                border: 1px solid {c['border']};
                border-radius: 16px;
                padding: 0.9rem 1.1rem;
                text-align: center;
            }}
            .mm-stat .value {{ font-size: 1.6rem; font-weight: 700; color: {c['accent']}; }}
            .mm-stat .label {{ font-size: 0.8rem; color: {c['subtext']}; }}

            /* Softer controls */
            .stButton > button, .stDownloadButton > button {{ border-radius: 12px; }}
            .stButton > button[kind="primary"] {{
                background-color: {c['accent']};
                border-color: {c['accent']};
            }}

            /* Chat bubbles */
            [data-testid="stChatMessage"] {{
                background-color: {c['card']};
                border: 1px solid {c['border']};
                border-radius: 16px;
                padding: 0.4rem 0.6rem;
            }}

            /* Tabs a touch roomier */
            .stTabs [data-baseweb="tab"] {{ padding: 0.5rem 1rem; }}

            /* Responsive: gently shrink on phones */
            @media (max-width: 640px) {{
                .block-container {{ padding-left: 0.9rem; padding-right: 0.9rem; }}
                .mm-hero h1 {{ font-size: 1.65rem; }}
                .mm-card {{ padding: 1.1rem 1.15rem; border-radius: 16px; }}
                .mm-stat .value {{ font-size: 1.35rem; }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_theme(theme: str) -> None:
    """
    Persist the user's theme preference to their profile (so it's remembered
    between sessions) and update the in-session copy.
    """
    theme = "Dark" if theme == "Dark" else "Light"
    profile = st.session_state.get("profile", {})
    if profile.get("theme") == theme:
        return
    profile["theme"] = theme
    st.session_state.profile["theme"] = theme
    # Only persist once the user actually has a saved profile.
    if profile.get("name"):
        save_profile(profile)


def theme_toggle_widget(key: str = "theme_toggle") -> None:
    """A friendly light/dark toggle that remembers the user's choice."""
    is_dark = st.session_state.profile.get("theme", "Light") == "Dark"
    dark = st.toggle("🌙 Dark mode", value=is_dark, key=key)
    if dark != is_dark:
        set_theme("Dark" if dark else "Light")
        st.rerun()


def hero_header(title: str, subtitle: str = "") -> None:
    """Render a centered hero title with an optional subtitle."""
    html = f'<div class="mm-hero"><h1>{title}</h1>'
    if subtitle:
        html += f'<p class="mm-subtext">{subtitle}</p>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def stat_tiles(stats: list) -> None:
    """Render a row of soft stat tiles. `stats` is a list of (label, value) pairs."""
    tiles = "".join(
        f'<div class="mm-stat"><div class="value">{v}</div>'
        f'<div class="label">{l}</div></div>'
        for l, v in stats
    )
    st.markdown(f'<div class="mm-stats">{tiles}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session-state bootstrap
# --------------------------------------------------------------------------- #
def init_state() -> None:
    """Load the profile into session_state once per session."""
    if "profile" not in st.session_state:
        st.session_state.profile = load_profile()
    if "editing_profile" not in st.session_state:
        st.session_state.editing_profile = False
    if "entry_draft" not in st.session_state:
        st.session_state.entry_draft = {"title": "", "content": ""}
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "auto"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "llama3.1:8b"
    if "llm_base_url" not in st.session_state:
        st.session_state.llm_base_url = "http://localhost:11434"
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""
    if "safety_level" not in st.session_state:
        st.session_state.safety_level = "none"


# --------------------------------------------------------------------------- #
# Onboarding / Profile form  (Phase 1)
# --------------------------------------------------------------------------- #
def render_profile_form(profile: dict, is_onboarding: bool) -> bool:
    """Render the onboarding / edit-profile form. Returns True when saved."""
    with st.form("profile_form", clear_on_submit=False):
        st.subheader("A little about you")
        st.caption(
            "Everything you share stays on your device. You can change it anytime."
        )

        name = st.text_input(
            "What should I call you?",
            value=profile.get("name", ""),
            placeholder="Your name or nickname",
        )

        age = st.number_input(
            "Age (optional)",
            min_value=13,
            max_value=120,
            value=int(profile["age"]) if isinstance(profile.get("age"), int) else 25,
            step=1,
            help="Used only to tailor the tone of support, never shared.",
        )

        challenges = st.multiselect(
            "What would you like support with? (choose any)",
            options=CHALLENGE_OPTIONS,
            default=[c for c in profile.get("challenges", []) if c in CHALLENGE_OPTIONS],
        )

        support_style = st.radio(
            "How would you like me to talk with you?",
            options=SUPPORT_STYLES,
            index=(
                SUPPORT_STYLES.index(profile["support_style"])
                if profile.get("support_style") in SUPPORT_STYLES
                else 0
            ),
            horizontal=True,
        )
        st.caption(SUPPORT_STYLE_BLURB[support_style])

        submitted = st.form_submit_button(
            "Save & continue" if is_onboarding else "Save changes",
            width="stretch",
        )

    if not submitted:
        return False

    if not name.strip():
        st.error("Please add a name so I know how to address you. 🌱")
        return False
    if not challenges:
        st.warning("Pick at least one area you'd like support with — or choose 'Stress'.")

    profile.update(
        {
            "name": name.strip(),
            "age": int(age),
            "challenges": challenges,
            "support_style": support_style,
        }
    )
    if save_profile(profile):
        st.session_state.profile = load_profile()
        st.session_state.editing_profile = False
        return True

    st.error("Something went wrong while saving. Please try again.")
    return False


def render_onboarding() -> None:
    """First-time onboarding experience."""
    hero_header(
        "Welcome to MindfulMe 🌿",
        "A quiet, supportive space to understand your emotions.",
    )
    st.markdown(
        '<div class="mm-card">'
        "<p class='mm-subtext'>I'm here to help you notice how you feel, reflect "
        "gently, and find small moments of calm. Before we begin, tell me a little "
        "about yourself — no pressure, and you can change this later.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if render_profile_form(st.session_state.profile, is_onboarding=True):
        st.balloons()
        st.rerun()


def render_edit_profile() -> None:
    """Standalone edit-profile screen with a cancel option."""
    hero_header("Edit your profile")
    if render_profile_form(st.session_state.profile, is_onboarding=False):
        st.success("Saved. Thank you for keeping things up to date. 🌿")
        st.rerun()

    if st.button("Cancel"):
        st.session_state.editing_profile = False
        st.rerun()


# --------------------------------------------------------------------------- #
# Home dashboard
# --------------------------------------------------------------------------- #
def _checkin_streak() -> int:
    """Count consecutive days (ending today or yesterday) with a check-in."""
    df = load_mood_logs()
    if df.empty:
        return 0
    days = set(pd.to_datetime(df["date"], errors="coerce").dropna().dt.date)
    if not days:
        return 0
    today = datetime.now().date()
    start = today if today in days else today - timedelta(days=1)
    if start not in days:
        return 0
    streak = 0
    cursor = start
    while cursor in days:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def render_welcome(profile: dict) -> None:
    """Personalised home dashboard."""
    name = profile.get("name", "friend")
    hero_header(
        f"{greeting_for_now()}, {name} 🌿",
        "Take a slow breath — you've already done something kind for yourself by being here.",
    )

    # ---- Quick stats ------------------------------------------------------ #
    mood_df = load_mood_logs()
    journal_df = load_journal_entries()
    streak = _checkin_streak()
    checkins = len(mood_df)
    recent = mood_df.tail(7)["mood_score"] if not mood_df.empty else []
    avg7 = round(float(recent.mean()), 1) if len(recent) else None
    stat_tiles(
        [
            ("🌱 Day streak", streak),
            ("📝 Check-ins", checkins),
            ("📖 Journal entries", len(journal_df)),
            ("📊 7-day avg", f"{avg7}/10" if avg7 else "–"),
        ]
    )

    # ---- Today's status --------------------------------------------------- #
    today = get_today_mood()
    if today:
        status = (
            f"{mood_emoji(today['mood_score'])} You checked in today and felt "
            f"<b>{today['mood_label'].lower()}</b>. Thank you for noticing. 🌱"
        )
    else:
        status = (
            "You haven't checked in yet today. A quick mood note only takes a "
            "moment — head to <b>Daily Check-in</b>. 🌈"
        )

    challenges = profile.get("challenges", [])
    pills = "".join(f'<span class="mm-pill">{c}</span>' for c in challenges)
    style = profile.get("support_style", "Gentle")
    st.markdown(
        f"""
        <div class="mm-card">
            <h3 class="mm-title">Today</h3>
            <p class="mm-subtext">{status}</p>
            <hr style="opacity:0.12">
            <p class="mm-subtext">You'd like support with:</p>
            <p>{pills or '<span class="mm-subtext">No areas selected yet.</span>'}</p>
            <p class="mm-subtext">I'll keep a <b>{style.lower()}</b> tone with you ~</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Use the sidebar to check in your mood, journal, review your forecast, "
        "or talk with your companion. I'm here whenever you're ready. 💚",
        icon="🧭",
    )


# --------------------------------------------------------------------------- #
# Daily Mood Check-in  (Phase 2)
# --------------------------------------------------------------------------- #
def render_mood_checkin(profile: dict) -> None:
    """A gentle daily mood check-in with a 1-10 emoji scale and a short note."""
    name = profile.get("name", "friend")
    hero_header(
        "How are you feeling right now? 🌈",
        f"No right or wrong answers, {name}. Just what's true for you.",
    )

    existing = get_today_mood()
    default_score = int(existing["mood_score"]) if existing else 5
    default_note = existing.get("note", "") if existing else ""

    if existing:
        st.caption(
            f"You already checked in today ({mood_emoji(existing['mood_score'])} "
            f"{existing['mood_label']}). You can update it below if things have shifted."
        )

    st.markdown('<div class="mm-card">', unsafe_allow_html=True)
    score = st.select_slider(
        "Move the slider to match your mood",
        options=list(MOOD_SCALE.keys()),
        value=default_score,
        format_func=lambda s: f"{MOOD_SCALE[s]['emoji']}  {s} — {MOOD_SCALE[s]['label']}",
    )
    st.markdown(
        f"<p class='mm-subtext'>Selected: <b>{mood_emoji(score)} {mood_label(score)}</b></p>",
        unsafe_allow_html=True,
    )

    note = st.text_area(
        "Anything you'd like to add? (optional)",
        value=default_note,
        placeholder="A word, a sentence, or a few feelings...",
        height=100,
    )

    if st.button("Save today's check-in", type="primary", width="stretch"):
        if save_mood_log(score, note):
            st.success(
                f"Saved {mood_emoji(score)} — thank you for checking in, {name}. 🌿"
            )
            st.rerun()
        else:
            st.error("Couldn't save your check-in just now. Please try again.")
    st.markdown("</div>", unsafe_allow_html=True)

    # A small reassuring message, tuned to how low/high the mood is.
    if score <= 3:
        st.info(
            "It's okay to feel low — feelings move like weather. Be gentle with "
            "yourself today, and know you don't have to carry it alone. 💚",
            icon="🫂",
        )
    elif score <= 6:
        st.info("Thanks for tuning in. Even a steady day deserves noticing. 🌤️", icon="🌿")
    else:
        st.success("It's lovely to hear today feels good. Savor it. ☀️")


# --------------------------------------------------------------------------- #
# Emotion analysis display  (Phase 3)
# --------------------------------------------------------------------------- #
def _parse_json_field(value, default):
    """Safely decode a JSON string stored in a CSV cell."""
    if not value or (isinstance(value, float)):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def ensure_entry_analysis(row: dict) -> dict:
    """
    Return an analysis result for a journal entry.

    The lexicon engine is fast and deterministic, so we analyze on demand. If
    the entry has never been analyzed we quietly persist the result so later
    phases (RAG / insights) can reuse it.
    """
    content = row.get("content", "") or ""
    result = analyze_text(content)
    if not row.get("sentiment"):
        try:
            update_journal_analysis(row.get("id"), analysis_to_storable(result))
        except Exception:  # noqa: BLE001 - persistence must never break the UI
            pass
    return result


def render_analysis_card(result: dict, heading: str = "A gentle reading 🌿") -> None:
    """Render an emotion analysis result in a calm, non-judgmental way."""
    sentiment = result.get("sentiment", "Neutral")
    intensity = result.get("intensity", "Low")
    dom = result.get("dominant_emotion_display", "Calmness")
    summary = result.get("summary", "")

    st.markdown(f"##### {heading}")
    st.markdown(
        f"""
        <div class="mm-card">
            <p>{summary}</p>
            <span class="mm-pill">{SENTIMENT_EMOJI.get(sentiment, "🌤️")} {sentiment} tone</span>
            <span class="mm-pill">🎭 {dom}</span>
            <span class="mm-pill">{INTENSITY_EMOJI.get(intensity, "🌱")} {intensity} intensity</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Emotion intensity bars (only the meaningful ones are kept by the model).
    emotions = result.get("emotions", {}) or {}
    if emotions:
        st.caption("What I noticed underneath")
        for emo, weight in emotions.items():
            label = EMOTION_DISPLAY.get(emo, str(emo).title())
            st.progress(float(weight), text=f"{label}  ·  {int(weight * 100)}%")

    # Possible thought patterns, offered with a soft reframe.
    patterns = result.get("thought_patterns", []) or []
    if patterns:
        st.caption("A thought pattern that might be visiting")
        for p in patterns:
            with st.expander(f"🧩 {p.get('label', 'Pattern')}"):
                if p.get("snippet"):
                    st.caption(f'You wrote: “{p["snippet"]}”')
                st.markdown(p.get("explanation", ""))
                if p.get("reframe"):
                    st.info(p["reframe"], icon="🌱")
    else:
        st.caption("I didn't spot any harsh thought patterns here — that's worth noting. 🌿")

    st.caption(
        f"Read with the {result.get('engine', 'lexicon')} engine. "
        "This is a gentle reflection, never a diagnosis."
    )


# --------------------------------------------------------------------------- #
# Journal  (Phase 2 + analysis in Phase 3)
# --------------------------------------------------------------------------- #
def render_journal_entry_card(row: dict) -> None:
    """Render one saved journal entry as a soft card."""
    title = row.get("title") or "Untitled"
    body = (row.get("content") or "").replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="mm-entry">
            <h4>{title}</h4>
            <div class="mm-meta">{pretty_timestamp(row.get('timestamp'))}</div>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_journal(profile: dict) -> None:
    """In-app journal: write a new entry and browse past ones."""
    name = profile.get("name", "friend")
    hero_header(
        "Your Journal 📖",
        f"A private page for your thoughts, {name}. Write freely — no one else reads this.",
    )

    tab_write, tab_past = st.tabs(["✍️ Write", "🕰️ Past entries"])

    # ---- Write ------------------------------------------------------------ #
    with tab_write:
        # Show the analysis of the entry the user just saved (survives rerun).
        if st.session_state.get("last_analysis"):
            st.success("Your entry is saved. Here's what I noticed — gently. 🌿")
            render_analysis_card(st.session_state.last_analysis)
            if st.button("Thanks, got it"):
                st.session_state.last_analysis = None
                st.rerun()
            st.divider()

        title = st.text_input(
            "Title (optional)",
            value=st.session_state.entry_draft.get("title", ""),
            placeholder="Give today a name, or leave it blank",
        )
        content = st.text_area(
            "What's on your mind?",
            value=st.session_state.entry_draft.get("content", ""),
            placeholder="Let it spill out — messy is welcome here.",
            height=260,
        )

        col_save, col_clear = st.columns([3, 1])
        with col_save:
            if st.button("Save entry", type="primary", width="stretch"):
                if not content.strip():
                    st.warning("Write a little something first — even one line counts. 🌱")
                else:
                    result = analyze_text(content)
                    if save_journal_entry(
                        title, content, analysis_to_storable(result)
                    ):
                        st.session_state.entry_draft = {"title": "", "content": ""}
                        st.session_state.last_analysis = result
                        st.rerun()
                    else:
                        st.error("Couldn't save your entry just now. Please try again.")
        with col_clear:
            if st.button("Clear", width="stretch"):
                st.session_state.entry_draft = {"title": "", "content": ""}
                st.rerun()

    # ---- Past entries ----------------------------------------------------- #
    with tab_past:
        df = load_journal_entries()
        if df.empty:
            st.info("No entries yet. Your first one is waiting whenever you're ready. 🌼")
        else:
            st.caption(f"{len(df)} entr{'y' if len(df) == 1 else 'ies'} so far.")
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                with st.expander(
                    f"{row_dict.get('title') or 'Untitled'}  ·  "
                    f"{pretty_timestamp(row_dict.get('timestamp'))}"
                ):
                    render_journal_entry_card(row_dict)
                    st.markdown("**Analysis**")
                    render_analysis_card(ensure_entry_analysis(row_dict))
                    if st.button("Delete entry", key=f"del_{row_dict['id']}"):
                        if delete_journal_entry(row_dict["id"]):
                            st.success("Entry deleted.")
                            st.rerun()


# --------------------------------------------------------------------------- #
# Mood History + Forecast  (Phase 4)
# --------------------------------------------------------------------------- #
def _plotly_template() -> str:
    """Pick a Plotly template that matches the current theme."""
    return "plotly_dark" if st.session_state.profile.get("theme") == "Dark" else "plotly_white"


def mood_history_chart(history, template: str) -> go.Figure:
    """A soft line chart of the user's mood history."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["ds"],
            y=history["y"],
            mode="lines+markers",
            name="Mood",
            line=dict(color="#4a9e7f", width=3, shape="spline"),
            marker=dict(size=8, color="#4a9e7f"),
            hovertemplate="%{x|%b %d}<br>Mood: %{y:.1f}/10<extra></extra>",
        )
    )
    fig.update_layout(
        template=template,
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Mood", range=[0.5, 10.5], dtick=1),
        xaxis=dict(title=""),
        showlegend=False,
    )
    return fig


def mood_forecast_chart(history, forecast, template: str) -> go.Figure:
    """History + forecast line with a soft confidence band."""
    fig = go.Figure()

    # Uncertainty band (draw upper, then lower with fill).
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"], y=forecast["yhat_upper"],
            mode="lines", line=dict(width=0),
            name="Upper", showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"], y=forecast["yhat_lower"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(122, 200, 169, 0.25)",
            name="Likely range", hoverinfo="skip",
        )
    )

    # Actual history.
    fig.add_trace(
        go.Scatter(
            x=history["ds"], y=history["y"],
            mode="lines+markers", name="How you felt",
            line=dict(color="#4a9e7f", width=3, shape="spline"),
            marker=dict(size=8, color="#4a9e7f"),
            hovertemplate="%{x|%b %d}<br>Mood: %{y:.1f}/10<extra></extra>",
        )
    )

    # Forecast line (only the future portion is emphasised).
    upcoming = forecast[forecast["is_future"]]
    if not upcoming.empty:
        # Bridge from last known point for visual continuity.
        bridge = pd.concat([history.tail(1)[["ds", "y"]].rename(columns={"y": "yhat"}), upcoming[["ds", "yhat"]]])
        fig.add_trace(
            go.Scatter(
                x=bridge["ds"], y=bridge["yhat"],
                mode="lines+markers", name="Forecast",
                line=dict(color="#e0a458", width=3, dash="dash"),
                marker=dict(size=7, color="#e0a458", symbol="diamond"),
                hovertemplate="%{x|%b %d}<br>Forecast: %{y:.1f}/10<extra></extra>",
            )
        )

    fig.update_layout(
        template=template,
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Mood", range=[0.5, 10.5], dtick=1),
        xaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_mood_forecast(profile: dict) -> None:
    """Mood history + gentle forecast page."""
    name = profile.get("name", "friend")
    hero_header(
        "Your Mood, Looking Ahead 📈",
        f"A picture of how you've been feeling, {name} — and a soft look forward.",
    )

    history = load_mood_logs()
    if history.empty:
        st.info(
            "No mood check-ins yet. Once you log a few days, I'll draw your "
            "history here and gently look ahead. 🌱",
            icon="🌈",
        )
        return

    days = st.slider("How many days ahead shall we look?", 3, 7, 7)

    with st.spinner("Looking over your recent moods…"):
        result = forecast_mood(days=days)

    st.caption(result.get("message", ""))
    if result.get("ok"):
        st.markdown(
            f"<div class='mm-card'><p>{forecast_summary(result)}</p></div>",
            unsafe_allow_html=True,
        )

    template = _plotly_template()
    hist_df = result.get("history")
    fc_df = result.get("forecast")

    if result.get("ok") and fc_df is not None and not fc_df.empty:
        st.plotly_chart(
            mood_forecast_chart(hist_df, fc_df, template),
            width="stretch",
        )
        method = result.get("method", "trend")
        label = "Prophet" if method == "prophet" else "trend estimate"
        st.caption(f"Forecast shown in dashed gold · method: {label}.")
    else:
        st.plotly_chart(mood_history_chart(hist_df, template), width="stretch")

    # A small table of recent check-ins.
    with st.expander("Recent check-ins"):
        recent = history.copy()
        recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%b %d, %Y")
        recent = recent.rename(columns={"date": "Date", "mood_score": "Mood (/10)", "mood_label": "Feeling"})
        cols = [c for c in ["Date", "Mood (/10)", "Feeling", "note"] if c in recent.columns]
        st.dataframe(recent[cols].tail(14).iloc[::-1], width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Emotional Patterns  (Phase 4)
# --------------------------------------------------------------------------- #
def render_patterns(profile: dict) -> None:
    """Show recurring emotional patterns detected from mood + journals."""
    name = profile.get("name", "friend")
    hero_header(
        "Your Patterns 🧭",
        f"Gentle things I've noticed across your check-ins and journal, {name}.",
    )

    if not has_enough_data():
        st.info(
            "I need a little more to go on — a few mood check-ins or journal "
            "entries. Patterns will start to appear soon. 🌿",
            icon="🔍",
        )
        return

    with st.spinner("Listening for patterns…"):
        patterns = detect_patterns()

    if not patterns:
        st.info(
            "Nothing stands out as a recurring pattern just yet — and that's "
            "perfectly okay. Keep being curious with yourself. 🌼",
            icon="🌱",
        )
        return

    st.caption(
        "These are observations, not rules about you. Notice them with curiosity, "
        "and take only what feels helpful."
    )
    for p in patterns:
        strength = float(p.get("strength", 0.0))
        st.markdown(
            f"""
            <div class="mm-card">
                <h4 class="mm-title">{p.get('emoji', '🔎')} {p.get('title', '')}</h4>
                <p class="mm-subtext">{p.get('explanation', '')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(strength, 1.0), text=f"Notice-strength · {int(strength * 100)}%")


# --------------------------------------------------------------------------- #
# AI Companion  (Phase 5)
# --------------------------------------------------------------------------- #
def _current_llm_config() -> LLMConfig:
    """Build an LLMConfig from the user's in-session companion settings."""
    return LLMConfig(
        provider=st.session_state.llm_provider,
        model=st.session_state.llm_model,
        base_url=st.session_state.llm_base_url,
        api_key=st.session_state.llm_api_key,
    )


def _greeting(profile: dict) -> str:
    """A warm opening message from the companion."""
    name = profile.get("name", "friend")
    return (
        f"Hi {name} 🌿 I'm here with you. You can tell me how today's going, "
        "vent a little, or ask me to look back over what you've written. "
        "No pressure — whatever feels right."
    )


def render_companion(profile: dict) -> None:
    """Supportive RAG-powered chat page."""
    name = profile.get("name", "friend")
    hero_header(
        "Your Companion 💬",
        f"A gentle, judgment-free space to talk things through, {name}.",
    )

    # ---- Connection status ------------------------------------------------- #
    config = _current_llm_config()
    client = LLMClient(config)
    if client.is_available():
        prov = client.provider
        if prov == "ollama":
            st.caption(f"🟢 Connected to a local model (`{config.model}` via Ollama).")
        else:
            st.caption(f"🟢 Connected via {prov} (`{config.model}`).")
    else:
        st.caption(
            "🟡 No AI model connected — I'll still reply warmly using a built-in "
            "companion. Connect Ollama or an API in the sidebar for richer replies."
        )

    # ---- Build the memory index once -------------------------------------- #
    with st.spinner("Gathering your memories…"):
        engine = get_rag_engine()
        doc_count = engine.build_index()

    # ---- Conversation ------------------------------------------------------ #
    if not st.session_state.chat_history:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": _greeting(profile), "sources": []}
        )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🌿" if message["role"] == "assistant" else "🙂"):
            st.markdown(message["content"])
            sources = message.get("sources") or []
            if sources:
                with st.expander("What I drew on from your past 🌱"):
                    for s in sources:
                        when = f" · {s['date']}" if s.get("date") else ""
                        st.caption(f"[{s.get('type', 'note')}{when}] {s.get('text', '')}")

    prompt = st.chat_input("Share what's on your mind…")
    if prompt:
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt, "sources": []}
        )
        companion = Companion(profile, config, engine=engine)
        with st.spinner("Thinking gently…"):
            result = companion.respond(prompt, history=st.session_state.chat_history[:-1])
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["reply"],
                "sources": result.get("sources", []),
            }
        )
        st.rerun()

    # ---- Small controls ---------------------------------------------------- #
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Clear chat", width="stretch"):
            st.session_state.chat_history = []
            st.rerun()
    with col_b:
        st.caption(
            f"{doc_count} memories indexed · retrieval backend: {engine.backend}. "
            "I'm a supportive companion, not a substitute for professional care. 💚"
        )


# --------------------------------------------------------------------------- #
# Safety  (Phase 6)
# --------------------------------------------------------------------------- #
_LEVEL_ORDER = {"none": 0, "elevated": 1, "high": 2}


def _max_level(a: str, b: str) -> str:
    """Return the more severe of two distress levels."""
    return a if _LEVEL_ORDER.get(a, 0) >= _LEVEL_ORDER.get(b, 0) else b


def assess_current_distress() -> str:
    """
    Look at today's mood note, recent journal entries and the last chat message
    to decide whether to surface the safety card. Cheap and non-invasive.
    """
    level = "none"

    today = get_today_mood()
    if today:
        d = detect_distress(str(today.get("note", "")), mood_score=today.get("mood_score"))
        level = _max_level(level, d["level"])

    try:
        df = load_journal_entries()
        for _, row in df.head(2).iterrows():
            d = detect_distress(str(row.get("content", "")))
            level = _max_level(level, d["level"])
    except Exception:  # noqa: BLE001 - safety check must never crash the app
        pass

    for message in reversed(st.session_state.chat_history):
        if message.get("role") == "user":
            level = _max_level(level, detect_distress(message.get("content", ""))["level"])
            break

    return level


def render_safety_card(level: str) -> None:
    """Show a calm but clear safety message with real crisis resources."""
    if level == "none":
        return

    message = safety_message(level)
    if level == "high":
        st.error(message, icon="🫶")
    else:
        st.warning(message, icon="💛")

    with st.expander("🆘 Immediate support & helplines", expanded=(level == "high")):
        for r in crisis_resources():
            contact = f"**{r['name']}** — {r['contact']}"
            st.markdown(contact)
            if r.get("url"):
                st.markdown(f"[{r['url']}]({r['url']})")
            st.caption(r.get("note", ""))
        st.caption(
            "If you are in immediate danger, please call your local emergency "
            "number right now. You deserve support. 💚"
        )


# --------------------------------------------------------------------------- #
# Weekly Insights + recommendations  (Phase 6)
# --------------------------------------------------------------------------- #
def _mood_sparkline(report: dict, template: str) -> go.Figure:
    """A small mood sparkline for the weekly report."""
    series = report.get("mood_series", [])
    fig = go.Figure(
        go.Scatter(
            x=[p["date"] for p in series],
            y=[p["score"] for p in series],
            mode="lines+markers",
            line=dict(color="#4a9e7f", width=3, shape="spline"),
            marker=dict(size=8, color="#4a9e7f"),
            hovertemplate="%{x}<br>Mood: %{y}/10<extra></extra>",
        )
    )
    fig.update_layout(
        template=template,
        height=240,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(range=[0.5, 10.5], dtick=1, title="Mood"),
        xaxis=dict(title=""),
        showlegend=False,
    )
    return fig


def render_insights(profile: dict) -> None:
    """Weekly emotional report + personalized suggestions."""
    name = profile.get("name", "friend")
    hero_header(
        "Your Weekly Insights 🌱",
        f"A gentle look back at your week, {name} — plus a few ideas for the days ahead.",
    )

    days = st.select_slider("Report window", options=[7, 14, 30], value=7)
    report = weekly_report(profile, days=days)

    # ---- Key numbers ------------------------------------------------------ #
    c1, c2, c3 = st.columns(3)
    c1.metric("Average mood", f"{report['avg_mood'] or '–'}/10")
    delta = report["mood_delta"]
    c2.metric(
        "vs. previous period",
        f"{delta:+.1f}" if delta is not None else "–",
        delta=f"{delta:+.1f}" if delta is not None else None,
    )
    c3.metric("Check-ins / entries", f"{report['checkins']} / {report['journal_count']}")

    # ---- Narrative -------------------------------------------------------- #
    st.markdown(
        f"<div class='mm-card'><p>{report['summary']}</p></div>",
        unsafe_allow_html=True,
    )

    # ---- Mood sparkline --------------------------------------------------- #
    if report.get("mood_series"):
        st.plotly_chart(_mood_sparkline(report, _plotly_template()), width="stretch")

    # ---- Highlights ------------------------------------------------------- #
    if report["highlights"]:
        with st.expander("Highlights this period", expanded=True):
            for h in report["highlights"]:
                st.markdown(f"- {h}")

    st.divider()

    # ---- Personalized suggestions ---------------------------------------- #
    st.subheader("🌿 Ideas for the days ahead")
    suggestions = report.get("recommendations") or recommend(profile, limit=4)
    for item in suggestions:
        st.markdown(
            f"""
            <div class="mm-card">
                <h4 class="mm-title">{item['emoji']} {item['title']}
                    <span class="mm-pill">{item['category']} · {item['duration']}</span>
                </h4>
                <p class="mm-subtext">{item['why']}</p>
                <p><b>Try:</b> {item['how']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Gratitude & reframing ------------------------------------------- #
    col_g, col_r = st.columns(2)
    with col_g:
        st.subheader("🌻 Gratitude")
        st.caption("A small pause to notice what's okay.")
        for prompt in gratitude_prompts(3):
            st.markdown(f"- {prompt}")
        gratitude_note = st.text_area(
            "Jot one down (optional)", key="gratitude_note", height=100,
            placeholder="Today, I'm grateful for…",
        )
        if gratitude_note.strip():
            st.success("Thank you for noticing that. 🌼")
    with col_r:
        st.subheader("🔄 Gentle reframes")
        st.caption("Softer ways to hold a hard thought.")
        for r in reframing_prompts():
            st.markdown(f"**{r['label']}**")
            st.caption(r["reframe"])

    # ---- Export ----------------------------------------------------------- #
    st.divider()
    st.download_button(
        "⬇️ Download this report (Markdown)",
        data=report_to_markdown(report, profile),
        file_name=f"mindfulme_report_{report['period_end']}.md",
        mime="text/markdown",
        width="stretch",
    )


# --------------------------------------------------------------------------- #
# Settings  (Phase 7)
# --------------------------------------------------------------------------- #
def render_companion_settings() -> None:
    """AI companion / model configuration widgets (shared by settings)."""
    provider_labels = {
        "auto": "Auto-detect",
        "ollama": "Ollama (local)",
        "openai": "OpenAI-compatible API",
        "none": "Off (built-in replies)",
    }
    choice = st.selectbox(
        "AI model source",
        options=list(provider_labels.keys()),
        format_func=lambda k: provider_labels[k],
        index=list(provider_labels.keys()).index(st.session_state.llm_provider),
        key="settings_llm_provider",
    )
    st.session_state.llm_provider = choice

    if choice in ("auto", "ollama"):
        st.session_state.llm_base_url = st.text_input(
            "Ollama URL", value=st.session_state.llm_base_url, key="settings_ollama_url"
        )
        st.session_state.llm_model = st.text_input(
            "Model", value=st.session_state.llm_model, key="settings_ollama_model"
        )
        st.caption("Tip: `ollama pull llama3.1` then `ollama serve`.")
    elif choice == "openai":
        st.session_state.llm_base_url = st.text_input(
            "API base URL",
            value=st.session_state.llm_base_url or "https://api.openai.com/v1",
            key="settings_api_url",
        )
        st.session_state.llm_model = st.text_input(
            "Model", value=st.session_state.llm_model or "gpt-4o-mini",
            key="settings_api_model",
        )
        st.session_state.llm_api_key = st.text_input(
            "API key", value=st.session_state.llm_api_key, type="password",
            key="settings_api_key",
        )
    st.caption("Your conversations and data stay on your device. 💚")


def render_settings(profile: dict) -> None:
    """Profile / Settings page: profile, appearance, companion, data, about."""
    hero_header("Settings ⚙️", "Make MindfulMe feel like your own.")

    tab_profile, tab_look, tab_ai, tab_data, tab_about = st.tabs(
        ["👤 Profile", "🎨 Appearance", "🤖 AI Companion", "🔐 Data & Privacy", "ℹ️ About"]
    )

    # ---- Profile ---------------------------------------------------------- #
    with tab_profile:
        st.caption("Update how MindfulMe addresses and supports you.")
        if render_profile_form(st.session_state.profile, is_onboarding=False):
            st.success("Saved. Thank you for keeping things up to date. 🌿")
            st.rerun()

    # ---- Appearance ------------------------------------------------------- #
    with tab_look:
        st.caption("Choose a look that feels calm to you. Your choice is remembered.")
        theme_toggle_widget(key="theme_toggle_settings")
        current = st.session_state.profile.get("theme", "Light")
        st.markdown(
            f"<div class='mm-card'><p>Current theme: <b>{current}</b> 🌿</p>"
            "<p class='mm-subtext'>Light is soft and airy; Dark is quiet and low-glare.</p></div>",
            unsafe_allow_html=True,
        )

    # ---- AI Companion ----------------------------------------------------- #
    with tab_ai:
        st.caption(
            "Connect a model for richer conversations. If none is available, a "
            "warm built-in companion still replies."
        )
        render_companion_settings()

    # ---- Data & Privacy --------------------------------------------------- #
    with tab_data:
        mood_df = load_mood_logs()
        journal_df = load_journal_entries()
        stat_tiles(
            [
                ("Mood check-ins", len(mood_df)),
                ("Journal entries", len(journal_df)),
            ]
        )
        st.caption("Everything lives in local files under `data/` — nothing leaves your device.")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Export mood logs (CSV)",
                data=mood_df.to_csv(index=False).encode("utf-8"),
                file_name="mindfulme_mood_logs.csv",
                mime="text/csv",
                width="stretch",
            )
        with c2:
            st.download_button(
                "⬇️ Export journal (CSV)",
                data=journal_df.to_csv(index=False).encode("utf-8"),
                file_name="mindfulme_journal.csv",
                mime="text/csv",
                width="stretch",
            )

        st.divider()
        st.markdown("**Danger zone**")
        st.caption(
            "This permanently deletes your profile, moods, journal and the local "
            "memory index. This cannot be undone."
        )
        confirm = st.checkbox("I understand this will erase all my data.")
        if st.button("Delete all my data", type="primary", disabled=not confirm):
            if clear_all_data():
                st.session_state.clear()
                st.success("All data cleared. Starting fresh…")
                st.rerun()
            else:
                st.error("Something went wrong while clearing data. Please try again.")

    # ---- About ------------------------------------------------------------ #
    with tab_about:
        st.markdown(
            f"""
            <div class="mm-card">
                <h3 class="mm-title">🌿 MindfulMe v{APP_VERSION}</h3>
                <p class="mm-subtext">Your AI personal mental wellness companion.</p>
                <p class="mm-subtext">Mood tracking · journaling · emotion analysis ·
                forecasting · patterns · a supportive companion · weekly insights.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            "MindfulMe is a supportive self-reflection tool, not a medical device "
            "or a substitute for professional care. If you are struggling, please "
            "reach out to a qualified professional or a crisis line.",
            icon="🫶",
        )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(profile: dict) -> str:
    """Sidebar with branding, navigation, and profile summary. Returns page."""
    with st.sidebar:
        st.markdown("## 🌿 MindfulMe")
        st.caption("Your personal mental wellness companion")
        st.divider()

        page = st.radio("Go to", NAV_SECTIONS, label_visibility="collapsed")
        st.divider()

        if profile.get("name"):
            st.markdown(f"**{profile['name']}**")
            if profile.get("age"):
                st.caption(f"Age {profile['age']}")
            if profile.get("challenges"):
                st.caption("Support areas: " + ", ".join(profile["challenges"]))
            st.caption(f"Support style: {profile.get('support_style', 'Gentle')}")

            if st.button("Edit profile", width="stretch"):
                st.session_state.editing_profile = not st.session_state.editing_profile
                st.rerun()

        st.divider()
        theme_toggle_widget(key="theme_toggle_sidebar")

        st.divider()
        st.caption(f"🌿 MindfulMe v{APP_VERSION} · take care of you.")

    return page


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    profile = st.session_state.profile

    inject_css(profile.get("theme", "Light"))

    # Onboarding takes over the whole screen until a profile exists.
    if not profile.get("onboarded") or not profile.get("name"):
        render_sidebar(profile)
        render_onboarding()
        return

    page = render_sidebar(profile)

    # Safety first: surface resources whenever recent data suggests distress.
    if not st.session_state.editing_profile:
        level = assess_current_distress()
        st.session_state.safety_level = level
        render_safety_card(level)

    if st.session_state.editing_profile:
        render_edit_profile()
    else:
        _render_page(page, profile)


def _render_page(page: str, profile: dict) -> None:
    """Dispatch to the selected page, with friendly error handling."""
    routes = {
        NAV_SECTIONS[0]: render_welcome,
        NAV_SECTIONS[1]: render_mood_checkin,
        NAV_SECTIONS[2]: render_journal,
        NAV_SECTIONS[3]: render_mood_forecast,
        NAV_SECTIONS[4]: render_patterns,
        NAV_SECTIONS[5]: render_companion,
        NAV_SECTIONS[6]: render_insights,
        NAV_SECTIONS[7]: render_settings,
    }
    renderer = routes.get(page, render_welcome)
    try:
        renderer(profile)
    except Exception as exc:  # noqa: BLE001 - keep the app warm, not broken
        st.error(
            "Something went a little sideways on this page — sorry about that. "
            "You can try again, or head somewhere else. 💚",
            icon="🌧️",
        )
        with st.expander("Technical details"):
            st.exception(exc)


if __name__ == "__main__":
    main()
