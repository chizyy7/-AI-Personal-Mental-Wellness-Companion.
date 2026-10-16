"""
mood_forecaster.py
------------------
Phase 4 - Mood Forecasting.

Predicts the user's mood for the next 3-7 days from their diary of daily mood
check-ins. The primary engine is Facebook Prophet; if there isn't enough history
(or Prophet is unavailable), we fall back to a simple, honest linear-trend model
so the feature still works on small datasets.

The public API is deliberately tiny:

    forecast = forecast_mood(days=7)
    if forecast["ok"]:
        ... forecast["history"], forecast["forecast"] ...

Both DataFrames use the Prophet-style columns:
    history : ['ds', 'y']
    forecast: ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'is_future']
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.helpers import load_mood_logs

# Prophet is chatty on import/fit; keep the console (and Streamlit) calm.
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)

MOOD_MIN, MOOD_MAX = 1.0, 10.0  # valid mood range


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def _daily_series() -> pd.DataFrame:
    """
    Collapse mood logs into one row per day (mean of any same-day check-ins).

    Returns a DataFrame with columns ['ds', 'y'] sorted by date.
    """
    df = load_mood_logs()
    empty = pd.DataFrame(columns=["ds", "y"])
    if df.empty:
        return empty
    df = df.copy()
    df["ds"] = pd.to_datetime(df["date"], errors="coerce")
    df["y"] = pd.to_numeric(df["mood_score"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])
    if df.empty:
        return empty
    return df.groupby("ds", as_index=False)["y"].mean().sort_values("ds").reset_index(drop=True)


def _clip(values) -> np.ndarray:
    """Clamp forecast values to the valid 1-10 mood range."""
    return np.clip(np.asarray(values, dtype=float), MOOD_MIN, MOOD_MAX)


# --------------------------------------------------------------------------- #
# Trend fallback
# --------------------------------------------------------------------------- #
def _trend_forecast(series: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Simple linear-trend + noise-band forecast used when there's too little data
    (or Prophet fails). Honest and dependency-free.
    """
    x = np.arange(len(series))
    y = series["y"].to_numpy(dtype=float)

    if len(series) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0.0, float(y.mean())

    # Residual spread gives a (very rough) uncertainty band.
    fitted = slope * x + intercept
    resid_std = float(np.std(y - fitted)) if len(series) >= 3 else 1.5
    band = max(1.0, min(2.5, resid_std * 1.5))

    last_date = series["ds"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_x = np.arange(len(series), len(series) + days)
    yhat = slope * future_x + intercept

    return pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": _clip(yhat),
            "yhat_lower": _clip(yhat - band),
            "yhat_upper": _clip(yhat + band),
            "is_future": True,
        }
    )


# --------------------------------------------------------------------------- #
# Prophet forecast
# --------------------------------------------------------------------------- #
def _prophet_forecast(series: pd.DataFrame, days: int) -> Optional[pd.DataFrame]:
    """
    Run Prophet and return a forecast DataFrame, or None if it fails.

    Weekly seasonality is only enabled once we have ~2 weeks of history, since
    it's meaningless (and unstable) on tiny datasets.
    """
    try:
        from prophet import Prophet
    except Exception:  # noqa: BLE001 - Prophet not installed / not importable
        return None

    try:
        model = Prophet(
            interval_width=0.8,
            daily_seasonality=False,
            weekly_seasonality=len(series) >= 14,
            yearly_seasonality=False,
        )
        # Prophet needs a little help staying calm on small data.
        model.fit(series, iter=300)

        future = model.make_future_dataframe(periods=days, freq="D")
        fc = model.predict(future)

        last_history_date = series["ds"].max()
        fc = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        fc["yhat"] = _clip(fc["yhat"])
        fc["yhat_lower"] = _clip(fc["yhat_lower"])
        fc["yhat_upper"] = _clip(fc["yhat_upper"])
        fc["is_future"] = fc["ds"] > last_history_date
        return fc.reset_index(drop=True)
    except Exception:  # noqa: BLE001 - any Prophet hiccup -> graceful fallback
        return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def forecast_mood(days: int = 7) -> Dict[str, object]:
    """
    Forecast mood for the next `days` days (3-7 recommended).

    Returns a dict:
        ok        : bool  - whether a forecast was produced
        method    : str   - 'prophet' | 'trend' | None
        history   : DataFrame['ds','y']
        forecast  : DataFrame['ds','yhat','yhat_lower','yhat_upper','is_future']
        message   : str   - gentle, human-readable context for the UI
    """
    days = int(max(1, min(14, days)))
    series = _daily_series()

    if series.empty:
        return {
            "ok": False,
            "method": None,
            "history": series,
            "forecast": pd.DataFrame(),
            "message": (
                "I don't have any mood check-ins yet. Once you log a few days, "
                "I'll be able to gently look ahead with you. 🌱"
            ),
        }

    history = series.copy()

    # Prefer Prophet; require at least 4 points for it to be meaningful.
    forecast = None
    method = None
    if len(series) >= 4:
        forecast = _prophet_forecast(series, days)
        if forecast is not None:
            method = "prophet"

    # Fallback to the transparent trend model.
    if forecast is None:
        forecast = _trend_forecast(series, days)
        method = "trend"

    if len(series) < 4:
        message = (
            "This is an early, gentle estimate based on just a few check-ins. "
            "The more you log, the clearer the picture becomes. 🌿"
        )
    elif method == "prophet":
        message = (
            "Here's a soft look at the days ahead, based on your recent "
            "patterns. It's a guide, not a verdict — feelings still surprise us. ☁️"
        )
    else:
        message = (
            "Prophet wasn't available, so this is a simple trend estimate. "
            "Still useful, just a little more cautious. 🌤️"
        )

    return {
        "ok": True,
        "method": method,
        "history": history,
        "forecast": forecast,
        "message": message,
    }


def forecast_summary(forecast_result: Dict[str, object]) -> str:
    """Return a one-line, warm summary of the coming days (or a gentle note)."""
    if not forecast_result.get("ok"):
        return str(forecast_result.get("message", ""))

    fc = forecast_result["forecast"]  # type: ignore[index]
    future = fc[fc["is_future"]]
    if future.empty:
        return "No upcoming days to look at just yet."

    avg = float(future["yhat"].mean())
    first = float(future.iloc[0]["yhat"])
    last = float(future.iloc[-1]["yhat"])
    direction = "steady" if abs(last - first) < 0.6 else ("gently rising" if last > first else "gently dipping")

    if avg >= 6.5:
        feel = "on the brighter side"
    elif avg >= 4.5:
        feel = "fairly mixed"
    else:
        feel = "on the heavier side"

    return (
        f"Over the next {len(future)} day(s), your mood looks {feel} and {direction}. "
        "Take it as a kind heads-up, not a promise. 💚"
    )
