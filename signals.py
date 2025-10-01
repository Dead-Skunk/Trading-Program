"""
signals.py - Multi-strategy signal generation engine for AutoTraderPro
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, Any
from config import (
    STRATEGY_TOGGLES,
    CONFIDENCE_CUTOFF,
    SIGNAL_COOLDOWN_SEC,
    get_logger,
)
from strategy_optimizer import get_strategy_weight
from regime import detect_regime
from ml_model import ml_predict

log = get_logger(__name__)
_last_signal_time: float = 0  # global cooldown tracker


# ==============================
# Indicators
# ==============================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span).mean()


def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    try:
        rs = gain / loss
        rsi_val = 100 - (100 / (1 + rs.iloc[-1]))
        return float(rsi_val) if not np.isnan(rsi_val) else 50.0
    except Exception:
        return 50.0


def macd(series: pd.Series) -> float:
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    return float((ema12 - ema26).iloc[-1])


def bollinger(series: pd.Series, window: int = 20) -> Dict[str, float]:
    mean = series.rolling(window).mean().iloc[-1]
    std = series.rolling(window).std().iloc[-1]
    return {"upper": mean + 2 * std, "lower": mean - 2 * std}


# ==============================
# Strategies
# ==============================
def strat_vwap_ema(df: pd.DataFrame) -> int:
    ema_fast, ema_slow = ema(df["close"], 9).iloc[-1], ema(df["close"], 21).iloc[-1]
    return 1 if ema_fast > ema_slow else -1


def strat_breakout(df: pd.DataFrame) -> int:
    bb = bollinger(df["close"])
    price = df["close"].iloc[-1]
    return 1 if price > bb["upper"] else (-1 if price < bb["lower"] else 0)


def strat_mean_reversion(df: pd.DataFrame) -> int:
    rsi_val = rsi(df["close"])
    return -1 if rsi_val > 70 else (1 if rsi_val < 30 else 0)


def strat_orb(df: pd.DataFrame) -> int:
    """Opening range breakout (first 15 bars)."""
    if len(df) < 15:
        return 0
    first_15m = df.head(15)
    high, low = first_15m["high"].max(), first_15m["low"].min()
    price = df["close"].iloc[-1]
    return 1 if price > high else (-1 if price < low else 0)


def strat_expected_move_fade(context: Dict[str, Any]) -> int:
    price = context.get("price")
    em = context.get("expected_move")
    if not price or not em:
        return 0
    upper, lower = context["underlying_price"] + em, context["underlying_price"] - em
    return -1 if price > upper else (1 if price < lower else 0)


def strat_gamma_scalping(context: Dict[str, Any]) -> int:
    gamma = context.get("gamma_exposure", 0)
    return -1 if gamma > 0 else (1 if gamma < 0 else 0)


def strat_options_flow(context: Dict[str, Any]) -> int:
    flow_score = context.get("flow_score", 0)
    if flow_score > 0.7:
        return 1
    elif flow_score < -0.7:
        return -1
    return 0


# ==============================
# Master Signal Generator
# ==============================
def generate_signals(df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate entry signals across multiple strategies.

    Args:
        df (pd.DataFrame): OHLCV dataframe.
        context (dict): Market context (price, iv, gamma, flow, ml_cutoff).

    Returns:
        dict: {
            "score": float,
            "signals": dict,
            "blocked": bool,
            "regime": str,
            "ml_prob": float
        }
    """
    global _last_signal_time
    now = time.time()

    # Enforce cooldown
    if now - _last_signal_time < SIGNAL_COOLDOWN_SEC:
        return {"score": 0, "signals": {}, "blocked": True}

    # Guarantee context has "price"
    if "price" not in context:
        try:
            context["price"] = float(df["close"].iloc[-1])
        except Exception:
            context["price"] = 0.0

    signals, weights = {}, {}
    score = 0.0

    # Detect regime
    regime = detect_regime(df, context)

    # Apply strategies with toggles + regime filters
    if STRATEGY_TOGGLES.get("vwap_ema_trend") and regime in ["trend", "neutral"]:
        signals["vwap_ema_trend"] = strat_vwap_ema(df)
        weights["vwap_ema_trend"] = get_strategy_weight("vwap_ema_trend")

    if STRATEGY_TOGGLES.get("breakout") and regime == "trend":
        signals["breakout"] = strat_breakout(df)
        weights["breakout"] = get_strategy_weight("breakout")

    if STRATEGY_TOGGLES.get("mean_reversion") and regime in ["mean_reversion", "neutral"]:
        signals["mean_reversion"] = strat_mean_reversion(df)
        weights["mean_reversion"] = get_strategy_weight("mean_reversion")

    if STRATEGY_TOGGLES.get("orb"):
        signals["orb"] = strat_orb(df)
        weights["orb"] = get_strategy_weight("orb")

    if STRATEGY_TOGGLES.get("expected_move_fade") and context.get("expected_move") is not None:
        signals["expected_move_fade"] = strat_expected_move_fade(context)
        weights["expected_move_fade"] = get_strategy_weight("expected_move_fade")

    if STRATEGY_TOGGLES.get("gamma_scalping") and context.get("gamma_exposure") is not None:
        signals["gamma_scalping"] = strat_gamma_scalping(context)
        weights["gamma_scalping"] = get_strategy_weight("gamma_scalping")

    if STRATEGY_TOGGLES.get("options_flow") and context.get("flow_score") is not None:
        signals["options_flow"] = strat_options_flow(context)
        weights["options_flow"] = get_strategy_weight("options_flow")

    # Weighted score
    score = sum(signals[s] * weights.get(s, 1.0) for s in signals)

    # ML probability filter
    if context.get("disable_ml", False):
        ml_prob = 1.0  # approve all
    else:
        ml_prob = ml_predict(df, context)
        if ml_prob < context.get("ml_cutoff", 0.55):
            return {"score": 0, "signals": signals, "blocked": True, "ml_prob": ml_prob}

    # Confidence cutoff
    if abs(score) < CONFIDENCE_CUTOFF * sum(abs(w) for w in weights.values()):
        return {"score": score, "signals": signals, "blocked": True, "ml_prob": ml_prob}

    # Passed all filters → valid signal
    _last_signal_time = now
    return {
        "score": score,
        "signals": signals,
        "blocked": False,
        "regime": regime,
        "ml_prob": ml_prob,
    }
