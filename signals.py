"""
signals.py - Signal generation engine for AutoTraderPro
"""

import time
import pandas as pd
from typing import Dict, Any
from config import (
    CONFIDENCE_CUTOFF,
    SIGNAL_COOLDOWN_SEC,
    STRATEGY_TOGGLES,
    get_logger,
)
from ml_model import ml_predict

log = get_logger(__name__)

# Cooldown tracking per symbol
_last_signal_time: Dict[str, float] = {}


# ==============================
# Utility Indicators
# ==============================
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (trader-style)."""
    return series.ewm(span=span, adjust=False).mean()


# ==============================
# Individual Strategies
# ==============================
def vwap_ema_trend(bars: pd.DataFrame) -> int:
    """Trend strategy: price above VWAP + EMA(20) = long bias."""
    if bars is None or bars.empty:
        return 0
    try:
        bars = bars.copy()
        bars["ema20"] = ema(bars["close"], 20)
        bars["cum_vol"] = bars["volume"].cumsum()
        bars["cum_pv"] = (bars["close"] * bars["volume"]).cumsum()
        bars["vwap"] = bars["cum_pv"] / bars["cum_vol"]

        latest = bars.iloc[-1]
        if latest["close"] > latest["vwap"] and latest["close"] > latest["ema20"]:
            return 1
        elif latest["close"] < latest["vwap"] and latest["close"] < latest["ema20"]:
            return -1
        return 0
    except Exception as e:
        log.error(f"VWAP/EMA trend error: {e}")
        return 0


def breakout(bars: pd.DataFrame, lookback: int = 20) -> int:
    """Breakout strategy: close > recent high = long, close < recent low = short."""
    if bars is None or len(bars) < lookback:
        return 0
    try:
        high = bars["high"].tail(lookback).max()
        low = bars["low"].tail(lookback).min()
        last_close = bars["close"].iloc[-1]

        if last_close > high:
            return 1
        elif last_close < low:
            return -1
        return 0
    except Exception as e:
        log.error(f"Breakout error: {e}")
        return 0


def mean_reversion(bars: pd.DataFrame, lookback: int = 20) -> int:
    """Mean reversion: fade closes 2 std dev from mean."""
    if bars is None or len(bars) < lookback:
        return 0
    try:
        mean = bars["close"].tail(lookback).mean()
        std = bars["close"].tail(lookback).std()
        last_close = bars["close"].iloc[-1]

        if last_close > mean + 2 * std:
            return -1
        elif last_close < mean - 2 * std:
            return 1
        return 0
    except Exception as e:
        log.error(f"Mean reversion error: {e}")
        return 0


def expected_move_fade(context: Dict[str, Any]) -> int:
    """Fade if move exceeds expected move band."""
    try:
        if not context:
            return 0
        price = context.get("price")
        exp_move = context.get("expected_move")
        if not price or not exp_move:
            return 0
        if price > exp_move:
            return -1
        elif price < -exp_move:
            return 1
        return 0
    except Exception as e:
        log.error(f"Expected move fade error: {e}")
        return 0


def orb(bars: pd.DataFrame) -> int:
    """Opening range breakout: fade against range extremes."""
    if bars is None or bars.empty:
        return 0
    try:
        if len(bars) < 30:
            return 0
        open_range = bars.iloc[:5]
        high, low = open_range["high"].max(), open_range["low"].min()
        last_close = bars["close"].iloc[-1]

        if last_close > high:
            return 1
        elif last_close < low:
            return -1
        return 0
    except Exception as e:
        log.error(f"ORB error: {e}")
        return 0


def gamma_scalping(context: Dict[str, Any]) -> int:
    """Gamma scalping: fade high positive/negative gamma exposure."""
    try:
        gex = context.get("gamma_exposure", 0)
        if gex > 1e9:
            return -1
        elif gex < -1e9:
            return 1
        return 0
    except Exception as e:
        log.error(f"Gamma scalping error: {e}")
        return 0


def options_flow(context: Dict[str, Any]) -> int:
    """Simple flow bias: follow call vs. put order flow."""
    try:
        flow_score = context.get("flow_score", 0)
        if flow_score > 0.6:
            return 1
        elif flow_score < -0.6:
            return -1
        return 0
    except Exception as e:
        log.error(f"Options flow error: {e}")
        return 0


# ==============================
# Master Signal Engine
# ==============================
def generate_signals(symbol: str, data: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Aggregate strategy signals + ML filter.

    Args:
        symbol (str): Ticker symbol.
        data (dict): Market context (bars, context vars).
        weights (dict): Strategy weights.

    Returns:
        dict: {
            "signal": int (-1,0,1),
            "score": float,
            "strategies": dict
        }
    """
    now = time.time()
    last_time = _last_signal_time.get(symbol, 0)
    if now - last_time < SIGNAL_COOLDOWN_SEC:
        return {"signal": 0, "score": 0.0, "strategies": {}}

    bars = data.get("bars")
    context = data.get("context", {})

    signals = {}
    try:
        if STRATEGY_TOGGLES.get("vwap_ema_trend"):
            signals["vwap_ema_trend"] = vwap_ema_trend(bars)
        if STRATEGY_TOGGLES.get("breakout"):
            signals["breakout"] = breakout(bars)
        if STRATEGY_TOGGLES.get("mean_reversion"):
            signals["mean_reversion"] = mean_reversion(bars)
        if STRATEGY_TOGGLES.get("expected_move_fade"):
            signals["expected_move_fade"] = expected_move_fade(context)
        if STRATEGY_TOGGLES.get("orb"):
            signals["orb"] = orb(bars)
        if STRATEGY_TOGGLES.get("gamma_scalping"):
            signals["gamma_scalping"] = gamma_scalping(context)
        if STRATEGY_TOGGLES.get("options_flow"):
            signals["options_flow"] = options_flow(context)
    except Exception as e:
        log.error(f"Signal generation error: {e}")

    # Weighted score
    score = sum(signals.get(s, 0) * weights.get(s, 1.0) for s in signals)
    n_active = sum(abs(signals.get(s, 0)) for s in signals)
    score = score / max(n_active, 1)

    # ML filter (cached features per call)
    try:
        ml_prob = ml_predict(data)
        if ml_prob is not None and ml_prob < CONFIDENCE_CUTOFF:
            log.info(f"ML filter rejected trade (prob={ml_prob:.2f})")
            signal = 0
        else:
            signal = 1 if score > 0 else -1 if score < 0 else 0
    except Exception as e:
        log.error(f"ML prediction error: {e}")
        signal = 0

    _last_signal_time[symbol] = now
    return {"signal": signal, "score": float(score), "strategies": signals}
