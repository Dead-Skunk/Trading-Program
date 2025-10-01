"""
regime.py - Market regime detection for AutoTraderPro
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from config import get_logger

log = get_logger(__name__)


# ==============================
# Indicators
# ==============================
def atr(
    series_high: pd.Series,
    series_low: pd.Series,
    series_close: pd.Series,
    period: int = 14
) -> float:
    """
    Average True Range (ATR).

    Args:
        series_high (pd.Series): High prices.
        series_low (pd.Series): Low prices.
        series_close (pd.Series): Close prices.
        period (int): Lookback period (default=14).

    Returns:
        float: Latest ATR value.
    """
    try:
        high_low = series_high - series_low
        high_close = (series_high - series_close.shift()).abs()
        low_close = (series_low - series_close.shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_val = tr.rolling(period).mean().iloc[-1]
        return float(atr_val) if not np.isnan(atr_val) else 0.0
    except Exception as e:
        log.error(f"ATR calc error: {e}")
        return 0.0


def realized_vol(series: pd.Series, window: int = 20) -> float:
    """
    Annualized realized volatility.

    Args:
        series (pd.Series): Price series.
        window (int): Lookback period (default=20).

    Returns:
        float: Realized volatility (annualized).
    """
    try:
        returns = np.log(series / series.shift(1)).dropna()
        return float(np.std(returns[-window:]) * np.sqrt(252))
    except Exception as e:
        log.error(f"Realized vol error: {e}")
        return 0.0


# ==============================
# Regime Detection
# ==============================
def detect_regime(df: pd.DataFrame, context: Dict[str, Any]) -> str:
    """
    Detect current market regime.

    Categories:
        - "trend": directional momentum strong
        - "mean_reversion": oscillatory, RSI extremes
        - "high_iv": implied volatility elevated
        - "neutral": no clear bias

    Args:
        df (pd.DataFrame): OHLCV dataframe.
        context (dict): Market context (iv, etc.).

    Returns:
        str: Regime label.
    """
    try:
        price = df["close"].iloc[-1]
        atr_val = atr(df["high"], df["low"], df["close"])
        atr_pct = atr_val / price if price else 0.0

        rsi_val = _rsi(df["close"])
        realized = realized_vol(df["close"])
        implied = context.get("iv", {}).get("current_iv", 0.0)

        # High IV regime if implied >> realized
        if implied and implied > realized * 1.5:
            return "high_iv"

        # Trend regime if ATR% high and RSI not extreme
        if atr_pct > 0.015 and 40 < rsi_val < 60:
            return "trend"

        # Mean reversion if RSI extreme
        if rsi_val > 70 or rsi_val < 30:
            return "mean_reversion"

        return "neutral"

    except Exception as e:
        log.error(f"Regime detection error: {e}")
        return "neutral"


# ==============================
# RSI Helper
# ==============================
def _rsi(series: pd.Series, period: int = 14) -> float:
    """
    Relative Strength Index (RSI).

    Args:
        series (pd.Series): Price series.
        period (int): Lookback period.

    Returns:
        float: Latest RSI value.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    try:
        rs = gain / loss
        rsi_val = 100 - (100 / (1 + rs.iloc[-1]))
        return float(rsi_val) if not np.isnan(rsi_val) else 50.0
    except Exception:
        return 50.0  # Neutral RSI if calculation fails
