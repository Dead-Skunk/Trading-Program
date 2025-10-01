"""
ml_model.py - Machine learning overlay for AutoTraderPro
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from config import JOURNAL_DIR, get_logger

log = get_logger(__name__)

# Global state
_model = None
_warned_not_enough_data = False
_features = ["rsi", "macd", "atr_pct", "iv_rank", "iv_percentile", "regime_trend", "flow_score"]


# ==============================
# Feature Extraction
# ==============================
def extract_features(df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract ML features from OHLCV + market context.

    Args:
        df (pd.DataFrame): OHLCV bars.
        context (dict): Market context (iv, regime, flow, etc.).

    Returns:
        dict: Feature vector.
    """
    try:
        price = df["close"].iloc[-1]
        atr_val = df["close"].rolling(14).std().iloc[-1]
        atr_pct = atr_val / price if price else 0

        rsi_val = _rsi(df["close"])
        macd_val = _macd(df["close"])

        return {
            "rsi": rsi_val,
            "macd": macd_val,
            "atr_pct": atr_pct,
            "iv_rank": context.get("iv", {}).get("iv_rank", 0),
            "iv_percentile": context.get("iv", {}).get("iv_percentile", 0),
            "regime_trend": 1 if context.get("regime") == "trend" else 0,
            "flow_score": context.get("flow_score", 0),
        }
    except Exception as e:
        log.error(f"Feature extraction error: {e}")
        return {f: 0 for f in _features}


# ==============================
# Model Training
# ==============================
def train_model():
    """Train ML model from journal history."""
    global _model, _warned_not_enough_data
    X, y = [], []

    for fname in os.listdir(JOURNAL_DIR):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(JOURNAL_DIR, fname), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if "outcome" in record and "features" in record:
                        X.append([record["features"].get(feat, 0) for feat in _features])
                        y.append(1 if record.get("pnl", 0) > 0 else 0)
                except Exception:
                    continue

    if len(X) < 50:
        if not _warned_not_enough_data:
            log.warning("⚠️ Not enough trades to train ML model (need ≥50)")
            _warned_not_enough_data = True
        return

    X = np.array(X)
    y = np.array(y)
    _model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    _model.fit(X, y)
    log.info(f"🤖 ML model trained on {len(X)} samples")


# ==============================
# Prediction
# ==============================
def ml_predict(df: pd.DataFrame, context: Dict[str, Any]) -> float:
    """
    Predict probability of success for a trade setup.

    Args:
        df (pd.DataFrame): OHLCV bars.
        context (dict): Market context.

    Returns:
        float: Probability (0–1).
    """
    global _model

    if context.get("disable_ml", False):
        return 1.0  # bypass ML filter

    if _model is None:
        train_model()
        if _model is None:
            return 0.5  # neutral until trained

    feats = extract_features(df, context)
    X = np.array([[feats[f] for f in _features]])
    try:
        prob = _model.predict_proba(X)[0][1]
        return float(prob)
    except Exception as e:
        log.error(f"ML predict error: {e}")
        return 0.5


# ==============================
# Helpers
# ==============================
def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    try:
        rs = gain / loss
        return float(100 - (100 / (1 + rs.iloc[-1]))) if loss.iloc[-1] != 0 else 100
    except Exception:
        return 50.0


def _macd(series: pd.Series) -> float:
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    return float((ema12 - ema26).iloc[-1])
