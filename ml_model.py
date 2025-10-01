"""
ml_model.py - Machine learning filter for AutoTraderPro
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config import JOURNAL_DIR, get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(JOURNAL_DIR, "ml_model.pkl")


# ==============================
# Feature Engineering
# ==============================
def extract_features(df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract ML features from market data.

    Args:
        df (pd.DataFrame): OHLCV data.
        context (dict): Market context.

    Returns:
        dict: Feature set.
    """
    try:
        features = {
            "close": float(df["close"].iloc[-1]),
            "ema_fast": float(df["close"].ewm(span=12, adjust=False).mean().iloc[-1]),
            "ema_slow": float(df["close"].ewm(span=26, adjust=False).mean().iloc[-1]),
            "atr": float((df["high"] - df["low"]).rolling(14).mean().iloc[-1]),
            "volume": float(df["volume"].iloc[-1]),
            "vix": float(context.get("vix", 0)),
        }
        return {k: (0.0 if pd.isna(v) else v) for k, v in features.items()}
    except Exception as e:
        log.error(f"Feature extraction error: {e}")
        return {
            "close": 0.0,
            "ema_fast": 0.0,
            "ema_slow": 0.0,
            "atr": 0.0,
            "volume": 0.0,
            "vix": 0.0,
        }


# ==============================
# Training
# ==============================
def train_ml_model() -> None:
    """Train or retrain ML model from journaled trades."""
    X, y = [], []
    try:
        for fname in os.listdir(JOURNAL_DIR):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(JOURNAL_DIR, fname), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        feats = rec.get("features", {})
                        if feats and "outcome" in rec:
                            X.append(list(feats.values()))
                            y.append(1 if rec["pnl"] > 0 else 0)
                    except Exception:
                        continue
    except Exception as e:
        log.error(f"ML training data scan failed: {e}")
        return

    if len(X) < 10:
        log.warning(f"⚠️ Not enough samples for training ({len(X)})")
        return

    X, y = np.array(X), np.array(y)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        log.info("📊 ML Training Report:\n" + classification_report(y_test, preds))

        joblib.dump(model, MODEL_PATH)
        log.info(f"✅ ML model trained + saved: {MODEL_PATH}")

        # Log feature importances
        importances = dict(zip([f"f{i}" for i in range(X.shape[1])], model.feature_importances_))
        log.info(f"🔎 Feature importances: {importances}")
    except Exception as e:
        log.error(f"ML training failed: {e}")


# ==============================
# Prediction
# ==============================
def ml_predict(data: Dict[str, Any]) -> Optional[float]:
    """
    Predict win probability for a trade setup.

    Args:
        data (dict): {bars, context}

    Returns:
        float or None: Probability of success.
    """
    if not os.path.exists(MODEL_PATH):
        log.warning("⚠️ No ML model found, skipping prediction")
        return None

    try:
        model = joblib.load(MODEL_PATH)
        df, context = data.get("bars"), data.get("context", {})
        if df is None or df.empty:
            return None
        feats = extract_features(df, context)
        X = np.array(list(feats.values())).reshape(1, -1)
        prob = model.predict_proba(X)[0, 1]
        return float(prob)
    except Exception as e:
        log.error(f"ML prediction error: {e}")
        return None
