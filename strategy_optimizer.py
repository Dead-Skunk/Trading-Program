"""
strategy_optimizer.py - Adaptive strategy weighting for AutoTraderPro
"""

import os
import json
from typing import Dict
from config import (
    JOURNAL_DIR,
    OPTIMIZER_MIN_TRADES,
    EXPECTANCY_SMOOTHING,
    get_logger,
)
from journal import calculate_expectancy

log = get_logger(__name__)

# Strategy weights memory
_strategy_weights: Dict[str, float] = {
    "vwap_ema_trend": 1.0,
    "breakout": 1.0,
    "mean_reversion": 1.0,
    "orb": 1.0,
    "expected_move_fade": 1.0,
    "gamma_scalping": 1.0,
    "options_flow": 1.0,
}

# Persist weights across sessions
WEIGHTS_FILE = os.path.join(JOURNAL_DIR, "strategy_weights.json")


# ==============================
# Load / Save Weights
# ==============================
def load_weights() -> None:
    """Load strategy weights from JSON file."""
    global _strategy_weights
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
                _strategy_weights = json.load(f)
            log.info("✅ Strategy weights loaded")
        except Exception as e:
            log.error(f"Failed to load weights: {e}")


def save_weights() -> None:
    """Persist current strategy weights to JSON file."""
    try:
        with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(_strategy_weights, f, indent=2)
    except Exception as e:
        log.error(f"Failed to save weights: {e}")


# ==============================
# Update Weights
# ==============================
def update_weights() -> None:
    """
    Recalculate strategy weights from expectancy.
    Applies EMA smoothing and caps extreme values.
    """
    updated = {}
    try:
        for strat in _strategy_weights:
            exp = calculate_expectancy(strat)
            if exp != 0:
                prev = _strategy_weights[strat]
                new_val = (EXPECTANCY_SMOOTHING * exp) + ((1 - EXPECTANCY_SMOOTHING) * prev)
                updated[strat] = max(min(new_val, 3.0), -3.0)  # cap weights
            else:
                updated[strat] = _strategy_weights[strat]

        # Count trades in journal
        trade_count = sum(
            1 for f in os.listdir(JOURNAL_DIR) if os.path.isfile(os.path.join(JOURNAL_DIR, f))
            for _ in open(os.path.join(JOURNAL_DIR, f), "r", encoding="utf-8")
        )

        if trade_count >= OPTIMIZER_MIN_TRADES:
            _strategy_weights.update(updated)
            save_weights()
            log.info(f"📊 Strategy weights updated: {updated}")
        else:
            log.info(f"⚠️ Not enough trades yet ({trade_count}/{OPTIMIZER_MIN_TRADES}) for optimizer")
    except Exception as e:
        log.error(f"Weight update error: {e}")


# ==============================
# Access Weight
# ==============================
def get_strategy_weight(strategy: str) -> float:
    """Retrieve current weight for a strategy (default=1.0)."""
    return _strategy_weights.get(strategy, 1.0)


# Initialize weights on import
load_weights()
