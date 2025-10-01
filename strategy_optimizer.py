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

# Default strategy weights
_strategy_weights: Dict[str, float] = {
    "vwap_ema_trend": 1.0,
    "breakout": 1.0,
    "mean_reversion": 1.0,
    "orb": 1.0,
    "expected_move_fade": 1.0,
    "gamma_scalping": 1.0,
    "options_flow": 1.0,
}

# File for persistence
WEIGHTS_FILE = os.path.join(JOURNAL_DIR, "strategy_weights.json")


# ==============================
# Load / Save
# ==============================
def load_weights() -> None:
    """Load strategy weights from JSON file into memory."""
    global _strategy_weights
    if not os.path.exists(WEIGHTS_FILE):
        log.info("⚠️ No saved strategy weights file found, using defaults")
        return
    try:
        with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            _strategy_weights.update(loaded)
            log.info(f"✅ Strategy weights loaded: {loaded}")
        else:
            log.warning("⚠️ Malformed weights file, ignoring")
    except Exception as e:
        log.error(f"Failed to load strategy weights: {e}")


def save_weights() -> None:
    """Persist current strategy weights to JSON atomically."""
    try:
        tmp_path = f"{WEIGHTS_FILE}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_strategy_weights, f, indent=2)
        os.replace(tmp_path, WEIGHTS_FILE)
        log.info(f"💾 Strategy weights saved: {WEIGHTS_FILE}")
    except Exception as e:
        log.error(f"Failed to save strategy weights: {e}")


# ==============================
# Update Weights
# ==============================
def update_weights() -> None:
    """
    Recalculate strategy weights using expectancy.
    Applies EMA smoothing and caps extreme values.
    """
    updated = {}
    try:
        for strat in _strategy_weights:
            exp = calculate_expectancy(strat)
            if exp != 0:
                prev = _strategy_weights[strat]
                new_val = (EXPECTANCY_SMOOTHING * exp) + ((1 - EXPECTANCY_SMOOTHING) * prev)
                updated[strat] = max(min(new_val, 3.0), -3.0)  # cap between -3 and 3
            else:
                updated[strat] = _strategy_weights[strat]

        # Count trades across journal
        trade_count = 0
        try:
            for f in os.listdir(JOURNAL_DIR):
                file_path = os.path.join(JOURNAL_DIR, f)
                if not os.path.isfile(file_path):
                    continue
                with open(file_path, "r", encoding="utf-8") as fh:
                    trade_count += sum(1 for _ in fh)
        except Exception as e:
            log.error(f"Trade count scan error: {e}")

        if trade_count >= OPTIMIZER_MIN_TRADES:
            _strategy_weights.update(updated)
            save_weights()
            log.info(f"📊 Strategy weights updated: {updated}")
        else:
            log.info(
                f"⚠️ Not enough trades yet ({trade_count}/{OPTIMIZER_MIN_TRADES}) "
                f"— skipping weight update"
            )
    except Exception as e:
        log.error(f"Weight update error: {e}")


# ==============================
# Access Weight
# ==============================
def get_strategy_weight(strategy: str) -> float:
    """Retrieve current weight for a given strategy (default=1.0)."""
    return _strategy_weights.get(strategy, 1.0)


# Initialize on import
load_weights()
