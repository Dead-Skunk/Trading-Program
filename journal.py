"""
journal.py - Trade journaling & expectancy tracking for AutoTraderPro
"""

import os
import json
import datetime
from typing import Dict, Any
from config import JOURNAL_DIR, get_logger
from ml_model import extract_features  # new import

log = get_logger(__name__)
os.makedirs(JOURNAL_DIR, exist_ok=True)


# ==============================
# Save Trade
# ==============================
def save_trade(trade: Dict[str, Any], df=None, context: Dict[str, Any] = None) -> None:
    """
    Save trade to daily journal file.

    Args:
        trade (dict): Trade object (must include schema keys).
        df (pd.DataFrame, optional): OHLCV data for feature extraction.
        context (dict, optional): Market context.
    """
    today = datetime.date.today().strftime("%Y%m%d")
    file_path = os.path.join(JOURNAL_DIR, f"{today}.txt")

    # Attach ML features if not already included
    if df is not None and context is not None:
        try:
            trade["features"] = extract_features(df, context)
        except Exception as e:
            log.error(f"Feature extraction failed: {e}")
            trade["features"] = {}
    else:
        trade.setdefault("features", {})

    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade) + "\n")
        log.info(
            f"✍️ Trade saved: {trade.get('id')} | "
            f"{trade.get('strategy','')} | entry={trade.get('entry')}"
        )
    except Exception as e:
        log.error(f"Failed to save trade: {e}")


# ==============================
# Update Trade Outcome
# ==============================
def update_trade_outcome(trade_id: str, outcome: str, pnl: float) -> None:
    """
    Update trade record outcome in journal file.

    Args:
        trade_id (str): Trade ID.
        outcome (str): Final outcome ("STOP", "TARGET", etc.).
        pnl (float): Profit/loss value.
    """
    today = datetime.date.today().strftime("%Y%m%d")
    file_path = os.path.join(JOURNAL_DIR, f"{today}.txt")

    if not os.path.exists(file_path):
        return

    updated = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                if record.get("id") == trade_id:
                    record["outcome"] = outcome
                    record["pnl"] = pnl
                updated.append(record)

        with open(file_path, "w", encoding="utf-8") as f:
            for record in updated:
                f.write(json.dumps(record) + "\n")

        log.info(f"✅ Trade {trade_id} updated: {outcome} ({pnl})")
    except Exception as e:
        log.error(f"Failed to update trade: {e}")


# ==============================
# Expectancy Calculation
# ==============================
def calculate_expectancy(strategy: str, lookback: int = 50) -> float:
    """
    Calculate expectancy for a given strategy over recent trades.

    Args:
        strategy (str): Strategy name or "all".
        lookback (int): Number of recent files to scan.

    Returns:
        float: Expectancy value.
    """
    trades = []
    for fname in sorted(os.listdir(JOURNAL_DIR))[-lookback:]:
        file_path = os.path.join(JOURNAL_DIR, fname)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if (strategy == "all" or record.get("strategy") == strategy) and "outcome" in record:
                        trades.append(record)
                except Exception:
                    continue

    if not trades:
        return 0.0

    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]

    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 1

    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    return float(expectancy)
