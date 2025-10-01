"""
performance.py - Capital & trade lifecycle analytics for AutoTraderPro
"""

import os
import json
import datetime
import math
from typing import Dict, Any, List
from config import JOURNAL_DIR, STARTING_EQUITY, get_logger

log = get_logger(__name__)
os.makedirs(JOURNAL_DIR, exist_ok=True)

# Internal tracking
_mfe_mae: Dict[str, Dict[str, float]] = {}  # {trade_id: {"mfe": x, "mae": y}}
_hold_times: Dict[str, float] = {}          # {trade_id: entry_timestamp}


# ==============================
# Track Trade (on entry)
# ==============================
def track_trade(trade: Dict[str, Any]) -> None:
    trade_id = trade.get("id")
    if not trade_id:
        return
    _mfe_mae[trade_id] = {"mfe": 0, "mae": 0}
    _hold_times[trade_id] = datetime.datetime.now().timestamp()
    log.info(f"📊 Tracking trade lifecycle: {trade_id}")


# ==============================
# Update MFE/MAE (during trade)
# ==============================
def update_trade_progress(trade_id: str, price: float, entry: float, confidence: float):
    """Update max favorable / adverse excursion."""
    if trade_id not in _mfe_mae:
        return
    pnl = (price - entry) * (1 if confidence > 0 else -1)
    _mfe_mae[trade_id]["mfe"] = max(_mfe_mae[trade_id]["mfe"], pnl)
    _mfe_mae[trade_id]["mae"] = min(_mfe_mae[trade_id]["mae"], pnl)


# ==============================
# Close Trade (record lifecycle stats)
# ==============================
def close_trade(trade: Dict[str, Any]) -> Dict[str, Any]:
    trade_id = trade.get("id")
    if not trade_id:
        return {}

    entry_time = _hold_times.pop(trade_id, datetime.datetime.now().timestamp())
    hold_time = datetime.datetime.now().timestamp() - entry_time

    mfe_mae = _mfe_mae.pop(trade_id, {"mfe": 0, "mae": 0})

    lifecycle = {
        "id": trade_id,
        "mfe": mfe_mae["mfe"],
        "mae": mfe_mae["mae"],
        "hold_time_sec": hold_time,
    }

    _save_lifecycle(lifecycle)
    log.info(f"📈 Trade lifecycle closed: {lifecycle}")
    return lifecycle


def _save_lifecycle(lifecycle: Dict[str, Any]):
    today = datetime.date.today().strftime("%Y%m%d")
    file_path = os.path.join(JOURNAL_DIR, f"{today}_lifecycle.txt")
    try:
        with open(file_path, "a") as f:
            f.write(json.dumps(lifecycle) + "\n")
    except Exception as e:
        log.error(f"Failed to save lifecycle: {e}")


# ==============================
# Capital Health
# ==============================
def capital_health(trades: List[Dict[str, Any]], equity: float) -> Dict[str, float]:
    """Compute utilization, drawdown, risk-of-ruin approximation."""
    try:
        max_equity = max([t.get("equity", equity) for t in trades] + [equity])
        drawdown = (max_equity - equity) / max_equity if max_equity else 0.0

        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losses = sum(1 for t in trades if t.get("pnl", 0) <= 0)
        total = wins + losses

        win_rate = wins / total if total else 0.5
        avg_win = (sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0) / wins) if wins else 1
        avg_loss = abs(
            (sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) <= 0) / losses)
        ) if losses else 1

        rr = avg_win / avg_loss if avg_loss else 1
        risk_of_ruin = 1 - (win_rate - (1 - win_rate) / rr) if rr else 1
        risk_of_ruin = max(min(risk_of_ruin, 1), 0)

        return {
            "drawdown": float(drawdown),
            "risk_of_ruin": float(risk_of_ruin),
            "utilization": float(equity / STARTING_EQUITY),
        }
    except Exception as e:
        log.error(f"Capital health error: {e}")
        return {"drawdown": 0, "risk_of_ruin": 0, "utilization": 0}


# ==============================
# PnL Attribution (for GUI)
# ==============================
def pnl_attribution(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate PnL attribution by strategy.

    Args:
        trades (list): List of trade objects.

    Returns:
        dict: {strategy: {"count": n, "net_pnl": x, "win_rate": y}}
    """
    results: Dict[str, Any] = {}
    try:
        if not trades:
            return results

        strategies = set(t.get("strategy", "unknown") for t in trades)
        for strat in strategies:
            strat_trades = [t for t in trades if t.get("strategy") == strat]
            wins = [t for t in strat_trades if t.get("pnl", 0) > 0]
            pnl_total = sum(t.get("pnl", 0) for t in strat_trades)

            results[strat] = {
                "count": len(strat_trades),
                "net_pnl": pnl_total,
                "win_rate": len(wins) / len(strat_trades) if strat_trades else 0.0,
            }

        return results
    except Exception as e:
        log.error(f"PnL attribution error: {e}")
        return {}
