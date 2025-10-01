"""
logic.py - Trade planning & risk management for AutoTraderPro
"""

from typing import Dict, Any, Optional
from datetime import datetime, date
import uuid
import pandas as pd

from config import (
    RISK_PER_TRADE_PCT,
    MAX_DAILY_LOSS_PCT,
    MAX_TRADES_PER_DAY,
    ATR_STOP_MULT,
    TARGET_RR,
    KELLY_CAP_PCT,
    CONFIDENCE_CUTOFF,
    get_logger,
)
from regime import atr as calc_atr  # unified ATR source

log = get_logger(__name__)
CONTRACT_MULTIPLIER = 100  # standard US equity options


# ==============================
# Account / Risk
# ==============================
class Account:
    """Tracks account equity, daily guardrails, and trade stats."""

    def __init__(self, starting_equity: float = 25000.0):
        self.starting_equity = starting_equity
        self.equity = starting_equity
        self.trades_today: int = 0
        self.net_pnl_today: float = 0.0
        self.trade_log = []
        self.last_reset: date = date.today()

    def reset_day(self) -> None:
        """Reset daily counters at new trading day."""
        self.trades_today = 0
        self.net_pnl_today = 0.0
        self.last_reset = date.today()

    def can_trade(self) -> bool:
        """Check if new trades are allowed under guardrails."""
        today = date.today()
        if today != self.last_reset:
            self.reset_day()

        if self.trades_today >= MAX_TRADES_PER_DAY:
            log.debug("Guardrail: max trades reached today")
            return False

        if abs(self.net_pnl_today) >= MAX_DAILY_LOSS_PCT * self.equity:
            log.debug("Guardrail: daily loss limit reached")
            return False

        return True

    def update_pnl(self, pnl: float) -> None:
        """Update account equity + daily PnL after a trade closes."""
        self.equity += pnl
        self.net_pnl_today += pnl
        self.trades_today += 1


# ==============================
# Trade Planning
# ==============================
def plan_trade(account: Account, entry: float, bars: pd.DataFrame, confidence: float) -> Dict[str, Any]:
    """
    Plan a trade with risk sizing, stops, and guardrails.

    Args:
        account (Account): Trading account.
        entry (float): Entry price.
        bars (pd.DataFrame): OHLCV bars (for ATR).
        confidence (float): Signal confidence score.

    Returns:
        dict: Trade object (valid=False if rejected).
    """
    trade: Dict[str, Any] = {"valid": False}

    if not account.can_trade():
        log.debug("Trade rejected: account guardrails triggered")
        return trade

    # ATR stop distance
    try:
        stop_dist = calc_atr(bars["high"], bars["low"], bars["close"]) * ATR_STOP_MULT
    except Exception:
        stop_dist = 0.0

    if stop_dist <= 0 or pd.isna(stop_dist):
        # fallback: 1% of price if ATR fails
        stop_dist = entry * 0.01

    risk_amt = account.equity * RISK_PER_TRADE_PCT
    pos_size = risk_amt / stop_dist if stop_dist > 0 else 0

    # Kelly cap
    kelly_size = account.equity * KELLY_CAP_PCT / entry if entry > 0 else 0
    final_size = int(max(1, min(pos_size, kelly_size)))

    if final_size < 1:
        log.debug(f"Trade rejected: position too small | size={final_size}")
        return trade

    if abs(confidence) < CONFIDENCE_CUTOFF:
        log.debug(f"Trade rejected: confidence too low ({confidence:.2f})")
        return trade

    # Build trade object
    stop = entry - stop_dist if confidence > 0 else entry + stop_dist
    target = entry + stop_dist * TARGET_RR if confidence > 0 else entry - stop_dist * TARGET_RR

    trade.update({
        "valid": True,
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": "",  # caller fills
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "size": final_size,
        "confidence": float(confidence),
        "outcome": "OPEN",
        "pnl": 0.0,
        "trailing_stop": stop,  # dynamic trailing stop
        "entry_time": datetime.now().timestamp(),
    })

    log.debug(
        f"Planned trade: size={final_size}, conf={confidence:.2f}, "
        f"entry={entry:.2f}, stop={stop:.2f}, target={target:.2f}"
    )
    return trade


# ==============================
# Exit Check
# ==============================
def check_exit(trade: Dict[str, Any], price: float, bars: pd.DataFrame, now: float) -> Optional[str]:
    """
    Check whether a trade should be exited.

    Args:
        trade (dict): Trade object.
        price (float): Current price.
        bars (pd.DataFrame): OHLCV bars.
        now (float): Current timestamp.

    Returns:
        str or None: "STOP", "TARGET", "TRAIL", "TIME", or None.
    """
    try:
        entry = trade["entry"]
        stop = trade["stop"]
        target = trade["target"]
        trailing = trade.get("trailing_stop", stop)
        direction = 1 if trade["confidence"] > 0 else -1

        # Stop loss
        if (direction > 0 and price <= stop) or (direction < 0 and price >= stop):
            return "STOP"

        # Profit target
        if (direction > 0 and price >= target) or (direction < 0 and price <= target):
            return "TARGET"

        # Trailing stop: move stop up in profitable direction
        if direction > 0 and price > entry:
            trade["trailing_stop"] = max(trailing, price - (price - entry) * 0.5)
            if price <= trade["trailing_stop"]:
                return "TRAIL"
        elif direction < 0 and price < entry:
            trade["trailing_stop"] = min(trailing, price + (entry - price) * 0.5)
            if price >= trade["trailing_stop"]:
                return "TRAIL"

        # Time-based exit (after 1 hour)
        if now - trade.get("entry_time", now) > 3600:
            return "TIME"

        return None
    except Exception as e:
        log.error(f"Exit check error: {e}")
        return None
