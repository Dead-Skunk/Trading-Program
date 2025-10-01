"""
logic.py - Trade planning & risk management for AutoTraderPro
"""

from typing import Dict, Any, Optional
from datetime import datetime, date
import uuid

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
# ACCOUNT / RISK
# ==============================
class Account:
    """Tracks account equity, daily limits, and trade count."""

    def __init__(self, starting_equity: float = 25000.0):
        self.starting_equity = starting_equity
        self.equity = starting_equity
        self.trades_today = 0
        self.daily_loss = 0.0
        self.trade_log = []
        self.last_reset = date.today()

    def reset_day(self) -> None:
        """Reset daily counters."""
        self.trades_today = 0
        self.daily_loss = 0.0
        self.last_reset = date.today()

    def can_trade(self) -> bool:
        """Check guardrails (max trades, max daily loss)."""
        today = date.today()
        if today != self.last_reset:
            self.reset_day()

        if self.trades_today >= MAX_TRADES_PER_DAY:
            log.debug(f"Guardrail: max trades per day reached ({MAX_TRADES_PER_DAY})")
            return False
        if abs(self.daily_loss) >= MAX_DAILY_LOSS_PCT * self.equity:
            log.debug(f"Guardrail: daily loss limit reached ({self.daily_loss})")
            return False
        return True

    def update_pnl(self, pnl: float) -> None:
        """Update equity and daily stats after a trade closes."""
        self.equity += pnl
        self.daily_loss += pnl
        self.trades_today += 1


# ==============================
# TRADE PLANNING
# ==============================
def plan_trade(account: Account, entry: float, bars, confidence: float) -> Dict[str, Any]:
    """
    Plan a trade with risk sizing and guardrails.

    Args:
        account (Account): Trading account.
        entry (float): Entry price.
        bars (pd.DataFrame): OHLCV bars (for ATR).
        confidence (float): Signal confidence.

    Returns:
        dict: Trade object (valid=False if rejected).
    """
    trade: Dict[str, Any] = {"valid": False}

    # Guardrails
    if not account.can_trade():
        log.debug("Rejected trade: account guardrails triggered")
        return trade

    # ATR stop distance
    try:
        stop_dist = calc_atr(bars["high"], bars["low"], bars["close"]) * ATR_STOP_MULT
    except Exception:
        stop_dist = 0.0

    risk_amt = account.equity * RISK_PER_TRADE_PCT
    pos_size = risk_amt / stop_dist if stop_dist > 0 else 0

    if pos_size <= 0:
        log.debug(f"Rejected trade: position size <= 0 | risk_amt={risk_amt}, stop_dist={stop_dist}")
        return trade

    # Apply Kelly cap
    kelly_size = account.equity * KELLY_CAP_PCT / entry if entry > 0 else 0
    final_size = min(pos_size, kelly_size)

    if final_size < 1:
        log.debug(f"Rejected trade: final_size < 1 | final_size={final_size}")
        return trade

    # Confidence check
    if abs(confidence) < CONFIDENCE_CUTOFF:
        log.debug(f"Rejected trade: confidence too low ({confidence:.2f}) vs cutoff {CONFIDENCE_CUTOFF}")
        return trade

    # Passed all filters → Build trade
    stop = entry - stop_dist if confidence > 0 else entry + stop_dist
    target = entry + stop_dist * TARGET_RR if confidence > 0 else entry - stop_dist * TARGET_RR

    trade.update({
        "valid": True,
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": "",  # filled by caller
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "size": int(final_size),
        "confidence": float(confidence),
        "outcome": "OPEN",
        "pnl": 0.0,
    })
    log.debug(f"Planned trade: size={int(final_size)}, confidence={confidence:.2f}, entry={entry}")
    return trade


# ==============================
# EXIT CHECK
# ==============================
def check_exit(trade: Dict[str, Any], price: float, bars, entry_time: float) -> Optional[str]:
    """
    Check if trade should be exited based on stop, target, or time.

    Args:
        trade (dict): Trade object.
        price (float): Current price.
        bars (pd.DataFrame): OHLCV bars (for ATR/trailing stops if needed).
        entry_time (float): Timestamp of entry.

    Returns:
        str or None: "STOP", "TARGET", or None.
    """
    try:
        entry = trade["entry"]
        stop = trade["stop"]
        target = trade["target"]
        direction = 1 if trade["confidence"] > 0 else -1

        if (direction > 0 and price <= stop) or (direction < 0 and price >= stop):
            return "STOP"
        if (direction > 0 and price >= target) or (direction < 0 and price <= target):
            return "TARGET"

        # TODO: implement TIME and TRAIL exits if desired
        return None
    except Exception as e:
        log.error(f"Exit check error: {e}")
        return None
