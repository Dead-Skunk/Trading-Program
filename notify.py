"""
notify.py - Discord notifications for AutoTraderPro
"""

import time
import requests
from typing import Dict, Any
from config import DISCORD_WEBHOOK, DISCORD_ENABLED, DEV_MODE, get_logger

log = get_logger(__name__)


# ==============================
# Internal Sender
# ==============================
def _send_discord(payload: Dict[str, Any]) -> bool:
    """
    Send a message to Discord webhook with retry + backoff.

    Args:
        payload (dict): Discord message payload.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not DISCORD_ENABLED or DEV_MODE:
        log.info(f"🔔 [DEV] Discord alert: {payload}")
        return True

    # Ensure payload content is not None
    if not payload.get("content"):
        log.warning("⚠️ Discord payload missing content")
        return False

    backoff = 1
    for attempt in range(5):
        try:
            resp = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
            if resp.status_code == 204:
                return True
            log.warning(f"Discord send failed (status {resp.status_code}), attempt {attempt+1}/5")
        except Exception as e:
            log.error(f"Discord send error: {e}")
        time.sleep(backoff)
        backoff *= 2
    return False


# ==============================
# Alert Wrappers
# ==============================
def alert_entry(trade: Dict[str, Any]) -> None:
    """Send entry alert for a new trade."""
    payload = {
        "content": (
            f"📈 **ENTRY** {trade.get('strategy','')} | "
            f"Entry: {trade.get('entry')} | "
            f"Stop: {trade.get('stop')} | "
            f"Target: {trade.get('target')} | "
            f"Size: {trade.get('size')} | "
            f"Confidence: {trade.get('confidence', 0):.2f}"
        )
    }
    _send_discord(payload)


def alert_exit(trade: Dict[str, Any], outcome: str, pnl: float) -> None:
    """Send exit alert when a trade closes."""
    payload = {
        "content": (
            f"📉 **EXIT** {trade.get('strategy','')} | "
            f"Outcome: {outcome} | "
            f"PnL: {pnl:.2f}"
        )
    }
    _send_discord(payload)


def alert_lockout(reason: str) -> None:
    """Send lockout alert (e.g., VAR exceeded, guardrail breach)."""
    payload = {"content": f"🚫 **LOCKOUT**: {reason}"}
    _send_discord(payload)


def alert_heartbeat(equity: float, trades_today: int) -> None:
    """Send heartbeat with equity + trade count."""
    payload = {
        "content": f"❤️ **Heartbeat** | Equity: {equity:.2f} | Trades today: {trades_today}"
    }
    _send_discord(payload)


def alert_error(module: str, error: str) -> None:
    """Send error alert for a module failure."""
    payload = {"content": f"⚠️ **Error in {module}**: {error}"}
    _send_discord(payload)
