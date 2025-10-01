"""
system_check.py - Pre-flight system check for AutoTraderPro
"""

import importlib
import requests
from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_DATA_URL,
    POLYGON_API_KEY,
    POLYGON_BASE_URL,
    DISCORD_ENABLED,
    DISCORD_WEBHOOK,
    get_logger,
)

log = get_logger(__name__)


# ==============================
# Dependency Check
# ==============================
def check_dependencies() -> bool:
    """Verify required Python dependencies are installed."""
    deps = ["pandas", "numpy", "aiohttp", "yfinance", "customtkinter", "scikit-learn"]
    ok = True
    for dep in deps:
        try:
            importlib.import_module(dep)
            log.info(f"✅ Dependency OK: {dep}")
        except ImportError:
            log.error(f"❌ Missing dependency: {dep}")
            ok = False
    return ok


# ==============================
# Alpaca Connectivity
# ==============================
def check_alpaca() -> bool:
    """Test Alpaca data API connectivity."""
    try:
        url = f"{ALPACA_DATA_URL}/stocks/SPY/bars?timeframe=1Min&limit=1"
        resp = requests.get(url, auth=(ALPACA_API_KEY, ALPACA_SECRET_KEY), timeout=10)
        if resp.status_code == 200:
            log.info("✅ Alpaca connectivity OK")
            return True
        else:
            log.error(f"❌ Alpaca check failed ({resp.status_code})")
            return False
    except Exception as e:
        log.error(f"❌ Alpaca error: {e}")
        return False


# ==============================
# Polygon Connectivity
# ==============================
def check_polygon() -> bool:
    """Test Polygon options API connectivity."""
    try:
        url = f"{POLYGON_BASE_URL}/v3/reference/options/contracts?underlying_ticker=SPY&limit=1&apiKey={POLYGON_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            log.info("✅ Polygon connectivity OK")
            return True
        else:
            log.error(f"❌ Polygon check failed ({resp.status_code})")
            return False
    except Exception as e:
        log.error(f"❌ Polygon error: {e}")
        return False


# ==============================
# Discord Webhook
# ==============================
def check_discord() -> bool:
    """Test Discord webhook connectivity."""
    if not DISCORD_ENABLED:
        log.info("⚠️ Discord not configured")
        return True
    try:
        resp = requests.post(DISCORD_WEBHOOK, json={"content": "🔔 AutoTraderPro system check"}, timeout=10)
        if resp.status_code == 204:
            log.info("✅ Discord webhook OK")
            return True
        else:
            log.error(f"❌ Discord check failed ({resp.status_code})")
            return False
    except Exception as e:
        log.error(f"❌ Discord error: {e}")
        return False


# ==============================
# Master Pre-flight
# ==============================
def run_system_check() -> bool:
    """Run all system checks before trading starts."""
    log.info("🚀 Running system check...")
    deps = check_dependencies()
    alpaca_ok = check_alpaca()
    polygon_ok = check_polygon()
    discord_ok = check_discord()
    return all([deps, alpaca_ok, polygon_ok, discord_ok])
