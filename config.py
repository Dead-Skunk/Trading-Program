"""
config.py - Master Control Panel for AutoTraderPro
Centralizes all system parameters and settings.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo

# ==============================
# ACCOUNT SETTINGS
# ==============================
STARTING_EQUITY: float = 12_508.0  # Active trading equity (buying power only)

# ==============================
# API KEYS & ENDPOINTS
# (kept hardcoded per user request)
# ==============================
ALPACA_API_KEY: str = ""
ALPACA_SECRET_KEY: str = ""
ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL: str = "https://data.alpaca.markets/v2"

POLYGON_API_KEY: str = ""
POLYGON_BASE_URL: str = "https://api.polygon.io"

DISCORD_WEBHOOK: str = (
    ""
)
DISCORD_ENABLED: bool = True

# ==============================
# SYSTEM SETTINGS
# ==============================
REFRESH_INTERVAL_SEC: int = 2        # API polling frequency
LOCAL_TZ = ZoneInfo("America/New_York")
DEV_MODE: bool = False               # True = no live trades/signals

# ==============================
# BACKTESTING SETTINGS
# ==============================
BACKTEST_MODE: bool = True
BACKTEST_SYMBOLS: list[str] = ["SPY"]
BACKTEST_START: str = "2024-01-01"
BACKTEST_END: str = "2024-12-01"
BACKTEST_TIMEFRAME: str = "1Min"     # Supported: "1Min", "5Min", "15Min", "1Day"
BACKTEST_EXPIRY_DAYS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7]  # Currently unused

# ==============================
# RISK MANAGEMENT
# ==============================
RISK_PER_TRADE_PCT: float = 0.0075   # 0.75% of equity
MAX_DAILY_LOSS_PCT: float = 0.03     # 3% daily stop
MAX_TRADES_PER_DAY: int = 6
KELLY_CAP_PCT: float = 0.02          # Max Kelly position size cap (2%)
VAR_THRESHOLD_PCT: float = 0.05      # Kill switch at 5% portfolio VAR
STRESS_TEST_GAP_PCT: float = 0.03    # Simulate -3% gap overnight

# ==============================
# SIGNAL ENGINE
# ==============================
CONFIDENCE_CUTOFF: float = 0.65      # Minimum score for valid signal
SIGNAL_COOLDOWN_SEC: int = 120       # Prevent duplicate signals
ATR_STOP_MULT: float = 1.5           # Stop-loss ATR multiplier
TARGET_RR: float = 2.0               # Reward-to-risk target

STRATEGY_TOGGLES: dict[str, bool] = {
    "vwap_ema_trend": True,
    "breakout": True,
    "mean_reversion": True,
    "expected_move_fade": True,
    "orb": True,
    "gamma_scalping": True,
    "options_flow": True,
}

# ==============================
# OPTIMIZER & ML
# ==============================
OPTIMIZER_MIN_TRADES: int = 10       # Require N trades before reweighting
EXPECTANCY_SMOOTHING: float = 0.2    # EMA smoothing for expectancy
ML_PROBABILITY_CUTOFF: float = 0.55  # Filter out coin-flip trades

# ==============================
# EXECUTION INTELLIGENCE
# ==============================
SPREAD_WARNING_PCT: float = 0.05     # Warn if spread > 5%
CONSERVATIVE_PNL: bool = True        # Use bid-side for wide spreads
GREEKS_REFRESH_SEC: int = 60         # Refresh Greeks every 60s

# ==============================
# LOGGING & MONITORING
# ==============================
LOG_DIR: str = "logs"
LOG_FILE: str = os.path.join(LOG_DIR, "system.log")
LOG_ROTATION_BYTES: int = 5_000_000  # 5 MB log rotation
LOG_BACKUP_COUNT: int = 5            # Keep 5 log backups

HEARTBEAT_INTERVAL_MIN: int = 15     # Send Discord heartbeat every 15 min
ERROR_SNAPSHOTS: bool = True         # Dump error traces to file

# ==============================
# LOCKOUT & KILL SWITCH
# ==============================
ENABLE_LOCKOUT: bool = True          # Lockout after daily stop/max trades
ENABLE_PANIC_HALT: bool = True       # Kill all signals if VAR breached

# ==============================
# JOURNALING
# ==============================
JOURNAL_DIR: str = "journal"
TRADE_LOG_FILE: str = os.path.join(JOURNAL_DIR, "trades.txt")

# ==============================
# LOGGER FACTORY
# ==============================
def get_logger(name: str) -> logging.Logger:
    """
    Create a logger with both console and rotating file handlers.

    Args:
        name (str): Logger name (e.g., module name).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        os.makedirs(LOG_DIR, exist_ok=True)

        # File handler with rotation
        file_handler = RotatingFileHandler(
            LOG_FILE, mode="a", maxBytes=LOG_ROTATION_BYTES,
            backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        )
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        logger.setLevel(logging.INFO)
    return logger


# Default root logger for system-wide use
log = get_logger("AutoTraderPro")
