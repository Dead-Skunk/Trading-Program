"""
backtest.py - Options backtesting engine for AutoTraderPro
"""

import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
import pytz

from config import (
    STARTING_EQUITY,
    BACKTEST_MODE,
    BACKTEST_SYMBOLS,
    BACKTEST_START,
    BACKTEST_END,
    BACKTEST_TIMEFRAME,
    BACKTEST_EXPIRY_DAYS,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_DATA_URL,
    SPREAD_WARNING_PCT,
    CONSERVATIVE_PNL,
    get_logger,
)
from signals import generate_signals
from logic import plan_trade, check_exit, Account
from journal import save_trade, update_trade_outcome, calculate_expectancy

log = get_logger("Backtest")


# ==============================
# Alpaca: Fetch underlying bars
# ==============================
def fetch_equity_bars(symbol: str) -> pd.DataFrame:
    """Fetch OHLCV bars from Alpaca."""
    url = f"{ALPACA_DATA_URL}/stocks/{symbol}/bars"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}
    params = {"timeframe": BACKTEST_TIMEFRAME, "start": BACKTEST_START, "end": BACKTEST_END}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            log.error(f"Alpaca fetch failed: {resp.status_code} {resp.text}")
            return pd.DataFrame()
        data = resp.json().get("bars", [])
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame()
        df.rename(
            columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"},
            inplace=True,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        log.error(f"fetch_equity_bars error: {e}")
        return pd.DataFrame()


# ==============================
# Backtest Engine
# ==============================
def run_backtest(symbol: str) -> Dict[str, Any]:
    """
    Run backtest for given symbol.

    Args:
        symbol (str): Ticker symbol.

    Returns:
        dict: Summary stats.
    """
    bars = fetch_equity_bars(symbol)
    if bars.empty:
        log.error("No historical data available")
        return {}

    account = Account(STARTING_EQUITY)
    trades: List[Dict[str, Any]] = []
    open_trades: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    expiry_days = BACKTEST_EXPIRY_DAYS or [0, 1, 2, 3, 4, 5]

    for i in range(30, len(bars)):
        window = bars.iloc[: i + 1]
        last_bar = window.iloc[-1]

        # Generate signals
        signal_data = {"bars": window, "context": {"price": last_bar["close"]}}
        weights = {}  # placeholder (strategy_optimizer can feed)
        sig = generate_signals(symbol, signal_data, weights)

        # Plan trade
        if sig["signal"] != 0:
            trade = plan_trade(account, last_bar["close"], window, sig["score"])
            if trade["valid"]:
                trade["strategy"] = "composite"
                trade["expiry"] = last_bar["timestamp"] + timedelta(days=int(np.random.choice(expiry_days)))
                save_trade(trade, window, signal_data["context"])
                trades.append(trade)
                open_trades.append(trade)
                log.info(f"Trade opened {trade['id']} at {trade['entry']}")

        # Process exits
        for trade in open_trades[:]:
            outcome = check_exit(trade, last_bar["close"], window, last_bar["timestamp"].timestamp())
            pnl = 0.0
            exit_type = None

            if outcome:
                if outcome in ["STOP", "TARGET", "TRAIL"]:
                    fill_price = last_bar["close"]
                    # Slippage + spread realism
                    spread = last_bar["close"] * SPREAD_WARNING_PCT
                    slip = spread * np.random.uniform(0.1, 0.5)
                    if CONSERVATIVE_PNL:
                        fill_price -= slip if outcome == "TARGET" else fill_price + slip
                    pnl = (fill_price - trade["entry"]) * trade["size"] * 100 * (
                        1 if trade["confidence"] > 0 else -1
                    )
                    exit_type = outcome
                elif outcome == "TIME":
                    pnl = (last_bar["close"] - trade["entry"]) * trade["size"] * 100 * (
                        1 if trade["confidence"] > 0 else -1
                    )
                    exit_type = "TIME"

            # Expiry check
            if last_bar["timestamp"] >= trade.get("expiry", last_bar["timestamp"]):
                pnl = (last_bar["close"] - trade["entry"]) * trade["size"] * 100 * (
                    1 if trade["confidence"] > 0 else -1
                )
                exit_type = "EXPIRE"

            if exit_type:
                trade["outcome"] = exit_type
                trade["pnl"] = pnl
                update_trade_outcome(trade["id"], exit_type, pnl)
                account.update_pnl(pnl)
                open_trades.remove(trade)
                results.append(trade)
                log.info(f"Trade {trade['id']} closed | {exit_type} | PnL={pnl:.2f}")

    expectancy = calculate_expectancy("all")
    return {
        "equity": account.equity,
        "trades": len(results),
        "expectancy": expectancy,
        "net_pnl": account.equity - STARTING_EQUITY,
    }
