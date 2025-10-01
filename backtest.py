"""
backtest.py - Options backtesting engine for AutoTraderPro
"""

import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, Any
from datetime import datetime
import pytz

from config import (
    STARTING_EQUITY,
    BACKTEST_MODE,
    BACKTEST_SYMBOLS,
    BACKTEST_START,
    BACKTEST_END,
    BACKTEST_TIMEFRAME,
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_DATA_URL,
    POLYGON_API_KEY,
    POLYGON_BASE_URL,
    SPREAD_WARNING_PCT,
    CONSERVATIVE_PNL,
    get_logger,
)
from signals import generate_signals
from logic import plan_trade, check_exit, Account
from journal import save_trade, update_trade_outcome, calculate_expectancy
from regime import atr as calc_atr

log = get_logger("Backtest")


# ==============================
# Alpaca: Fetch underlying bars
# ==============================
def fetch_equity_bars(symbol: str) -> pd.DataFrame:
    """Fetch OHLCV bars from Alpaca for backtest period."""
    url = f"{ALPACA_DATA_URL}/stocks/{symbol}/bars"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY}

    ny = pytz.timezone("America/New_York")
    start_dt = ny.localize(datetime.strptime(BACKTEST_START, "%Y-%m-%d")).astimezone(pytz.UTC)
    end_dt = ny.localize(datetime.strptime(BACKTEST_END, "%Y-%m-%d")).astimezone(pytz.UTC)

    params = {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "timeframe": BACKTEST_TIMEFRAME,
        "limit": 10000,
    }

    all_bars, next_page = [], None
    while True:
        if next_page:
            params["page_token"] = next_page
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            log.error(f"Alpaca error {resp.status_code}: {resp.text}")
            return pd.DataFrame()
        data = resp.json()
        all_bars.extend(data.get("bars", []))
        next_page = data.get("next_page_token")
        if not next_page:
            break

    df = pd.DataFrame(all_bars)
    if df.empty:
        return pd.DataFrame()
    df.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"},
        inplace=True,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df.set_index("timestamp", inplace=True)
    return df[["open", "high", "low", "close", "volume"]].dropna()


# ==============================
# Polygon: Fetch options chain
# ==============================
def fetch_options_chain(symbol: str, date: str) -> pd.DataFrame:
    """Fetch options chain snapshot from Polygon."""
    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{symbol}"
    headers = {"Authorization": f"Bearer {POLYGON_API_KEY}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        log.error(f"Polygon chain error {resp.status_code}: {resp.text}")
        return pd.DataFrame()
    results = resp.json().get("results", {})
    options = results.get("options", []) if isinstance(results, dict) else results

    return pd.DataFrame(
        [
            {
                "contract": c.get("details", {}).get("contract_name"),
                "strike": c.get("details", {}).get("strike_price"),
                "expiration_date": c.get("details", {}).get("expiration_date"),
                "option_type": c.get("details", {}).get("type"),  # normalized key
                "bid": c.get("last_quote", {}).get("bid", 0),
                "ask": c.get("last_quote", {}).get("ask", 0),
                "mid": (c.get("last_quote", {}).get("bid", 0) + c.get("last_quote", {}).get("ask", 0)) / 2,
                "iv": c.get("greeks", {}).get("iv"),
                "delta": c.get("greeks", {}).get("delta"),
                "gamma": c.get("greeks", {}).get("gamma"),
                "theta": c.get("greeks", {}).get("theta"),
                "vega": c.get("greeks", {}).get("vega"),
                "volume": c.get("day", {}).get("volume", 0),
                "open_interest": c.get("open_interest", 0),
            }
            for c in options
        ]
    )


# ==============================
# Backtest Engine
# ==============================
def run_backtest() -> Dict[str, Any]:
    """Run historical simulation of strategy performance."""
    if not BACKTEST_MODE:
        log.error("Backtest mode is disabled in config.py")
        return {}

    equity = STARTING_EQUITY
    trades, durations = [], []
    account = Account()

    for symbol in BACKTEST_SYMBOLS:
        eq_df = fetch_equity_bars(symbol)
        if eq_df.empty:
            continue

        for ts, row in eq_df.iterrows():
            df_bar = eq_df.loc[:ts].copy()
            if df_bar.empty:
                continue

            signal_dict = generate_signals(df_bar, {"symbol": symbol, "disable_ml": True})
            if not signal_dict or signal_dict.get("blocked", False):
                continue

            score = signal_dict.get("score", 0)
            chain = fetch_options_chain(symbol, ts.strftime("%Y-%m-%d"))
            if chain.empty:
                continue

            for _, contract in chain.iterrows():
                spread = (contract["ask"] - contract["bid"]) / contract["mid"] if contract["mid"] else 1
                if spread > SPREAD_WARNING_PCT:
                    continue

                # Plan trade using unified ATR
                trade = plan_trade(account, contract["mid"], df_bar, score)
                if not trade.get("valid"):
                    continue

                trade["strategy"] = max(signal_dict["signals"], key=signal_dict["signals"].get, default="multi")
                trade.update({"contract": contract["contract"], "option_type": contract["option_type"]})

                # Save with features
                save_trade(trade, df_bar, {"symbol": symbol})

                trades.append(trade)
                entry_time = time.time()

                for ts2, row2 in eq_df.loc[ts:].iterrows():
                    exit_signal = check_exit(trade, row2["close"], df_bar, entry_time=entry_time)
                    if exit_signal:
                        exit_price = contract["bid"] if CONSERVATIVE_PNL else contract["mid"]
                        pnl = (exit_price - trade["entry"]) * 100 * trade["size"] * (1 if score > 0 else -1)
                        trade["pnl"] = pnl
                        trade["outcome"] = exit_signal
                        equity += pnl
                        account.update_pnl(pnl)
                        durations.append((ts2 - ts).total_seconds())
                        update_trade_outcome(trade["id"], exit_signal, pnl)
                        break

    results = {
        "trades": len(trades),
        "final_equity": equity,
        "net_pnl": equity - STARTING_EQUITY,
        "expectancy": calculate_expectancy("all") if trades else 0,
        "win_rate": len([t for t in trades if t.get("pnl", 0) > 0]) / len(trades) if trades else 0,
        "profit_factor": abs(
            sum(t["pnl"] for t in trades if t["pnl"] > 0)
            / sum(t["pnl"] for t in trades if t["pnl"] < 0)
        )
        if any(t["pnl"] < 0 for t in trades)
        else float("inf"),
        "avg_duration_min": np.mean(durations) / 60 if durations else 0,
    }
    return results


if __name__ == "__main__":
    print(run_backtest())
