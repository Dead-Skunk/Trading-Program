"""
performance.py - Performance analytics for AutoTraderPro
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, Any
from config import JOURNAL_DIR, get_logger

log = get_logger(__name__)


# ==============================
# Load Trades
# ==============================
def load_trades(lookback: int = 50) -> pd.DataFrame:
    """
    Load recent trades from journal files.

    Args:
        lookback (int): Number of files (days) to load.

    Returns:
        pd.DataFrame: Trades dataframe.
    """
    trades = []
    try:
        files = sorted(os.listdir(JOURNAL_DIR))[-lookback:]
        for fname in files:
            file_path = os.path.join(JOURNAL_DIR, fname)
            if not os.path.isfile(file_path):
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if "pnl" in record:
                            trades.append(record)
                    except Exception:
                        continue
    except Exception as e:
        log.error(f"Load trades failed: {e}")
        return pd.DataFrame()

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df.get("timestamp", datetime.now()))
    return df


# ==============================
# Performance Metrics
# ==============================
def compute_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute performance stats (win rate, expectancy, Sharpe, Sortino).

    Args:
        df (pd.DataFrame): Trades dataframe.

    Returns:
        dict: Performance stats.
    """
    if df is None or df.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
        }

    wins = df[df["pnl"] > 0]["pnl"]
    losses = df[df["pnl"] <= 0]["pnl"]

    win_rate = len(wins) / len(df) if len(df) > 0 else 0
    avg_win = wins.mean() if not wins.empty else 0
    avg_loss = losses.mean() if not losses.empty else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    returns = df["pnl"].values
    if len(returns) > 1:
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9)
        downside = returns[returns < 0]
        sortino = np.mean(returns) / (np.std(downside) + 1e-9) if len(downside) > 0 else sharpe
    else:
        sharpe, sortino = 0.0, 0.0

    return {
        "total_trades": int(len(df)),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
    }


# ==============================
# Equity Curve
# ==============================
def equity_curve(df: pd.DataFrame, starting_equity: float = 25000.0) -> pd.DataFrame:
    """
    Build equity curve from trades.

    Args:
        df (pd.DataFrame): Trades dataframe.
        starting_equity (float): Starting balance.

    Returns:
        pd.DataFrame: Equity curve.
    """
    if df is None or df.empty:
        return pd.DataFrame({"equity": [starting_equity]})

    df = df.sort_values("timestamp")
    df["equity"] = starting_equity + df["pnl"].cumsum()
    return df[["timestamp", "equity"]]


# ==============================
# Plot Equity Curve
# ==============================
def plot_equity_curve(df: pd.DataFrame, starting_equity: float = 25000.0) -> None:
    """
    Plot equity curve using matplotlib.

    Args:
        df (pd.DataFrame): Trades dataframe.
        starting_equity (float): Starting balance.
    """
    if df is None or df.empty:
        log.warning("No trades to plot")
        return

    try:
        import matplotlib.pyplot as plt

        eq = equity_curve(df, starting_equity)
        plt.figure(figsize=(10, 5))
        plt.plot(eq["timestamp"], eq["equity"], label="Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.title("Equity Curve")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        log.error(f"Plot failed: {e}")
