"""
main.py - Entry point for AutoTraderPro
"""

import asyncio
import time
import threading
from datetime import datetime
import pandas as pd

from config import (
    REFRESH_INTERVAL_SEC,
    HEARTBEAT_INTERVAL_MIN,
    get_logger,
)
from system_check import run_system_check
from data_fetch import data_loop
from logic import Account, plan_trade, check_exit
from signals import generate_signals
from regime import atr as calc_atr
from gui import AutoTraderGUI
from notify import alert_entry, alert_exit, alert_lockout, alert_heartbeat, alert_error
from journal import save_trade, update_trade_outcome
from performance import track_trade, close_trade, capital_health, pnl_attribution
from risk_ext import calc_var

log = get_logger(__name__)

# Global state
account = Account()
active_trades = {}
closed_trades = []  # track history for attribution
last_heartbeat = time.time()
current_symbol = "SPY"
app: AutoTraderGUI | None = None


# ==============================
# Symbol Change Callback
# ==============================
def on_symbol_change(symbol: str):
    global current_symbol
    current_symbol = symbol
    log.info(f"📊 Tracking symbol changed to {symbol}")
    if app:
        app.log_terminal(f"📊 Now tracking: {symbol}")


# ==============================
# Manual Exit Callback
# ==============================
def on_manual_exit():
    for trade_id, trade in list(active_trades.items()):
        update_trade_outcome(trade_id, "MANUAL_EXIT", 0)
        alert_exit(trade, "MANUAL_EXIT", 0)
        log.info(f"👋 Manual exit: {trade_id}")
        if app:
            app.log_terminal(f"👋 Manual exit: {trade_id}")
        active_trades.pop(trade_id, None)


# ==============================
# Data Handler
# ==============================
def handle_data(data):
    global last_heartbeat

    try:
        bars = data.get("bars")
        if bars is None or bars.empty:
            return

        # Build context
        context = {
            "price": float(bars["close"].iloc[-1]),
            "underlying_price": data.get("quote", {}).get("ap", 0),
            "expected_move": data.get("iv", {}).get("expected_move"),
            "gamma_exposure": data.get("iv", {}).get("gamma"),
            "flow_score": data.get("flow", [0])[0] if data.get("flow") else 0,
            "ml_cutoff": 0.55,
        }

        # Generate signal
        signal = generate_signals(bars, context)

        if not signal.get("blocked") and account.can_trade():
            confidence = signal["score"]
            entry = context["price"]

            trade = plan_trade(account, entry, bars, confidence)
            if trade["valid"]:
                trade["strategy"] = max(signal["signals"], key=signal["signals"].get, default="multi")
                active_trades[trade["id"]] = trade

                # Save with features
                save_trade(trade, bars, context)
                alert_entry(trade)
                track_trade(trade)

                if app:
                    app.log_terminal(f"📈 Entry: {trade['strategy']} | {entry}")
                    app.analysis_text.insert("end", f"Signal: {signal}\n")
                    app.ticket_text.insert("end", f"New Trade: {trade}\n")

        # Heartbeat
        if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL_MIN * 60:
            alert_heartbeat(account.equity, account.trades_today)
            last_heartbeat = time.time()

        # Monitor active trades
        for trade_id, trade in list(active_trades.items()):
            outcome = check_exit(trade, context["price"], bars, time.time())
            if outcome:
                pnl = (
                    (context["price"] - trade["entry"])
                    * trade["size"]
                    * (1 if trade["confidence"] > 0 else -1)
                )
                trade["pnl"] = pnl
                trade["outcome"] = outcome
                account.update_pnl(pnl)
                update_trade_outcome(trade_id, outcome, pnl)
                alert_exit(trade, outcome, pnl)

                # Track closed trade
                closed_trades.append(trade)
                close_trade(trade)

                active_trades.pop(trade_id, None)

                if app:
                    app.log_terminal(f"📉 Exit: {outcome} | PnL={pnl:.2f}")
                    app.ticket_text.insert("end", f"Exit Trade: {trade}\n")

        # Kill switch
        if calc_var(account.equity) >= account.equity * 0.05:
            alert_lockout("VAR exceeded")
            log.error("🚨 Global panic halt triggered")
            active_trades.clear()
            if app:
                app.log_terminal("🚨 Panic halt triggered")

        # ==============================
        # Update GUI Panels
        # ==============================
        if app:
            # Update trades
            app.update_trades(list(active_trades.values()) + closed_trades[-5:])

            # Update PnL attribution
            pnl_stats = pnl_attribution(closed_trades)
            health_stats = capital_health(closed_trades, account.equity)
            combined_stats = {"equity": account.equity, **health_stats, **pnl_stats}
            app.update_pnl(combined_stats)

            # Analysis panel update
            app.update_analysis(f"Signals: {signal}\nContext: {context}")

    except Exception as e:
        log.error(f"Data handling error: {e}")
        alert_error("main.handle_data", str(e))
        if app:
            app.log_terminal(f"⚠️ Error: {e}")


# ==============================
# Async Runner
# ==============================
def start_data_loop():
    asyncio.run(data_loop(current_symbol, handle_data))


# ==============================
# Main Entry
# ==============================
def main():
    global app

    if not run_system_check():
        log.error("❌ System check failed. Exiting.")
        return

    # Start data loop thread
    t = threading.Thread(target=start_data_loop, daemon=True)
    t.start()

    # Start GUI
    app = AutoTraderGUI(on_symbol_change, on_manual_exit)
    app.log_terminal("🚀 AutoTraderPro started")
    app.mainloop()


if __name__ == "__main__":
    main()
