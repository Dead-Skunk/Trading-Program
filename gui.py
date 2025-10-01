"""
gui.py - Unified CustomTkinter dashboard for AutoTraderPro
"""

import tkinter as tk
import customtkinter as ctk
from typing import Dict, Any, List
from config import REFRESH_INTERVAL_SEC, get_logger

log = get_logger(__name__)


class AutoTraderGUI(ctk.CTk):
    def __init__(self, on_symbol_change, on_manual_exit):
        super().__init__()

        self.title("AutoTraderPro Dashboard")
        self.geometry("1400x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.on_symbol_change = on_symbol_change
        self.on_manual_exit = on_manual_exit

        # Tab View
        self.tabview = ctk.CTkTabview(self, width=1400, height=900)
        self.tabview.pack(fill="both", expand=True)

        # Tabs
        self.tab_dashboard = self.tabview.add("Dashboard")
        self.tab_pnl = self.tabview.add("PnL Attribution")
        self.tab_heatmap = self.tabview.add("Sector Heatmap")

        # Dashboard Layout
        self._build_dashboard()

        # PnL Attribution Layout
        self.pnl_text = ctk.CTkTextbox(self.tab_pnl, width=1350, height=800)
        self.pnl_text.pack(padx=10, pady=10)

        # Heatmap Layout
        self.heatmap_text = ctk.CTkTextbox(self.tab_heatmap, width=1350, height=800)
        self.heatmap_text.pack(padx=10, pady=10)

        # Live terminal log
        self.terminal_text = ctk.CTkTextbox(self.tab_dashboard, width=1350, height=200)
        self.terminal_text.pack(padx=10, pady=10, side="bottom")

        # Internal state
        self.current_analysis: str = ""
        self.current_contracts: List[Dict[str, Any]] = []
        self.current_trades: List[Dict[str, Any]] = []
        self.current_pnl: Dict[str, Any] = {}

        # Periodic refresh
        self.after(REFRESH_INTERVAL_SEC * 1000, self.refresh)

    # ==============================
    # Dashboard Components
    # ==============================
    def _build_dashboard(self):
        frame_top = ctk.CTkFrame(self.tab_dashboard)
        frame_top.pack(fill="x", padx=10, pady=10)

        # Watchlist
        self.watchlist_box = ctk.CTkTextbox(frame_top, width=200, height=200)
        self.watchlist_box.pack(side="left", padx=10)
        self.watchlist_box.insert("end", "SPY\nQQQ\nIWM")

        # Symbol Input
        self.symbol_entry = ctk.CTkEntry(frame_top, placeholder_text="Enter Symbol (e.g. SPY)")
        self.symbol_entry.pack(side="left", padx=10)
        self.symbol_button = ctk.CTkButton(frame_top, text="Load", command=self._change_symbol)
        self.symbol_button.pack(side="left", padx=10)

        # Ticker Analysis
        self.analysis_text = ctk.CTkTextbox(frame_top, width=500, height=200)
        self.analysis_text.pack(side="left", padx=10)

        # Contracts Panel
        self.contracts_text = ctk.CTkTextbox(frame_top, width=400, height=200)
        self.contracts_text.pack(side="left", padx=10)

        # Trade Ticket
        self.ticket_text = ctk.CTkTextbox(self.tab_dashboard, width=1350, height=200)
        self.ticket_text.pack(padx=10, pady=10)

        # Manual Exit Button
        self.exit_button = ctk.CTkButton(self.tab_dashboard, text="Manual Exit", command=self.on_manual_exit)
        self.exit_button.pack(padx=10, pady=5)

    # ==============================
    # Refresh Logic
    # ==============================
    def refresh(self):
        """Refresh dashboard panels with latest data."""
        try:
            # Analysis
            self.analysis_text.delete("1.0", "end")
            self.analysis_text.insert("end", f"📊 Analysis:\n{self.current_analysis}\n")

            # Contracts
            self.contracts_text.delete("1.0", "end")
            if self.current_contracts:
                for c in self.current_contracts[:5]:
                    self.contracts_text.insert(
                        "end",
                        f"{c.get('contract','')} | {c.get('option_type','')} "
                        f"Strike={c.get('strike')} | IV={c.get('iv')}\n",
                    )
            else:
                self.contracts_text.insert("end", "📈 No contracts loaded\n")

            # Trades
            self.ticket_text.delete("1.0", "end")
            if self.current_trades:
                for t in self.current_trades[-5:]:
                    self.ticket_text.insert(
                        "end",
                        f"🎟️ {t.get('strategy','')} | Entry={t.get('entry')} "
                        f"| Stop={t.get('stop')} | Target={t.get('target')} "
                        f"| Size={t.get('size')} | Outcome={t.get('outcome')}\n",
                    )
            else:
                self.ticket_text.insert("end", "No active trades\n")

            # PnL Attribution
            self.pnl_text.delete("1.0", "end")
            if self.current_pnl:
                for k, v in self.current_pnl.items():
                    self.pnl_text.insert("end", f"{k}: {v}\n")
            else:
                self.pnl_text.insert("end", "💰 PnL attribution updating...\n")

            # Heatmap
            self.heatmap_text.delete("1.0", "end")
            self.heatmap_text.insert("end", "🔥 Sector correlation heatmap (TODO)\n")

        except Exception as e:
            log.error(f"GUI refresh error: {e}")

        self.after(REFRESH_INTERVAL_SEC * 1000, self.refresh)

    # ==============================
    # Event Handlers
    # ==============================
    def _change_symbol(self):
        symbol = self.symbol_entry.get().strip().upper()
        if symbol:
            self.on_symbol_change(symbol)

    def log_terminal(self, message: str):
        self.terminal_text.insert("end", f"{message}\n")
        self.terminal_text.see("end")

    # ==============================
    # External Updaters (called from main.py)
    # ==============================
    def update_analysis(self, text: str):
        self.current_analysis = text

    def update_contracts(self, contracts: List[Dict[str, Any]]):
        self.current_contracts = contracts

    def update_trades(self, trades: List[Dict[str, Any]]):
        self.current_trades = trades

    def update_pnl(self, pnl_data: Dict[str, Any]):
        self.current_pnl = pnl_data
