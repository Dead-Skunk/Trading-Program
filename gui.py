"""
gui.py - Unified AutoTraderPro Dashboard
"""

import asyncio
import customtkinter as ctk
from typing import Dict, Any, Callable
from config import REFRESH_INTERVAL_SEC, get_logger

log = get_logger(__name__)


# ==============================
# Dashboard GUI
# ==============================
class AutoTraderDashboard(ctk.CTk):
    """Unified dashboard for AutoTraderPro."""

    def __init__(self, on_symbol_change: Callable[[str], None], on_arm: Callable[[], None]):
        super().__init__()

        self.title("AutoTraderPro Dashboard")
        self.geometry("1200x800")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.on_symbol_change = on_symbol_change
        self.on_arm = on_arm
        self.refresh_task = None

        # Configure grid
        self.grid_columnconfigure(0, weight=3)  # left panel
        self.grid_columnconfigure(1, weight=2)  # right panel
        self.grid_rowconfigure(0, weight=5)
        self.grid_rowconfigure(1, weight=2)

        # Left: Ticker Analysis
        self.analysis_frame = ctk.CTkFrame(self, corner_radius=10)
        self.analysis_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.symbol_entry = ctk.CTkEntry(self.analysis_frame, placeholder_text="Enter symbol (e.g. SPY)")
        self.symbol_entry.pack(fill="x", padx=5, pady=5)

        self.change_btn = ctk.CTkButton(
            self.analysis_frame, text="Load Symbol", command=self._handle_symbol_change
        )
        self.change_btn.pack(padx=5, pady=5)

        self.analysis_text = ctk.CTkTextbox(self.analysis_frame, wrap="word", height=20)
        self.analysis_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Right: Contract Details
        self.contracts_frame = ctk.CTkFrame(self, corner_radius=10)
        self.contracts_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.arm_btn = ctk.CTkButton(self.contracts_frame, text="Arm System", command=self.on_arm)
        self.arm_btn.pack(padx=5, pady=5)

        self.contracts_text = ctk.CTkTextbox(self.contracts_frame, wrap="word")
        self.contracts_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Bottom: Live Terminal Feed
        self.feed_frame = ctk.CTkFrame(self, corner_radius=10)
        self.feed_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        self.feed_text = ctk.CTkTextbox(self.feed_frame, wrap="word", height=10)
        self.feed_text.pack(fill="both", expand=True, padx=5, pady=5)

    # ==============================
    # UI Updates
    # ==============================
    def update_analysis(self, data: Dict[str, Any]) -> None:
        """Update ticker analysis panel."""
        try:
            self.analysis_text.delete("1.0", "end")
            self.analysis_text.insert("end", str(data))
        except Exception as e:
            log.error(f"GUI analysis update error: {e}")

    def update_contracts(self, data: Dict[str, Any]) -> None:
        """Update contracts panel."""
        try:
            self.contracts_text.delete("1.0", "end")
            self.contracts_text.insert("end", str(data))
        except Exception as e:
            log.error(f"GUI contracts update error: {e}")

    def update_feed(self, msg: str) -> None:
        """Append message to terminal feed."""
        try:
            self.feed_text.insert("end", msg + "\n")
            self.feed_text.see("end")
        except Exception as e:
            log.error(f"GUI feed update error: {e}")

    # ==============================
    # Symbol + Arm Handlers
    # ==============================
    def _handle_symbol_change(self) -> None:
        symbol = self.symbol_entry.get().strip().upper()
        if symbol:
            try:
                self.on_symbol_change(symbol)
                self.update_feed(f"🔄 Symbol changed to {symbol}")
            except Exception as e:
                log.error(f"Symbol change handler error: {e}")

    # ==============================
    # Refresh Loop
    # ==============================
    async def start_refresh_loop(self, fetch_callback: Callable[[], Dict[str, Any]]) -> None:
        """
        Start auto-refresh loop for dashboard.

        Args:
            fetch_callback (callable): Function returning latest dashboard data.
        """
        if self.refresh_task:
            log.warning("Refresh loop already running")
            return

        async def loop():
            while True:
                try:
                    data = fetch_callback()
                    if data:
                        self.update_analysis(data.get("analysis", {}))
                        self.update_contracts(data.get("contracts", {}))
                        if "feed" in data:
                            self.update_feed(data["feed"])
                except Exception as e:
                    log.error(f"Dashboard refresh error: {e}")
                await asyncio.sleep(REFRESH_INTERVAL_SEC)

        self.refresh_task = asyncio.create_task(loop())

    # ==============================
    # Lifecycle
    # ==============================
    def on_close(self) -> None:
        """Clean shutdown when window closes."""
        try:
            if self.refresh_task:
                self.refresh_task.cancel()
            self.destroy()
        except Exception as e:
            log.error(f"GUI shutdown error: {e}")
