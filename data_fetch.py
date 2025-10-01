"""
data_fetch.py - Market data integration for AutoTraderPro
Uses Alpaca (equities), Polygon (options), Yahoo fallback.
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
from typing import Dict, Any, List, Optional

from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_DATA_URL,
    POLYGON_API_KEY,
    POLYGON_BASE_URL,
    REFRESH_INTERVAL_SEC,
    get_logger,
)

log = get_logger(__name__)

# Headers for Polygon API
POLYGON_HEADERS = {"Authorization": f"Bearer {POLYGON_API_KEY}"}


# ==============================
# HELPER: Safe async request with retries
# ==============================
async def safe_request(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Perform GET request with retries + exponential backoff.

    Args:
        session (aiohttp.ClientSession): active HTTP session
        url (str): request URL
        params (dict, optional): query params

    Returns:
        dict or None: parsed JSON on success
    """
    for i in range(3):
        try:
            async with session.get(url, params=params, headers=POLYGON_HEADERS, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
                log.warning(f"Retry {i+1} for {url} (status {resp.status})")
        except Exception as e:
            log.error(f"Request error {url}: {e}")
        await asyncio.sleep(2**i)
    return None


# ==============================
# ALPACA (Equities Data)
# ==============================
async def fetch_alpaca_bars(
    session: aiohttp.ClientSession,
    symbol: str,
    timeframe: str = "1Min",
    limit: int = 100,
) -> pd.DataFrame:
    """Fetch OHLCV bars from Alpaca."""
    url = f"{ALPACA_DATA_URL}/stocks/{symbol}/bars"
    params = {"timeframe": timeframe, "limit": limit}
    auth = aiohttp.BasicAuth(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    try:
        async with session.get(url, params=params, auth=auth, timeout=10) as resp:
            if resp.status != 200:
                log.error(f"Alpaca bars error {resp.status}: {await resp.text()}")
                return pd.DataFrame()
            data = await resp.json()
    except Exception as e:
        log.error(f"Alpaca bars request failed: {e}")
        return pd.DataFrame()

    bars = pd.DataFrame(data.get("bars", []))
    if bars.empty:
        return pd.DataFrame()

    bars.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"},
        inplace=True,
    )
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    return bars


async def fetch_alpaca_quote(session: aiohttp.ClientSession, symbol: str) -> Dict[str, Any]:
    """Fetch latest equity quote from Alpaca."""
    url = f"{ALPACA_DATA_URL}/stocks/{symbol}/quotes/latest"
    auth = aiohttp.BasicAuth(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    try:
        async with session.get(url, auth=auth, timeout=10) as resp:
            if resp.status != 200:
                log.error(f"Alpaca quote error {resp.status}")
                return {}
            data = await resp.json()
            return data.get("quote", {}) or {}
    except Exception as e:
        log.error(f"Alpaca quote request failed: {e}")
        return {}


async def fetch_alpaca_news(session: aiohttp.ClientSession, symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch recent news headlines from Alpaca."""
    url = f"{ALPACA_DATA_URL}/news"
    params = {"symbols": symbol, "limit": limit}
    auth = aiohttp.BasicAuth(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    try:
        async with session.get(url, params=params, auth=auth, timeout=10) as resp:
            if resp.status != 200:
                log.error(f"Alpaca news error {resp.status}")
                return []
            data = await resp.json()
            return data.get("news", []) or []
    except Exception as e:
        log.error(f"Alpaca news request failed: {e}")
        return []


# ==============================
# POLYGON (Options Data)
# ==============================
async def fetch_polygon_chain(session: aiohttp.ClientSession, symbol: str) -> List[Dict[str, Any]]:
    """Fetch option contracts for underlying symbol."""
    url = f"{POLYGON_BASE_URL}/v3/reference/options/contracts"
    data = await safe_request(session, url, {"underlying_ticker": symbol, "limit": 100})
    return data.get("results", []) if data else []


async def fetch_polygon_snapshot(session: aiohttp.ClientSession, option_symbol: str) -> Dict[str, Any]:
    """Fetch option snapshot (Greeks, IV, bid/ask, OI)."""
    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{option_symbol}"
    data = await safe_request(session, url)
    return data.get("results", {}) if data else {}


async def fetch_polygon_iv(session: aiohttp.ClientSession, symbol: str) -> Dict[str, Any]:
    """
    Placeholder for implied volatility stats.
    TODO: Replace with options_analytics calculations.
    """
    log.debug("fetch_polygon_iv placeholder hit — returning empty dict")
    return {}


async def fetch_polygon_flow(session: aiohttp.ClientSession, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Placeholder for options trades flow.
    TODO: Implement using valid Polygon endpoint.
    """
    log.debug("fetch_polygon_flow placeholder hit — returning empty list")
    return []


# ==============================
# YAHOO (Fallback VIX/ETFs)
# ==============================
def fetch_vix() -> float:
    """Fetch latest VIX close from Yahoo Finance."""
    try:
        data = yf.Ticker("^VIX").history(period="1d")
        return float(data["Close"].iloc[-1]) if not data.empty else 0.0
    except Exception as e:
        log.warning(f"VIX fetch failed (likely after-hours): {e}")
        return 0.0


def fetch_etf(symbol: str) -> pd.DataFrame:
    """Fetch ETF bars from Yahoo Finance (fallback)."""
    try:
        return yf.Ticker(symbol).history(period="5d", interval="1m")
    except Exception as e:
        log.error(f"Failed to fetch ETF {symbol}: {e}")
        return pd.DataFrame()


# ==============================
# MASTER LOOP
# ==============================
async def fetch_all(symbol: str) -> Dict[str, Any]:
    """
    Fetch all data sources in parallel for a symbol.

    Returns:
        dict: {
            bars, quote, news, chain, iv, flow, vix
        }
    """
    async with aiohttp.ClientSession() as session:
        try:
            bars_task = fetch_alpaca_bars(session, symbol)
            quote_task = fetch_alpaca_quote(session, symbol)
            news_task = fetch_alpaca_news(session, symbol)
            chain_task = fetch_polygon_chain(session, symbol)
            iv_task = fetch_polygon_iv(session, symbol)      # TODO
            flow_task = fetch_polygon_flow(session, symbol)  # TODO

            bars, quote, news, chain, iv, flow = await asyncio.gather(
                bars_task, quote_task, news_task, chain_task, iv_task, flow_task
            )

            return {
                "bars": bars if isinstance(bars, pd.DataFrame) else pd.DataFrame(),
                "quote": quote or {},
                "news": news or [],
                "chain": chain or [],
                "iv": iv or {},
                "flow": flow or [],
                "vix": fetch_vix(),
            }
        except Exception as e:
            log.error(f"fetch_all error: {e}")
            return {}


async def data_loop(symbol: str, callback):
    """
    Continuous loop to fetch + deliver data to callback.

    Args:
        symbol (str): ticker symbol
        callback (function): function to receive data dict
    """
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                data = await fetch_all(symbol)
                if data and isinstance(data.get("bars"), pd.DataFrame) and not data["bars"].empty:
                    callback(data)
            except Exception as e:
                log.error(f"Data loop error: {e}")
            await asyncio.sleep(REFRESH_INTERVAL_SEC)
