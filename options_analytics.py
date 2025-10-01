"""
options_analytics.py - Advanced options analytics for AutoTraderPro
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from config import SPREAD_WARNING_PCT, get_logger

log = get_logger(__name__)


# ==============================
# IV Skew
# ==============================
def calc_iv_skew(chain: pd.DataFrame, atm_strike: float, window: float = 0.05) -> Dict[str, float]:
    """
    Compare OTM vs ITM IV for calls and puts.

    Args:
        chain (pd.DataFrame): Options chain with columns [strike, option_type, iv].
        atm_strike (float): At-the-money strike.
        window (float): Moneyness window (default=0.05 = ±5%).

    Returns:
        dict: {call_skew, put_skew}
    """
    try:
        otm_calls = chain[(chain["strike"] > atm_strike * (1 + window)) & (chain["option_type"] == "call")]
        itm_calls = chain[(chain["strike"] < atm_strike * (1 - window)) & (chain["option_type"] == "call")]

        otm_puts = chain[(chain["strike"] < atm_strike * (1 - window)) & (chain["option_type"] == "put")]
        itm_puts = chain[(chain["strike"] > atm_strike * (1 + window)) & (chain["option_type"] == "put")]

        call_skew = float(otm_calls["iv"].mean() - itm_calls["iv"].mean()) if not otm_calls.empty and not itm_calls.empty else 0.0
        put_skew = float(otm_puts["iv"].mean() - itm_puts["iv"].mean()) if not otm_puts.empty and not itm_puts.empty else 0.0

        return {"call_skew": call_skew, "put_skew": put_skew}
    except Exception as e:
        log.error(f"IV skew error: {e}")
        return {"call_skew": 0.0, "put_skew": 0.0}


# ==============================
# Term Structure
# ==============================
def calc_term_structure(chain: pd.DataFrame) -> Dict[str, float]:
    """
    Compare IV across short/mid/long maturities.

    Args:
        chain (pd.DataFrame): Options chain with columns [expiration_date, iv].

    Returns:
        dict: {short_iv, mid_iv, long_iv}
    """
    try:
        chain = chain.copy()
        chain["days_to_exp"] = (pd.to_datetime(chain["expiration_date"]) - pd.Timestamp.today()).dt.days

        short_iv = chain[chain["days_to_exp"] <= 7]["iv"].mean()
        mid_iv = chain[(chain["days_to_exp"] > 7) & (chain["days_to_exp"] <= 30)]["iv"].mean()
        long_iv = chain[chain["days_to_exp"] > 30]["iv"].mean()

        return {
            "short_iv": float(short_iv) if not np.isnan(short_iv) else 0.0,
            "mid_iv": float(mid_iv) if not np.isnan(mid_iv) else 0.0,
            "long_iv": float(long_iv) if not np.isnan(long_iv) else 0.0,
        }
    except Exception as e:
        log.error(f"Term structure error: {e}")
        return {"short_iv": 0.0, "mid_iv": 0.0, "long_iv": 0.0}


# ==============================
# Open Interest Heatmap
# ==============================
def oi_heatmap(chain: pd.DataFrame) -> pd.DataFrame:
    """
    Build OI heatmap: strikes vs open interest.

    Args:
        chain (pd.DataFrame): Options chain with columns [strike, open_interest].

    Returns:
        pd.DataFrame: Pivot table (strike → OI).
    """
    try:
        pivot = chain.pivot_table(index="strike", values="open_interest", aggfunc="sum")
        return pivot.sort_index()
    except Exception as e:
        log.error(f"OI heatmap error: {e}")
        return pd.DataFrame()


# ==============================
# Smart Fill Price
# ==============================
def smart_fill_price(bid: float, ask: float, bias: str = "mid") -> float:
    """
    Suggest execution price given bid/ask spread.

    Args:
        bid (float): Bid price.
        ask (float): Ask price.
        bias (str): "mid" (default) or "conservative".

    Returns:
        float: Suggested execution price.
    """
    try:
        spread = (ask - bid) / bid if bid > 0 else 0
        if spread > SPREAD_WARNING_PCT:
            log.warning(f"⚠️ Wide spread detected ({spread:.2%})")

        if bias == "mid":
            return (bid + ask) / 2
        elif bias == "conservative":
            return bid if bid > 0 else ask
        else:
            return (bid + ask) / 2
    except Exception as e:
        log.error(f"Smart fill error: {e}")
        return (bid + ask) / 2 if bid and ask else 0.0


# ==============================
# Liquidity Scoring
# ==============================
def liquidity_score(contract: Dict[str, Any]) -> float:
    """
    Score contract liquidity based on OI, volume, and spread.

    Args:
        contract (dict): Must include bid, ask, open_interest, volume.

    Returns:
        float: Score 0–1 (higher = more liquid).
    """
    try:
        bid, ask = contract.get("bid", 0), contract.get("ask", 0)
        spread = (ask - bid) / bid if bid > 0 else 0
        oi = contract.get("open_interest", 0)
        vol = contract.get("volume", 0)

        score = 0
        if spread <= SPREAD_WARNING_PCT:
            score += 1
        if oi > 500:
            score += 1
        if vol > 100:
            score += 1

        return score / 3.0  # normalize 0 → 1
    except Exception as e:
        log.error(f"Liquidity score error: {e}")
        return 0.0
