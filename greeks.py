"""
greeks.py - Option Greeks and advanced volatility analytics
"""

import math
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import norm
from config import get_logger

log = get_logger(__name__)


# ==============================
# Black-Scholes Greeks
# ==============================
def black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> Dict[str, float]:
    """
    Compute Black-Scholes option Greeks.

    Args:
        S (float): Underlying price.
        K (float): Strike price.
        T (float): Time to expiration (in years).
        r (float): Risk-free rate.
        sigma (float): Implied volatility (annualized).
        option_type (str): "call" or "put".

    Returns:
        dict: {delta, gamma, vega, theta, rho}
    """
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}

        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type.lower() == "call":
            delta = norm.cdf(d1)
            rho = K * T * math.exp(-r * T) * norm.cdf(d2)
            theta = (
                -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                - r * K * math.exp(-r * T) * norm.cdf(d2)
            )
        else:  # put
            delta = -norm.cdf(-d1)
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
            theta = (
                -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
                + r * K * math.exp(-r * T) * norm.cdf(-d2)
            )

        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100  # per 1 vol point

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
    except Exception as e:
        log.error(f"Greeks error: {e}")
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}


# ==============================
# Expected Move
# ==============================
def expected_move(price: float, iv: float, days: int) -> float:
    """
    Expected move = price * iv * sqrt(days/365).

    Args:
        price (float): Current price.
        iv (float): Implied volatility (annualized).
        days (int): Days to expiration.

    Returns:
        float: Expected move in price terms.
    """
    try:
        return float(price * iv * math.sqrt(days / 365))
    except Exception as e:
        log.error(f"Expected move error: {e}")
        return 0.0


# ==============================
# Gamma Exposure (GEX)
# ==============================
def gamma_exposure(chain: pd.DataFrame, underlying_price: float) -> float:
    """
    Calculate total gamma exposure for an options chain.

    Requires columns: strike, gamma, open_interest, option_type.

    Args:
        chain (pd.DataFrame): Options chain.
        underlying_price (float): Current underlying price.

    Returns:
        float: Aggregate gamma exposure.
    """
    if chain is None or chain.empty:
        return 0.0

    required_cols = {"strike", "gamma", "open_interest", "option_type"}
    if not required_cols.issubset(chain.columns):
        log.warning("Gamma exposure: missing required columns")
        return 0.0

    try:
        gex = 0.0
        for _, row in chain.iterrows():
            row_dict = row.to_dict()
            direction = 1 if str(row_dict.get("option_type", "")).lower() == "call" else -1
            gex += (
                float(row_dict.get("gamma", 0))
                * float(row_dict.get("open_interest", 0))
                * 100
                * float(underlying_price)
                * direction
            )
        return float(gex)
    except Exception as e:
        log.error(f"GEX calc error: {e}")
        return 0.0


# ==============================
# IV Skew
# ==============================
def iv_skew(chain: pd.DataFrame, atm_strike: float, moneyness: float = 0.05) -> Dict[str, float]:
    """
    Compute IV skew between ITM, ATM, and OTM options.

    Args:
        chain (pd.DataFrame): Options chain with columns [strike, option_type, iv].
        atm_strike (float): At-the-money strike.
        moneyness (float): Window % around ATM for classification.

    Returns:
        dict: {call_skew, put_skew, atm_iv}
    """
    if chain is None or chain.empty:
        return {"call_skew": 0.0, "put_skew": 0.0, "atm_iv": 0.0}

    try:
        otm_calls = chain[(chain["strike"] > atm_strike * (1 + moneyness)) & (chain["option_type"] == "call")]
        itm_calls = chain[(chain["strike"] < atm_strike * (1 - moneyness)) & (chain["option_type"] == "call")]
        atm_calls = chain[
            (chain["strike"].between(atm_strike * 0.98, atm_strike * 1.02)) & (chain["option_type"] == "call")
        ]

        otm_puts = chain[(chain["strike"] < atm_strike * (1 - moneyness)) & (chain["option_type"] == "put")]
        itm_puts = chain[(chain["strike"] > atm_strike * (1 + moneyness)) & (chain["option_type"] == "put")]
        atm_puts = chain[
            (chain["strike"].between(atm_strike * 0.98, atm_strike * 1.02)) & (chain["option_type"] == "put")
        ]

        return {
            "call_skew": float(otm_calls["iv"].mean() - itm_calls["iv"].mean())
            if not otm_calls.empty and not itm_calls.empty
            else 0.0,
            "put_skew": float(otm_puts["iv"].mean() - itm_puts["iv"].mean())
            if not otm_puts.empty and not itm_puts.empty
            else 0.0,
            "atm_iv": float(pd.concat([atm_calls["iv"], atm_puts["iv"]]).mean())
            if not atm_calls.empty or not atm_puts.empty
            else 0.0,
        }
    except Exception as e:
        log.error(f"IV skew error: {e}")
        return {"call_skew": 0.0, "put_skew": 0.0, "atm_iv": 0.0}


# ==============================
# Term Structure
# ==============================
def term_structure(chain: pd.DataFrame) -> Dict[str, float]:
    """
    Compare implied volatility term structure.

    Requires columns: expiration_date, iv.

    Args:
        chain (pd.DataFrame): Options chain.

    Returns:
        dict: {short_iv, mid_iv, long_iv}
    """
    if chain is None or chain.empty:
        return {"short_iv": 0.0, "mid_iv": 0.0, "long_iv": 0.0}

    if not {"expiration_date", "iv"}.issubset(chain.columns):
        log.warning("Term structure: missing required columns")
        return {"short_iv": 0.0, "mid_iv": 0.0, "long_iv": 0.0}

    try:
        chain = chain.copy()
        chain["days_to_exp"] = (pd.to_datetime(chain["expiration_date"]) - pd.Timestamp.today()).dt.days

        short_term = chain[chain["days_to_exp"] <= 7]["iv"].mean()
        mid_term = chain[(chain["days_to_exp"] > 7) & (chain["days_to_exp"] <= 30)]["iv"].mean()
        long_term = chain[chain["days_to_exp"] > 30]["iv"].mean()

        return {
            "short_iv": float(short_term) if not np.isnan(short_term) else 0.0,
            "mid_iv": float(mid_term) if not np.isnan(mid_term) else 0.0,
            "long_iv": float(long_term) if not np.isnan(long_term) else 0.0,
        }
    except Exception as e:
        log.error(f"Term structure error: {e}")
        return {"short_iv": 0.0, "mid_iv": 0.0, "long_iv": 0.0}
