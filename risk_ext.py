"""
risk_ext.py - Advanced risk analytics for AutoTraderPro
"""

import numpy as np
from config import (
    VAR_THRESHOLD_PCT,
    STRESS_TEST_GAP_PCT,
    get_logger,
)

log = get_logger(__name__)


# ==============================
# Kelly Criterion
# ==============================
def kelly_fraction(win_rate: float, rr: float) -> float:
    """
    Compute Kelly fraction (optimal bet size).

    Args:
        win_rate (float): Probability of winning (0–1).
        rr (float): Reward-to-risk ratio.

    Returns:
        float: Kelly fraction (0–1).
    """
    try:
        p = max(min(win_rate, 1), 0)
        q = 1 - p
        b = rr if rr > 0 else 0
        f = (b * p - q) / b if b > 0 else 0
        return max(f, 0.0)
    except Exception as e:
        log.error(f"Kelly error: {e}")
        return 0.0


# ==============================
# Value-at-Risk (Monte Carlo)
# ==============================
def calc_var(
    equity: float,
    mu: float = 0,
    sigma: float = 0.02,
    n: int = 10000,
    alpha: float = 0.05,
) -> float:
    """
    Monte Carlo Value-at-Risk (VaR).

    Args:
        equity (float): Portfolio equity.
        mu (float): Mean daily return (default=0).
        sigma (float): Daily volatility (default=0.02 = 2%).
        n (int): Number of simulations (default=10000).
        alpha (float): Confidence level (default=0.05 = 95%).

    Returns:
        float: Portfolio VaR estimate.
    """
    try:
        returns = np.random.normal(mu, sigma, n)
        portfolio = equity * (1 + returns)
        var = equity - np.percentile(portfolio, 100 * alpha)
        return float(var)
    except Exception as e:
        log.error(f"VAR calc error: {e}")
        return 0.0


# ==============================
# Stress Test
# ==============================
def stress_test(equity: float) -> float:
    """
    Simulate overnight gap loss.

    Args:
        equity (float): Portfolio equity.

    Returns:
        float: Simulated loss from stress event.
    """
    try:
        loss = equity * STRESS_TEST_GAP_PCT
        return float(loss)
    except Exception as e:
        log.error(f"Stress test error: {e}")
        return 0.0
