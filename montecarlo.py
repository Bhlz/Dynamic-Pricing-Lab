"""
Dynamic Pricing Lab — Monte Carlo Belief Distribution
Sample-based estimation of P(price is optimal) for Thompson Sampling.
"""

import numpy as np
import pandas as pd
from model import DemandModel


def _approx_negbin_sample(
    mu: float, alpha: float, rng: np.random.Generator, n: int
) -> np.ndarray:
    """Draw n samples from NegBin(mu, alpha)."""
    if mu <= 0.01:
        return np.zeros(n, dtype=int)
    r = alpha
    p = alpha / (alpha + mu)
    return rng.negative_binomial(r, p, size=n)


def compute_belief(
    model: DemandModel,
    base_row: dict,
    price_grid: np.ndarray,
    objective: str,       # "revenue" or "profit"
    unit_cost: float,
    alpha: float = 2.0,
    n_samples: int = 1000,
    seed: int = 77,
) -> dict:
    """
    For each candidate price, sample demand and compute objective.
    Returns:
        belief_probs: np.ndarray of shape (len(price_grid),) with P(optimal)
        expected_obj: np.ndarray of expected objective per price
        obj_p10, obj_p90: bands
    """
    rng = np.random.default_rng(seed)
    n_prices = len(price_grid)

    # ── predict mean demand per candidate price ─────────────────────
    rows = []
    for p in price_grid:
        r = dict(base_row)
        r["log_price"] = np.log(max(p, 1))
        if "log_comp" in r and r.get("log_comp", 0) > 0:
            cp = np.exp(r["log_comp"])
            r["log_ratio"] = np.log(max(p / cp, 0.1))
        rows.append(r)

    X = pd.DataFrame(rows)
    d_mean, d_p10, d_p90 = model.predict(X)

    # ── sample demand per price ─────────────────────────────────────
    # (n_prices × n_samples) matrix
    demand_samples = np.zeros((n_prices, n_samples))
    for i in range(n_prices):
        mu = max(d_mean[i], 0.01)
        demand_samples[i] = _approx_negbin_sample(mu, alpha, rng, n_samples)

    # ── compute objective per sample ────────────────────────────────
    prices_col = price_grid[:, None]  # (n_prices, 1)
    if objective == "profit":
        obj_matrix = (prices_col - unit_cost) * demand_samples
    else:
        obj_matrix = prices_col * demand_samples

    # ── which price wins per sample? ────────────────────────────────
    winners = np.argmax(obj_matrix, axis=0)  # (n_samples,)
    win_counts = np.bincount(winners, minlength=n_prices)
    belief_probs = win_counts / n_samples

    # ── summary stats ───────────────────────────────────────────────
    expected_obj = obj_matrix.mean(axis=1)
    obj_p10 = np.percentile(obj_matrix, 10, axis=1)
    obj_p90 = np.percentile(obj_matrix, 90, axis=1)

    return {
        "belief_probs": belief_probs,
        "expected_obj": expected_obj,
        "obj_p10": obj_p10,
        "obj_p90": obj_p90,
        "d_mean": d_mean,
        "d_p10": d_p10,
        "d_p90": d_p90,
    }
