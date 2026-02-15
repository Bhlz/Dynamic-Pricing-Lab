"""
Dynamic Pricing Lab — Feature Engineering
Static feature builder + rolling live feature store.
"""

import numpy as np
import pandas as pd
from collections import deque
from config import Config


# ─────────────────────────────────────────────────────────────────────
#  Static feature builder (works on full DataFrame)
# ─────────────────────────────────────────────────────────────────────

FEATURE_COLS_BASE = [
    "dow", "hour", "log_traffic", "promo", "log_price", "log_stock",
]
FEATURE_COLS_COMP = ["log_comp", "log_ratio"]


def build_features(df: pd.DataFrame, comp_enabled: bool = True) -> pd.DataFrame:
    """Add derived feature columns to a copy of df."""
    out = df.copy()
    out["log_traffic"] = np.log1p(out["traffic"])
    out["log_price"] = np.log(out["price"].clip(lower=1))
    out["log_stock"] = np.log1p(out["stock"])

    if comp_enabled and "comp_price" in out.columns:
        out["log_comp"] = np.log(out["comp_price"].clip(lower=1))
        out["log_ratio"] = np.log(
            (out["price"] / out["comp_price"].clip(lower=1)).clip(0.1, 10)
        )
    else:
        out["log_comp"] = 0.0
        out["log_ratio"] = 0.0

    return out


def feature_columns(comp_enabled: bool = True):
    cols = list(FEATURE_COLS_BASE)
    if comp_enabled:
        cols.extend(FEATURE_COLS_COMP)
    return cols


# ─────────────────────────────────────────────────────────────────────
#  Rolling feature store for live simulation
# ─────────────────────────────────────────────────────────────────────

class RollingFeatureStore:
    """Maintains a sliding window of recent observations for rolling aggregates."""

    def __init__(self, comp_enabled: bool = True, max_len: int = 120):
        self.comp_enabled = comp_enabled
        self._buffer: deque = deque(maxlen=max_len)

    def push(self, obs: dict):
        """Append a new observation to the buffer."""
        self._buffer.append(obs)

    def rolling_features(self) -> dict:
        """Compute rolling aggregates from buffer."""
        if len(self._buffer) == 0:
            return {
                "traffic_5m": 0, "traffic_15m": 0, "traffic_60m": 0,
                "sales_15m": 0,
            }
        buf = list(self._buffer)
        traffics = [o["traffic"] for o in buf]
        sales = [o["sales"] for o in buf]

        return {
            "traffic_5m": float(np.mean(traffics[-5:])) if len(traffics) >= 1 else 0,
            "traffic_15m": float(np.mean(traffics[-15:])) if len(traffics) >= 1 else 0,
            "traffic_60m": float(np.mean(traffics[-60:])) if len(traffics) >= 1 else 0,
            "sales_15m": float(np.sum(sales[-15:])) if len(sales) >= 1 else 0,
        }

    def current_feature_row(self, obs: dict, price_override: float = None) -> dict:
        """Build a single feature row for the optimizer.
        If price_override is given, uses that instead of obs['price']."""
        price = price_override if price_override is not None else obs["price"]
        row = {
            "dow": obs["dow"],
            "hour": obs["hour"],
            "log_traffic": np.log1p(obs["traffic"]),
            "promo": obs["promo"],
            "log_price": np.log(max(price, 1)),
            "log_stock": np.log1p(obs.get("stock", 0)),
        }
        if self.comp_enabled:
            cp = max(obs.get("comp_price", 100), 1)
            row["log_comp"] = np.log(cp)
            row["log_ratio"] = np.log(max(price / cp, 0.1))
        else:
            row["log_comp"] = 0.0
            row["log_ratio"] = 0.0

        # merge rolling
        row.update(self.rolling_features())
        return row
