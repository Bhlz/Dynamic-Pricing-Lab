"""
Dynamic Pricing Lab — Configuration
All tunable parameters with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ShockEvent:
    tick: int
    kind: str          # "traffic", "comp_price", "promo"
    value: float       # multiplier or delta
    duration: int = 1  # ticks the shock lasts


@dataclass
class Config:
    # ── Historical simulation ────────────────────────────────────────
    hist_days: int = 180
    hist_granularity_hours: int = 1   # 1 observation per hour

    # ── Live simulation --──────────────────────────────────────────────
    live_hours: int = 6
    live_granularity_minutes: int = 1  # 1 tick = 1 simulated minute
    live_ticks: int = 360              # 6h * 60min

    # ── Price grid ───────────────────────────────────────────────────
    p_min: float = 60.0
    p_max: float = 160.0
    grid_candidates: int = 101

    # ── Guardrails ───────────────────────────────────────────────────
    max_price_change_per_tick: float = 2.0

    # ── Competition ──────────────────────────────────────────────────
    comp_enabled: bool = True

    # ── Stock ────────────────────────────────────────────────────────
    stock_enabled: bool = True
    stock_initial: int = 250

    # ── Cost / Profit ────────────────────────────────────────────────
    unit_cost: float = 65.0
    min_margin: float = 5.0

    # ── Demand truth parameters ──────────────────────────────────────
    base_demand: float = 4.0           # log-scale intercept
    elasticity_price: float = -1.25    # coef on log(price)
    elasticity_comp: float = -0.70     # coef on log(price/comp_price)
    coef_traffic: float = 0.55         # coef on log1p(traffic)
    coef_promo: float = 0.20           # promo lift  (additive in log)
    negbin_alpha: float = 2.0          # dispersion (higher = more variance)

    # Seasonality amplitudes
    hour_amplitude: float = 0.6
    dow_amplitude: float = 0.15

    # ── Live shocks ──────────────────────────────────────────────────
    shocks: List[ShockEvent] = field(default_factory=lambda: [
        ShockEvent(tick=80,  kind="traffic",    value=-0.35, duration=1),
        ShockEvent(tick=160, kind="comp_price", value=-0.12, duration=1),
        ShockEvent(tick=240, kind="promo",      value=1.0,   duration=30),
    ])

    # ── Model retraining ─────────────────────────────────────────────
    retrain_every: int = 30  # retrain model every N ticks

    # ── Monte Carlo ──────────────────────────────────────────────────
    mc_samples: int = 1000

    # ── Historical confounding strength ──────────────────────────────
    confound_traffic: float = 0.08   # high traffic → price += this * traffic_z
    confound_promo: float = -5.0     # promo → price += this

    def price_grid(self):
        """Return array of candidate prices."""
        import numpy as np
        return np.linspace(self.p_min, self.p_max, self.grid_candidates)
