"""
Dynamic Pricing Lab — Demand Simulator (the "real world")
Generates historical data and runs the live simulation tick-by-tick.
"""

import numpy as np
import pandas as pd
from config import Config, ShockEvent


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────

def _hour_seasonality(hour: int, amplitude: float) -> float:
    """Peak around 11-14h, trough at 3-5h."""
    return amplitude * np.sin(np.pi * (hour - 3) / 12)


def _dow_seasonality(dow: int, amplitude: float) -> float:
    """Slight weekend lift (dow 5,6)."""
    if dow >= 5:
        return amplitude * 0.5
    return -amplitude * 0.1 * (dow - 2)


def _true_log_demand(
    price: float,
    traffic: float,
    promo: int,
    comp_price: float,
    hour: int,
    dow: int,
    cfg: Config,
    comp_enabled: bool = True,
) -> float:
    """Compute log(mu) of the true demand function."""
    log_mu = cfg.base_demand
    log_mu += cfg.elasticity_price * np.log(np.clip(price, 1, None))
    if comp_enabled and comp_price > 0:
        log_mu += cfg.elasticity_comp * np.log(
            np.clip(price / comp_price, 0.1, 10)
        )
    log_mu += cfg.coef_traffic * np.log1p(traffic)
    log_mu += cfg.coef_promo * promo
    log_mu += _hour_seasonality(hour, cfg.hour_amplitude)
    log_mu += _dow_seasonality(dow, cfg.dow_amplitude)
    return log_mu


def _sample_negbin(mu: float, alpha: float, rng: np.random.Generator) -> int:
    """Sample from NegBin(mu, alpha).  Var = mu + mu^2/alpha."""
    if mu <= 0:
        return 0
    # scipy-style: r = alpha, p = alpha/(alpha+mu)
    r = alpha
    p = alpha / (alpha + mu)
    return int(rng.negative_binomial(r, p))


# ─────────────────────────────────────────────────────────────────────
#  Historical data generator
# ─────────────────────────────────────────────────────────────────────

def generate_historical(cfg: Config, seed: int = 42) -> pd.DataFrame:
    """Generate ~180 days × 24 hours of historical observations."""
    rng = np.random.default_rng(seed)
    records = []

    n_hours = cfg.hist_days * 24
    comp_price = 100.0  # start

    for i in range(n_hours):
        day = i // 24
        hour = i % 24
        dow = day % 7

        # ── traffic (intradaily seasonality + noise) ──
        base_traffic = 80 + 40 * np.sin(np.pi * (hour - 4) / 12)
        traffic = max(1, base_traffic + rng.normal(0, 15))

        # ── promo (random ~7 % of the time) ──
        promo = int(rng.random() < 0.07)

        # ── competition price (random walk) ──
        if cfg.comp_enabled:
            comp_price += rng.normal(0, 0.5)
            comp_price = np.clip(comp_price, 70, 140)
        else:
            comp_price = 100.0

        # ── confounded historical price ──
        traffic_z = (traffic - 80) / 20  # rough z-score
        price = 100 + cfg.confound_traffic * traffic_z * 100
        if promo:
            price += cfg.confound_promo
        price += rng.normal(0, 5)
        price = np.clip(price, cfg.p_min, cfg.p_max)

        # ── true demand ──
        log_mu = _true_log_demand(
            price, traffic, promo, comp_price, hour, dow, cfg
        )
        mu = np.exp(log_mu)
        demand = _sample_negbin(mu, cfg.negbin_alpha, rng)

        # ── stock constraint ──
        stock = max(50, int(200 + rng.normal(0, 30)))
        sales = min(demand, stock)

        records.append(
            dict(
                day=day, hour=hour, dow=dow,
                traffic=round(traffic, 2),
                promo=promo,
                comp_price=round(comp_price, 2),
                price=round(price, 2),
                stock=stock,
                demand=demand,
                sales=sales,
            )
        )

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────
#  Live world
# ─────────────────────────────────────────────────────────────────────

class LiveWorld:
    """Tick-by-tick live simulation environment."""

    def __init__(self, cfg: Config, start_hour: int = 10, seed: int = 99):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.tick_idx = 0
        self.start_hour = start_hour
        self.comp_price = 100.0
        self.stock = cfg.stock_initial
        self.promo_active = False
        self.promo_remaining = 0

        # pre-index shocks by tick for O(1) lookup
        self._shocks: dict = {}
        for s in cfg.shocks:
            self._shocks[s.tick] = s

        # persistent traffic modifier (for shock)
        self._traffic_mult = 1.0
        self._comp_shock_applied = False

    # ── public api ───────────────────────────────────────────────────

    def tick(self, chosen_price: float) -> dict:
        """Advance one simulated minute. Returns observation dict."""
        t = self.tick_idx
        minute_of_day = self.start_hour * 60 + t
        hour = (minute_of_day // 60) % 24
        dow = (minute_of_day // 1440) % 7

        # ── process shocks ──
        self._apply_shocks(t)

        # ── traffic ──
        base = 80 + 40 * np.sin(np.pi * (hour - 4) / 12)
        traffic = max(1, (base + self.rng.normal(0, 12)) * self._traffic_mult)

        # ── promo ──
        promo = int(self.promo_active)

        # ── comp_price random walk ──
        if self.cfg.comp_enabled:
            self.comp_price += self.rng.normal(0, 0.3)
            self.comp_price = np.clip(self.comp_price, 70, 140)

        # ── demand ──
        log_mu = _true_log_demand(
            chosen_price, traffic, promo, self.comp_price,
            hour, dow, self.cfg, self.cfg.comp_enabled,
        )
        mu = np.exp(log_mu)
        demand = _sample_negbin(mu, self.cfg.negbin_alpha, self.rng)

        # ── stock ──
        if self.cfg.stock_enabled:
            sales = min(demand, self.stock)
            self.stock = max(0, self.stock - sales)
        else:
            sales = demand

        obs = dict(
            tick=t,
            hour=hour,
            dow=dow,
            traffic=round(traffic, 2),
            promo=promo,
            comp_price=round(self.comp_price, 2),
            price=round(chosen_price, 2),
            stock=self.stock,
            demand=demand,
            sales=sales,
            mu=round(mu, 4),
        )

        self.tick_idx += 1

        # decrement promo timer
        if self.promo_remaining > 0:
            self.promo_remaining -= 1
            if self.promo_remaining == 0:
                self.promo_active = False

        return obs

    # ── internals ────────────────────────────────────────────────────

    def _apply_shocks(self, t: int):
        shock = self._shocks.get(t)
        if shock is None:
            return
        if shock.kind == "traffic":
            self._traffic_mult = 1.0 + shock.value  # e.g. 0.65
        elif shock.kind == "comp_price":
            if not self._comp_shock_applied:
                self.comp_price *= (1.0 + shock.value)
                self._comp_shock_applied = True
        elif shock.kind == "promo":
            self.promo_active = True
            self.promo_remaining = shock.duration
