"""
Dynamic Pricing Lab — Pricing Optimizer
Grid-based optimization with guardrails + optional Thompson Sampling.
"""

import numpy as np
import pandas as pd
from config import Config
from model import DemandModel
from montecarlo import compute_belief


class PricingOptimizer:
    """Decides the optimal price each tick."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def optimize(
        self,
        model: DemandModel,
        base_row: dict,
        last_price: float,
        objective: str = "revenue",
        explore: bool = False,
        comp_enabled: bool = True,
    ) -> dict:
        """
        Run full optimization for one tick.

        Returns dict with:
            recommended_price, candidates (DataFrame), belief_probs,
            mc_result (full Monte Carlo output)
        """
        cfg = self.cfg
        grid = cfg.price_grid()

        # ── Guardrail: restrict grid to feasible range ──────────────
        feasible = np.ones(len(grid), dtype=bool)

        # bounds
        feasible &= (grid >= cfg.p_min) & (grid <= cfg.p_max)

        # max delta from last price
        feasible &= np.abs(grid - last_price) <= cfg.max_price_change_per_tick

        # profit floor
        if objective == "profit":
            feasible &= grid >= (cfg.unit_cost + cfg.min_margin)

        if not np.any(feasible):
            # fallback: just keep last price
            feasible = np.abs(grid - last_price) < 0.5
            if not np.any(feasible):
                feasible[len(grid) // 2] = True

        # ── Monte Carlo belief (always computed) ─────────────────────
        mc = compute_belief(
            model=model,
            base_row=base_row,
            price_grid=grid,
            objective=objective,
            unit_cost=cfg.unit_cost,
            alpha=cfg.negbin_alpha,
            n_samples=cfg.mc_samples,
        )

        # ── Build candidate table ────────────────────────────────────
        candidates = pd.DataFrame({
            "price": grid,
            "d_mean": mc["d_mean"],
            "d_p10": mc["d_p10"],
            "d_p90": mc["d_p90"],
            "obj_expected": mc["expected_obj"],
            "obj_p10": mc["obj_p10"],
            "obj_p90": mc["obj_p90"],
            "belief_prob": mc["belief_probs"],
            "feasible": feasible,
        })
        candidates["risk_width"] = candidates["obj_p90"] - candidates["obj_p10"]

        # ── Select price ─────────────────────────────────────────────
        feas_mask = candidates["feasible"]
        feas_df = candidates[feas_mask]

        if explore and feas_df["belief_prob"].sum() > 0:
            # Thompson: sample proportional to belief
            probs = feas_df["belief_prob"].values.copy()
            probs = np.maximum(probs, 0)
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs = np.ones(len(probs)) / len(probs)
            idx = np.random.choice(feas_df.index, p=probs)
            rec_price = float(candidates.loc[idx, "price"])
        else:
            # Greedy: argmax expected objective among feasible
            idx = feas_df["obj_expected"].idxmax()
            rec_price = float(candidates.loc[idx, "price"])

        return {
            "recommended_price": round(rec_price, 2),
            "candidates": candidates,
            "mc_result": mc,
        }
