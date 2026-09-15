"""Buy / Hold / Sell rating derived from the simulated distribution.

The rating is a function of two things at the rating horizon:
  * edge  - annualised median excess return of SPXL over the T-bill
  * p_beat - probability that SPXL beats the T-bill over the horizon
Both are mapped to [-1, 1], averaged into a score, and thresholded. Every input that shaped the
distribution (valuation, macro, vol regime, sentiment, fund costs) is echoed in ``reasons``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .montecarlo import SimulationResult


@dataclass
class Rating:
    label: str            # BUY / HOLD / SELL
    conviction: str       # High / Medium / Low
    score: float          # -1 .. 1
    horizon: int
    edge_annual: float
    p_beat_rf: float
    p_positive: float
    median_return: float
    mean_return: float
    rf_return: float
    p_drawdown_20: float
    reasons: List[str] = field(default_factory=list)


def annualize(total_return: float, horizon_days: int) -> float:
    base = 1.0 + total_return
    if base <= 0:
        return -1.0
    return float(base ** (252.0 / horizon_days) - 1.0)


def rate(sim: SimulationResult, horizon: int, cfg: Config, context: Optional[Dict[str, str]] = None) -> Rating:
    summ = sim.summary(horizon)
    med, rf = summ["median_return"], summ["rf_return"]
    edge = annualize(med, horizon) - annualize(rf, horizon)
    p_beat = summ["p_beat_rf"]

    s_edge = float(np.clip(edge / cfg.edge_scale, -1.0, 1.0))
    s_prob = float(np.clip((p_beat - 0.5) / cfg.prob_scale, -1.0, 1.0))
    score = 0.5 * s_edge + 0.5 * s_prob

    if score >= cfg.buy_score and edge > 0:
        label = "BUY"
    elif score <= cfg.sell_score:
        label = "SELL"
    else:
        label = "HOLD"
    mag = abs(score)
    conviction = "High" if mag >= 0.7 else "Medium" if mag >= 0.4 else "Low"

    months = horizon / 21.0
    reasons = [
        f"Median {months:.0f}-month return {med:+.1%} vs T-bill {rf:+.1%} "
        f"(annualised edge {edge:+.1%}, score {s_edge:+.2f})",
        f"P(beat T-bill) {p_beat:.0%}, P(positive) {summ['p_positive']:.0%} (score {s_prob:+.2f})",
        f"Tail: 5% worst case {summ['var_5']:+.1%}, P(-20% drawdown on the path) {summ['p_drawdown_20']:.0%}",
    ]
    for key in ("drift", "vol", "costs", "sentiment", "macro"):
        if context and context.get(key):
            reasons.append(context[key])
    return Rating(label=label, conviction=conviction, score=score, horizon=horizon, edge_annual=edge,
                  p_beat_rf=p_beat, p_positive=summ["p_positive"], median_return=med,
                  mean_return=summ["mean_return"], rf_return=rf, p_drawdown_20=summ["p_drawdown_20"],
                  reasons=reasons)
