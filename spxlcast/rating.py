"""Buy / Hold / Sell rating derived from the simulated distribution.

Two views of the distribution at the rating horizon, each scaled to [-1, 1] and averaged:
  * typical outcome - annualised MEDIAN excess return of SPXL over the T-bill (edge / edge_scale)
  * expected value   - annualised MEAN excess return per unit of vol, a Sharpe-like ratio
                       (sharpe / sharpe_scale)
The median is dragged by volatility decay, the mean is not, so for a 3x fund the two can disagree;
using both avoids a structural bias either way. A BUY additionally requires the typical outcome to
beat cash. The score carries a Monte Carlo standard error so borderline labels are flagged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .montecarlo import SimulationResult


@dataclass
class Rating:
    label: str            # BUY / HOLD / SELL
    conviction: str       # High / Medium / Low
    score: float          # -1 .. 1
    score_se: float       # Monte Carlo standard error of the score
    borderline: bool      # a threshold lies within one standard error of the score
    horizon: int
    edge_annual: float
    sharpe_annual: float
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


def horizon_phrase(horizon: int) -> str:
    if horizon % 21 == 0:
        m = horizon // 21
        return f"{m}-month" if m % 12 else f"{m // 12}-year"
    if horizon < 21 and horizon % 5 == 0:
        return f"{horizon // 5}-week"
    return f"{horizon}-day"


def _components(r: np.ndarray, rf: float, horizon: int, cfg: Config) -> Tuple[float, float, float]:
    """(score, annualised median edge, annualised Sharpe-like ratio) for a block of returns."""
    med = float(np.median(r))
    edge = annualize(med, horizon) - annualize(rf, horizon)
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    sharpe = float((r.mean() - rf) / sd * np.sqrt(252.0 / horizon)) if sd > 0 else 0.0
    s_edge = float(np.clip(edge / cfg.edge_scale, -1.0, 1.0))
    s_sharpe = float(np.clip(sharpe / cfg.sharpe_scale, -1.0, 1.0))
    return 0.5 * s_edge + 0.5 * s_sharpe, edge, sharpe


def rate(sim: SimulationResult, horizon: int, cfg: Config, context: Optional[Dict[str, str]] = None) -> Rating:
    summ = sim.summary(horizon)
    r = sim.returns(horizon)
    rf = summ["rf_return"]
    score, edge, sharpe = _components(r, rf, horizon, cfg)
    if not np.isfinite(score):
        raise ValueError("simulated returns are not finite; check the drift and vol inputs")

    # Monte Carlo standard errors from independent path blocks.
    k = max(2, min(cfg.score_blocks, len(r) // 50))
    blocks = [_components(b, rf, horizon, cfg) for b in np.array_split(r, k)]
    score_se = float(np.std([b[0] for b in blocks], ddof=1) / np.sqrt(k))
    edge_se = float(np.std([b[1] for b in blocks], ddof=1) / np.sqrt(k))

    if score >= cfg.buy_score and edge > 0:
        label = "BUY"
    elif score <= cfg.sell_score:
        label = "SELL"
    else:
        label = "HOLD"
    mag = abs(score)
    conviction = "High" if mag >= 0.7 else "Medium" if mag >= 0.4 else "Low"
    borderline = any(abs(mag - t) <= score_se for t in (abs(cfg.buy_score), abs(cfg.sell_score), 0.4, 0.7))
    if score >= cfg.buy_score and abs(edge) <= edge_se:   # the BUY gate on the median edge is itself within noise
        borderline = True

    med = summ["median_return"]
    hz = horizon_phrase(horizon)
    reasons = [
        f"Typical outcome: median {hz} return {med:+.1%} vs T-bill {rf:+.1%} "
        f"(annualised edge {edge:+.1%} -> {0.5 * np.clip(edge / cfg.edge_scale, -1, 1):+.2f} of the score)",
        f"Expected value: mean {summ['mean_return']:+.1%}, vol {summ['std_return']:.0%} -> Sharpe-like {sharpe:+.2f}/yr "
        f"({0.5 * np.clip(sharpe / cfg.sharpe_scale, -1, 1):+.2f} of the score)",
        f"P(beat T-bill) {summ['p_beat_rf']:.0%}, P(positive) {summ['p_positive']:.0%}; "
        f"5% worst case {summ['var_5']:+.1%}; P(-20% drawdown on the path) {summ['p_drawdown_20']:.0%}",
    ]
    if borderline:
        reasons.append(f"Borderline: the score ({score:+.2f}, SE {score_se:.2f}) or the median edge ({edge:+.1%}, "
                       f"SE {edge_se:.1%}) sits within one Monte Carlo standard error of a label or conviction "
                       f"threshold; more paths will not settle a genuinely marginal case")
    for key in ("drift", "vol", "costs", "sentiment", "macro"):
        if context and context.get(key):
            reasons.append(context[key])
    return Rating(label=label, conviction=conviction, score=score, score_se=score_se, borderline=borderline,
                  horizon=horizon, edge_annual=edge, sharpe_annual=sharpe, p_beat_rf=summ["p_beat_rf"],
                  p_positive=summ["p_positive"], median_return=med, mean_return=summ["mean_return"],
                  rf_return=rf, p_drawdown_20=summ["p_drawdown_20"], reasons=reasons)
