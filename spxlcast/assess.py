"""What the model can back up about SPXL today, in place of a buy / hold / sell verdict.

A 1990-2026 backtest (scripts/backtest_rating.py) found that the rating had no timing value while
the forecast distribution itself was well calibrated. So the page states three things the
distribution does support, each placed against that backtest's month-ends:

1. Leverage cost: the S&P 500 total return per year SPXL needs to break even over the long run (its
   average log growth is then zero; the "hurdle"), with its parts in the fund's terms per year:
   financing of the borrowed notional, fees, and volatility drag.
2. Drawdown risk: the chance SPXL closes at least 20% below today's price within 3 months, and how
   often such a fall actually followed in past months at the same level.
3. The 90% price range at 3 months and its median.

Levels (low / normal / high) are terciles of the 1990-2026 month-ends, read from ``reference.json``,
which ``scripts/backtest_rating.py --write-reference`` regenerates. Without that file every level is
"n/a"; the numbers themselves do not depend on it.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REFERENCE_PATH = Path(__file__).resolve().parent / "reference.json"
DIP_HORIZON = 63            # sessions: 3 months
DIP_SIZE = 0.20
LEVERAGE_LEVELS = ("low", "normal", "high")
DRAWDOWN_LEVELS = ("low", "normal", "elevated")
NA = "n/a"


@dataclass
class LeverageCost:
    hurdle: float                  # S&P 500 total return per year at which SPXL's average log growth is zero
    financing: float               # per year, fund terms: (L - 1) x (3m bill + spread)
    fees: float                    # expense ratio
    drag: float                    # volatility drag L(L-1)/2 x sigma^2
    sigma: float                   # the 1-year index vol behind the drag and the hurdle
    percentile: Optional[float]    # the hurdle's percentile among the reference month-ends (0-100)
    level: str                     # low / normal / high, or n/a
    expected_index_return: float   # the model's long-run S&P 500 estimate: context, not a 6-month forecast


@dataclass
class DrawdownRisk:
    p_dip20_3m: float              # chance of a close >= 20% below today's price within 3 months
    typical: Optional[float]       # its median over the reference month-ends
    percentile: Optional[float]
    level: str                     # low / normal / elevated, or n/a
    history_pred: Optional[float]  # past month-ends at this level: mean predicted chance ...
    history_real: Optional[float]  # ... and how often the dip actually followed
    history_n: Optional[int]


@dataclass
class Assessment:
    leverage: LeverageCost
    drawdown: DrawdownRisk
    range_3m: Optional[Tuple[float, float, float]]   # (5th percentile, median, 95th percentile) price
    notes: List[str] = field(default_factory=list)
    period: Optional[str] = None                      # the reference's month-ends, "1990-01..2026-08"


# ---------------------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------------------
def load_reference(path=None) -> Optional[dict]:
    try:
        with open(path or REFERENCE_PATH, encoding="utf-8") as fh:
            ref = json.load(fh)
    except (OSError, ValueError, RecursionError):
        return None
    return ref if isinstance(ref, dict) else None


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if math.isfinite(v) else None


def _section(reference: Any, key: str) -> dict:
    s = reference.get(key) if isinstance(reference, dict) else None
    return s if isinstance(s, dict) else {}


def _cutoffs(cutoffs: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(cutoffs, (list, tuple)) or len(cutoffs) != 2:
        return None
    lo, hi = _num(cutoffs[0]), _num(cutoffs[1])
    return (lo, hi) if lo is not None and hi is not None and lo <= hi else None


def level_of(value: Any, cutoffs: Any, labels: Sequence[str]) -> str:
    """labels[0] below the lower cutoff, labels[1] from it up to the upper one, labels[2] from there."""
    v, c = _num(value), _cutoffs(cutoffs)
    if v is None or c is None:
        return NA
    return labels[int(v >= c[0]) + int(v >= c[1])]


def _knots(values: Sequence[float], percentiles: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted distinct values, each with the mean percentile of its ties."""
    order = np.argsort(values, kind="stable")
    xs, inv = np.unique(np.asarray(values)[order], return_inverse=True)
    return xs, np.bincount(inv, weights=np.asarray(percentiles)[order]) / np.bincount(inv)


def percentile_of(value: Any, grid: Any, cutoffs: Any = None) -> Optional[float]:
    """Where ``value`` sits (0-100) in a grid of values at evenly spaced percentiles (0, 5, ..., 100),
    linear in between; tied grid values share the mean of their percentiles. The tercile ``cutoffs``,
    when given, are pinned at 100/3 and 200/3, so the percentile never contradicts the level shown."""
    v = _num(value)
    if v is None or not isinstance(grid, (list, tuple)) or len(grid) < 2:
        return None
    g = [_num(x) for x in grid]
    if any(x is None for x in g) or any(b < a for a, b in zip(g, g[1:])):
        return None
    if v < g[0]:
        return 0.0
    if v > g[-1]:
        return 100.0
    ps = list(np.linspace(0.0, 100.0, len(g)))
    xs, pv = _knots(g, ps)
    c = _cutoffs(cutoffs)
    if c is not None:
        xc, pc = _knots(g + list(c), ps + [100.0 / 3.0, 200.0 / 3.0])
        if np.all(np.diff(pc) >= 0):    # cutoffs that disagree with the grid are left out
            xs, pv = xc, pc
    return float(np.interp(v, xs, pv))


# ---------------------------------------------------------------------------------------
# The two measures
# ---------------------------------------------------------------------------------------
def leverage_cost(leverage: float, expense_ratio: float, financing_rate: float, sigma_1y: float,
                  expected_index_return: float, reference: Optional[dict]) -> LeverageCost:
    """``financing_rate`` is already multiplied by (L - 1), as in etf.ETFParams."""
    L, s = float(leverage), float(sigma_1y)
    # etf.ETFParams.breakeven_index_return, restated so this module needs no market-data imports
    hurdle = math.exp((expense_ratio + financing_rate) / L + 0.5 * L * s * s) - 1.0 if L > 0 else float("nan")
    sec = _section(reference, "hurdle")
    return LeverageCost(hurdle=hurdle, financing=float(financing_rate), fees=float(expense_ratio),
                        drag=0.5 * L * (L - 1.0) * s * s, sigma=s,
                        percentile=percentile_of(hurdle, sec.get("grid"), sec.get("cutoffs")),
                        level=level_of(hurdle, sec.get("cutoffs"), LEVERAGE_LEVELS),
                        expected_index_return=float(expected_index_return))


def drawdown_risk(p_dip20_3m: Any, reference: Optional[dict]) -> DrawdownRisk:
    sec = _section(reference, "p_dip20_3m")
    p = _num(p_dip20_3m)
    level = level_of(p, sec.get("cutoffs"), DRAWDOWN_LEVELS)
    by_level = sec.get("by_level")
    hist = by_level.get(level) if isinstance(by_level, dict) and level != NA else None
    hist = hist if isinstance(hist, dict) else {}
    n = _num(hist.get("n"))
    return DrawdownRisk(p_dip20_3m=float("nan") if p is None else p, typical=_num(sec.get("median")),
                        percentile=percentile_of(p, sec.get("grid"), sec.get("cutoffs")), level=level,
                        history_pred=_num(hist.get("pred")), history_real=_num(hist.get("real")),
                        history_n=int(n) if n is not None else None)


def sigma_1y(vol) -> float:
    """The 1-year index vol behind the hurdle, as pipeline.run_forecast computes it."""
    return vol.one_year_vol if vol.one_year_vol is not None else vol.total_vol(252)


def assess(fc, reference_path=None) -> Assessment:
    """The assessment of a pipeline.Forecast (the reference is read from ``reference_path``, by
    default the packaged reference.json)."""
    ref = load_reference(reference_path)
    notes: List[str] = []
    if ref is None:
        notes.append("The history file (reference.json) is missing or unreadable, so no levels are shown")
    etf = fc.etf
    lev = leverage_cost(etf.leverage, etf.expense_ratio, etf.financing_rate, sigma_1y(fc.vol),
                        fc.expected.final, ref)
    p_dip, rng = None, None
    if DIP_HORIZON in fc.sim.horizons:
        p_dip = fc.sim.summary(DIP_HORIZON)["p_drawdown_20"]
        q = fc.sim.quantiles(DIP_HORIZON, (5, 50, 95))
        rng = (q[5.0], q[50.0], q[95.0])
    else:
        notes.append("The 3-month horizon was not simulated, so the drawdown risk and price range are not shown")
    period = ref.get("period") if ref is not None else None
    return Assessment(leverage=lev, drawdown=drawdown_risk(p_dip, ref), range_3m=rng, notes=notes,
                      period=period if isinstance(period, str) else None)


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def assessment_to_dict(a: Assessment) -> Dict[str, Any]:
    r = a.range_3m
    return _clean({"leverage": asdict(a.leverage), "drawdown": asdict(a.drawdown),
                   "range_3m": None if r is None else {"p5": r[0], "median": r[1], "p95": r[2]},
                   "horizon_days": DIP_HORIZON, "dip_size": DIP_SIZE, "period": a.period, "notes": list(a.notes)})
