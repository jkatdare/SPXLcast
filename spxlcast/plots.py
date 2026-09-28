"""Optional matplotlib output: fan chart of simulated SPXL prices and a horizon histogram."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .pipeline import Forecast
from .report import ordinal


def save_fan_chart(fc: Forecast, path, prices: Optional[List[float]] = None, fmt: Optional[str] = None):
    """Write the chart to ``path`` (a file name or a binary file object; ``fmt`` names the image format
    when the name has no extension or a file object is given)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = fc.sim
    fan = sim.fan
    qs = list(sim.fan_quantiles)
    days = np.arange(fan.shape[0])
    h = fc.cfg.rating_horizon

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [3, 2]})

    def col(q):
        return fan[:, qs.index(q)]

    ax1.fill_between(days, col(5), col(95), color="#4C72B0", alpha=0.15, label="5-95%")
    ax1.fill_between(days, col(25), col(75), color="#4C72B0", alpha=0.35, label="25-75%")
    ax1.plot(days, col(50), color="#1f3c88", lw=2, label="median")
    ax1.axhline(fc.spot, color="black", lw=0.8, ls="--", label=f"spot {fc.spot:.2f}")
    for p in prices or []:
        ax1.axhline(p, color="#C44E52", lw=1, ls=":", label=f"price {p:.2f}")
    ax1.set_title(f"{fc.cfg.etf} simulated price fan  ({sim.n_paths:,} paths)")
    ax1.set_xlabel("trading days ahead")
    ax1.set_ylabel("price")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    term = sim.terminal[h]
    ax2.hist(term, bins=80, color="#4C72B0", alpha=0.8)
    ax2.axvline(fc.spot, color="black", lw=0.8, ls="--")
    ax2.axvline(np.median(term), color="#1f3c88", lw=2)
    for p in prices or []:
        pct = sim.percentile_of_price(p, h)
        ax2.axvline(p, color="#C44E52", lw=1, ls=":")
        ax2.text(p, ax2.get_ylim()[1] * 0.9, f"{ordinal(pct)} pct", color="#C44E52", rotation=90,
                 va="top", ha="right", fontsize=8)
    ax2.set_title(f"Distribution at {h} trading days")
    ax2.set_xlabel("price")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, format=fmt)
    plt.close(fig)
    return path
