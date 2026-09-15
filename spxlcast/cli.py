"""Command-line interface.

    spxlcast forecast [--price 200 --price 190] [--horizons 21 63 126 252] [--json out.json] [--plot out.png]
    spxlcast price 190 200            # focus on where given prices sit in the distribution
    spxlcast metrics                  # influencer levels, valuation, macro, constituents
    spxlcast news                     # sentiment breakdown and headlines
    spxlcast calibrate                # leveraged-ETF calibration diagnostics
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from typing import List, Optional

from rich.console import Console

from .config import Config
from .pipeline import forecast_to_dict, run_forecast
from .report import render_all

COMMANDS = ("forecast", "price", "metrics", "news", "calibrate")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--price", type=float, action="append", default=None,
                   help="price level to locate in the distribution (repeatable)")
    p.add_argument("--horizons", type=int, nargs="+", default=None, help="horizons in trading days (default 21 63 126 252)")
    p.add_argument("--rating-horizon", type=int, default=None, help="horizon used for the rating (default 126)")
    p.add_argument("--paths", type=int, default=None, help="Monte Carlo paths (default 20000)")
    p.add_argument("--seed", type=int, default=None, help="random seed (default 42)")
    p.add_argument("--no-news", action="store_true", help="skip news sentiment")
    p.add_argument("--no-fred", action="store_true", help="skip FRED macro series")
    p.add_argument("--no-macro-adj", action="store_true", help="disable macro adjustments to the index drift")
    p.add_argument("--refresh", action="store_true", help="ignore the on-disk cache")
    p.add_argument("--index-drift", type=float, default=None, help="override S&P 500 expected total return, e.g. 0.08")
    p.add_argument("--vol", type=float, default=None, help="override annualised index vol, e.g. 0.18")
    p.add_argument("--pe", type=float, default=None, help="override trailing P/E of the index")
    p.add_argument("--div-yield", type=float, default=None, help="override dividend yield, e.g. 0.013")
    p.add_argument("--eps-growth", type=float, default=None, help="override long-run nominal EPS growth, e.g. 0.055")
    p.add_argument("--json", dest="json_path", default=None, help="write the full result to a JSON file")
    p.add_argument("--plot", dest="plot_path", default=None, help="write a fan chart PNG to this path")
    p.add_argument("--quiet", action="store_true", help="only print the rating line")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spxlcast", description="Fundamentals-driven SPXL forecast, rating and price percentiles")
    sub = parser.add_subparsers(dest="command")
    p_fc = sub.add_parser("forecast", help="full report (default)")
    _add_common(p_fc)
    p_pr = sub.add_parser("price", help="locate one or more prices in the forecast distribution")
    p_pr.add_argument("prices", type=float, nargs="+")
    _add_common(p_pr)
    for name, help_text in (("metrics", "influencer levels, valuation and macro"), ("news", "news sentiment"),
                            ("calibrate", "leveraged-ETF calibration and drivers")):
        _add_common(sub.add_parser(name, help=help_text))
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    updates = {}
    if args.horizons:
        updates["horizons"] = tuple(sorted(set(args.horizons)))
    if args.rating_horizon:
        updates["rating_horizon"] = args.rating_horizon
    if args.paths:
        updates["n_paths"] = args.paths
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.no_news:
        updates["use_news"] = False
    if args.no_fred:
        updates["use_fred"] = False
    if args.no_macro_adj:
        updates["use_macro_adjustments"] = False
    if args.refresh:
        updates["refresh"] = True
    if args.index_drift is not None:
        updates["override_index_drift"] = args.index_drift
    if args.vol is not None:
        updates["override_vol"] = args.vol
    if args.pe is not None:
        updates["override_trailing_pe"] = args.pe
    if args.div_yield is not None:
        updates["override_dividend_yield"] = args.div_yield
    if args.eps_growth is not None:
        updates["long_run_eps_growth"] = args.eps_growth
    cfg = replace(cfg, **updates)
    if cfg.rating_horizon not in cfg.horizons:
        cfg = replace(cfg, horizons=tuple(sorted(set(cfg.horizons) | {cfg.rating_horizon})))
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv.insert(0, "forecast")
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = config_from_args(args)
    prices: List[float] = list(args.price or [])
    if args.command == "price":
        prices = list(args.prices) + prices

    console = Console()
    if not console.is_terminal:  # piped or captured output: do not squeeze tables into 80 columns
        console = Console(width=max(console.width, 120))
    with console.status("Fetching data and simulating...", spinner="dots"):
        fc = run_forecast(cfg)

    if args.quiet:
        console.print(f"{cfg.etf} {fc.spot:,.2f}  rating {fc.rating.label} ({fc.rating.conviction.lower()}, "
                      f"score {fc.rating.score:+.2f}) over {fc.rating.horizon}d; median {fc.rating.median_return:+.1%}, "
                      f"P(beat T-bill) {fc.rating.p_beat_rf:.0%}")
        for p in prices:
            rows = fc.price_lookup(p)
            r = next(x for x in rows if x["horizon"] == fc.rating.horizon)
            console.print(f"  price {p:,.2f}: {r['percentile']:.0f}th percentile at {fc.rating.horizon}d, "
                          f"P(touch at/below) {r['p_touch_below']:.0%}, P(touch at/above) {r['p_touch_above']:.0%}")
    else:
        sections = {
            "forecast": None,
            "price": {"header", "rating", "price", "ladder", "notes"},
            "metrics": {"header", "metrics", "sensitivity", "notes"},
            "news": {"header", "news", "notes"},
            "calibrate": {"header", "drivers", "sensitivity", "notes"},
        }[args.command]
        render_all(fc, console, prices=prices, sections=sections)

    if args.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(forecast_to_dict(fc, prices), fh, indent=2, default=str)
        console.print(f"[dim]wrote {args.json_path}[/dim]")
    if args.plot_path:
        from .plots import save_fan_chart
        os.makedirs(os.path.dirname(os.path.abspath(args.plot_path)), exist_ok=True)
        save_fan_chart(fc, args.plot_path, prices)
        console.print(f"[dim]wrote {args.plot_path}[/dim]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
