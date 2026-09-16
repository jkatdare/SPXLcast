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
from .report import ordinal, render_all, render_score
from .tracklog import DEFAULT_LOG, append_log, score_log

COMMANDS = ("forecast", "price", "metrics", "news", "calibrate", "log", "score")


def _positive_int(text: str) -> int:
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def _positive_float(text: str) -> float:
    v = float(text)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def _nonnegative_float(text: str) -> float:
    v = float(text)
    if v < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return v


def _drift_float(text: str) -> float:
    v = float(text)
    if v <= -1.0 or v > 5.0:
        raise argparse.ArgumentTypeError("must be an annual return above -100% (e.g. 0.08)")
    return v


def _paths_int(text: str) -> int:
    v = int(text)
    if v < 100:
        raise argparse.ArgumentTypeError("must be at least 100 paths")
    return v


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--price", type=float, action="append", default=None,
                   help="price level to locate in the distribution (repeatable)")
    p.add_argument("--horizons", type=_positive_int, nargs="+", default=None, help="horizons in trading days (default 21 63 126 252)")
    p.add_argument("--rating-horizon", type=_positive_int, default=None, help="horizon used for the rating (default 126)")
    p.add_argument("--paths", type=_paths_int, default=None, help="Monte Carlo paths (default 50000)")
    p.add_argument("--seed", type=int, default=None, help="random seed (default 42)")
    p.add_argument("--no-news", action="store_true", help="skip news sentiment")
    p.add_argument("--no-fred", action="store_true", help="skip FRED macro series")
    p.add_argument("--no-macro-adj", action="store_true", help="disable valuation/regime adjustments to the index drift")
    p.add_argument("--refresh", action="store_true", help="ignore the on-disk cache (still refreshes it)")
    p.add_argument("--index-drift", type=_drift_float, default=None, help="override S&P 500 expected total return, e.g. 0.08")
    p.add_argument("--vol", type=_nonnegative_float, default=None, help="override annualised index vol, e.g. 0.18")
    p.add_argument("--pe", type=float, default=None, help="override trailing P/E of the index")
    p.add_argument("--div-yield", type=float, default=None, help="override dividend yield, e.g. 0.013")
    p.add_argument("--eps-growth", type=float, default=None, help="override long-run nominal EPS growth, e.g. 0.055")
    p.add_argument("--swap-spread", type=float, default=None, help="override the all-in financing spread, e.g. 0.0075")
    p.add_argument("--skew", type=_positive_float, default=None, help="daily shock skew gamma (1 = symmetric, default 0.9)")
    p.add_argument("--log-file", dest="log_file", default=None,
                   help=f"append this run's forecast to a track-record CSV (default for `log`/`score`: {DEFAULT_LOG})")
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
                            ("calibrate", "leveraged-ETF calibration and drivers"),
                            ("log", "run the forecast quietly and append it to the track-record CSV"),
                            ("score", "score the track-record CSV against realised prices")):
        _add_common(sub.add_parser(name, help=help_text))
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    updates = {}
    if args.horizons is not None:
        updates["horizons"] = tuple(sorted(set(args.horizons)))
    if args.rating_horizon is not None:
        updates["rating_horizon"] = args.rating_horizon
    if args.paths is not None:
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
        updates["override_eps_growth"] = args.eps_growth
    if args.swap_spread is not None:
        updates["swap_spread"] = args.swap_spread
    if args.skew is not None:
        updates["skew_gamma"] = args.skew
    cfg = replace(cfg, **updates)
    if cfg.rating_horizon not in cfg.horizons:
        cfg = replace(cfg, horizons=tuple(sorted(set(cfg.horizons) | {cfg.rating_horizon})))
    return cfg


def _utf8_stdout() -> None:
    """Windows consoles and redirected output default to cp1252; headlines can contain anything."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":            # the status page has its own small parser
        from .serve import main as serve_main
        return serve_main(argv[1:])
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv.insert(0, "forecast")
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    _utf8_stdout()
    cfg = config_from_args(args)
    prices: List[float] = list(args.price or [])
    if args.command == "price":
        prices = list(args.prices) + prices

    console = Console()
    if not console.is_terminal or console.width < 110:  # keep tables readable when piped or narrow
        console = Console(width=max(console.width, 120))

    if args.command == "score":
        render_score(score_log(args.log_file or DEFAULT_LOG, cfg=cfg), console, args.log_file or DEFAULT_LOG)
        return 0

    with console.status("Fetching data and simulating...", spinner="dots"):
        fc = run_forecast(cfg)

    log_file = args.log_file or (DEFAULT_LOG if args.command == "log" else None)
    if log_file:
        append_log(fc, log_file)

    if args.quiet or args.command == "log":
        console.print(f"{cfg.etf} {fc.spot:,.2f} ({fc.spot_status} {fc.spot_date})  rating {fc.rating.label} "
                      f"({fc.rating.conviction.lower()}, score {fc.rating.score:+.2f} +/- {fc.rating.score_se:.2f}) "
                      f"over {fc.rating.horizon}d; median {fc.rating.median_return:+.1%}, mean {fc.rating.mean_return:+.1%}, "
                      f"P(beat T-bill) {fc.rating.p_beat_rf:.0%}")
        for p in prices:
            rows = fc.price_lookup(p)
            r = next(x for x in rows if x["horizon"] == fc.rating.horizon)
            console.print(f"  price {p:,.2f}: {ordinal(r['percentile'])} percentile at {fc.rating.horizon}d, "
                          f"P(dips to it) {r['p_touch_below']:.0%}, P(rises to it) {r['p_touch_above']:.0%}")
        if log_file:
            console.print(f"[dim]appended to {log_file}[/dim]")
    else:
        sections = {
            "forecast": None,
            "price": {"header", "rating", "price", "ladder", "notes"},
            "metrics": {"header", "metrics", "sensitivity", "notes"},
            "news": {"header", "news", "notes"},
            "calibrate": {"header", "drivers", "sensitivity", "notes"},
        }[args.command]
        render_all(fc, console, prices=prices, sections=sections)
        if log_file:
            console.print(f"[dim]appended to {log_file}[/dim]")

    if args.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(forecast_to_dict(fc, prices), fh, indent=2, default=str, allow_nan=False)
        console.print(f"[dim]wrote {args.json_path}[/dim]")
    if args.plot_path:
        from .plots import save_fan_chart
        os.makedirs(os.path.dirname(os.path.abspath(args.plot_path)), exist_ok=True)
        save_fan_chart(fc, args.plot_path, prices)
        console.print(f"[dim]wrote {args.plot_path}[/dim]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
