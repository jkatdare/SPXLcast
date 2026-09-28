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
import math
import os
import sys
import time
from dataclasses import replace
from typing import List, Optional

from rich.console import Console
from rich.markup import escape

from .config import Config
from .pipeline import forecast_to_dict, run_forecast
from .report import horizon_label, ordinal, quiet_summary, render_all, render_score
from .tracklog import DEFAULT_LOG, NO_PRICES, append_log, score_log

COMMANDS = ("forecast", "price", "metrics", "news", "calibrate", "log", "score")


def _positive_int(text: str) -> int:
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def _nonnegative_int(text: str) -> int:
    v = int(text)
    if v < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return v


def _finite_float(text: str) -> float:
    v = float(text)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError("must be a finite number")
    return v


def _positive_float(text: str) -> float:
    v = _finite_float(text)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def _nonnegative_float(text: str) -> float:
    v = _finite_float(text)
    if v < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return v


def _drift_float(text: str) -> float:
    # each simulated path draws its drift up to 50 points either side of this, and must stay above -100%
    v = _finite_float(text)
    if v <= -0.5 or v > 5.0:
        raise argparse.ArgumentTypeError("must be an annual return above -50% and at most 500% (e.g. 0.08)")
    return v


def _paths_int(text: str) -> int:
    v = int(text)
    if v < 100:
        raise argparse.ArgumentTypeError("must be at least 100 paths")
    return v


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--price", type=_positive_float, action="append", default=None,
                   help="price level to locate in the distribution (repeatable)")
    p.add_argument("--horizons", type=_positive_int, nargs="+", default=None, help="horizons in trading days (default 5 10 21 63 126 252)")
    p.add_argument("--rating-horizon", type=_positive_int, default=None,
                   help="horizon of the buy-limit ladder and of the logged (retired) rating (default 126)")
    p.add_argument("--paths", type=_paths_int, default=None, help="Monte Carlo paths (default 50000)")
    p.add_argument("--seed", type=_nonnegative_int, default=None, help="random seed (default 42)")
    p.add_argument("--no-news", action="store_true", help="skip news sentiment")
    p.add_argument("--no-fred", action="store_true", help="skip FRED macro series")
    p.add_argument("--no-macro-adj", action="store_true", help="disable valuation/regime adjustments to the index drift")
    p.add_argument("--refresh", action="store_true",
                   help="ignore the download cache (still refreshes it; the price-history archive is kept)")
    p.add_argument("--index-drift", type=_drift_float, default=None, help="override S&P 500 expected total return, e.g. 0.08")
    p.add_argument("--vol", type=_nonnegative_float, default=None, help="override annualised index vol, e.g. 0.18")
    p.add_argument("--pe", type=_finite_float, default=None, help="override trailing P/E of the index")
    p.add_argument("--div-yield", type=_finite_float, default=None, help="override dividend yield, e.g. 0.013")
    p.add_argument("--eps-growth", type=_finite_float, default=None, help="override long-run nominal EPS growth, e.g. 0.055")
    p.add_argument("--swap-spread", type=_finite_float, default=None, help="override the all-in financing spread, e.g. 0.0075")
    p.add_argument("--skew", type=_positive_float, default=None, help="daily shock skew gamma (1 = symmetric, default 0.9)")
    p.add_argument("--log-file", dest="log_file", default=None,
                   help=f"append this run's forecast to a track-record CSV (default for `log`/`score`: {DEFAULT_LOG})")
    p.add_argument("--json", dest="json_path", default=None, help="write the full result to a JSON file")
    p.add_argument("--archive", dest="archive_dir", default=None,
                   help="the run archive: forecasts store their inputs, simulator arguments and new headlines "
                        "there; `score` reads each run's fine percentile grid from it")
    p.add_argument("--plot", dest="plot_path", default=None, help="write a fan chart PNG to this path")
    p.add_argument("--quiet", action="store_true", help="only print the one-line summary")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spxlcast", description="Fundamentals-driven SPXL forecast: leverage cost, "
                                                                  "drawdown risk and price percentiles")
    sub = parser.add_subparsers(dest="command")
    p_fc = sub.add_parser("forecast", help="full report (default)")
    _add_common(p_fc)
    p_pr = sub.add_parser("price", help="locate one or more prices in the forecast distribution")
    p_pr.add_argument("prices", type=_positive_float, nargs="+")
    _add_common(p_pr)
    for name, help_text in (("metrics", "influencer levels, valuation and macro"), ("news", "news sentiment"),
                            ("calibrate", "leveraged-ETF calibration and drivers"),
                            ("log", "run the forecast quietly and append it to the track-record CSV")):
        _add_common(sub.add_parser(name, help=help_text))
    p_sc = sub.add_parser("score", help="score the track-record CSV against realised prices")
    p_sc.add_argument("--model-version", dest="model_version", default=None,
                      help="score only the rows logged by this model version (default: all, pooled)")
    _add_common(p_sc)
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
        rep = score_log(args.log_file or DEFAULT_LOG, cfg=cfg, model_version=args.model_version,
                        archive_dir=args.archive_dir)
        render_score(rep, console, args.log_file or DEFAULT_LOG)
        return 1 if NO_PRICES in rep.notes else 0

    with console.status("Fetching data and simulating...", spinner="dots"):
        fc = run_forecast(cfg)

    log_file = args.log_file or (DEFAULT_LOG if args.command == "log" else None)
    logged = None
    if log_file:
        try:
            append_log(fc, log_file)
        except Exception as exc:  # noqa: BLE001 - still write the outputs; the run then exits non-zero
            logged = exc
    archived = None
    if args.archive_dir:
        from .archive import archive_run
        try:
            archived = archive_run(fc, args.archive_dir, prices)
        except Exception as exc:  # noqa: BLE001 - the forecast and the track record matter more
            archived = exc

    # Outputs are written before the report is rendered, so a rendering problem cannot leave a logged
    # run without its forecast.json / fan.png.
    if args.json_path:
        doc = forecast_to_dict(fc, prices)
        _write_atomic(args.json_path, lambda fh: fh.write(
            json.dumps(doc, indent=2, default=str, allow_nan=False).encode("utf-8")))
    if args.plot_path:
        from .plots import save_fan_chart
        fmt = os.path.splitext(args.plot_path)[1].lstrip(".") or None
        _write_atomic(args.plot_path, lambda fh: save_fan_chart(fc, fh, prices, fmt=fmt))

    if args.quiet or args.command == "log":
        console.print(f"{cfg.etf} {fc.spot:,.2f} ({fc.spot_status} {fc.spot_date})  {quiet_summary(fc)}",
                      markup=False, highlight=False, soft_wrap=True)
        for p in prices:
            rows = fc.price_lookup(p)
            r = next(x for x in rows if x["horizon"] == fc.rating.horizon)
            dips = p <= fc.spot
            console.print(f"  price {p:,.2f}: {ordinal(r['percentile'])} percentile of the price in "
                          f"{horizon_label(r['horizon'])}, {r['p_touch_below' if dips else 'p_touch_above']:.0%} chance "
                          f"it {'dips' if dips else 'rises'} to it before then")
        if log_file and logged is None:
            console.print(f"appended to {log_file}", style="dim", markup=False, highlight=False)
    else:
        sections = {
            "forecast": None,
            "price": {"header", "assessment", "price", "ladder", "notes"},
            "metrics": {"header", "metrics", "sensitivity", "notes"},
            "news": {"header", "news", "notes"},
            "calibrate": {"header", "drivers", "sensitivity", "notes"},
        }[args.command]
        try:
            render_all(fc, console, prices=prices, sections=sections)
        except Exception as exc:  # noqa: BLE001 - the report is presentation; the run's outputs are written
            logging.getLogger(__name__).debug("report rendering failed", exc_info=True)
            console.print(f"warning: the report could not be rendered in full: {type(exc).__name__}: {exc}",
                          style="yellow", markup=False, highlight=False)
        if log_file and logged is None:
            console.print(f"appended to {log_file}", style="dim", markup=False, highlight=False)
    if isinstance(archived, Exception):
        console.print(f"[yellow]warning: archive failed: {escape(str(archived))}[/yellow]")
    elif archived:
        console.print(f"[dim]archived {archived['run']} ({archived['new_stories']} new stories)[/dim]")
    for path in (args.json_path, args.plot_path):
        if path:
            console.print(f"wrote {path}", style="dim", markup=False, highlight=False)
    if logged is not None:
        console.print(f"error: the run was not logged to {log_file}: {type(logged).__name__}: {logged}",
                      style="red", markup=False, highlight=False)
        return 1
    return 0


def _write_atomic(path: str, write) -> None:
    """Write through a temp file in the same directory and rename it into place, so a reader (the live
    loop, a download) never sees a truncated or half-written file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.urandom(6).hex()}.tmp"    # unique: a manual run may overlap the scheduled one
    try:
        with open(tmp, "xb") as fh:
            write(fh)
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:     # Windows: a reader (the live loop) holds the file open for a moment
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
