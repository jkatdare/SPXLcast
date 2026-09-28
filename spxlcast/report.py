"""Console rendering of a Forecast with rich."""
from __future__ import annotations

import math
from typing import Iterable, List, Optional

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .assess import DIP_HORIZON, DRAWDOWN_LEVELS, LEVERAGE_LEVELS, NA
from .config import MARKET_TICKERS
from .pipeline import Forecast
from .tracklog import MIN_INDEPENDENT


def pct(x: Optional[float], digits: int = 1, sign: bool = True) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"


def num(x: Optional[float], digits: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{digits}f}"


def ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def horizon_label(days: int) -> str:
    if days % 21 == 0:
        m = days // 21
        return f"{m}M" if m % 12 else f"{m // 12}Y"
    if days < 21 and days % 5 == 0:
        return f"{days // 5}W"
    return f"{days}d"


# ---------------------------------------------------------------------------------------
def render_header(fc: Forecast, console: Console) -> None:
    idx = fc.fundamentals.index_level
    text = Text()
    text.append(f"{fc.cfg.etf}  ", style="bold")
    text.append(f"{fc.spot:,.2f}", style="bold cyan")
    text.append(f"   {fc.spot_status} {fc.spot_date}   |   S&P 500 {num(idx, 0)}   |   ")
    text.append(f"VIX {num(fc.macro.vix, 1)}   |   3m bill {pct(fc.macro.rf_3m, 2, False)}   |   "
                f"10y {pct(fc.macro.y10, 2, False)}")
    fetched = fc.snap.fetched_at.get("prices", fc.snap.asof)
    console.print(Panel(text, title="SPXLcast",
                        subtitle=f"prices fetched {fetched:%Y-%m-%d %H:%M} UTC  |  run {fc.snap.asof:%H:%M} UTC",
                        box=box.ROUNDED))
    if fc.spot_status == "intraday":
        console.print("[dim]The session is open: the spot, VIX and yields are live quotes, not closes.[/dim]")


def _span(period: Optional[str]) -> str:
    """'1990-01..2026-08' -> 'from 1990 to 2026'."""
    try:
        a, b = str(period).split("..")
        return f"from {int(a[:4])} to {int(b[:4])}"
    except (TypeError, ValueError):
        return "in the backtest"


def render_assessment(fc: Forecast, console: Console) -> None:
    """Leverage cost, drawdown risk and the 3-month range, in plain words (see assess.py)."""
    a = fc.assessment
    if a is None:
        console.print("[dim]No assessment for this run.[/dim]")
        return
    lc, dd, span, L = a.leverage, a.drawdown, _span(a.period), fc.etf.leverage
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column()

    def level(name: str, value: str) -> None:
        grid.add_row(name, Text(value, style="bold cyan" if value != NA else "dim"))

    def where(percentile: Optional[float], value: str, labels, ends) -> None:
        if percentile is not None and value in labels:
            i = labels.index(value)
            n = min(max(round(percentile), (0, 34, 67)[i]), (33, 66, 100)[i])   # rounding stays inside the third
            grid.add_row("", f"That is higher than in {n}% of the months {span} (low = the {ends[0]} third of those "
                             f"months, {labels[2]} = the {ends[1]} third).")

    total = lc.financing + lc.fees + lc.drag
    level("Leverage cost", lc.level)
    grid.add_row("", f"The S&P 500 has to return {pct(lc.hurdle, 1, False)} a year on average, dividends included, "
                     f"for SPXL just to break even over the long run.")
    where(lc.percentile, lc.level, LEVERAGE_LEVELS, ("cheapest", "costliest"))
    grid.add_row("", f"Compared with {L:.0f} times the S&P 500's return, SPXL gives up about {pct(total, 1, False)} a "
                     f"year: borrowing {pct(lc.financing, 1, False)} (it borrows {L - 1:.0f} times its own value and pays "
                     f"the 3-month Treasury bill rate, {pct(fc.macro.rf_3m, 2, False)}, plus {fc.cfg.swap_spread:.2%} on "
                     f"it), fees {pct(lc.fees, 2, False)}, and volatility drag {pct(lc.drag, 1, False)} (SPXL resets to "
                     f"{L:.0f}x every day, which loses money when the market zigzags: the bigger the swings, the bigger "
                     f"the loss. The model expects the S&P 500 to swing about {pct(lc.sigma, 0, False)} a year).")
    grid.add_row("", f"The S&P 500 has to make up only about a third of that ({pct(total / L, 1, False)}), plus a "
                     f"little for its own swings: hence the {pct(lc.hurdle, 1, False)}.")
    grid.add_row("", f"For comparison: the model's long-run S&P 500 estimate is {pct(lc.expected_index_return)} a year. "
                     f"It is an average over many years, not a forecast for the coming months, and since 1990 it has "
                     f"run below what the S&P 500 actually returned.")
    grid.add_row("", "")

    level("Drawdown risk", dd.level)
    if math.isfinite(dd.p_dip20_3m):
        grid.add_row("", f"{pct(dd.p_dip20_3m, 0, False)} chance SPXL closes at or below {fc.spot * 0.8:,.2f}, a fall "
                         f"of 20% or more from {fc.spot:,.2f}, on some day in the next 3 months (even if it recovers "
                         f"afterwards).")
        where(dd.percentile, dd.level, DRAWDOWN_LEVELS, ("calmest", "riskiest"))
        if dd.history_real is not None and dd.history_pred is not None and dd.history_n:
            grid.add_row("", f"In the backtest's {dd.history_n} months at this level {span}, the model would have said "
                             f"{pct(dd.history_pred, 0, False)} on average, and such a fall followed "
                             f"{pct(dd.history_real, 0, False)} of the time. Before 2009 these use a 3x fund rebuilt from "
                             f"the S&P 500, as SPXL did not exist yet; and neighbouring months share most of their 3 "
                             f"months, so there are fewer separate cases than months.")
    grid.add_row("", "")

    if a.range_3m is not None:
        lo, mid, hi = a.range_3m
        grid.add_row("3-month range", f"In 3 months SPXL ends between {lo:,.2f} and {hi:,.2f} in 90% of the model's "
                                      f"simulations (5% end lower, 5% higher); the typical (middle) outcome is "
                                      f"{mid:,.2f}. In the 1990-2026 backtest, ranges like this held about 9 times in 10.")
    else:
        grid.add_row("3-month range", Text("n/a", style="dim"))
    grid.add_row("", "")
    grid.add_row("", Text("This is not a buy or sell signal. In the 1990-2026 backtest, months at high leverage cost or "
                          "elevated drawdown risk were not followed by lower returns on average (elevated risk did bring "
                          "more 20% falls), and the months the old buy/hold/sell rating marked as a buy did no better than "
                          "the rest, which is why it is no longer shown.", style="dim"))
    for n in a.notes:
        grid.add_row("", Text(f"note: {n}", style="yellow"))
    console.print(Panel(grid, title="Assessment", box=box.ROUNDED))


def quiet_summary(fc: Forecast) -> str:
    """The assessment in one line (``--quiet`` and ``log``)."""
    a = fc.assessment
    if a is None:
        return "no assessment for this run"
    lc, dd = a.leverage, a.drawdown
    parts = [f"leverage cost {lc.level} (the S&P 500 needs {pct(lc.hurdle, 1, False)}/yr for SPXL to break even "
             f"over the long run)"]
    parts.append(f"drawdown risk {dd.level}" + (f" ({dd.p_dip20_3m:.0%} chance of a fall of 20% or more within 3 months)"
                                                if math.isfinite(dd.p_dip20_3m) else ""))
    if a.range_3m is not None:
        lo, mid, hi = a.range_3m
        parts.append(f"3-month range {lo:,.2f} to {hi:,.2f} (90% of outcomes), typical {mid:,.2f}")
    return "; ".join(parts)


def render_forecast(fc: Forecast, console: Console) -> None:
    t = Table(title=f"{fc.cfg.etf} price distribution (spot {fc.spot:,.2f})", box=box.SIMPLE_HEAVY)
    for c in ("Horizon", "Mean", "P5", "P25", "Median", "P75", "P95", "P(up)", "P(>bill)",
              "P(>S&P)", "P(dd 20%)"):
        t.add_column(c, justify="right")
    for h in fc.sim.horizons:
        s = fc.sim.summary(h)
        q = fc.sim.quantiles(h, (5, 25, 50, 75, 95))
        mean_price = fc.spot * (1 + s["mean_return"])
        t.add_row(horizon_label(h), f"{mean_price:,.0f}", f"{q[5.0]:,.0f}", f"{q[25.0]:,.0f}",
                  f"{q[50.0]:,.0f}", f"{q[75.0]:,.0f}", f"{q[95.0]:,.0f}",
                  pct(s["p_positive"], 0, False), pct(s["p_beat_rf"], 0, False),
                  pct(s["p_beat_index"], 0, False), pct(s["p_drawdown_20"], 0, False))
    console.print(t)
    console.print("[dim]P(up): ends above spot. P(>bill): beats the T-bill. P(>S&P): beats the unlevered index. "
                  "P(dd 20%): touches -20% from spot at some point before the horizon.[/dim]")
    t2 = Table(title="Returns (SPXL vs. unlevered S&P 500 total return vs. T-bill)", box=box.SIMPLE)
    for c in ("Horizon", "SPXL median", "SPXL mean", "SPXL 5% worst", "S&P median", "T-bill"):
        t2.add_column(c, justify="right")
    for h in fc.sim.horizons:
        s = fc.sim.summary(h)
        t2.add_row(horizon_label(h), pct(s["median_return"]), pct(s["mean_return"]), pct(s["var_5"]),
                   pct(s["index_median_return"]), pct(s["rf_return"]))
    console.print(t2)


def render_price_lookup(fc: Forecast, console: Console, prices: Iterable[float]) -> None:
    for p in prices:
        rows = fc.price_lookup(p)
        below_spot = p < fc.spot
        t = Table(title=f"Price {p:,.2f} ({rows[0]['vs_spot']:+.1%} vs spot {fc.spot:,.2f})", box=box.SIMPLE_HEAVY)
        for c in ("Horizon", "Percentile of end price", "P(end above)", "P(dips to level) = buy-limit fill",
                  "P(rises to level) = sell-limit fill"):
            t.add_column(c, justify="right")
        for r in rows:
            touch_below = pct(r["p_touch_below"], 0, False) if below_spot else "already below"
            touch_above = pct(r["p_touch_above"], 0, False) if not below_spot else "already above"
            t.add_row(horizon_label(r["horizon"]), ordinal(r["percentile"]), pct(r["p_end_above"], 0, False),
                      touch_below, touch_above)
        console.print(t)
    console.print("[dim]Percentile = share of simulated end-of-horizon prices at or below the level "
                  "(a low percentile means the level is a cheap outcome). Fill probabilities use the whole simulated "
                  "path: a buy-limit placed below spot fills if the price dips to it at any point before the horizon.[/dim]")


def render_ladder(fc: Forecast, console: Console, horizon: Optional[int] = None) -> None:
    h = horizon or fc.cfg.rating_horizon
    t = Table(title=f"Buy-limit ladder over the next {horizon_label(h)} (chance the price dips to the level)",
              box=box.SIMPLE)
    for c in ("P(fill)", "Limit price", "vs spot", "Median end vs entry (if filled)", "P(profit if filled)"):
        t.add_column(c, justify="right")
    for r in fc.limit_ladder(h):
        t.add_row(pct(r["p_fill"], 0, False), f"{r['price']:,.2f}", pct(r["vs_spot"]),
                  pct(r["median_end_if_bought"]), pct(r["p_profit_if_bought"], 0, False))
    console.print(t)
    console.print("[dim]The last two columns only count the paths on which the order actually fills, so they already "
                  "reflect that a dip to the level is bad news on average.[/dim]")


def render_drivers(fc: Forecast, console: Console) -> None:
    e, f, m, v, etf = fc.expected, fc.fundamentals, fc.macro, fc.vol, fc.etf
    t = Table(title="S&P 500 expected total return (annualised) - how it is built", box=box.SIMPLE)
    t.add_column("Component")
    t.add_column("Value", justify="right")
    t.add_column("Detail")
    infl = m.breakeven_10y if m.breakeven_10y is not None else fc.cfg.expected_inflation_default
    t.add_row("Earnings-yield model", pct(e.earnings_yield_model), f"E/P {f.earnings_yield:.2%} + inflation {infl:.2%}")
    t.add_row("Dividend-growth model", pct(e.dividend_growth_model),
              f"div yield {f.dividend_yield:.2%} + nominal EPS growth {f.eps_growth:.2%}")
    t.add_row("Base (50/50 blend)", pct(e.base), "")
    for k, val in e.adjustments.items():
        t.add_row(f"  adj: {k}", pct(val), "countercyclical valuation term" if k == "valuation" else "risk-regime penalty")
    for n in e.notes:
        t.add_row("", "", f"[dim]{escape(n)}[/dim]")
    detail = ("command-line override (not clipped)" if fc.cfg.override_index_drift is not None
              else f"clipped to [{fc.cfg.drift_floor:+.0%}, {fc.cfg.drift_cap:+.0%}]")
    t.add_row("[bold]Final index drift[/bold]", f"[bold]{pct(e.final)}[/bold]", detail)
    if fc.sentiment is not None:
        t.add_row("News tilt (first %d days)" % fc.cfg.sentiment_days, pct(fc.sentiment.drift_adjustment),
                  f"{fc.sentiment.label} ({fc.sentiment.score:+.2f}); kept small, not calibrated")
    console.print(t)

    t = Table(title="Volatility (from the VIX term structure) and SPXL mechanics", box=box.SIMPLE)
    t.add_column("Item")
    t.add_column("Value", justify="right")
    t.add_column("Detail")
    for h, s in v.pillars.items():
        t.add_row(f"Index vol pillar {horizon_label(h)}", pct(s, 1, False), "")
    hr = fc.cfg.rating_horizon
    sig = v.total_vol(hr)
    t.add_row(f"Index vol to {horizon_label(hr)}", pct(sig, 1, False), "variance-consistent average")
    t.add_row("Daily shock shape", f"t({fc.cfg.t_dof:.0f}), skew {fc.cfg.skew_gamma:.2f}",
              "fat tails; skew < 1 means larger down moves than up moves")
    t.add_row("Leverage", f"{etf.leverage:.2f}x", "stated 3x, checked against realised beta")
    t.add_row("Expense ratio", pct(etf.expense_ratio, 2, False), "")
    t.add_row("Financing cost", pct(etf.financing_rate, 2, False),
              f"({etf.leverage - 1:.0f}x) x (3m bill + {fc.cfg.swap_spread:.2%} all-in spread)")
    t.add_row(f"Volatility drag to {horizon_label(hr)}", pct(etf.theoretical_drag(sig), 1, False),
              f"L(L-1)/2 x sigma^2 at the {horizon_label(hr)} vol (emerges in the simulation); the assessment's figure "
              f"uses the 1-year vol")
    sig_1y = v.one_year_vol if v.one_year_vol is not None else v.total_vol(min(252, len(v.daily)))
    t.add_row("Break-even index return", pct(etf.breakeven_index_return(sig_1y), 1, False),
              "the leverage-cost hurdle: average index return at which SPXL's long-run (log) growth is zero, at the "
              "1-year vol; the simulated 1-year median sits higher, because volatility comes in bursts")
    if etf.calibration:
        c = etf.calibration
        t.add_row("Realised beta / R2", f"{c.beta:.2f} / {c.r2:.3f}",
                  f"{c.n} days, {c.n_outliers} dislocation days excluded; tracking noise {c.resid_sd_daily:.2%}/day")
    for n in v.notes + etf.notes:
        t.add_row("", "", f"[dim]{escape(n)}[/dim]")
    console.print(t)


def render_metrics(fc: Forecast, console: Console) -> None:
    snap, f, m = fc.snap, fc.fundamentals, fc.macro
    t = Table(title="Influencers - current levels", box=box.SIMPLE_HEAVY)
    for c, j in (("Ticker", "left"), ("Description", "left"), ("Last", "right"), ("1D", "right"),
                 ("1M", "right"), ("3M", "right"), ("1Y", "right")):
        t.add_column(c, justify=j)
    point_tickers = {"^VIX", "^VIX3M", "^VIX6M", "^VVIX", "^SKEW", "^IRX", "2YY=F", "^FVX", "^TNX", "^TYX"}
    for tk, desc in MARKET_TICKERS.items():
        last = snap.last(tk)
        if last is None:
            continue
        cells = []
        for d in (1, 21, 63, 252):
            if tk in point_tickers:
                ch = snap.diff(tk, d)
                cells.append("n/a" if ch is None else f"{ch:+.2f}")
            else:
                ch = snap.change(tk, d)
                cells.append(pct(ch))
        t.add_row(tk, desc, f"{last:,.2f}", *cells)
    console.print(t)
    console.print("[dim]Yield and volatility rows show changes in points; price rows show % changes. "
                  "Changes are aligned to the trading calendar; n/a means the series has a gap there.[/dim]")

    t = Table(title="S&P 500 valuation and macro backdrop", box=box.SIMPLE)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_column("Source")
    src = {**f.sources, **m.sources}
    curve_note = ""
    if m.curve_10y_3m is not None and m.curve_10y_3m < 0:
        curve_note = "inverted = recession signal"
    elif m.curve_10y_3m is None:
        curve_note = src.get("curve_10y_3m", "")
    rows = [
        ("Trailing P/E", num(f.trailing_pe, 1), src.get("trailing_pe", "")),
        ("Earnings yield (E/P)", pct(f.earnings_yield, 2, False), src.get("earnings_yield", src.get("trailing_pe", ""))),
        ("Dividend yield", pct(f.dividend_yield, 2, False), src.get("dividend_yield", "")),
        ("Nominal EPS growth assumed", pct(f.eps_growth, 2, False), src.get("eps_growth", "")),
        ("Price/Book", num(1.0 / f.book_to_price, 2) if f.book_to_price else "n/a", "Yahoo funds_data"),
        ("Price/Sales", num(1.0 / f.sales_to_price, 2) if f.sales_to_price else "n/a", "Yahoo funds_data"),
        ("3m T-bill (bond-equivalent)", pct(m.rf_3m, 2, False), src.get("rf_3m", "")),
        ("2y yield", pct(m.y2, 2, False), src.get("y2", "")),
        ("10y Treasury", pct(m.y10, 2, False), src.get("y10", "")),
        ("30y Treasury", pct(m.y30, 2, False), src.get("y30", "")),
        ("Curve 10y-3m", pct(m.curve_10y_3m, 2), curve_note),
        ("10y breakeven inflation", pct(m.breakeven_10y, 2, False), src.get("breakeven_10y", "default used")),
        ("Real 10y yield", pct(m.real_10y, 2), src.get("real_10y", "")),
        ("E/P minus real 10y", pct(f.earnings_yield - m.real_10y, 2) if m.real_10y is not None else "n/a", "context only"),
        ("HY credit spread (OAS)", pct(m.hy_oas, 2, False), src.get("hy_oas", "FRED unavailable")),
        ("CPI YoY", pct(m.cpi_yoy, 2, False), src.get("cpi_yoy", "FRED unavailable")),
        ("Unemployment", pct(m.unemployment, 1, False), src.get("unemployment", "FRED unavailable")),
        ("VIX / 3M / 6M", f"{num(m.vix, 1)} / {num(m.vix3m, 1)} / {num(m.vix6m, 1)}", src.get("vix_term", "Yahoo")),
        ("VVIX / SKEW", f"{num(m.vvix, 1)} / {num(m.skew, 1)}", "Yahoo"),
    ]
    for r in rows:
        t.add_row(*(escape(str(c)) for c in r))       # sources carry text from the data providers
    console.print(t)

    if snap.holdings is not None and not snap.holdings.empty:
        t = Table(title="Top S&P 500 constituents (weight, valuation, 1M move, news tone)", box=box.SIMPLE)
        for c, j in (("Symbol", "left"), ("Name", "left"), ("Weight", "right"), ("Trailing P/E", "right"),
                     ("Forward P/E", "right"), ("1M", "right"), ("News", "right")):
            t.add_column(c, justify=j)
        by_ticker = fc.sentiment.by_ticker if fc.sentiment else {}
        for _, row in snap.holdings.head(fc.cfg.news_holdings_top_n).iterrows():
            sym = row["symbol"]
            info = snap.info(sym)
            tpe, fpe = info.get("trailingPE"), info.get("forwardPE")
            ns = by_ticker.get(sym)
            t.add_row(escape(str(sym)), escape(str(row["name"])[:28]), pct(row["weight"], 1, False),
                      num(tpe, 1) if tpe else "n/a", num(fpe, 1) if fpe else "n/a",
                      pct(snap.change(sym, 21)) if snap.close(sym) is not None else "n/a",
                      f"{ns:+.2f}" if ns is not None else "n/a")
        console.print(t)
        console.print("[dim]News tone is per company; share classes (e.g. GOOG/GOOGL) are merged into one.[/dim]")


def render_sensitivities(fc: Forecast, console: Console) -> None:
    if not fc.sensitivities:
        return
    t = Table(title=f"What moves {fc.cfg.etf}: daily OLS betas over the last year", box=box.SIMPLE)
    t.add_column("Driver")
    t.add_column("Shock", justify="right")
    t.add_column(f"{fc.cfg.etf} move", justify="right")
    t.add_column("R2", justify="right")
    t.add_column("Days", justify="right")
    for s in fc.sensitivities:
        t.add_row(s.driver, s.unit, f"{s.beta:+.2%}", f"{s.r2:.2f}", str(s.n))
    console.print(t)
    console.print("[dim]Read: a +2.9% move for a +1% shock to SPY means SPXL moves about 2.9% per 1% SPY move on the same "
                  "day. Yield shocks are +1 percentage point, VIX +1 point. Descriptive statistics, not a trading signal.[/dim]")


def render_news(fc: Forecast, console: Console, max_items: int = 5) -> None:
    s = fc.sentiment
    if s is None:
        console.print("[dim]News sentiment disabled.[/dim]")
        return
    style = "green" if s.score > 0.15 else "red" if s.score < -0.15 else "yellow"
    console.print(Panel(Text.assemble((f"{s.label}  ", f"bold {style}"),
                                      (f"score {s.score:+.2f}  from {s.n_used} unique recent stories "
                                       f"({s.n_articles} fetched); near-term drift tilt {s.drift_adjustment:+.1%}/yr", "")),
                        title="News sentiment", box=box.ROUNDED))
    # headlines, feed names and notes are outside text: escaped, so a '[/...]' in one is shown, not parsed
    for n in s.notes:
        console.print(f"[dim]{escape(n)}[/dim]")
    if s.by_ticker:
        line = "  ".join(f"{k} {v:+.2f}" for k, v in sorted(s.by_ticker.items(), key=lambda kv: -abs(kv[1])))
        console.print(f"[dim]By feed/company: {escape(line)}[/dim]")
    for title, items in (("Most positive (weighted)", s.top_positive), ("Most negative (weighted)", s.top_negative)):
        if not items:
            continue
        t = Table(title=title, box=box.SIMPLE, show_lines=False)
        t.add_column("Score", justify="right")
        t.add_column("Weight", justify="right")
        t.add_column("Feed")
        t.add_column("When")
        t.add_column("Headline")
        for x in items[:max_items]:
            t.add_row(f"{x.score:+.2f}", f"{x.weight:.2f}", escape(",".join(dict.fromkeys(x.feeds))[:14]),
                      x.item.published.strftime("%m-%d %H:%M"), escape(x.item.title[:110]))
        console.print(t)


def render_score(rep, console: Console, path: str) -> None:
    """Track-record scoring (see tracklog.score_log)."""
    head = Text()
    head.append(f"{rep.n_rows} logged forecast dates", style="bold")
    if rep.first_date:
        head.append(f"  {rep.first_date} to {rep.last_date}")
    head.append(f"   |   {rep.n_scoreable} scoreable so far   |   {path}")
    if rep.model_version:
        head.append(f"   |   model {rep.model_version} only")
    elif rep.versions:
        head.append("   |   model " + ", ".join(f"{v} ({n} rows)" for v, n in rep.versions.items()))
    console.print(Panel(head, title="Track record", box=box.ROUNDED))
    for n in rep.notes:
        console.print(f"[yellow]note:[/yellow] {escape(n)}")        # notes carry paths and logged text
    if not rep.horizons:
        _render_sentiment_score(rep, console)
        return

    def ci(pair, fmt: str) -> str:
        lo, hi = pair
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return escape("[n/a]")                   # brackets would otherwise be read as rich markup
        return escape(f"[{fmt.format(lo)}, {fmt.format(hi)}]")

    t = Table(title="Calibration: where realised prices fell in the forecast distribution (targets: mean PIT 0.50, "
                    "5-95 band 90%, 25-75 band 50%, tails 5% each) [90% interval]", box=box.SIMPLE_HEAVY)
    for c in ("Horizon", "n", "Indep.", "Mean PIT", "In 5-95", "In 25-75", "Below 5 / above 95",
              "P(dd20) pred/real", "P(up20) pred/real"):
        t.add_column(c, justify="right")
    for hs in rep.horizons:
        t.add_row(horizon_label(hs.horizon), str(hs.n), f"{hs.n_eff:.1f}",
                  f"{hs.mean_pit:.2f} {ci(hs.mean_pit_ci, '{:.2f}')}",
                  f"{pct(hs.cov_5_95, 0, False)} {ci(hs.cov_5_95_ci, '{:.0%}')}", pct(hs.cov_25_75, 0, False),
                  f"{pct(hs.frac_below_5, 0, False)} / {pct(hs.frac_above_95, 0, False)}",
                  f"{hs.pred_dd20:.0%}/{hs.real_dd20:.0%}", f"{hs.pred_up20:.0%}/{hs.real_up20:.0%}")
    console.print(t)
    t = Table(title="Accuracy: CRPS of the log return (lower = better) against a naive lognormal at the raw VIX "
                    "with a T-bill drift; skill above zero = the model beat it [90% interval]", box=box.SIMPLE_HEAVY)
    for c in ("Horizon", "n", "Real mean ret", "Pred median ret", "CRPS model", "CRPS naive", "Skill", "Scored on"):
        t.add_column(c, justify="right")
    for hs in rep.horizons:
        skill = "n/a" if not math.isfinite(hs.crps_skill) else f"{hs.crps_skill:+.1%} {ci(hs.crps_skill_ci, '{:+.1%}')}"
        t.add_row(horizon_label(hs.horizon), str(hs.n), pct(hs.mean_realised_return),
                  pct(hs.mean_predicted_median_return), f"{hs.crps_model:.4f}",
                  f"{hs.crps_naive:.4f}" if math.isfinite(hs.crps_naive) else "n/a", skill,
                  f"{hs.n_fine} fine / {hs.n - hs.n_fine} coarse grid")
    console.print(t)
    partial = [f"{horizon_label(hs.horizon)} {hs.n_crps} of {hs.n}" for hs in rep.horizons if 0 < hs.n_crps < hs.n]
    if partial:
        console.print(f"[dim]CRPS and skill cover only the rows that logged the benchmark's inputs (VIX, T-bill, "
                      f"cost): {', '.join(partial)}.[/dim]")
    console.print("[dim]PIT = where the realised price fell in the predicted distribution (0 = below everything, "
                  "1 = above everything). A mean far from 0.5 is bias; band coverage far from target is mis-sized "
                  "dispersion. Indep. = how many independent outcomes the rows amount to: forecasts a day apart share "
                  "most of their window, so a year of daily 1-month forecasts holds about 12. Intervals are on that "
                  f"many observations and appear once there are {MIN_INDEPENDENT}; until then the numbers are "
                  "anecdotes, not evidence. Fine = the run's 103-point grid from the archive; coarse = the 9 logged "
                  "quantiles.[/dim]")
    if rep.by_drawdown_risk:
        def share(x) -> str:
            return f"{x:.0%}" if x is not None and math.isfinite(x) else "n/a"

        t = Table(title=f"Drawdown risk check: how often a 20% fall within 3 months ({DIP_HORIZON} sessions) followed, "
                        f"by the level the page showed [90% interval]", box=box.SIMPLE)
        for c in ("Drawdown risk", "n", "Indep.", "Predicted", "Happened", "Backtest predicted / happened"):
            t.add_column(c, justify="right")
        for label in DRAWDOWN_LEVELS:
            if label in rep.by_drawdown_risk:
                r = rep.by_drawdown_risk[label]
                t.add_row(label, str(r["n"]), f"{r['n_eff']:.1f}", share(r["pred"]),
                          f"{share(r['real'])} {ci(r['real_ci'], '{:.0%}')}",
                          f"{share(r['backtest_pred'])} / {share(r['backtest_real'])}")
        console.print(t)
        console.print("[dim]Level = the drawdown risk the page showed; rows logged before it existed are placed by their "
                      "logged 3-month dip chance. Happened = SPXL closed 20% or more below the logged price within 3 "
                      "months. Backtest = the same at that level over the 1990-2026 month-ends. The bracket shows how "
                      "far 'Happened' could be from the true rate with this few separate 3-month windows (Indep.); "
                      "while it is wide, the live record neither confirms nor contradicts the page, and the backtest "
                      "column is the long-run check.[/dim]")
    _render_sentiment_score(rep, console)


def _render_sentiment_score(rep, console: Console) -> None:
    if rep.sentiment_n:
        corr = ("n/a (fewer than 10 pairs or no variation)" if rep.sentiment_corr is None
                or not math.isfinite(rep.sentiment_corr) else f"{rep.sentiment_corr:+.2f}")
        console.print(f"News score vs next-10-session SPXL return: correlation {corr} over {rep.sentiment_n} dates. "
                      f"[dim]A value near zero means the news tilt has no predictive content and can stay informational.[/dim]")


def render_notes(fc: Forecast, console: Console) -> None:
    for n in fc.snap.notes:
        console.print(f"[yellow]note:[/yellow] {escape(n)}")


def render_all(fc: Forecast, console: Console, prices: Optional[List[float]] = None,
               sections: Optional[Iterable[str]] = None) -> None:
    sections = set(sections) if sections else {"header", "assessment", "forecast", "price", "ladder", "drivers",
                                                "metrics", "sensitivity", "news", "notes"}
    if "header" in sections:
        render_header(fc, console)
    if "assessment" in sections:
        render_assessment(fc, console)
    if "forecast" in sections:
        render_forecast(fc, console)
    if "price" in sections and prices:
        render_price_lookup(fc, console, prices)
    if "ladder" in sections:
        render_ladder(fc, console)
    if "drivers" in sections:
        render_drivers(fc, console)
    if "metrics" in sections:
        render_metrics(fc, console)
    if "sensitivity" in sections:
        render_sensitivities(fc, console)
    if "news" in sections:
        render_news(fc, console)
    if "notes" in sections:
        render_notes(fc, console)
