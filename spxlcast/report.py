"""Console rendering of a Forecast with rich."""
from __future__ import annotations

from typing import Iterable, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import MARKET_TICKERS
from .pipeline import Forecast

RATING_STYLE = {"BUY": "bold green", "HOLD": "bold yellow", "SELL": "bold red"}


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
        return f"{m}M" if m < 12 else f"{m // 12}Y" if m % 12 == 0 else f"{m}M"
    return f"{days}d"


# ---------------------------------------------------------------------------------------
def render_header(fc: Forecast, console: Console) -> None:
    idx = fc.fundamentals.index_level
    text = Text()
    text.append(f"{fc.cfg.etf}  ", style="bold")
    text.append(f"{fc.spot:,.2f}", style="bold cyan")
    text.append(f"   close {fc.spot_date}   |   S&P 500 {num(idx, 0)}   |   ")
    text.append(f"VIX {num(fc.macro.vix, 1)}   |   3m bill {pct(fc.macro.rf_3m, 2, False)}   |   "
                f"10y {pct(fc.macro.y10, 2, False)}")
    console.print(Panel(text, title="SPXLcast", subtitle=f"data as of {fc.snap.asof:%Y-%m-%d %H:%M UTC}",
                        box=box.ROUNDED))


def render_rating(fc: Forecast, console: Console) -> None:
    r = fc.rating
    style = RATING_STYLE.get(r.label, "bold")
    body = Text()
    body.append(f"{r.label}", style=style)
    body.append(f"   conviction {r.conviction.lower()}   score {r.score:+.2f}   "
                f"horizon {horizon_label(r.horizon)} ({r.horizon} trading days)\n\n")
    for reason in r.reasons:
        body.append(f" - {reason}\n")
    console.print(Panel(body, title="Rating", box=box.ROUNDED, border_style=style.split()[-1]))


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
    t.add_row("Dividend-growth model", pct(e.dividend_growth_model), f"div yield {f.dividend_yield:.2%} + EPS growth {f.eps_growth:.2%}")
    t.add_row("Base (50/50 blend)", pct(e.base), "")
    for k, val in e.adjustments.items():
        t.add_row(f"  adj: {k}", pct(val), "")
    for n in e.notes:
        t.add_row("", "", f"[dim]{n}[/dim]")
    t.add_row("[bold]Final index drift[/bold]", f"[bold]{pct(e.final)}[/bold]",
              f"clipped to [{fc.cfg.drift_floor:+.0%}, {fc.cfg.drift_cap:+.0%}]")
    if fc.sentiment is not None:
        t.add_row("News tilt (first %d days)" % fc.cfg.sentiment_days, pct(fc.sentiment.drift_adjustment),
                  f"{fc.sentiment.label} ({fc.sentiment.score:+.2f})")
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
    t.add_row("Leverage", f"{etf.leverage:.2f}x", "stated 3x, checked against realised beta")
    t.add_row("Expense ratio", pct(etf.expense_ratio, 2, False), "")
    t.add_row("Financing cost", pct(etf.financing_rate, 2, False), f"({etf.leverage - 1:.0f}x) x (3m bill + swap spread)")
    t.add_row("Volatility decay", pct(etf.theoretical_drag(sig), 1, False),
              "L(L-1)/2 x sigma^2 at the horizon vol (emerges in the simulation)")
    t.add_row("Break-even index return", pct(etf.breakeven_index_return(sig), 1, False),
              "index total return needed for SPXL to be flat over a year")
    if etf.calibration:
        c = etf.calibration
        t.add_row("Realised beta / R2", f"{c.beta:.2f} / {c.r2:.3f}", f"{c.n} days; tracking noise {c.resid_sd_daily:.2%}/day")
    for n in v.notes + etf.notes:
        t.add_row("", "", f"[dim]{n}[/dim]")
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
    console.print("[dim]Yield and volatility rows show changes in points; price rows show % changes.[/dim]")

    t = Table(title="S&P 500 valuation and macro backdrop", box=box.SIMPLE)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_column("Source")
    src = {**f.sources, **m.sources}
    rows = [
        ("Trailing P/E", num(f.trailing_pe, 1), src.get("trailing_pe", "")),
        ("Earnings yield (E/P)", pct(f.earnings_yield, 2, False), src.get("earnings_yield", src.get("trailing_pe", ""))),
        ("Dividend yield", pct(f.dividend_yield, 2, False), src.get("dividend_yield", "")),
        ("Assumed EPS growth", pct(f.eps_growth, 2, False), src.get("eps_growth", "")),
        ("Price/Book", num(1.0 / f.book_to_price, 2) if f.book_to_price else "n/a", "Yahoo funds_data"),
        ("Price/Sales", num(1.0 / f.sales_to_price, 2) if f.sales_to_price else "n/a", "Yahoo funds_data"),
        ("3m T-bill", pct(m.rf_3m, 2, False), src.get("rf_3m", "")),
        ("2y Treasury", pct(m.y2, 2, False), src.get("y2", "")),
        ("10y Treasury", pct(m.y10, 2, False), src.get("y10", "")),
        ("30y Treasury", pct(m.y30, 2, False), src.get("y30", "")),
        ("Curve 10y-3m", pct(m.curve_10y_3m, 2), "inverted = recession signal" if m.curve_10y_3m < 0 else ""),
        ("10y breakeven inflation", pct(m.breakeven_10y, 2, False), src.get("breakeven_10y", "default used")),
        ("Real 10y yield", pct(m.real_10y, 2), src.get("real_10y", "")),
        ("Equity risk premium (E/P - real 10y)", pct(f.earnings_yield - m.real_10y, 2) if m.real_10y is not None else "n/a", ""),
        ("HY credit spread (OAS)", pct(m.hy_oas, 2, False), src.get("hy_oas", "FRED unavailable")),
        ("CPI YoY", pct(m.cpi_yoy, 2, False), src.get("cpi_yoy", "FRED unavailable")),
        ("Unemployment", pct(m.unemployment, 1, False), src.get("unemployment", "FRED unavailable")),
        ("VIX / 3M / 6M", f"{num(m.vix, 1)} / {num(m.vix3m, 1)} / {num(m.vix6m, 1)}", "Yahoo"),
        ("VVIX / SKEW", f"{num(m.vvix, 1)} / {num(m.skew, 1)}", "Yahoo"),
    ]
    for r in rows:
        t.add_row(*r)
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
            t.add_row(sym, str(row["name"])[:28], pct(row["weight"], 1, False),
                      num(tpe, 1) if tpe else "n/a", num(fpe, 1) if fpe else "n/a",
                      pct(snap.change(sym, 21)) if snap.close(sym) is not None else "n/a",
                      f"{ns:+.2f}" if ns is not None else "n/a")
        console.print(t)


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
                                      (f"score {s.score:+.2f}  from {s.n_used} recent articles "
                                       f"({s.n_articles} fetched); near-term drift tilt {s.drift_adjustment:+.1%}/yr", "")),
                        title="News sentiment", box=box.ROUNDED))
    if s.by_ticker:
        line = "  ".join(f"{k} {v:+.2f}" for k, v in sorted(s.by_ticker.items(), key=lambda kv: -abs(kv[1])))
        console.print(f"[dim]By ticker: {line}[/dim]")
    for title, items in (("Most positive", s.top_positive), ("Most negative", s.top_negative)):
        if not items:
            continue
        t = Table(title=title, box=box.SIMPLE, show_lines=False)
        t.add_column("Score", justify="right")
        t.add_column("Ticker")
        t.add_column("When")
        t.add_column("Headline")
        for x in items[:max_items]:
            t.add_row(f"{x.score:+.2f}", x.item.ticker, x.item.published.strftime("%m-%d %H:%M"), x.item.title[:110])
        console.print(t)


def render_notes(fc: Forecast, console: Console) -> None:
    for n in fc.snap.notes:
        console.print(f"[yellow]note:[/yellow] {n}")
    console.print("[dim]Not investment advice. Model output from public data and stated assumptions; "
                  "a 3x leveraged fund can lose most of its value in a sustained decline.[/dim]")


def render_all(fc: Forecast, console: Console, prices: Optional[List[float]] = None,
               sections: Optional[Iterable[str]] = None) -> None:
    sections = set(sections) if sections else {"header", "rating", "forecast", "price", "ladder", "drivers",
                                                "metrics", "sensitivity", "news", "notes"}
    if "header" in sections:
        render_header(fc, console)
    if "rating" in sections:
        render_rating(fc, console)
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
