spxlcast.com

Fundamentals-driven price forecast for **SPXL** (Direxion Daily S&P 500 Bull 3X), with a
**Buy / Hold / Sell** rating and a **percentile lookup** for any price level.

The forecast is a Monte Carlo distribution of SPXL prices built from:

* **Valuation of the S&P 500** - earnings yield, dividend yield, long-run earnings growth
* **Macro conditions** - short and long Treasury yields, yield-curve slope, credit spreads,
  inflation and labour data (FRED, when reachable)
* **Option-implied volatility** - the VIX, VIX3M and VIX6M term structure, less a tenor-dependent
  variance-risk-premium haircut that was backtested against realised outcomes
* **Leveraged-ETF mechanics** - 3x daily rebalancing, expense ratio, swap financing at the short
  rate plus an all-in spread on the borrowed 2x notional, calibrated against SPXL's own history
* **News sentiment** - Yahoo Finance headlines for market-wide feeds and the ten largest
  constituents, scored with VADER plus a finance lexicon and subject-aware phrase rules

All market data comes from Yahoo Finance through `yfinance`; FRED is optional.

## Install

Python 3.10+.

```bash
pip install -r requirements.txt
```

## Use

```bash
# full report: rating, distribution, drivers, influencer metrics, news
python -m spxlcast

# where do 190 and 200 sit in the distribution? (percentile + chance of a limit order filling)
python -m spxlcast price 190 200

# only the influencer metrics / only the news / only the ETF calibration
python -m spxlcast metrics
python -m spxlcast news
python -m spxlcast calibrate

# track record: append today's forecast, then (weeks later) grade the logged forecasts
python -m spxlcast log
python -m spxlcast score

# machine-readable output and a fan chart
python -m spxlcast forecast --price 200 --json output/forecast.json --plot output/fan.png

# one-line answer
python -m spxlcast --quiet --price 200
```

Useful options (all subcommands): `--horizons 5 10 21 63 126 252`, `--rating-horizon 126`,
`--paths 50000`, `--seed 42`, `--no-news`, `--no-fred`, `--no-macro-adj`, `--refresh` (bypass and
rebuild the cache), and overrides `--index-drift 0.08`, `--vol 0.18`, `--pe 22`, `--div-yield 0.013`,
`--eps-growth 0.055`, `--swap-spread 0.0075`, `--skew 1.0`, and `--archive DIR` (store the run's
inputs, simulator arguments and scored headlines; see `spxlcast/archive.py`).

## How to read the output

**Rating.** Judged at the rating horizon (default 6 months = 126 trading days) from two views of
the simulated distribution, each scaled to [-1, 1] and averaged into a score:

* *typical outcome* - annualised median SPXL return minus the T-bill return, divided by 15%
* *expected value* - annualised mean excess return per unit of volatility (a Sharpe-like ratio),
  divided by 0.5

Score >= +0.30 with a positive median edge is **BUY**, score <= -0.30 is **SELL**, otherwise
**HOLD**. Conviction is High / Medium / Low by |score|. The score is printed with its Monte Carlo
standard error and the rating is flagged as borderline when a threshold lies within one standard
error. Every driver of the distribution is listed under the rating so the verdict is auditable.
For a 3x fund the median is dragged down by volatility decay while the mean is not, which is why
both views are used.

**Price percentile.** For a price P and each horizon:

* *Percentile of end price* - share of simulated end-of-horizon prices at or below P. A 23rd
  percentile means only 23% of outcomes end that low.
* *P(dips to level)* - chance the price trades at or below P at some point before the horizon,
  i.e. the chance a buy-limit order at P fills.
* *P(rises to level)* - the same for a sell-limit above spot.

The **buy-limit ladder** inverts this: it lists the prices with a 90/75/50/25/10% chance of being
touched within the rating horizon, with the median end price and the chance of profit computed only
over the paths on which the order fills.

## Model

1. **S&P 500 expected total return** (annualised, arithmetic): 50/50 blend of
   * earnings-yield model: `E/P + expected inflation` (10y breakeven, or 2.5% default)
   * dividend-growth model: `dividend yield + nominal EPS growth`, with nominal growth = 3% real
     + expected inflation

   plus small **regime penalties** for an inverted 10y-3m curve, high-yield spreads above 5%, CPI
   above 3.5% and a Sahm-rule labour signal, capped at 1.5% in total. Interest rates are charged to
   SPXL once, through its financing cost, not through the drift. An optional countercyclical
   valuation term (`valuation_sensitivity` in config) is off by default: over 1881-2023 it added no
   predictive value to the blend (see the drift backtest below). The result is clipped to
   [-10%, +20%]. Each simulated path draws its own drift from a normal with a 2% standard deviation
   around this estimate, so the bands honestly reflect that the expected return is uncertain.
2. **Volatility**: VIX / VIX3M / VIX6M less haircuts of 3 / 5 / 7 vol points (implied vol exceeds
   subsequently realised vol by more at longer tenors), converted into forward variances so the
   total variance to each pillar matches; beyond six months the vol mean-reverts toward 16%. On top
   of that term structure each path carries **stochastic volatility**: a persistent log-vol
   deviation (AR(1), persistence 0.97, stationary sd 0.35) with a leverage effect (correlation -0.5
   between today's return shock and tomorrow's vol), scaled so the average variance still matches
   the term structure. This gives volatility clustering and realistic drawdown-touch probabilities.
3. **News sentiment**: stories within 7 days, deduplicated across feeds, weighted by recency
   (2-day half-life) and by relevance judged from the headline: a market-wide story counts fully
   on any feed, a story about one of the ten largest constituents counts in proportion to that
   company's index weight (six times the weight, capped at 1), and a story about neither is scaled
   by a further 0.25. VADER is extended with a finance vocabulary and clause-aware phrase rules that
   know "rate cuts" are good and "yields jump" is bad for stocks. The score in [-1, 1] shifts the
   drift of the first 10 trading days by up to +/-5% annualised in the displayed price tables. The
   rating is computed on a second simulation without the tilt, so an uncalibrated news score can
   never flip a label.
4. **Simulation**: 50,000 paths of daily S&P 500 total returns with skewed Student-t shocks
   (4 degrees of freedom, skew 0.9 for larger down moves than up moves, capped at +/-20% a day)
   under the stochastic-vol process above.
   Each day SPXL earns `3 x index return - (expense + 2 x (3m bill + 0.75% spread)) / 252 + tracking
   noise`; volatility decay is not assumed, it emerges from compounding. Leverage and tracking noise
   are checked against a two-year robust regression of SPXL on SPY that excludes close-versus-NAV
   dislocation days.
5. **Rating and percentiles** are read directly off the simulated paths (terminal prices, running
   minimum and maximum).

Everything is configurable in `spxlcast/config.py`.

## Track record

Every run can append its forecast to a CSV, and the same tool later grades those forecasts against
what actually happened. This is the only test of the fundamentals-based drift that does not depend
on historical assumptions.

```bash
python -m spxlcast log            # quiet forecast, appended to logs/forecast_log.csv
python -m spxlcast score          # PIT, band coverage, drawdown-touch hit rates, returns by rating,
                                  # and whether the news score predicted the next two weeks
```

Any subcommand accepts `--log-file PATH` to append its run as well. The scorer keeps one row per
spot date (a close beats an intraday quote) and needs the shortest horizon to elapse before it has
anything to report. The 1-week and 2-week horizons exist for fast feedback: daily forecasts
overlap, so a year of logging holds about 50 independent 1-week outcomes but only 12 1-month ones,
and the news tilt only acts over the first 10 days.

Each row also records `model_version` (`MODEL_VERSION` in `config.py`, bumped whenever a change
alters the numbers a run produces), `build` (the git commit of the code), `data_flags` (input
problems such as `fred:none` or `vix6m:missing`; empty when clean), `fred_series` and
`news_fetched`. `score --model-version X` scores one version only; by default versions are pooled
with a note when there is more than one.

## Hosting the daily run on Azure

`infra/deploy.ps1` creates a scheduled Container Apps Job that runs the forecast hourly through
the US session plus once after the close (13:40 to 21:40 UTC, weekdays), appends to the track
record on an Azure Files share and rescores it, plus an always-on web app that serves a status
page (latest report, fan chart, score, CSV/JSON downloads). Intraday runs give a live rating and
price percentiles; the after-close run is the one the track record keeps. Between full runs the
web app re-prices the latest forecast at the live SPXL quote every minute of the session (the
simulated distribution is one of returns, so prices scale with the quote and the fixed price
checks are re-read off stored percentile grids), shows that at the top of the page and records
the quote in `logs/spot_log.csv`. The job and the web app are defined in `infra/job.yaml` and
`infra/web.yaml`; the image is built in the cloud by Azure Container Registry, so no local Docker
is needed. From the repo root after `az login`:

```powershell
.\infra\deploy.ps1 -DryRun                 # prints every command it would run, changes nothing
.\infra\deploy.ps1                          # eastus2, resource group spxlcast-rg
```

The FRED key is read from `.env` or the `FRED_API_KEY` environment variable and stored as a
Container Apps secret. Approximate cost: the registry (Basic, about $5 a month) and the always-on
web app (about $14 a month) are the fixed charges; the hourly job costs under $1 a month. `az group delete -n spxlcast-rg`
removes everything. `python -m spxlcast serve --root DIR [--live]` runs the same status page
(and, with `--live`, the minute loop) locally.
To run the steps by hand instead of through the script, follow `infra/DEPLOY.md`.

Each job run also archives its inputs, exact simulator arguments and newly seen headlines on the
share (`archive/`, not served by the page), and a scheduled GitHub workflow
(`.github/workflows/healthcheck.yml`) checks the site four times a weekday, opening an issue when a
scheduled run never logged, a run did not finish, the live price goes stale or the page is down.
See `infra/DEPLOY.md` sections 10 and Archive.

## Data notes

* FRED is read through its official API when `FRED_API_KEY` is set (free key from
  fred.stlouisfed.org, kept in a git-ignored `.env`); the key-less chart endpoint is the fallback.
  Yahoo yields (`^IRX`, `^TNX`, ...) are used when neither works; `^IRX` is a bank-discount rate and
  is converted to a bond-equivalent yield. Breakeven inflation then falls back to the configured
  default and the credit / CPI / unemployment adjustments are skipped.
* When the session is open the spot, VIX and yields are live quotes; the header says `intraday`
  instead of `close` and the price cache expires at the bell.
* Results are cached in `.cache/` (prices 6h, fund info 12h, news 1h). Failed or empty fetches are
  never cached. `--refresh` bypasses the cache and rebuilds it. Price history also accumulates in a
  per-ticker archive, because Yahoo intermittently returns only the latest bar for `^VIX3M` and
  `^VIX6M`; a partial response therefore never erases history already seen, and the metrics table
  prints n/a for a window that falls into a gap rather than a misleading number.
* A 3x fund can lose most of its value in a sustained decline; the model's own 5% tail shows how
  large that risk is.

## Tests and backtests

```bash
python -m pytest
python scripts/backtest_calibration.py --drift 0.07     # the engine: bands and touch probabilities
python scripts/backtest_drift.py                        # the drift model: 150 years of Shiller data
```

**Engine calibration.** The first script builds the model's inputs at every month-end since 2016
from data available then, simulates with the project's own engine, and scores the realised outcome
over the next 1, 3 and 6 months. With the current settings (10 years, 114 origins, constant 7%
drift) the share of realised SPXL outcomes inside the 5-95% band is 91% / 92% / 89% at 1 / 3 / 6
months against a 90% target, and the predicted chances of touching -10% / -20% before the horizon
match realised frequencies closely (47% vs 44% and 25% vs 25% at three months, 58% vs 53% and 36%
vs 35% at six). The first version of the model (flat 2-point haircut, symmetric shocks, no
stochastic vol) covered 99% at six months and predicted a 54% chance of a 20% drawdown against the
35% realised. Outcomes still sit high in the distribution on average (mean percentile about 0.61
at 6 months) for SPY and SPXL alike, because the index returned about 15% a year over the sample
against the 7% assumed; that is the sample, not the fund mechanics.

**Drift model.** The second script rebuilds the expected-return model month by month from 1881
with Shiller's data and compares it with the realised nominal total return over the next 1, 5 and
10 years. The model's drift carries real information (correlation 0.23 / 0.48 / 0.62 with realised
returns at 1 / 5 / 10 years) and beats a constant 7% on error at every horizon with a bias under
1%. The valuation term added nothing to this and biased the post-1990 era low, which is why it is
off by default. Over 2016-2023 the model would have said about 6% a year while the market delivered
12-15%, which explains the upward bias in the engine backtest.
