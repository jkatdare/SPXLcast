spxlcast.com

Fundamentals-driven price forecast for **SPXL** (Direxion Daily S&P 500 Bull 3X), with a plain
**assessment** of what holding it costs and risks right now (leverage cost, drawdown risk, the 3-month
price range) and a **percentile lookup** for any price level. It does not tell you to buy or sell.

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
# full report: assessment, distribution, drivers, influencer metrics, news
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
rebuild the download cache; the price-history archive is kept), and overrides `--index-drift 0.08`,
`--vol 0.18`, `--pe 22`, `--div-yield 0.013`, `--eps-growth 0.055`, `--swap-spread 0.0075`,
`--skew 1.0`, and `--archive DIR` (store the run's inputs, simulator arguments and scored headlines;
see `spxlcast/archive.py`). Prices must be positive and every number finite, `--seed` is a
non-negative integer, and `--index-drift` must be above -50%, because each simulated path draws its
drift up to 50 points either side of it.

## How to read the output

**Assessment.** The report (and the status page) opens with three statements the model can back up.
Each level is placed against the 440 month-ends from 1990 to 2026 of the full-model backtest below:
low, normal and high (or elevated) are the bottom, middle and top thirds of those months. The cutoffs
and the history live in `spxlcast/reference.json`; without that file the levels read n/a.

* *Leverage cost* - the S&P 500 total return per year, on average and dividends included, that SPXL
  needs to break even over the long run: the "hurdle". It comes from what SPXL gives up each year
  compared with 3 times the S&P 500's return: borrowing (it borrows twice its value at the 3-month
  bill plus 0.75%), fees (the expense ratio) and volatility drag (SPXL resets to 3x every day, which
  loses money when the market zigzags; the bigger the swings, the bigger the loss). The S&P 500 has to
  make up about a third of that, plus a little for its own swings. Low is below 5.6%, high is 7.6% and
  above. Over a single year SPXL's typical outcome can be ahead at a lower S&P 500 return, because
  volatility comes in bursts; the hurdle is where it goes nowhere over the long run. The model's
  long-run S&P 500 estimate is shown next to it for comparison: it is an average over many years, not
  a forecast for the coming months, and since 1990 it has run below what the S&P 500 actually returned
  (see the drift backtest below).
* *Drawdown risk* - the chance SPXL closes at least 20% below today's price on some day in the next
  3 months (63 trading days), even if it recovers afterwards. Low is below 16.9%, elevated is 32.1%
  and above. The report adds how often such a fall actually followed in the backtest's months at the
  same level: 8.8% of the time at low (11.7% predicted), 22.4% at normal (24.5%) and 35.4% at
  elevated (42.5%). Before 2009 those months use a 3x fund rebuilt from the S&P 500; on SPXL alone
  the elevated level ran well high (see the backtest below).
* *3-month price range* - SPXL ends inside it in 90% of the simulations (5% below, 5% above); the
  typical (middle) price is shown too. In the backtest, 3-month ranges held 93% of outcomes.

None of this is a buy or sell signal, and the tool gives none. High cost or risk has not meant lower
returns: in the backtest, months at high leverage cost or elevated drawdown risk were not followed by
lower returns on average, though elevated risk did bring more 20% falls. The tool used to give a
BUY / HOLD / SELL rating, but the same test found it did not pick better times to own SPXL: over the
next 6 months, months rated BUY returned 4.2 points less than the others on average (90% interval
-18.0 to +9.6 points). The forecast ranges held up in the same test: its 90% ranges held 91-93% of
outcomes from one week to six months, though the chance of a 20% fall ran 4-6 points high at three
to six months. So the output now describes cost and risk, which the backtest supports better (the
ranges held, the fall chances sort months by risk but run high, and the cost estimate matched SPXL's
on average), instead of giving a verdict. The rating is still computed and logged for research
(`rating`, `score`, `score_se` and `conviction` in the track record, `rating` in `forecast.json` and
`spot_log.csv`; see Model, item 5), but it is not shown.

**Price percentile.** For a price P and each horizon:

* *Percentile of end price* - share of simulated end-of-horizon prices at or below P. A 23rd
  percentile means only 23% of outcomes end that low.
* *P(dips to level)* - chance the price trades at or below P at some point before the horizon,
  i.e. the chance a buy-limit order at P fills.
* *P(rises to level)* - the same for a sell-limit above spot.

The **buy-limit ladder** inverts this: it lists the prices with a 90/75/50/25/10% chance of being
touched within 6 months (`--rating-horizon`), with the median end price and the chance of profit
computed only over the paths on which the order fills. At short horizons many paths never trade
below the spot, so when no price below it has a 90% (or 75%) chance of filling, those rungs are
replaced by one at-market rung at the spot, shown as 100%.

## Model

1. **S&P 500 expected total return** (annualised, arithmetic): 50/50 blend of
   * earnings-yield model: `E/P + expected inflation` (10y breakeven, or 2.5% default)
   * dividend-growth model: `dividend yield + nominal EPS growth`, with nominal growth = 3% real
     + expected inflation

   plus small **regime penalties** for an inverted 10y-3m curve, high-yield spreads above 5%, CPI
   above 3.5% and a Sahm-rule labour signal, capped at 1.5% in total. Interest rates are charged to
   SPXL once, through its financing cost, not through the drift. CPI inflation and the Sahm gap are
   computed on calendar months, not rows: a single missing month (BLS published no October 2025 CPI
   or unemployment figures) is filled with the mean of its neighbours, and when the month a year
   earlier is still missing, CPI inflation is reported unavailable and its penalty skipped. An
   optional countercyclical valuation term (`valuation_sensitivity` in config) is off by default:
   over 1881-2026 it added no predictive value to the blend (see the drift backtest below). The
   result is clipped to [-10%, +20%]. Each simulated path draws its own drift from a normal with a
   2% standard deviation around this estimate, so the bands honestly reflect that the expected
   return is uncertain.
2. **Volatility**: VIX / VIX3M / VIX6M less haircuts of 3 / 5 / 7 vol points (implied vol exceeds
   subsequently realised vol by more at longer tenors), converted into forward variances so the
   total variance to each pillar matches; beyond six months the vol mean-reverts toward 16%. Just
   after the open Yahoo has today's VIX but not yet today's VIX3M/VIX6M: those are then moved from
   their last close with the VIX, by the beta of their daily log changes on the VIX's over the past
   year, so all three pillars are of one instant. On top of that term structure each path carries
   **stochastic volatility**: a persistent log-vol deviation (AR(1), persistence 0.97, stationary
   sd 0.35) with a leverage effect (correlation -0.5 between today's return shock and tomorrow's
   vol), scaled so the average variance still matches the term structure. This gives volatility
   clustering and realistic drawdown-touch probabilities.
3. **News sentiment**: stories within 7 days, deduplicated across feeds, weighted by recency
   (2-day half-life) and by relevance judged from the headline: a market-wide story counts fully
   on any feed, a story about one of the ten largest constituents counts in proportion to that
   company's index weight (six times the weight, capped at 1), and a story about neither is scaled
   by a further 0.25. VADER is extended with a finance vocabulary and clause-aware phrase rules that
   know "rate cuts" are good and "yields jump" is bad for stocks. The score in [-1, 1] shifts the
   drift of the first 10 trading days by up to +/-5% annualised in the displayed price tables and the
   assessment (at the extreme score that moves the 3-month chance of a 20% dip by about one point).
   The retired rating is computed on a second simulation without the tilt, so an uncalibrated news
   score can never flip its label.
4. **Simulation**: 50,000 paths of daily S&P 500 total returns with skewed Student-t shocks
   (4 degrees of freedom, skew 0.9 for larger down moves than up moves, capped at +/-20% a day)
   under the stochastic-vol process above.
   Each day SPXL earns `3 x index return - (expense + 2 x (3m bill + 0.75% spread)) / 252 + tracking
   noise`; volatility decay is not assumed, it emerges from compounding. Leverage and tracking noise
   are checked against a two-year robust regression of SPXL on SPY that excludes close-versus-NAV
   dislocation days.
5. **Assessment and percentiles** are read directly off the simulated paths (terminal prices, running
   minimum and maximum), except the leverage-cost hurdle, which is the closed-form break-even
   `ETFParams.breakeven_index_return` at the 1-year vol: the fund's average log growth is zero when
   `ln(1 + index return) = (fees + financing) / 3 + 3/2 x vol^2`. At a constant vol its median is then
   flat too; with the simulation's bursts of volatility the 1-year median sits a few points higher and
   approaches flat only over longer holds. The levels compare today's hurdle and 3-month dip chance
   with the backtest's month-ends (`spxlcast/assess.py`, `spxlcast/reference.json`).
   The BUY / HOLD / SELL rating (`spxlcast/rating.py`) is still computed and logged but no longer
   shown: the average of the 6-month median return over T-bills divided by 15% and a Sharpe-like
   ratio divided by 0.5, BUY at +0.30 or more with a positive median edge, SELL at -0.30 or less. It
   had no timing value in the backtest: the rank correlation of its score with the next 6 months'
   SPXL return over T-bills was -0.03 (90% interval -0.22 to +0.17).

Everything is configurable in `spxlcast/config.py`.

## Track record

Every run can append its forecast to a CSV, and the same tool later grades those forecasts against
what actually happened. This is the only test of the fundamentals-based drift that does not depend
on historical assumptions.

```bash
python -m spxlcast log            # quiet forecast, appended to logs/forecast_log.csv
python -m spxlcast score          # PIT, band coverage, drawdown-touch hit rates, drawdown risk by
                                  # level, and whether the news score predicted the next two weeks
```

Each row records the assessment the run showed (`hurdle`, `leverage_cost`, `drawdown_risk`) next to
the retired rating's columns. The scorer's "Drawdown risk check" table checks the page's main
risk statement: for each level, the predicted chance of a 20% dip within 3 months (63 sessions)
against how often SPXL actually closed 20% or more below the logged price within them, with the
backtest's figures at that level beside it. Rows logged before the column existed are placed by
their logged 3-month dip chance (`h63_p_dd20`) against the cutoffs in `spxlcast/reference.json`.

The scorer is honest about overlap: forecasts a day apart share most of their outcome window, so
each horizon reports how many independent outcomes the rows amount to (a year of daily 1-month
forecasts holds about 12) and 90% intervals computed in closed form on that number of independent
outcomes (a t interval for mean PIT and for CRPS skill, a Wilson interval for band coverage and for
the dip frequency by drawdown-risk level) once there are at least three; in simulations of perfectly
calibrated forecasts the mean-PIT and skill intervals cover about 82-90% at three to seven
independent outcomes and 88-93% from about a dozen on, and the band-coverage interval is conservative
(94-98%).
It also grades the whole distribution with CRPS against a naive lognormal at the raw VIX with a
T-bill drift, and with `--archive DIR` it scores each run on its archived 103-point percentile
grid rather than the nine logged quantiles.

Any subcommand accepts `--log-file PATH` to append its run as well. The scorer keeps one row per
spot date: the run made on that New York day (a close beats an intraday quote, then the latest run).
A later run that logs the same close (the winter 08:40 ET pre-open run, or a holiday run) counts only
when nothing ran on the day itself. Each row is scored on the ratio of later closes to the close on
its own date, so distributions and splits after the window (Yahoo back-adjusts its history) do not
move a closed score. While the session is open, today's partial bar is not treated as a close. Until
a forecast reaches its horizon, the score says which one resolves first. The 1-week and 2-week
horizons exist for fast feedback: daily forecasts overlap, so a year of logging holds about 50
independent 1-week outcomes but only 12 1-month ones, and the news tilt only acts over the first 10
days. A row cut short by an interrupted write is moved to `logs/forecast_log.csv.partial`, and a
rewrite that adds columns goes through a temporary file, so the log is never truncated.

Each row also records `model_version` (`MODEL_VERSION` in `config.py`, bumped whenever a change
alters the numbers a run produces), `build` (the git commit of the code), `data_flags` (input
problems such as `fred:none` or `vix6m:missing`; empty when clean), `fred_series` and
`news_fetched`. `score --model-version X` scores one version only; by default versions are pooled
with a note when there is more than one. Besides missing inputs, `data_flags` marks stale ones:
`fred:stale` (a daily FRED series more than five sessions behind, or a monthly one older than 100
days), `spot:stale` (SPXL's last bar two or more sessions behind the index calendar), and
`vix:stale`, `vix3m:stale` and `vix6m:stale` (that index is from an earlier session than the spot or
another of the three, as on the 09:40 ET run, before Yahoo has today's VIX3M and VIX6M; those two
are then moved with the VIX, see Model). `etf:uncalibrated` means there was no usable leverage
calibration: a fit with R2 below `calibration_min_r2` (0.99) points to bad or mixed-basis prices and
is rejected, so the run uses the stated 3x with no tracking noise and its notes say "calibration
rejected".

## Hosting the daily run on Azure

`infra/deploy.ps1` creates a scheduled Container Apps Job that runs the forecast hourly through
the US session plus once after the close (13:40 to 21:40 UTC, weekdays), appends to the track
record on an Azure Files share and rescores it, plus an always-on web app that serves a status
page (latest report, fan chart, score, CSV/JSON downloads). Intraday runs give a live assessment and
price percentiles; the after-close run is the one the track record keeps. Between full runs the
web app re-prices the latest forecast at the live SPXL quote every minute of the session (the
simulated distribution is one of returns, so prices scale with the quote and the fixed price
checks are re-read off stored percentile grids), shows that at the top of the page with the latest
run's `leverage cost <level> · drawdown risk <level>` (both are about returns, so they change only
with a full run; the 3-month range scales with the quote like every other price) and records
the quote in `logs/spot_log.csv`. That file's `base_run_at` names the full run in the track record's
`run_at` form (`2026-09-22T21:40:21Z`); rows written before this form carry
`2026-09-22T21:40:21.218019+00:00`, which must be floored to the second (never rounded) to join.
Outside the session the loop fetches nothing: it marks the block closed after the bell and restates
it at each new full run's own price. The job and the web app are defined in `infra/job.yaml` and
`infra/web.yaml`; the image is built in the cloud by Azure Container Registry, so no local Docker
is needed. From the repo root after `az login`:

```powershell
.\infra\deploy.ps1 -DryRun                 # prints every command it would run, changes nothing
.\infra\deploy.ps1                          # eastus2, resource group spxlcast-rg
```

The FRED key is read from `.env` or the `FRED_API_KEY` environment variable and stored as a
Container Apps secret. Approximate cost: the registry (Basic, about $5 a month) and the always-on
web app (about $14 a month) are the fixed charges; the hourly job costs under $1 a month. `az group delete -n spxlcast-rg`
removes everything. `python -m spxlcast serve --root DIR [--live] [--interval SECONDS]` runs the
same status page (and, with `--live`, the minute loop; `--interval` sets the seconds between quotes,
5 to 3600, default 60) locally.
To run the steps by hand instead of through the script, follow `infra/DEPLOY.md`.

Each job run also archives its inputs, exact simulator arguments and newly seen headlines on the
share (`archive/`, not served by the page), and a scheduled GitHub workflow
(`.github/workflows/healthcheck.yml`) checks the site four times a weekday, opening an issue when a
scheduled run never logged, a run did not finish, the live loop stopped writing during the session
or before the close, the live quote has not moved for 30 minutes of the session (a feed answering
with an old price), the score step failed or the page is down.
See `infra/DEPLOY.md` sections 10 and Archive.

## Data notes

* FRED is read through its official API when `FRED_API_KEY` is set (free key from
  fred.stlouisfed.org, kept in a git-ignored `.env`); the key-less chart endpoint is the fallback.
  Yahoo yields (`^IRX`, `^TNX`, ...) are used when neither works; `^IRX` is a bank-discount rate and
  is converted to a bond-equivalent yield. Breakeven inflation then falls back to the configured
  default and the credit / CPI / unemployment adjustments are skipped.
* When the session is open the spot, VIX and yields are live quotes; the header says `intraday`
  instead of `close` and the price cache expires at the bell. The session follows the NYSE calendar
  in `spxlcast/config.py` (`nyse_session`): holidays are closed all day, and early-close days (3 July
  or 24 December from Monday to Thursday, and the day after Thanksgiving) close at 13:00, so the live
  loop does not tick then and a job run after 13:00 ET on such a day logs the close.
* Results are cached in `.cache/` (prices 6h, fund info 12h, news 30 min). Failed or empty fetches
  are never cached. `--refresh` bypasses the download cache and rebuilds it; the per-ticker archive
  is kept (delete `.cache/archive_*.pkl` to rebuild it). Price history accumulates in that archive, because Yahoo intermittently returns only the
  latest bar for `^VIX3M` and `^VIX6M`; a partial response therefore never erases history already
  seen, and the metrics table prints n/a for a window that falls into a gap rather than a misleading
  number. The latest download wins on every date it covers: Yahoo re-adjusts its whole history at
  each dividend or split, and older archived rows are rescaled to match. Today's bar is archived only
  after the close, so an intraday price never stands in for a past session's close. Non-finite
  closes, and zero or negative closes of stocks and funds, are dropped.
* A 3x fund can lose most of its value in a sustained decline; the model's own 5% tail shows how
  large that risk is.

## Tests and backtests

```bash
python -m pytest
python scripts/backtest_calibration.py --drift 0.07     # the engine: bands and touch probabilities
python scripts/backtest_drift.py                        # the drift model: 145 years of Shiller data
python scripts/backtest_rating.py                       # the full model, the rating and the assessment, 1990-2026
python scripts/backtest_rating.py --write-reference     # ... and copy its levels to spxlcast/reference.json
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
with Shiller's data (now published at shillerdata.com; the script finds the current file there and
warns if it has to fall back to the Yale copy, which stopped updating in 2023) and compares it with
the realised nominal total return over the next 1, 5 and 10 years. Over 1881-2026 the model's drift
carries real information (correlation 0.22 / 0.47 / 0.60 with realised returns at 1 / 5 / 10
years) and beats a constant 7% on error at every horizon, with a bias of -1.6% at one year and
0.2% or less at five and ten. The optional valuation term, tried at the 0.5 sensitivity it was
built with, added nothing (correlation 0.22 / 0.45 / 0.60, and a larger error at five and ten
years) and pulled the post-1990 forecasts further below a market that kept re-rating upward, which
is why it is off by default. Since 1990 the model has run low (bias -4.8% a year at one year,
-3.1% at five, -1.6% at ten): over the engine backtest's window, 2016-09 to 2026-06, it would have
said about 6.6% a year while the S&P 500 returned 15.4%, which explains the upward bias in the
engine backtest.

**Full model, rating and assessment.** The third script runs the live model's own functions at
every month-end from 1990 to 2026 (440 of them) on data available at the time: Shiller earnings and
dividends of the last quarter already reported (two months after it ends; Shiller interpolates the months in
between, so reading one of those would leak part of an unreported quarter), the 10-year breakeven
from 2003 (trailing CPI inflation before), FRED rates, CPI and unemployment as released (CPI
inflation and the Sahm gap by calendar month, through the live model's own functions), and the
VIX curve (VIX3M/VIX6M imputed from the VIX before 2008). One regime input is missing: FRED keeps
only three years of the ICE BofA high-yield spread, so the credit-stress penalty cannot fire at
404 of the 440 month-ends (before 2023-09), including every credit crisis in the sample; the report
says so. Outcomes are SPXL from 2009 and, before that, a synthetic 3x fund that tracks SPXL with
0.998 daily correlation and a 0.3%/yr gap where both exist. The 90% intervals are computed in
closed form on the number of independent outcome windows (the 434 six-month outcomes amount to
about 73 independent ones), the same construction as the track record's. Findings:

* The forecast distribution is well calibrated and beats a naive benchmark. The 90% band held
  91-93% of outcomes from one week to six months, and the model's CRPS is 3% (1 week) to 15%
  (6 months) better than a lognormal at the raw VIX with a T-bill drift, with 90% intervals above
  zero at every horizon. The median is slightly low (mean PIT 0.54-0.57) and the chance of a 20%
  dip is overstated by 4-6 points at 3-6 months.
* The fundamentals drift adds nothing measurable at these horizons: against the same engine with
  a constant 7% drift, 6-month skill is +0.4% (90% interval -0.9% to +1.7%). Valuation predicts
  5-10 year returns (above), not the next six months.
* The rating does not predict returns. Rank correlation of the score with the next 6 months'
  SPXL excess return is -0.03 (-0.22 to +0.17); months rated BUY did no better than the rest
  (-4.2%, -18.0% to +9.6%). The label mostly tracks financing cost and volatility: BUY in 83% of
  months in the low-rate years 2008-2021, HOLD in 95% since 2022. SELL fired in 12 months
  (August-September 1998, October-December 2000, October 2008 to March 2009, March 2020), mostly
  at volatility spikes near market bottoms. SPXL averaged +39% over the following six months but
  beat T-bills in only half of them; four episodes are far too few to tell (+26% more than the
  other months, 90% interval -53% to +105%). No BUY/SELL threshold worked in both halves of the
  sample. So the rating was retired from view (it is still computed and logged) and replaced by the
  assessment below.
* The assessment's levels are the terciles of these month-ends, written to
  `output/backtest/reference.json` (`--write-reference` also copies it to `spxlcast/reference.json`,
  where the report and page read it). Leverage cost: low below 5.6%, normal 5.6% to 7.6%, high 7.6%
  and above (median 6.9%, range 3.3% to 17.3%). Drawdown risk: low below 16.9%, normal 16.9% to
  32.1%, elevated 32.1% and above (median 24.2%, range 6.1% to 69.9%).
* The drawdown-risk levels sort risk over the whole sample, less clearly on SPXL itself. A 20% dip
  within 3 months followed 8.8% of the time at low (11.7% predicted; 90% interval 5% to 16%; 147
  months, about 66 independent), 22.4% at normal (24.5% predicted; 16% to 31%; 143 months, 82) and
  35.4% at elevated (42.5% predicted; 26% to 46%; 147 months, 64). On SPXL alone (2009 on) a dip
  followed 14% of the time at low, 25% at normal and 27% at elevated, where 44% was predicted (90%
  interval 16% to 43%; 59 months, about 28 independent); before 2009 (the simulated fund) 4%, 19% and
  41%. Overall the prediction runs high (26% predicted, 22% happened), with the largest gap at the
  elevated level since 2009, which is why the page quotes what actually happened at each level.
* The leverage cost matches SPXL's realised cost on average. On SPXL itself (2009-01 to 2026-02,
  206 month-ends, about 35 independent six-month windows) the predicted yearly cost plus volatility
  drag (fees + financing + L(L-1)/2 x vol^2 at the 1-year vol) averaged 13.1% against 14.4% realised
  (3 x the S&P 500's log return minus SPXL's, per year, over the next 6 months): a difference of
  +1.3% (90% interval -2.0% to +4.5%), correlation +0.26 (-0.03 to +0.51). The level is not a
  timing signal either: 6-month returns over T-bills in high-cost months differed from the rest by
  +9.4% (-5.6% to +24.4%), in low-cost months by -4.1% (-16.2% to +8.0%).

### Signal research

Since the rating had no timing value, a separate study searched for a real timing signal for SPXL, with
no technical analysis: twelve valuation, macro, rate, credit, inflation, jobless-claims, bank-lending and
volatility indicators, plus an equal-weight blend of nine of them. Everything was fixed and hashed in
advance (`research/PREREGISTRATION.md`), set on 1990-2007 and tested once on 2008-2026. **No timing
signal was found**: none of the 13 passed the statistical test after correcting for trying 13, and the
blend's switching rule lost to buy-and-hold SPXL (11.8% against 17.9% a year). So the page gives no
timing indicator. The plain-language summary, every registered table, the exploratory checks made for
three independent reviews, and the limitations are in [research/RESULTS.md](research/RESULTS.md).

```bash
python scripts/research_signals.py report   # regenerates research/test_results.md and research/RESULTS.md
```
