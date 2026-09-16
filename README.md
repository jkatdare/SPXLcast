# SPXLcast

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

# machine-readable output and a fan chart
python -m spxlcast forecast --price 200 --json output/forecast.json --plot output/fan.png

# one-line answer
python -m spxlcast --quiet --price 200
```

Useful options (all subcommands): `--horizons 21 63 126 252`, `--rating-horizon 126`,
`--paths 50000`, `--seed 42`, `--no-news`, `--no-fred`, `--no-macro-adj`, `--refresh` (bypass and
rebuild the cache), and overrides `--index-drift 0.08`, `--vol 0.18`, `--pe 22`, `--div-yield 0.013`,
`--eps-growth 0.055`, `--swap-spread 0.0075`, `--skew 1.0`.

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

   plus a countercyclical **valuation term**, `0.5 x (E/P - 5%)` capped at +/-1.5% (P/E 20 is
   neutral), and small **regime penalties** for an inverted 10y-3m curve, high-yield spreads above
   5%, CPI above 3.5% and a Sahm-rule labour signal, capped at 1.5% in total. The valuation term is
   deliberately absolute rather than relative to bond yields, so interest rates are charged to
   SPXL only once, through its financing cost. The result is clipped to [-10%, +20%].
2. **Volatility**: VIX / VIX3M / VIX6M less haircuts of 2 / 4 / 6 vol points (implied vol exceeds
   subsequently realised vol by more at longer tenors), converted into forward variances so the
   total variance to each pillar matches; beyond six months the vol mean-reverts toward 16%.
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
   (4 degrees of freedom, skew 0.9 for larger down moves than up moves, capped at +/-20% a day).
   Each day SPXL earns `3 x index return - (expense + 2 x (3m bill + 0.75% spread)) / 252 + tracking
   noise`; volatility decay is not assumed, it emerges from compounding. Leverage and tracking noise
   are checked against a two-year robust regression of SPXL on SPY that excludes close-versus-NAV
   dislocation days.
5. **Rating and percentiles** are read directly off the simulated paths (terminal prices, running
   minimum and maximum).

Everything is configurable in `spxlcast/config.py`.

## Data notes

* Yahoo yields (`^IRX`, `^TNX`, ...) are used when FRED is unreachable; `^IRX` is a bank-discount
  rate and is converted to a bond-equivalent yield. Breakeven inflation then falls back to the
  configured default and the credit / CPI / unemployment adjustments are skipped.
* When the session is open the spot, VIX and yields are live quotes; the header says `intraday`
  instead of `close` and the price cache expires at the bell.
* Results are cached in `.cache/` (prices 6h, fund info 12h, news 1h). Failed or empty fetches are
  never cached. `--refresh` bypasses the cache and rebuilds it. Price history also accumulates in a
  per-ticker archive, because Yahoo intermittently returns only the latest bar for `^VIX3M` and
  `^VIX6M`; a partial response therefore never erases history already seen, and the metrics table
  prints n/a for a window that falls into a gap rather than a misleading number.
* A 3x fund can lose most of its value in a sustained decline; the model's own 5% tail shows how
  large that risk is.

## Tests and calibration backtest

```bash
python -m pytest
python scripts/backtest_calibration.py --drift 0.07
```

The backtest builds the model's inputs at every month-end since 2016 from data available then,
simulates with the project's own engine, and scores the realised outcome over the next 1, 3 and 6
months. With the current settings (10 years, 114 origins, constant 7% drift), the share of realised
SPXL outcomes inside the model's 5-95% band is 94% / 93% / 93% at 1 / 3 / 6 months against a 90%
target, versus 92% / 95% / 99% with the previous settings (flat 2-point haircut, symmetric shocks,
0.40% spread). The realised-to-model variance ratio at
6 months improved from 0.49 to 0.72, and the predicted chance of a 20% drawdown within 6 months
fell from 54% to 42% against a realised 35%. Outcomes still sit high in the distribution on average
(mean percentile about 0.62 at 6 months) for SPY and SPXL alike, because the index returned about
15% a year over the sample against the 7% assumed; that is the sample, not the fund mechanics.
