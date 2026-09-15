# SPXLcast

Fundamentals-driven price forecast for **SPXL** (Direxion Daily S&P 500 Bull 3X), with a
**Buy / Hold / Sell** rating and a **percentile lookup** for any price level.

The forecast is a Monte Carlo distribution of SPXL prices built from:

* **Valuation of the S&P 500** - earnings yield, dividend yield, long-run earnings growth
* **Macro conditions** - short and long Treasury yields, real yields, yield-curve slope, credit
  spreads, inflation and labour data (FRED, when reachable)
* **Option-implied volatility** - the VIX, VIX3M and VIX6M term structure
* **Leveraged-ETF mechanics** - 3x daily rebalancing, expense ratio, swap financing at the short
  rate, calibrated against SPXL's own history
* **News sentiment** - Yahoo Finance headlines for SPXL, SPY, the S&P 500 and its ten largest
  constituents, scored with VADER plus a finance lexicon

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
`--paths 20000`, `--seed 42`, `--no-news`, `--no-fred`, `--no-macro-adj`, `--refresh` (ignore the
cache), and overrides `--index-drift 0.08`, `--vol 0.18`, `--pe 22`, `--div-yield 0.013`,
`--eps-growth 0.055`.

## How to read the output

**Rating.** Judged at the rating horizon (default 6 months = 126 trading days) from two numbers
taken off the simulated distribution:

* *edge* - annualised median SPXL return minus the T-bill return
* *P(beat T-bill)* - share of simulated paths where SPXL beats cash

Each is scaled to [-1, 1] (edge / 15%, and (P - 0.5) / 0.10), averaged into a score, and
thresholded: score >= +0.30 with positive edge is **BUY**, score <= -0.30 is **SELL**, otherwise
**HOLD**. Conviction is High / Medium / Low by |score|. Every driver of the distribution is listed
under the rating so the verdict is auditable.

**Price percentile.** For a price P and each horizon:

* *Percentile of end price* - share of simulated end-of-horizon prices at or below P. A 23rd
  percentile means only 23% of outcomes end that low.
* *P(dips to level)* - chance the price trades at or below P at some point before the horizon,
  i.e. the chance a buy-limit order at P fills.
* *P(rises to level)* - the same for a sell-limit above spot.

The **buy-limit ladder** inverts this: it lists the prices with a 90/75/50/25/10% chance of being
touched within the rating horizon, and the median end price relative to that entry.

## Model

1. **S&P 500 expected total return** (annualised): 50/50 blend of
   * earnings-yield model: `E/P + expected inflation` (10y breakeven, or 2.5% default)
   * dividend-growth model: `dividend yield + 5.5% nominal EPS growth`

   then macro adjustments, each capped and documented in the report: equity-risk premium versus
   the real 10y yield, yield-curve inversion, high-yield credit stress, hot inflation, and a Sahm-rule
   style labour signal. The result is clipped to [-10%, +20%].
2. **Volatility**: VIX / VIX3M / VIX6M less a 2-point variance-risk-premium haircut, converted into
   forward variances so the total variance to each pillar matches the options market; beyond six
   months the vol mean-reverts toward a 16% long-run level.
3. **News sentiment**: articles within 7 days, weighted by recency (2-day half-life) and index
   relevance; the score in [-1, 1] shifts the drift of the first 21 trading days by up to +/-10%
   annualised.
4. **Simulation**: 20,000 paths of daily S&P 500 total returns with Student-t shocks (4 degrees of
   freedom, capped at +/-20% a day). Each day SPXL earns `3 x index return - (expense + financing)/252
   + tracking noise`; volatility decay is not assumed, it emerges from compounding. Leverage,
   tracking noise and the empirical cost drag are checked against a two-year regression of SPXL on
   SPY.
5. **Rating and percentiles** are read directly off the simulated paths (terminal prices, running
   minimum and maximum).

Everything is configurable in `spxlcast/config.py`.

## Data notes

* Yahoo yields (`^IRX`, `^TNX`, ...) are used when FRED is unreachable; breakeven inflation then
  falls back to the configured default and the credit / CPI / unemployment adjustments are skipped.
* Results are cached in `.cache/` (prices 6h, fund info 12h, news 1h). Use `--refresh` to bypass.
* Not investment advice. A 3x fund can lose most of its value in a sustained decline; the model's
  own 5% tail shows how large that risk is.

## Tests

```bash
python -m pytest
```
