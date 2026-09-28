# SPXL timing-signal search: pre-registration

Version 1.0, frozen 2026-09-27 (UTC time and SHA-256 of this file and of `research/prereg.json` in
`research/PREREG_HASH.txt`). `prereg.json` is the machine-readable copy of Part 2. If the two ever
disagree, this file governs. Nothing in this plan may change after the freeze except through the
deviation procedure in section 15.

---

## Part 1. In plain English

### The question

The 1990-2026 backtest (`scripts/backtest_rating.py`) found that SPXLcast's forecast ranges are well
calibrated, but its old BUY / HOLD / SELL rating could not tell good times to hold SPXL from bad ones.
This plan asks one question, once, with the rules fixed in advance: is there a signal built from
valuation, the economy, interest rates, credit, inflation, jobs, bank lending conditions,
option-implied volatility or risk measures that tells, at a month-end, whether holding SPXL for the
next six months is likely to pay better than T-bills?

Chart-reading is ruled out. No signal may use price momentum, trends, moving averages, chart
patterns, or the direction in which SPXL or the S&P 500 has been moving.

### Why the rules are written down first

Try enough signals on the same history and one of them will look good by luck. So everything that
decides the answer is fixed and hashed before anyone looks at the data that will judge it: which
signals, how each is measured, which direction counts as good, how they are combined, the test, and
the bar for success. The outcomes from January 2008 on stay sealed until then, and the test is run
once.

### What has already been seen

Not all of 2008-2026 is fresh. The README reports full-sample (1990-2026) results. The old rating's
rank correlation with the next six months' SPXL return over T-bills was -0.03. Months when holding
SPXL was expensive (the leverage-cost "hurdle" was high) were followed by better returns, not worse.
The model's valuation-based expected return added nothing at six months. And the rating's SELL months
at volatility spikes (1998, 2000, 2008-09, 2020) were followed by strong rebounds, so VIX spikes
followed by rebounds have been noticed. The big events of 2008-2026 (the financial crisis, zero
interest rates, the 2020 crash and rebound, the 2022 inflation shock) are also common knowledge.
Three of the twelve signals (the VIX level, the hurdle, the old rating score) are therefore
re-tests, not clean out-of-sample tests, and several others are partly exposed. Section 2 of Part 2
lists everything.

### The twelve signals

| # | Signal | What it measures | Favourable when | Clean test? |
|---|---|---|---|---|
| C01 | Variance risk premium | VIX squared minus the S&P 500's variance over the last month | high | partly (uses the VIX) |
| C02 | Output gap | industrial production against its trend, on the data published at the time | low (below trend) | yes |
| C03 | Short-rate change | the 3-month Treasury yield against a year earlier | falling | mostly (2022 hikes widely known) |
| C04 | Term spread | 10-year minus 3-month Treasury yield | steep | mostly (an inverted-curve penalty was inside the tested model) |
| C05 | Default spread | Baa minus Aaa corporate bond yields | wide | yes |
| C06 | Inflation | consumer prices against a year earlier | low | mostly (2021-23 known; a CPI penalty was inside the tested model) |
| C07 | Bank lending standards | net share of banks tightening business-loan standards (Fed survey) | easing | yes |
| C08 | Jobless claims | new unemployment claims against a year earlier | falling | mostly (2008 and 2020 spikes known) |
| C09 | Excess CAPE yield | the S&P 500's long-run earnings yield minus the real bond yield | high (cheap) | partly (valuation was tested in the model) |
| C10 | Implied variance | VIX squared | low | no (VIX spikes and rebounds seen) |
| C11 | Leverage-cost hurdle | the S&P 500 return SPXL needs to break even (`spxlcast/assess.py`) | low | no (seen) |
| C12 | Old rating score | the retired BUY / HOLD / SELL score | high | no (seen: -0.03) |

The favourable directions come from theory and from research published before 2008, never from a
look at the data.

### The combination

The nine fresh signals (C01-C09) are combined into one. Each is turned into a score (how far it
sits from its 1990-2007 average, in standard deviations), flipped so that favourable is positive,
capped at 3, and the nine scores are averaged with equal weight. No weights are fitted, so there is
nothing to overfit. The three re-tests (C10-C12) are kept out of it.

### How it is judged

* **Split.** 1990 to mid-2007 is training. It is used only to set each signal's average and spread
  and the cut-off for the money test below; it does not choose signals or directions. January 2008
  on is the test.
* **The statistical test.** For every month-end from January 2008 to February 2026 (the last with
  six full months of data), rank the signal, and rank what SPXL then returned over T-bills in the
  next six months. The rank correlation must be positive and strong enough to pass a one-sided test.
  The test counts overlapping six-month windows honestly (the 218 month-ends amount to about 37
  independent windows) and allows for having tried 13 things (12 signals plus the combination) with
  the Holm correction at 5%.
* **The money test.** At each month-end from January 2008, hold SPXL if the combination is at or
  above the line that marked its lowest third in 1990 to mid-2007, otherwise hold T-bills, paying
  0.1% on each switch. Through August 2026 this rule must: grow faster than buy-and-hold SPXL; have
  a better Sharpe ratio (return per unit of risk); have a smaller worst peak-to-trough loss; beat a
  no-timing mix holding the same average amount of SPXL, on growth and on Sharpe; and keep its
  Sharpe edge over buy-and-hold in both halves of the period (2008-2016 and 2017-2026).

### What counts as success

A signal is a **real timing signal** only if it passes both tests. If it passes the statistical
test but not the money test, it predicts something but not usefully. If the money test passes but
the statistical test does not, the gain could be luck. Otherwise: **no timing signal found**.

### What to expect

The data are thin. Six-month windows overlap, so 18 years hold only about 37 independent ones. To
pass at the strictest step of the Holm correction, a signal needs a rank correlation of about 0.45.
Published predictors, where they work at all, are nearer 0.1 to 0.2, and they weaken after
publication. The most likely
result is "no timing signal found". That would be an answer, not a reason to try variants on the
same years. Every result is published, whatever it shows.

### Afterwards

Once the test has run, 2008-2026 is spent as test data. A new idea can then be tested only on data
that do not exist yet, such as the live track record from October 2026 on.

---

## Part 2. Technical plan

### 1. Scope and hard constraints

* **Files.** The work may create or edit only `scripts/research_signals.py`, `research/*`,
  `tests/test_research_*.py` and `output/research/*` (git-ignored). No other tracked file is edited.
  `scripts/backtest_rating.py`, `scripts/backtest_drift.py` and `spxlcast/*` may be imported, not
  changed.
* **No technical analysis (construction rule).** No input may be built from the direction, trend,
  momentum or moving average of SPXL, the S&P 500 or any other price. Allowed inputs are valuation
  ratios (levels only), macro data, interest rates, credit spreads, inflation, labour data, survey
  data, option-implied volatility, and realised volatility used strictly as a risk measure (squared
  returns, sign-blind). Every candidate below meets this rule by construction.
  Correlations with past S&P returns are reported as diagnostics (section 13), never used to add,
  drop or adjust a candidate.
* **Seal.** No predictive statistic that uses an outcome window ending after 2007-12-31 may be
  computed before the Test phase (section 15). Test-period predictor values may be computed and
  stored, but not summarised beyond missing-value counts, before the Test phase.

### 2. Exposure ledger (what has already been seen)

This ledger is part of the frozen record. Results that touch it carry the labels in section 12
("not a clean out-of-sample test" or "partly exposed").

1. **README.md full-sample results (1990-2026, including the sealed period).**
   * Retired rating score: Spearman rank correlation with the next 6 months' SPXL excess return over
     T-bills -0.03 (90% interval -0.22 to +0.17). BUY months -4.2 points against the rest (-18.0 to
     +9.6). SELL in 12 months (Aug-Sep 1998, Oct-Dec 2000, Oct 2008-Mar 2009, Mar 2020), mostly at
     volatility spikes near market bottoms, followed by +39% average SPXL over 6 months (beat T-bills
     in half of them). No BUY/SELL threshold worked in both halves (1990-2007 and 2008-2026). BUY in
     83% of months 2008-2021, HOLD in 95% since 2022.
   * Leverage-cost hurdle: 6-month excess returns in high-cost months +9.4 points against the rest
     (-5.6 to +24.4), low-cost months -4.1 (-16.2 to +8.0). The research brief also records that the
     hurdle's full-sample 6-month rank correlation was seen (about -0.03, like the score's). The
     predicted fund cost matched the realised on SPXL (13.1% against 14.4% a year).
   * VIX spikes followed by strong rebounds (1998, 2008-09, 2020) were noticed in full-sample output.
     Any result for a VIX-level signal (C10, C11, C12, and partly C01) is therefore not a clean test.
   * Drawdown-risk levels: a 20% dip within 3 months followed 8.8% / 22.4% / 35.4% of the time at
     low / normal / elevated (11.7% / 24.5% / 42.5% predicted); on SPXL alone since 2009, 14% / 25% /
     27%.
   * Forecast distribution: 90% bands held 91-93% of outcomes; CRPS 3-15% better than a naive
     lognormal; mean PIT 0.54-0.57; 20% dip chance overstated by 4-6 points at 3-6 months.
   * Fundamentals drift (E/P and D/P blend with the inverted-curve, HY-spread, CPI > 3.5% and Sahm
     penalties): 6-month CRPS skill against a constant 7% drift +0.4% (-0.9% to +1.7%). The HY-spread
     penalty could not fire before 2023-09.
   * Drift backtest 1881-2026: correlation with realised returns 0.22 / 0.47 / 0.60 at 1 / 5 / 10
     years; the countercyclical valuation term added nothing; since 1990 the drift ran 4.8% a year
     low at one year; the S&P 500 returned about 15.4% a year over 2016-09..2026-06.
2. **Model parameters fitted on data that include the test period:** the 0.75% swap spread and
   tracking noise (calibrated on SPXL since 2009), the 3 / 5 / 7 vol-point VIX haircuts, the VIX3M /
   VIX6M imputation before 2008 (a log-linear fit on 2008-2026; it feeds C11 and C12 before 2008),
   and the tercile cut-offs in `spxlcast/reference.json`.
3. **General knowledge** held by the judge and the three proposers (all language models) of
   2008-2026 market and macro history: the 2008-09 crisis, the zero lower bound (2009-2015,
   2020-2021), the 2020 crash, rebound and claims spike, strong returns at high valuations in the
   2010s, the 2021-23 inflation surge and 2022 rate shock and bear market, the April 2020 negative oil
   price, and the missing October 2025 CPI. Mitigation: definitions and signs are taken from
   research published before 2008, and fixed now.
4. **Literature that uses test-period data** (for example Goyal, Welch and Zafirov 2024; Moreira and
   Muir 2017; Chava, Gallmeyer and Park 2015; Bekaert and Hoerova 2014; Sahm 2019; the excess CAPE
   yield note of 2020) counts as seen.
5. **The training window is published evidence** for several candidates (Bollerslev, Tauchen and
   Zhou's VRP sample is exactly 1990-2007; Cooper and Priestley, Ang and Bekaert and Welch and Goyal
   overlap it). Training results are a replication, not new evidence. Only the test period is new,
   and it is post-publication, so decay is expected (McLean and Pontiff 2016).
6. **What the judge opened:** README.md; the code of `spxlcast/assess.py`, `spxlcast/evaluation.py`,
   `spxlcast/rating.py`, `spxlcast/env.py`, `spxlcast/data._fred_api`, `scripts/backtest_rating.py`
   and config values; the header line (column names only) of `output/backtest/rating_backtest.csv`;
   the date range and fetch time of `output/backtest/rating_inputs.pkl`; FRED series metadata and
   ALFRED vintage dates (no observation values). The judge computed only calendar quantities (origin
   counts, effective sample sizes, power) from the session calendar. It did not open
   `rating_backtest.txt`, `rating_backtest.png`, the rows of `rating_backtest.csv`, the drift
   backtest outputs, `observations.csv`, `summary.csv` or the `run_*.txt` files, and computed no
   statistic that uses predictors or outcomes.
7. **What the proposers opened:** each reports reading only README.md and code signatures
   (`scripts/backtest_drift.load_shiller`, `spxlcast.data._fred_api`), with no data or results
   files opened and no statistic computed.

### 3. Data, calendar and notation

* **Session calendar.** The trading days of `scripts/backtest_rating.build_daily(inputs, Config())`
  built from the pinned inputs below (the `^SP500TR` calendar).
* **Origin t.** The last session of each calendar month (`backtest_rating.month_end_positions`),
  from 1990-01-31 to 2026-08-31.
* **t-** is the session immediately before the origin t. All daily market inputs of C01-C10 use
  values dated on or before t-, so every input is known before the close of t. C11 and C12 are read
  as the backtest computed them, from data dated t (the VIX settles 15 minutes after the equity
  close; disclosed, as in the backtest).
* **Stale values.** A daily series takes its last value dated on or before t- if that value is at
  most 5 sessions old, else it is missing. Weekly and monthly series follow their release rules
  below.
* **Positions.** A decision made at origin t is implemented at the close of t. The outcome window
  for horizon h is sessions t+1..t+h.
* **Data cutoff.** The close of 2026-09-25 (the last session before this freeze). Nothing after it
  is used.
* **Pinned inputs** (SHA-256):
  * `output/backtest/rating_inputs.pkl` `d21e55055f35d91e5084f336b115b6dedcd99e8819598a1a15544851084d868e`
    (fetched 2026-09-26T15:04Z; Yahoo closes to 2026-09-25, FRED rates, Shiller data): the source of
    all outcomes (`build_daily`), of `^VIX` and `^GSPC` closes and of the Shiller earnings.
  * `output/backtest/rating_backtest.csv` `a272f0cadabefc75f1c88d4026444c3961cd9ce490f75ee65a763968c2fb4be4`:
    the source of C11 (`hurdle`) and C12 (`score`) only. The loader reads the columns `date`, `pos`,
    `hurdle` and `score` and nothing else.
  * `output/backtest/ie_data.xls` `044196dafe44c3030b2facbdea023975b3f6aa68b4e52f8f9bafc403e19589c1`
    (the Shiller file behind the pickle).
  * The other FRED / ALFRED series are downloaded once in the Train phase into
    `output/research/predictor_inputs.pkl`, whose SHA-256 is recorded in `research/TRAIN_REPORT.md`
    before the Test phase; the Test phase uses that same file.
  * If a pinned file's hash does not match, the script stops.
* **FRED key.** Read through `spxlcast.env.fred_api_key()`; never printed or written, and never
  placed in an exception message (requests puts the URL in its messages; see
  `spxlcast.data._fred_api`).

### 4. Targets

From the pinned `build_daily` frame `d` (the realised fund is the synthetic 3x fund through
2008-12-31 and SPXL from 2009-01-02; `tbill_ret` is the previous day's DGS3MO / 252):

* `R_bill(t,h) = prod over i = t+1..t+h of (1 + tbill_ret_i) - 1`
* **SPXL excess return** `Y(t,h) = fund[t+h] / fund[t] - 1 - R_bill(t,h)` (identical to the
  backtest's `excess_h`).
* **S&P 500 excess return** `S(t,h) = tr[t+h] / tr[t] - 1 - R_bill(t,h)` (`^SP500TR`; the
  backtest's `sp_h - tbill_h`).
* **20% dip** `D20(t) = 1` if `min over i = t+1..t+63 of fund[i] / fund[t] <= 0.8`, else 0 (the
  backtest's `real_dd20_63`).

**Primary target:** `Y(t,126)`, the SPXL excess return over T-bills over the next 6 months.
**Secondary targets:** `Y(t,21)`, `Y(t,63)`; `S(t,21)`, `S(t,63)`, `S(t,126)`; `D20(t)`.

### 5. Split and embargo

* **Train origins** for horizon h: month-ends from 1990-01-31 whose window's last session is on or
  before 2007-12-31. From the calendar: h = 126: 1990-01-31..2007-06-29 (210 origins, about 35.8
  independent windows); h = 63: ..2007-09-28 (213, 71.7); h = 21: ..2007-10-31 (214, 209.2).
* **Test origins** for horizon h: month-ends from 2008-01-31 whose window's last session is on or
  before 2026-09-25. h = 126: 2008-01-31..2026-02-27 (218 origins, about 37.1 independent);
  h = 63: ..2026-05-29 (221, 74.2); h = 21: ..2026-07-31 (223, 218.2). D20 uses h = 63.
* Origins between the two (for example 2007-07..2007-12 at 6 months) belong to neither.
* **Parameter window:** the month-ends 1990-01-31..2007-06-29. Every quantity the Train phase sets
  (section 8) is computed from predictor values at these month-ends only, for all horizons.

### 6. The candidates

Signs: **+** means a higher value is favourable (predicts a higher SPXL excess return); **-** the
reverse. `s_j` is +1 or -1 accordingly. "Composite" says whether the candidate enters the
combination (section 7).

#### C01 VRP - variance risk premium (sign +, composite yes)
* **Definition.** `VRP_t = VIX(t-)^2 / 12 - RV21(t-)`, with `RV21(t-)` the sum over the 21 sessions
  ending at t- of `(100 * ln(G_i / G_{i-1}))^2`, G the S&P 500 price-index close. Units: percent
  squared per month.
* **Sources.** Yahoo `^VIX` and `^GSPC` closes from the pinned pickle (FRED `VIXCLS` as a
  cross-check only).
* **History.** VIX from 1990-01-02 (CBOE's back-calculation before 2003, from option prices of the
  time).
* **Lag.** Market closes of session t-.
* **Rationale.** The variance premium measures what investors pay to insure against variance, a
  proxy for aggregate risk aversion; a high premium precedes a high equity premium at 1-6 months,
  peaking near one quarter (Bollerslev, Tauchen and Zhou 2009; Drechsler and Yaron 2011). For a 3x
  fund, for a given VIX a higher premium also means lower recent realised variance and so, through
  persistence, less expected volatility drag: both channels point the same way. It is far less
  persistent than valuation ratios, so small-sample bias is minor.
* **Exposure.** Partial: it uses the VIX, and the premium turns sharply negative right after crashes
  (known for 2008 and 2020).
* **TA note (flagged).** Uses realised variance of S&P daily returns only as a risk measure, which
  the owner's rule allows. No semivariance, signed, drawdown or skew statistic may replace it.

#### C02 GAP - real-time output gap (sign -, composite yes)
* **Definition.** Take the INDPRO vintage in effect on the origin date t (FRED API
  `realtime_start = realtime_end = t`). Let m be its latest observation month. Fit by OLS
  `ln IP_k = a + b * k + c * k^2` over months k from 1948-01 (or the vintage's first observation, if
  later) through m, with k counted in months. `GAP_t` is the residual at m. Re-estimated at every
  origin.
* **Sources.** FRED / ALFRED `INDPRO` (vintages from 1927-01-26; 983 before 2008).
* **History.** Series from 1919; every 1990-2026 origin has a vintage.
* **Lag.** The vintage in effect at t (G.17 releases around mid-month, so m is normally t's month
  minus one). Revisions and re-basing are handled by the vintage; the log trend absorbs re-basing.
* **Rationale.** Risk premia are countercyclical: output above trend marks low risk aversion and low
  expected returns, output below trend the reverse. Cooper and Priestley (2009) find the real-time
  gap predicts excess returns at 1-12 months, in and out of sample, on pre-2008 data. Risks: the
  end-point instability of a recursive trend and manufacturing's shrinking share. A negative gap
  also means higher volatility, which works against the sign for SPXL.
* **Exposure.** Clean apart from general knowledge.
* **TA note.** None.

#### C03 SRATE - 12-month change in the 3-month Treasury yield (sign -, composite yes)
* **Definition.** `SRATE_t = y3(t-) - y3(t'-)` in percentage points, where y3 is FRED `DGS3MO` and
  t' is the origin 12 month-ends earlier.
* **Sources.** FRED `DGS3MO` (from 1981-09-01).
* **Lag.** Market yields dated on or before t-.
* **Rationale.** A short rate above its level a year ago marks monetary tightening: higher discount
  rates and tighter credit. Ang and Bekaert (2007) find the short rate the most robust short-horizon
  predictor, with a negative sign; Campbell (1991) and Hodrick (1992) use the detrended bill rate.
  The 12-month change is used instead of the classic "rate minus its 12-month average" so that no
  moving-average construct appears anywhere. Financing at the bill rate cancels in an excess return,
  so there is no mechanical effect. At the zero lower bound the signal is near zero for long spells,
  which lowers its power.
* **Exposure.** The 2022 hiking cycle and bear market are common knowledge.
* **TA note.** None (an interest rate).

#### C04 TERM - term spread (sign +, composite yes)
* **Definition.** `TERM_t = DGS10(t-) - DGS3MO(t-)` in percentage points.
* **Sources.** FRED `DGS10`, `DGS3MO` (`T10Y3M` as a cross-check).
* **Lag.** Market yields dated on or before t-.
* **Rationale.** A steep curve marks early-cycle conditions and high risk premia; term-structure
  variables forecast excess stock returns (Campbell 1987; Fama and French 1989). An inverted curve
  precedes recessions, usually by more than 6 months. Out-of-sample evidence is weak (Welch and Goyal
  2008).
* **Exposure.** The live drift's inverted-curve penalty was part of the full-sample drift test
  (not isolated).
* **TA note.** None.

#### C05 DEF - default spread (sign +, composite yes)
* **Definition.** `DEF_t = DBAA(t-) - DAAA(t-)` in percentage points (Moody's seasoned Baa minus Aaa
  corporate yields).
* **Sources.** FRED `DBAA` (from 1986-01-02), `DAAA` (from 1983-01-03).
* **Lag.** Dated on or before t-; not revised.
* **Rationale.** The default spread is high in bad times and tracks the business-cycle component of
  expected returns (Keim and Stambaugh 1986; Fama and French 1989). The sign is + because the
  primary target is a simple excess return, whose expectation is about 3 x the S&P 500's excess
  return less costs, so the risk-premium channel sets it. High-spread states are also high-variance
  states, which hurts SPXL's typical outcome; that channel is tested separately by C10 and C11.
  Levels only: spread changes move with past equity returns.
* **Exposure.** Close to clean: the live model's HY-spread penalty could not fire before 2023-09.
* **TA note.** None.

#### C06 INFL - CPI inflation, year on year (sign -, composite yes)
* **Definition.** Take the `CPIAUCNS` vintage in effect on the origin date t (ALFRED) and let m be
  its latest observation month. `INFL_t = ln(CPI_m / CPI_{m-12})`. If `CPI_{m-12}` is missing, the
  previous origin's value is carried forward; missing months are never interpolated. Fallback, only
  if the ALFRED fetch fails: m = the latest month on or before the origin month minus one with a
  published value.
* **Sources.** FRED / ALFRED `CPIAUCNS` (not seasonally adjusted, so no seasonal-factor revisions;
  vintages from 1949).
* **Lag.** About 10-15 days after the reference month, through the vintage.
* **Rationale.** High inflation precedes low real and excess stock returns (Fama and Schwert 1977;
  Fama 1981): it signals weaker real activity and tighter policy, and brings higher volatility. Weak
  out of sample (Welch and Goyal 2008). Inflation illusion implies the opposite sign at long
  horizons; the short-horizon sign is registered.
* **Exposure.** The 2021-23 inflation and 2022 bear market are common knowledge; the live drift's
  CPI > 3.5% penalty was part of the full-sample drift test.
* **TA note.** None.

#### C07 SLOOS - bank lending standards (sign -, composite yes)
* **Definition.** `SLOOS_t` = FRED `DRTSCILM` (net percentage of domestic banks tightening standards
  on commercial and industrial loans to large and middle-market firms) from the latest observation
  usable at t, held constant between releases. An observation dated at the start of quarter q is
  usable from the last session of the first month after quarter q ends (for example, the observation
  dated 1990-04-01 is usable from 1990-07-31).
* **Sources.** FRED `DRTSCILM` (from 1990-04-01; ALFRED vintages from 2010-04-20 only).
* **Lag.** The rule above is deliberately conservative, because release dates before 2010 are not in
  ALFRED. QA branch fixed now: for every observation whose first ALFRED vintage date lies within 120
  days of its observation date, the rule's usable date must be on or after that vintage date; if it
  ever is not, all usable dates move one month later. No revisions.
* **Rationale.** Tightening credit supply precedes slower lending, investment and output (Lown and
  Morgan 2006; Bassett, Chosak, Driscoll and Zakrajsek 2014). Chava, Gallmeyer and Park (2015)
  report that tighter standards predict lower aggregate returns from one quarter to a year; their
  sample includes 2008, so this prior is partly post-2008. Persistent and quarterly.
* **Exposure.** Clean apart from general knowledge and the post-2008 paper.
* **TA note.** None (a survey of loan officers).

#### C08 CLAIMS - initial jobless claims, year on year (sign -, composite yes)
* **Definition.** Let w be the latest week-ending Saturday with `w + 5 days <= t` (claims are
  released on the Thursday after the week ends). `CLAIMS_t = ln(sum of ICNSA over weeks w-3..w) -
  ln(sum of ICNSA over weeks w-55..w-52)`.
* **Sources.** FRED `ICNSA` (not seasonally adjusted, from 1967-01-07), latest vintage. ALFRED
  vintages exist only from 2009; the advance figure is revised once a week later by a small amount,
  which is accepted and disclosed.
* **Lag.** 5 days after the week ends.
* **Rationale.** Claims are the timeliest hard labour-market series and a leading indicator (Stock
  and Watson 1989). Rising layoffs mark recession onsets, when drawdowns and high volatility, the
  main risks for a 3x fund, concentrate. Academic evidence for index returns is thin, and bad
  unemployment news can be good for stocks in expansions (Boyd, Hu and Jagannathan 2005): a weak
  prior, included for the labour category.
* **Exposure.** The 2008 and 2020 spikes are common knowledge. The 2020 spike is capped by the
  composite's z clip; ranks are unaffected.
* **TA note.** None.

#### C09 ECY - excess CAPE yield (sign +, composite yes)
* **Definition.** Let q = `backtest_rating.reported_quarter(t)` (the last quarter-end whose earnings
  were out two months after it ended) and m = the latest CPI month in the `CPIAUCNS` vintage at t
  (as in C06). `CAPE_t = (P(t-) / CPI_m) / mean over the 120 months k = q-119..q of (E_k / CPI_k)`,
  with P the `^GSPC` close and E Shiller's trailing 12-month earnings from the pinned pickle.
  `RR10_t = DGS10(t-) / 100 - ((CPI_m / CPI_{m-120})^(1/10) - 1)`. `ECY_t = 1 / CAPE_t - RR10_t`.
  Shiller's own CAPE and Excess CAPE Yield columns are not used (they use same-month interpolated
  earnings and monthly-average prices). Level only.
* **Sources.** Shiller `E` (pinned pickle, from `output/backtest/ie_data.xls`), Yahoo `^GSPC`, FRED
  `DGS10`, `CPIAUCNS`.
* **Lag.** Earnings: last reported quarter (quarter end plus 2 months), the backtest's rule; months
  between past reported quarters are Shiller interpolations of already reported values. CPI: vintage
  at t. Price and yield: t-.
* **Rationale.** Cheap equities relative to real bond yields should earn more (Campbell and Shiller
  1988, 1998; Asness 2003). Evidence is strong at 5-10 years and weak at 1-6 months. It is the
  valuation baseline, with a low prior. Very persistent (Stambaugh bias).
* **Exposure.** Partial: the model's valuation drift added no 6-month skill and the valuation term
  added nothing over 1881-2026; strong returns at high valuations in the 2010s are common knowledge.
* **TA note.** Allowed as valuation. The ratio's numerator is the price level, so it is used as a
  level only; changes in CAPE would be past returns and are forbidden.

#### C10 IVAR - implied variance (sign -, composite no)
* **Definition.** `IVAR_t = (VIX(t-) / 100)^2`.
* **Sources.** Yahoo `^VIX` from the pinned pickle.
* **Lag.** t-.
* **Rationale.** For a daily-reset 3x fund, log growth is about 3 x the index excess return less
  about 3 x the index variance per year less costs; the S&P premium does not rise in proportion to
  variance (French, Schwert and Stambaugh 1987; Glosten, Jagannathan and Runkle 1993). High implied
  variance should therefore mark worse typical SPXL outcomes. The sign is contested: Martin (2017)
  implies the S&P's expected simple excess return rises with implied variance.
* **Exposure.** Not a clean test: VIX spikes followed by strong rebounds (1998, 2008-09, 2020) were
  seen in full-sample output, and the hurdle, driven mostly by the VIX, was seen to have no timing
  value.
* **TA note.** None (option-implied).

#### C11 HURDLE - leverage-cost hurdle (sign -, composite no)
* **Definition.** Column `hurdle` of the pinned `rating_backtest.csv` at origin t: the S&P 500 total
  return per year at which SPXL's average log growth is zero,
  `exp((fees + 2 x (3-month bill + 0.75%)) / 3 + 1.5 x sigma_1y^2) - 1`, from
  `spxlcast.assess.leverage_cost` with the 1-year vol of the model's VIX term structure.
* **Sources.** Pinned `rating_backtest.csv` (VIX curve from Yahoo, VIX3M / VIX6M imputed before
  2008; DGS3MO).
* **Lag.** Data dated t, as in the backtest.
* **Rationale.** The model's own cost measure: a higher hurdle means SPXL needs more from the market,
  so it should mark worse times to hold it. The rate part mostly cancels in an excess return, so it
  is in effect a volatility measure.
* **Exposure.** Not a clean test: its full-sample tercile results and (per the research brief) its
  rank correlation were seen.
* **TA note.** None.

#### C12 SCORE - retired rating score (sign +, composite no)
* **Definition.** Column `score` of the pinned `rating_backtest.csv` at origin t: the average of the
  simulated 6-month annualised median excess return over T-bills divided by 15% and the annualised
  Sharpe-like ratio divided by 0.5, each clipped to [-1, 1] (`spxlcast/rating.py`), from the full
  model with the fundamentals drift and no news.
* **Sources.** Pinned `rating_backtest.csv`.
* **Lag.** Data dated t, as in the backtest.
* **Rationale.** The model's intended direction: a higher score was meant to mark better times to
  own SPXL.
* **Exposure.** Not a clean test: full-sample rank correlation -0.03 at 6 months and threshold
  behaviour in both halves were seen.
* **TA note.** None.

### 7. The combination signal (COMP, sign +)

* `z_{j,t} = (x_{j,t} - mu_j) / sd_j`, where `mu_j` and `sd_j` (sample sd, ddof = 1) are taken over
  the parameter-window month-ends at which `x_j` is available.
* `COMP_t = mean over j in M_t of clip(s_j * z_{j,t}, -3, +3)`, where `M_t` is the set of members
  C01-C09 available at t. `COMP_t` is missing if fewer than 7 of the 9 members are available.
* Equal weights, fixed now; no weights are fitted and no member is dropped after the freeze.
  Correlated members (for example C03 and C04, C01 and C05) keep their equal weights.

### 8. What the Train phase may set, and what is fixed now

**Fixed now:** the candidates and their definitions, sources and lags; the signs; the composite's
members, weights and clip; the targets, horizons and split; the statistic, inference, correction and
alpha; the economic rule, costs and pass criteria; the sanity checks and reports.

**Set by the Train phase, from predictor values at parameter-window month-ends only (no outcomes):**
* `mu_j`, `sd_j` for C01-C09 (also reported for C10-C12);
* the composite threshold `tau` = the 33.33rd percentile (numpy `percentile`, linear interpolation)
  of `COMP_t`; and for the secondary economic table, `tau_j` = the same percentile of `s_j * x_{j,t}`
  for each candidate;
* the vol-managed constant `c` = the median of `IVAR_t`, and `w_bar` = the mean of
  `min(1, c / IVAR_t)` (section 11);
* the SLOOS lag branch (section 6, C07);
* the SHA-256 of `output/research/predictor_inputs.pkl` and of `output/research/train_params.json`.

Train-period statistics against outcomes (section 10, "replication") are computed and reported but
may not change anything.

### 9. Primary statistical test

For each predictor P in {C01, ..., C12, COMP} (K = 13 tests):

1. Sample: test origins for h = 126 at which `P_t` and `Y(t,126)` are both available.
2. `r = spearman(s_P * P_t, Y(t,126))` with `spxlcast.evaluation.spearman` (average ranks for
   ties). `s_COMP = +1`.
3. `n_eff = spxlcast.evaluation.effective_n(positions, 126)`, positions being the session indices of
   the sample's origins.
4. `z = atanh(r) * sqrt(n_eff - 3) / sqrt(1 + r^2 / 2)` (the Fisher z with Bonett and Wright's
   variance, as `backtest_rating._corr_ci`). One-sided p-value in the registered direction:
   `p = 1 - Phi(z)`. If `r` is undefined or `n_eff <= 4`, `p = 1`. Equivalently, p < 0.05 exactly when
   the lower end of the backtest's 90% interval is above zero.
5. **Holm** across the K = 13 p-values at family-wise alpha = 0.05: sort ascending,
   `p_(1) <= ... <= p_(13)`; reject `H_(i)` while `p_(i) <= 0.05 / (13 - i + 1)`; stop at the first
   failure. A rejected hypothesis is a **primary statistical pass** for that predictor.
6. Also reported for each: `r`, n, `n_eff`, the 90% interval `_corr_ci(r, n_eff)`, raw and
   Holm-adjusted p.

**Power (calendar-based, fixed now).** At h = 126, `n_eff` is about 37.1. An unadjusted one-sided 5%
test needs r >= 0.28; the first Holm step (0.05 / 13 = 0.0038) needs r >= 0.45. Approximate power
at a true rank correlation of 0.1 / 0.2 / 0.3: 0.14 / 0.32 / 0.55 unadjusted, 0.02 / 0.07 / 0.18 at
the first Holm step. At h = 63 (`n_eff` about 74) the Holm threshold is r >= 0.31; at h = 21
(about 218), r >= 0.18. A null is the most likely outcome even if a modest effect exists; a null
means "no timing signal shown", not "try another variant".

### 10. Secondary tests (reported, never decisive on their own)

The same 13 predictors, statistic, inference and one-sided direction, with Holm applied within each
family of 13:

| Family | Target | Horizon for `n_eff` | Favourable direction |
|---|---|---|---|
| F2 | `Y(t,21)` | 21 | r > 0 |
| F3 | `Y(t,63)` | 63 | r > 0 |
| F4 | `S(t,126)` | 126 | r > 0 |
| F5 | `S(t,63)` | 63 | r > 0 |
| F6 | `S(t,21)` | 21 | r > 0 |
| F7 | `D20(t)` | 63 | favourable predicts fewer dips: r > 0 against `-D20(t)` |

**Replication (train period):** the primary test and F2-F7 on train origins, reported side by side
with the test period, labelled as published-sample replication.

### 11. Economic test

**Rule (primary, composite).** At each test origin t from 2008-01-31 through 2026-07-31, hold SPXL
(the realised fund) from the close of t to the close of the next month-end if `COMP_t >= tau`, else
hold T-bills (`tbill_ret`). The last holding ends at the close of 2026-08-31 (223 monthly holding
periods, 2008-02-01..2026-08-31). A switch between SPXL and T-bills costs 0.10% of the portfolio,
charged on the first day of the new holding. If `COMP_t` is missing, the previous position is kept
(the count is reported).

**Benchmarks.**
* **BH:** buy-and-hold SPXL (the realised fund) over the same sessions, no cost.
* **MIX_w:** w in SPXL and 1 - w in T-bills, rebalanced at each month-end, with w the rule's fraction
  of the 223 months in SPXL; cost 0.10% x |weight traded|. This is a no-timing portfolio with the
  same average exposure, which separates timing from simply holding less leverage (a 3x fund's
  volatility drag means a constant partial holding can beat buy-and-hold by itself).

**Metrics** on daily returns, as `backtest_rating.strategy_section`:
* CAGR = `W_end^(252 / N) - 1`, N sessions;
* Sharpe = mean / sd of daily (return - `tbill_ret`) x sqrt(252);
* maximum drawdown = max over days of `1 - W_t / max_{s <= t} W_s`.

**ECONOMIC PASS** if and only if all four hold:
* **E1** CAGR(rule) > CAGR(BH) and CAGR(rule) > CAGR(MIX_w);
* **E2** Sharpe(rule) > Sharpe(BH) and Sharpe(rule) > Sharpe(MIX_w);
* **E3** maximum drawdown(rule) < maximum drawdown(BH) (shallower);
* **E4** Sharpe(rule) > Sharpe(BH) within each half: sessions 2008-02-01..2016-12-30 and
  2017-01-03..2026-08-31.

The same rule and criteria are applied, with that predictor's own `tau_j`, to every candidate that
passes the primary test. For all 12 candidates the rule is reported as a secondary table.

**Secondary economic analyses (descriptive, never decisive):**
* the composite rule with 0 and 25 bp switching costs, and with the threshold at the train median;
* stationary bootstrap (Politis and Romano 1994) 90% intervals for Sharpe(rule) - Sharpe(BH) and
  CAGR(rule) - CAGR(BH): monthly holding-period returns resampled in pairs, mean block length 6
  months, 10,000 resamples, seed 20260927;
* a volatility-managed exposure rule (Moreira and Muir style; not clean, uses the VIX level):
  SPXL weight `w_t = min(1, c / IVAR_t)`, rest T-bills, rebalanced monthly at 0.10% x |weight
  traded|, against BH and against a constant `w_bar`;
* time in SPXL, number of switches, turnover, total costs paid.

### 12. Verdict

| Outcome | Label |
|---|---|
| A predictor passes the primary test (Holm) **and** the economic test | **Timing signal confirmed** for that predictor. For C10, C11 or C12 add "not a clean out-of-sample test"; for C01 or C09 add "partly exposed" (section 2). |
| Primary pass, economic fail | **Predictive, not usable by the registered rule.** |
| Composite economic pass, no primary pass | **Not confirmed**; the gain could be luck. A candidate for a new pre-registration on future data only. |
| Neither | **No timing signal found.** |

Robustness labels (added to the verdict, not changing it): "episode-dependent" if the one-sided p
rises above 0.10 when any one episode in section 13 is left out; "one half only" if r is not
positive in both halves.

### 13. Sanity checks (all run and reported)

1. **Seal.** The Train and Build phases load no outcome window ending after 2007-12-31 (code guard
   and unit test, section 15).
2. **Pinned inputs.** The three pinned hashes match; `predictor_inputs.pkl` and `train_params.json`
   match the hashes in `research/TRAIN_REPORT.md`.
3. **Point in time.** For each candidate at 20 train month-ends, the value computed on the full data
   equals the value computed after deleting every observation dated after t- (market data) or
   released after t (macro data). Unit tests with synthetic series for each release rule (CPI and
   INDPRO vintages, SLOOS usable date, claims Thursday rule, Shiller earnings lag).
4. **Target reproduction.** Train phase: rebuilt `Y`, `S` and `D20` equal the pinned CSV's
   `excess_h`, `sp_h - tbill_h` and `real_dd20_63` at the train origins of each horizon (tolerance
   1e-9); the CSV's outcome columns are read through the seal (rows outside the train origins are
   dropped at load). Test phase: the same at all origins.
5. **Legacy reproduction (Test phase).** The score's full-sample 6-month Spearman must come out at
   -0.03 +/- 0.01 (README). A mismatch means a join or target error, fixed as a deviation before
   anything is interpreted.
6. **Synthetic fund.** Daily correlation of the synthetic fund with SPXL where both exist (2009 on)
   is at least 0.99 (predictor-free).
7. **Coverage.** Non-missing share of every candidate and of COMP at train and test origins
   (expected: all at least 95%, SLOOS from 1990-07). Test-period coverage below 90% flags that test.
8. **Standardisation.** `mu_j` and `sd_j` reported; no `sd_j` is zero; members available per month.
9. **Descriptives (train only until unsealed).** Time-series plots and summary statistics of every
   predictor at train month-ends.
10. **TA diagnostic.** Train-period Spearman of each candidate with the trailing 1-, 3-, 6- and
    12-month S&P 500 price return, and of COMP; reported for the owner's judgement only.
11. **Collinearity.** Train-period Spearman matrix of the 12 candidates.
12. **Robustness of the primary results (Test phase, reported):** the two halves (origins
    2008-01..2016-12 and 2017-01..2026-02); leaving out each episode's origins in turn: 2008-01..2009-06
    (financial crisis), 2019-09..2020-06 (2020 crash), 2021-07..2022-12 (inflation and rate shock);
    a moving-block bootstrap 90% interval for r (blocks of 12 origins, 10,000 resamples, seed
    20260927); the composite with expanding-window standardisation (z from values at month-ends up
    to t, from 1990-01, minimum 60) instead of train-fixed; partial Spearman of each passing
    predictor controlling for `ln VIX(t-)` and `DGS3MO(t-)` (does it add to what the model already
    uses); the synthetic fund's spread at 0.25% and 1.25% (affects the 2008 test months only).
13. **Look log.** Every run of `scripts/research_signals.py` appends UTC time, phase, command line,
    spec hashes and `git describe --always --dirty` to `output/research/look_log.csv`; the log is
    summarised in the results.

### 14. Reported whatever the outcome

`research/RESULTS.md` (committed by the owner) and `output/research/test/*` contain: the 13 primary
tests (r, n, `n_eff`, 90% interval, raw and Holm p, pass or fail); families F2-F7; the train
replication; the economic test for the composite with benchmarks and E1-E4; the secondary economic
tables; every sanity check; the deviations; the look log summary; the verdict in the words of
section 12, including "No timing signal found" if that is the result. Nulls are published in full.

### 15. Phases, code guard and deviations

* **Phase 0, Freeze.** This file, `prereg.json` and `PREREG_HASH.txt`.
* **Phase 1, Build.** `scripts/research_signals.py` (subcommands `build`, `train`, `test`) and
  `tests/test_research_*.py` implement this plan exactly. Required tests: the seal
  (`tests/test_research_seal.py`), point in time (`tests/test_research_pit.py`), and the statistics
  (`tests/test_research_stats.py`: the p-value agrees with `_corr_ci`, Holm on known inputs, the
  composite on synthetic data).
* **Phase 2, Train.** Downloads `predictor_inputs.pkl`; builds every predictor at every origin;
  sets the section 8 quantities; runs the train-side checks and the replication; writes
  `output/research/train_params.json` and `research/TRAIN_REPORT.md` (with both hashes).
* **Phase 3, Test.** `py scripts/research_signals.py test --unseal <SHA-256 of PREREGISTRATION.md>`.
  Runs once.
* **Code guard.** The outcome loader truncates the daily frame at 2007-12-31 before computing any
  outcome unless unsealed. Unsealing requires: the token equals the SHA-256 of
  `research/PREREGISTRATION.md`; both files' SHA-256 match `research/PREREG_HASH.txt`;
  `train_params.json` exists and matches the hash in `research/TRAIN_REPORT.md`. The legacy
  predictor loader reads only `date`, `pos`, `hurdle`, `score` from the CSV and checks the pinned
  hash; the CSV's outcome columns are read only by the target-reproduction check, through the same
  seal.
* **Deviations.** Any departure is written to `research/DEVIATIONS.md` with its reason, the date and
  whether it was made before or after unsealing. Before unsealing, a deviation needs a data or code
  reason (for example a series that cannot be fetched), never a train-period result. After
  unsealing, only bug fixes are allowed; the original and the corrected results are both reported.
  A candidate whose data cannot be built is reported as "not tested"; K stays 13 and its p is 1.

### 16. Considered and excluded

| Proposal | Reason |
|---|---|
| Oil price change (1 month) | Price momentum of a commodity: excluded under the owner's no-momentum rule, read conservatively |
| Credit-spread change (3 months), GZ excess bond premium change | Changes move closely with past equity returns (disguised momentum); the EBP is re-estimated on the full sample and is not point in time |
| Relative bill rate (rate minus its 12-month average) | Replaced by the 12-month change (C03) so that no moving-average construct appears |
| Baa minus 10-year Treasury (BAA10Y) | Mixes credit risk with a maturity mismatch; Baa minus Aaa is the canonical default spread |
| NFCI, ANFCI, STLFSI, KCFSI | Re-estimated with full-sample weights, not point in time; include equity inputs |
| cay (consumption-wealth ratio) | Needs recursive DOLS on NIPA and Z.1 vintages; one quarter stale; household wealth moves with equity prices |
| AAII bull-bear spread, put/call ratios | Licence and download approval; survey answers chase recent prices, close to technical analysis |
| University of Michigan sentiment | Weak prior for the S&P 500 (effects mostly in small stocks) |
| Sahm rule (real time) | Designed in 2019 with hindsight including 2008; overlaps C08; inside the tested drift |
| VIX3M / VIX6M slope, VVIX, implied correlation | Start in 2006-2007: nothing to train on |
| ICE BofA high-yield spread | The FRED API serves about 3 years |
| TED spread, ISM PMI, OECD CLI | Discontinued, not on FRED, or embeds share prices |
| Short interest, net payout yield | No free point-in-time data |
| CBOE SKEW, calendar and FOMC-cycle effects | Weak prior, or calendar effects that decayed after publication |
| HAR-based ex-ante VRP, realised-variance scaling | One definition per idea; realised variance is kept inside C01 only |

### 17. How the three proposals were consolidated

* Kept where two or more proposers agreed and the data are point in time from 1990: VRP, implied
  variance, term spread, short rate, default spread, inflation, output gap, excess CAPE yield,
  claims. Added bank lending standards (one proposer) as the only point-in-time survey of financial
  conditions. Added the hurdle and the score as required, disclosed re-tests.
* Primary horizon 6 months and SPXL as the primary target, per the owner's brief (proposals
  suggested 3 months, or the S&P 500).
* One registered statistic (Spearman with the backtest's overlap-aware Fisher z) instead of
  out-of-sample R-squared and Clark-West, so the result reads on the same scale as the disclosed
  -0.03.
* Equal-weight composite of train-standardised signed scores (all three proposers' preferred
  "improper linear model"), with no selection on training results.
* The Kelly and vol-scaled exposure rules became a secondary economic analysis; the matched-exposure
  benchmark was added to the pass criteria so that de-leveraging is not mistaken for timing.

### 18. References

Ang and Bekaert (2007) RFS 20; Asness (2003) JPM; Bassett, Chosak, Driscoll and Zakrajsek (2014)
JME; Bekaert and Hoerova (2014) J. Econometrics; Bollerslev, Tauchen and Zhou (2009) RFS 22;
Bonett and Wright (2000) Psychometrika; Boyd, Hu and Jagannathan (2005) JF; Campbell (1987) JFE;
Campbell (1991) Economic Journal; Campbell and Shiller (1988) RFS and JF, (1998) JPM; Chava,
Gallmeyer and Park (2015) JME; Cheng and Madhavan (2009) J. Investment Management; Cooper and
Priestley (2009) RFS 22; Drechsler and Yaron (2011) RFS; Fama (1981) AER; Fama and French (1989)
JFE; Fama and Schwert (1977) JFE; French, Schwert and Stambaugh (1987) JFE; Glosten, Jagannathan and
Runkle (1993) JF; Goyal, Welch and Zafirov (2024) RFS; Harvey, Liu and Zhu (2016) RFS; Hodrick (1992)
RFS; Holm (1979) Scandinavian J. Statistics; Keim and Stambaugh (1986) JFE; Lown and Morgan (2006)
JMCB; Martin (2017) QJE; McLean and Pontiff (2016) JF; Moreira and Muir (2017) JF; Politis and
Romano (1994) JASA; Stambaugh (1999) JFE; Stock and Watson (1989) NBER Macroeconomics Annual; Welch
and Goyal (2008) RFS.
