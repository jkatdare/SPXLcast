# Recalibrating the 20% dip odds: pre-registration

Version 1.0. Frozen before any new number was computed: the UTC time and the SHA-256 of this file and
of `research/calibration_grid.json` (the machine-readable copy of the settings and the rule) are in
`research/CALIBRATION_HASH.txt`. If the two files disagree, this one governs. Nothing here changes
after the freeze; anything done differently is listed, with the reason, in
`research/calibration_results.md`, which reports every number whatever the outcome.

---

## Part 1. In plain English

### The problem

The page's drawdown-risk level is the chance that SPXL closes at least 20% below today's price at
some point in the next 3 months. The same kind of number gives the chance that a buy-limit order
fills. The 1990-2026 backtest (`scripts/backtest_rating.py`, 440 month-ends) found these chances run
high: at 3 months the model said 26.3% on average and a 20% fall followed 22.2% of the time (42.5%
against 35.4% at the elevated level); at 6 months 38% against 32%. Its 90% price ranges held 91-93%
of outcomes, a little more than the 90% they should, so they are slightly too wide as well.

### What may change, and what may not

Only the settings that decide how much SPXL is expected to swing:

* the **VIX haircut**: how many points are taken off the VIX (1 month), VIX3M (3 months) and VIX6M
  (6 months) before they are used as the expected swing size. Today 3, 5 and 7 points.
* the **size of the swings in swing size** (stochastic volatility): how much calm and stormy
  stretches differ from path to path. Today 0.35.

Fifteen combinations are tried: five haircut sets (3/5/7 as today, 3/6/8, 3/7/9, 4/6/8, 4/7/9) times
three stochastic-vol sizes (0.25, 0.35 as today, 0.45). Everything else stays as it is. In particular
the model's expected return is **not** tuned: the timing-signal study (`research/RESULTS.md`) and the
drift backtest showed that fitting returns to the past would be overfitting, and the expected return
is supposed to come from valuation, not from what the market happened to do.

### How the choice is made and checked

* **Choose on 1990 to mid-2007.** Each combination is scored on the month-ends from January 1990 to
  June 2007. The last 6-month outcome window ends in December 2007, so nothing from 2008 on is used.
* **The score.** For each month-end, the gap between the predicted chance of a 20% fall and what
  happened (1 if it fell 20%, 0 if not), squared, and averaged over the month-ends: the Brier score.
  Lower is better. It is taken at 1, 3 and 6 months and the three are averaged. A forecast cannot
  game it by always saying a low number: it is punished every time the fall does happen.
* **Two safety rules.** A combination only counts if (1) it does not make the forecast of the whole
  price range worse by more than 0.5% at 1, 3 or 6 months (measured with CRPS, a score of the whole
  range; lower is better), and (2) its 90% ranges still hold between 87% and 94% of outcomes at
  every horizon from one week to six months.
* **The pick.** The combination with the best (lowest) average Brier score that passes both safety
  rules. Today's setting always counts. It is re-checked with twice as many simulated paths.
* **Check once on 2008 to 2026.** The pick and today's setting are then scored once on the month-ends
  from January 2008 on, which played no part in the choice. The tool switches to the pick only if,
  there too, its Brier score is lower than today's setting's, its whole-range score is not worse by
  more than 0.5% at 1, 3 or 6 months, and its 90% ranges hold 87-94% of outcomes at every horizon.
  Otherwise nothing changes, and the result is written up anyway.

### What has already been seen (so the check is not perfectly fresh)

* The full 1990-2026 results for **today's setting** are published in the README and in
  `output/backtest/rating_backtest.txt`: the overall numbers above, the split by level (before 2009 a
  20% fall followed 4%, 19% and 41% of the time at low, normal and elevated; on SPXL itself since
  2009 14%, 25% and 27%, where 44% was predicted at elevated), and the forecast-range scores by era.
* **Today's setting was itself chosen on 2016-2026** with `scripts/backtest_calibration.py` (it
  compared haircuts from 1/3.5/5.5 to 3/5/7 and stochastic-vol sizes 0.25 to 0.35 there). So today's
  setting has already been fitted to part of the check period. That tilts the check in its favour:
  it makes switching harder, not easier.
* The fifteen combinations lean toward smaller swings because the full-sample numbers showed the
  fall chances run high. That knowledge chose the candidates; it does not choose the winner, which
  the 1990-2007 score does.

---

## Part 2. Technical specification

### 1. Data and code

* Inputs: the cached `output/backtest/rating_inputs.pkl` (SHA-256
  `d21e55055f35d91e5084f336b115b6dedcd99e8819598a1a15544851084d868e`, fetched 2026-09-26), never
  `--refresh`. Point-in-time model inputs exactly as in `scripts/backtest_rating.py` (`inputs_at`),
  VIX3M/VIX6M before 2008 imputed (`--term-structure impute`, the default).
* Outcome: the backtest's realised fund (`build_daily`): a synthetic 3x fund before 2009-01-01, SPXL
  from then on. The 20% dip happened at horizon h if the fund's lowest close over the next h
  sessions is at most 0.8 times its close on the month-end.
* The simulation is the live engine (`spxlcast.montecarlo.simulate`) fed as in `run_origin`: the
  model's own fundamentals drift (`expected_index_return`), `vol_term_structure` with the setting's
  haircuts, and every other engine argument at its 0.3.0 value. Only the model simulation is run
  (not the rating or the constant-7% twin). Horizons 5, 10, 21, 63 and 126 sessions. Seed 42 for
  every setting, so all settings share their random numbers.
* The added code (`scripts/backtest_rating.py --engine-grid`, `--end`, `--only`) is written after
  this file is frozen; it must implement this section, and any mismatch found later is reported.

### 2. Windows

* **Train:** origins at the backtest's month-ends dated 1990-01 through 2007-06 (last origin
  2007-06-29; its 126-session window ends 2007-12-28). About 210 origins.
* **Check:** origins at the month-ends dated 2008-01 through the last complete month in the cached
  inputs (2026-08). Each horizon uses the origins that have a full outcome window (6-month outcomes
  through the 2026-02 origin).

### 3. The grid (15 settings; `research/calibration_grid.json`)

| | `sv_logvol_sd` 0.25 | 0.35 | 0.45 |
|---|---|---|---|
| `vrp_vol_points` (3, 5, 7) | x | **current** | x |
| (3, 6, 8) | x | x | x |
| (3, 7, 9) | x | x | x |
| (4, 6, 8) | x | x | x |
| (4, 7, 9) | x | x | x |

Held fixed at their 0.3.0 values: `sv_persistence` 0.97, `sv_leverage` -0.5, `skew_gamma` 0.9,
`t_dof` 4, `max_daily_move` 0.20, `drift_uncertainty_sd` 0.02, `vol_floor` 0.08, `long_run_vol`
0.16, `swap_spread` 0.0075, the expense ratio, tracking noise 0.0008 a day, and the whole
expected-return model (not tuned).

### 4. Metrics (per setting, per horizon h, over the window's origins with an outcome at h)

* p = share of simulated paths whose lowest fund close up to h is at most 0.8 x the start; y = 1 if
  the realised dip happened, else 0.
* **Brier_h** = mean of (p - y)^2.
* **CRPS_h** = mean of `spxlcast.evaluation.crps_sample` of the simulated log fund returns at h
  against the realised log return.
* **Cov90_h** = share of realised log returns within the 5th to 95th percentile of the simulated
  ones (inclusive), for h in 5, 10, 21, 63, 126.
* **Objective B** = (Brier_21 + Brier_63 + Brier_126) / 3. Lower is better.
* Reported for information only (never used for the decision): mean PIT, 50% band coverage, mean
  predicted and realised dip frequency, and each Brier difference from the current setting with an
  overlap-aware 90% interval (`mean_interval` on `effective_n`, as elsewhere in the backtest). The
  rule does not require the improvement to be statistically clear.

### 5. Selection on the train window (10,000 paths per origin)

1. Guardrail G1: CRPS_h(setting) <= 1.005 x CRPS_h(current) for each h in 21, 63, 126.
2. Guardrail G2: 0.87 <= Cov90_h(setting) <= 0.94 for each h in 5, 10, 21, 63, 126 (unrounded).
3. Eligible = settings passing G1 and G2, plus the current setting in any case.
4. Selected = the eligible setting with the lowest B. On an exact tie the current setting wins.
5. Confirmation at 20,000 paths (same seed, same window): the selected and the current setting are
   rerun. The selection stands if the selected still has a lower B than the current and passes G1
   and G2. If not, the next eligible settings in order of their 10,000-path B are confirmed the same
   way, one at a time; if none confirms, the current setting is selected (nothing to check).

### 6. Check on the check window (20,000 paths per origin, run once)

The selected and the current setting are scored once on the check window. **Adopt** the selected
setting only if all three hold there:

* A1: B(selected) < B(current);
* A2: CRPS_h(selected) <= 1.005 x CRPS_h(current) for each h in 21, 63, 126;
* A3: 0.87 <= Cov90_h(selected) <= 0.94 for each h in 5, 10, 21, 63, 126.

Otherwise keep the current setting and say so.

### 7. After the decision

* **Adopted:** set the new defaults in `spxlcast/config.py` (with a comment pointing here), bump
  `MODEL_VERSION` to 0.4.0, rerun the full backtest (`py scripts/backtest_rating.py --paths 20000
  --write-reference`) so `spxlcast/reference.json` and `output/backtest/rating_backtest.txt` describe
  the new engine, rerun `scripts/backtest_calibration.py` (the README quotes it), and update every
  quoted number in the README. The full-backtest numbers then mix the train and check windows and
  are descriptive, not a test.
* **Not adopted:** no default changes, `MODEL_VERSION` stays 0.3.0, and the results are documented.
* Either way `research/calibration_results.md` holds the train table for all 15 settings, the
  20,000-path confirmation and the check table.
