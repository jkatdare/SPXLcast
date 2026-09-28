# Deviations from the pre-registration

Kept under `research/PREREGISTRATION.md` section 15. Every entry records its reason, its date and
whether it was made before or after unsealing. Before unsealing a deviation needs a data or code
reason, never a train-period result.

All entries below were made on 2026-09-27, **before unsealing**, during Phase 1 (Build). D3-D5
and note N12 come from an independent point-in-time audit made the same day, after the first build.
At that point no statistic of any candidate against any target had been computed, and no test-period
predictor value had been summarised beyond missing counts.

## Deviations

### D1. C09: CPI months that were never published (data reason)

* **Plan.** `CAPE_t = (P(t-) / CPI_m) / mean over the 120 months k = q-119..q of (E_k / CPI_k)`. The
  plan does not say what to do when a `CPI_k` does not exist.
* **Reason.** The BLS published no October 2025 CPI (exposure ledger, item 3), so `E_k / CPI_k` is
  undefined for k = 2025-10.
* **Implemented.** Months without a published CPI are left out of the mean, never interpolated
  (C06's rule). C09 is computed only if at least 114 of the 120 months have a CPI; otherwise it is
  missing.
* **Effect.** 7 test-side origins (2026-02-27..2026-08-31) use a 119-month mean. Of these, only
  2026-02-27 is a primary-test origin (h = 126). The other six enter only the economic test through
  the composite. No train-side origin is affected. The rule was chosen before any C09 value was
  looked at.

### D2. `predictor_inputs.pkl` is downloaded in the Build phase, not the Train phase (code reason)

* **Plan.** Section 3 says "the other FRED / ALFRED series are downloaded once in the Train phase
  into `output/research/predictor_inputs.pkl`".
* **Reason.** The Build phase has to compute every candidate at every origin, and it cannot do that
  without these series.
* **Implemented.** `build` downloads each series once. The raw responses are cached under
  `output/research/raw/` with a manifest of their SHA-256 values. The pickle is assembled once and
  never rewritten: later runs read the existing file, and `--refresh` refuses to run once
  `research/TRAIN_REPORT.md` records a hash for it.
* **For the Train phase.** Record SHA-256
  `133e9d8692620fdc992a31c529bdc456c1dca5fde907f9955734793cf55b5560` for this file in
  `research/TRAIN_REPORT.md`. The Test phase uses the same file. (The Build phase first wrote a file
  with SHA-256 `50b5e205...b79a`. The point-in-time audit, D5, added `UNRATE` to it. The other ten
  series and the eight vintage checks are byte-identical to that first file.)
* **Contents.**
  * ALFRED `INDPRO` and `CPIAUCNS`: every row whose real-time period intersects
    1990-01-01..2026-09-25.
  * ALFRED `DRTSCILM` and `ICNSA`: their full real-time histories up to 2026-09-25.
  * `DGS3MO`, `DGS10`, `DBAA`, `DAAA`, and `VIXCLS` and `T10Y3M` (cross-checks only): the vintage
    in effect on 2026-09-25.
  * Eight direct ALFRED queries (`realtime_start = realtime_end = t`) for `INDPRO` and `CPIAUCNS`
    at four train month-ends. These are used only to check the vintage reconstruction.
  * ALFRED `UNRATE`: its full real-time history up to 2026-09-25. Only its first-publication dates
    are read (D5), never its values.

### D3. C07: DRTSCILM values as published at the time, not the 2019 revision (data reason)

* **Plan.** C07 is "FRED `DRTSCILM` ... from the latest observation usable at t". The plan says
  "No revisions". The Build phase read every value from the vintage in effect on the data cutoff
  (note N3).
* **Reason (found by the point-in-time audit).** The series is revised, 38 observations in all.
  * On 2019-11-04 the Board revised all 30 observations from 1990-04-01 to 1997-07-01, by up to 7.7
    points. For example, 1990:Q3 went from 39.5 to 46.7 and 1991:Q2 from 15.5 to 20.0.
  * On the same date it changed seven later observations. Five from before 2008 moved by 0.1.
    2008-01-01 and 2008-07-01 are test-period values and were not inspected.
  * One observation (2004-04-01) was revised on 2012-01-31.
  * The earliest ALFRED vintage (2010-04-20) holds the values the Board published before 2019. The
    Board's own chart data of February 2012 shows the same values.
  * Those values match the contemporaneous record, and the revised values do not. Schreft and Owens
    (Richmond Fed *Economic Review*, March/April 1991) report the net shares tightening across firm
    sizes: over 50% in May 1990 (published 56.9, revised 54.4), at most 37% in January 1991
    (published 36.0, revised 38.6), and 17% in May 1991 (published 15.5, revised 20.0).
  * So the cutoff vintage used the revised series as if it were final, which is look-ahead.
* **Implemented.** `sloos_values`: at origin t, the value is taken from the ALFRED vintage in effect
  on t. Before ALFRED's first vintage (2010-04-20), the value is taken from that first vintage. The
  usable-date rule and the QA branch are unchanged.
* **Effect.** C07 changes at 108 train-side origins (1990-07-31..2007-09-28; largest change 7.7
  points) and at 6 test-side origins (2008). The six test-side origins use observations 2008-01-01
  and 2008-07-01. Only the count of changed test-side values was looked at. No outcome was involved.

### D4. C08: a week counts only once it has been published (data reason)

* **Plan.** "Let w be the latest week-ending Saturday with `w + 5 days <= t` (claims are released on
  the Thursday after the week ends)."
* **Reason (found by the point-in-time audit).** During the 2025 federal shutdown, the Department
  of Labor published no national weekly claims figures from late September until 2025-11-20.
  * ALFRED dates the first publication of weeks 2025-09-27..2025-11-08 at 2025-11-20.
  * At origin 2025-10-31 (a primary-test and economic-test origin), the Thursday rule used weeks
    2025-10-04..2025-10-25, none of which had been published.
  * The Build-phase check (note N6) missed this. It kept only weeks published within 14 days of
    their week-end, which filtered out exactly the weeks published late.
* **Implemented.** `claims_release_dates` gives the release date of each week. It is the Thursday
  rule, or ALFRED's first-publication date where that is later.
  * The ALFRED date is used only where ALFRED records one, which it does from 2009-05-30 on. Weeks
    first seen at ALFRED's first vintage (2009-05-28) keep the rule.
  * The date is never earlier than the rule, so holiday-Wednesday releases do not move any origin.
  * w is the latest week published by t. The check now covers every week ALFRED dates, and it fails
    if any origin used a week before its publication.
* **Effect.** One origin changes. At 2025-10-31, w moves from 2025-10-25 to 2025-09-20, the last
  week published before the shutdown. The other late week, 2018-03-17 (first vintage 2018-03-29,
  probably a missed vintage), affects no origin. No other origin changes.
* **Not verifiable.** Weeks before 2009 have no ALFRED dates. The 1995-12-16..1996-01-06 shutdown
  closed the Department of Labor and may have delayed the reports for weeks ending 1995-12-16 and
  1995-12-23, which the train origin 1995-12-29 uses. No record of those release dates was found,
  so the rule is kept there.

### D5. C12: missing where the backtest used a number not yet published (data reason)

* **Plan.** C12 is the `score` column of the pinned `rating_backtest.csv` at t, "read as the
  backtest computed them, from data dated t".
* **Reason (found by the point-in-time audit).** The backtest (`backtest_rating.released`) assumes
  that every monthly observation up to the month before the origin is published by the origin. That
  failed twice:
  * **1996-01-31** (train). December 1995 CPI was published on 1996-02-01, delayed by the shutdown
    (BLS release USDL-96-28, embargoed until 8:30 a.m. on 1 February 1996). It enters the score
    through the trailing 10-year inflation and the CPI penalty.
  * **2025-10-31** (test). The September 2025 unemployment rate was published on 2025-11-20. It
    enters the score through the unemployment level and the Sahm gap.
  * C11, the hurdle, uses neither input, so it is unaffected.
* **Implemented.** `legacy_unreleased_inputs` compares, at every origin, the latest CPI and
  unemployment month the backtest used with its first-publication date. CPI dates come from ALFRED
  `CPIAUCNS`, since `CPIAUCSL` is in the same BLS release. Unemployment dates come from ALFRED
  `UNRATE`, which was added to `predictor_inputs.pkl` for this purpose. C12 is set missing where
  either month was not yet published (panel column `c12_unreleased_input`). The backtest itself is
  not changed.
* **Effect.** C12 is missing at 1996-01-31 and 2025-10-31: 209 of 210 train origins and 217 of 218
  test origins at h = 126. C12 was already a disclosed re-test, not a clean test.
* **Not fixed, disclosed.** C12 still reads the backtest's `CPIAUCSL` and `UNRATE` from their
  latest vintages, so seasonal-factor revisions are treated as final. It also still carries the
  exposure-ledger items (section 2, item 2).

## Implementation notes (no change to the plan)

* **N1. Targets in the Build phase.**
  * Section 13.1 of the governing `.md` allows the Build phase to load outcome windows that end on
    or before 2007-12-31. `prereg.json` (`phases.1_build`) says "no outcomes loaded". The `.md`
    governs.
  * `build` computes Y, S and D20 only through the sealed loader. It also runs the section 13.4
    target reproduction at train origins. The rebuilt targets match the pinned CSV exactly
    (maximum absolute difference 0 for all seven targets at 210 / 213 / 214 origins).
  * In `panel_test.pkl` the target columns are present but all missing.
* **N2. The seal.**
  * `outcome_frame()` truncates the backtest inputs (Yahoo closes and FRED series) at 2007-12-31
    before `build_daily` runs. This is a stronger form of "truncates the daily frame at 2007-12-31
    before computing any outcome". A unit test shows the sealed frame equals the prefix of the full
    frame.
  * The session calendar is read from the pinned `^SP500TR` closes. That is exactly `build_daily`'s
    index, so the Build phase never builds the post-2007 fund.
* **N3. "Latest vintage".** For `ICNSA`, `DBAA`, `DAAA`, `DGS3MO` and `DGS10`, "latest vintage"
  means the vintage in effect on the data cutoff, 2026-09-25. Nothing released after the cutoff is
  used. `DRTSCILM` no longer uses the cutoff vintage (D3).
* **N4. Sources for C03, C04 and C09.**
  * `DGS3MO` and `DGS10` come from `predictor_inputs.pkl`, per section 3.
  * On every date from 1981 to 2007 they are identical to the copies in the pinned
    `rating_inputs.pkl`.
* **N5. SLOOS QA branch.**
  * Computed from ALFRED first-release dates only; no values are used.
  * 67 observations (from 2010-01-01) were first released within 120 days of their date. For all 67,
    the rule's usable date is on or after the first release, so the branch is **shift 0**.
  * Section 8 lists this as a Train-phase quantity. The Train phase should record the same value.
* **N6. Claims release rule (corrected by the audit; see D4).**
  * The Build phase checked only weeks first published within 14 days of their week-end. It found
    three late weeks (2009-05-16, 2018-03-17, 2025-11-08) and no origin that used an unpublished
    week.
  * That filter hid the shutdown weeks 2025-09-27..2025-11-01, published 2025-11-20. 2009-05-16 is
    an artefact of ALFRED's first vintage (2009-05-28).
  * The corrected check covers the 904 weeks ALFRED dates, from 2009-05-30. It finds eight weeks
    published after w + 5 days, and no origin that uses a week before its publication.
* **N7. Vintage reconstruction.** The vintage rebuilt from the real-time rows is identical to FRED's
  direct `realtime_start = realtime_end = t` answer for `INDPRO` and `CPIAUCNS` at 1990-01-31,
  1996-07-31, 2001-10-31 and 2007-06-29.
* **N8. C06 fallback.**
  * The ALFRED fetch worked, so the fallback was not used.
  * The carry-forward rule did not fire at any origin.
* **N9. Panel layout.**
  * `panel_train.pkl` holds the 216 month-ends 1990-01-31..2007-12-31, with a membership flag for
    each horizon.
  * 2007-11-30 and 2007-12-31 are embargo month-ends that belong to no train set. They are kept
    because section 13.12's expanding-window standardisation uses every month-end up to t. Their
    targets are missing.
  * `panel_test.pkl` holds 2008-01-31..2026-08-31. 2026-08-31 belongs to no test set; the plan's
    origin range ends there.
* **N10. Test files.**
  * The point-in-time unit tests are in `tests/test_research_signals.py`, the file name requested
    for this phase, instead of `tests/test_research_pit.py`.
  * The seal tests are in `tests/test_research_seal.py`.
  * `tests/test_research_stats.py` belongs to the Train and Test phases.
* **N11. What the Build phase looked at** (in addition to exposure ledger items 6-7):
  * the structure, date ranges and missing counts of the pinned inputs;
  * the header of `rating_backtest.csv`;
  * FRED metadata and ALFRED release dates, including test-period release dates (dates only);
  * candidate values at five train month-ends (1990-01, 1990-07, 1998-08, 2002-09, 2007-06) as a
    sanity check;
  * train-side minimum and maximum of each candidate (plausibility check, in `panel_meta.json`);
  * test-side missing counts.

  It did not look at any test-period predictor value, any outcome value, or any statistic that
  relates predictors to outcomes.
* **N12. Point-in-time audit (2026-09-27, before unsealing).** Each candidate was checked against
  release calendars, ALFRED vintages and the cached raw data. The fixes are D3-D5. What was
  checked, and what was left as registered:
  * **Daily market data (C01, C03-C05, C10).**
    * Every input is dated on or before t-.
    * `^GSPC`, `^VIX` and the `^SP500TR` session calendar align on every session from 1990 to 2026.
    * ALFRED's earliest vintages were compared with the cutoff vintage: `DGS3MO`/`DGS10` from
      2005-06-28, `DBAA`/`DAAA` from 2014-04-02, `VIXCLS` from 2010-11-22.
      * `DBAA` and `DAAA` are unrevised.
      * `DGS3MO` differs on one day (1999-10-01, 0.1) and `DGS10` on one (1991-01-29, 0.02).
        Neither is a t- date of any origin.
    * Treasury, Moody's and CBOE values for t- are public by the close of t.
    * Before 1993 no VIX-type index was published. `^VIX` for 1990-1992 is CBOE's 2003
      back-calculation from the option prices of the time: not look-ahead, but not a series anyone
      could have watched then.
  * **INDPRO and CPI vintages (C02, C06, C09).**
    * The first-release dates in ALFRED are mostly 9-17 days after the reference month ends.
    * Every lag over 25 days is explained:
      * CPI for 1995-12 (1996-02-01, which matches BLS release USDL-96-28), 1996-01, 2013-09,
        2015-01 and 2025-10;
      * INDPRO for 2013-09, 2025-09 and 2025-10.
    * The vintage rule handles these, so no origin uses a month released after t.
  * **SLOOS usable date (C07).**
    * FRED dates each survey at the start of the quarter in which it was run.
    * The 1990-91 surveys ran in May, August and October 1990 and January and May 1991 (Richmond
      Fed 1991).
    * The Board's archive lists every 1997-2010 survey as released within its quarter or early in
      the next month, including the special surveys of September 1998 and March 2001.
    * The rule (last session of the first month after the quarter) therefore lags every checked
      release by at least one month. 1992-1996 release dates were not found.
  * **Claims (C08).**
    * The shutdown gap is fixed by D4.
    * ICNSA is used at its cutoff vintage, as registered. From 2009 the latest value differs from
      the first print by a median of 0.55% (90th percentile 1.7%, maximum 12.9%). The large
      revisions are mostly 2020-2023 state corrections, which is more than the "small" in the
      plan's text. This is kept as registered and disclosed.
  * **Shiller earnings (C09).**
    * Shiller's monthly E interpolates linearly between quarter-end months (March, June, September,
      December). This holds for 403 of 412 checked months from 1975 on; the other 9, in 1989-91,
      are off by at most 0.07.
    * So E through the last reported quarter q uses no later quarter.
    * The two-month rule is registered. It can include a late reporter: AIG published its
      2008-Q4 loss on 2009-03-02, after origin 2009-02-27, whose q is 2008-12. Assume that loss
      moved index EPS by about $7. It reaches the 120-month mean through three interpolated months,
      so it moves 1/CAPE by roughly 0.0002. This is left as registered.
    * The most recent Shiller quarters may be S&P figures compiled after t. This matters only for
      the last few origins; 2026-08-31 enters no test.
  * **C11 and C12.**
    * C11 uses only the VIX curve, the bill rate and fees.
    * C12 is fixed by D5. Its other disclosed departures are data dated t, VIX3M/VIX6M imputation
      and parameters fitted on 2008-2026, and revised seasonally adjusted CPI and unemployment.
  * **What the audit looked at** (besides items already listed in N11):
    * train-side values of the `DRTSCILM` vintages and the Board's 2012 and 2019 chart data for
      1990-1993;
    * first-publication dates of `CPIAUCNS`, `INDPRO`, `ICNSA`, `DRTSCILM` and `UNRATE` over
      1990-2026 (dates only);
    * the sizes of revisions to the raw `ICNSA` weekly counts from 2009 (an input, not a candidate);
    * Shiller E for 1988-1990 and 2025-04..2026-06, while checking the interpolation;
    * per candidate, the count of test-side origins whose value changed with the fixes (C07: 6,
      C08: 1, C12: 1);
    * a truncation-invariance check at 12 origins, 1995-2026, which compares values for equality
      only.
  * **What it did not do.** It computed no statistic that uses an outcome, and summarised no
    test-period candidate value. Its scratch scripts imported `scripts/research_signals.py`
    without running it, so the only look-log entry from the audit is the one `build` run.

## Train phase notes (2026-09-27, before unsealing)

Phase 2 ran as `py scripts/research_signals.py train` at 2026-09-27T17:48:48Z. A second, identical run at
17:49:43Z checked that the spec is deterministic: same SHA-256, freeze time kept. The frozen spec is
`research/frozen_spec.json`, SHA-256 `c02bbd667e56782a2b4a0e6dcaad93de77a72de4338e048ce32a351047226fed`
(`research/FROZEN_HASH.txt`). None of the notes below changes the plan.

* **N13. File names.**
  * The Train-phase request names `research/frozen_spec.json`, `research/train_results.md` and
    `research/FROZEN_HASH.txt`. The plan (sections 8 and 15) names `output/research/train_params.json` and
    `research/TRAIN_REPORT.md`, which the code guard reads.
  * All five are written. `train_params.json` holds the same bytes as `frozen_spec.json`.
    `TRAIN_REPORT.md` records the SHA-256 of `predictor_inputs.pkl` (`133e9d86...`, as D2 requires) and
    of `train_params.json`. `train_results.md` holds the results.
  * `train` refuses to change a frozen spec. A re-run that produces the same bytes keeps the original
    freeze time. `--refreeze` exists only for a data or code reason recorded here, before unsealing.
    `train` also refuses to run once `output/research/test/` exists.
* **N14. Details the plan leaves open, fixed in `frozen_spec.json` before the first `train` run**
  (field `interpretations`). Each one was written into the code before any train-period outcome
  statistic had been computed. None depends on a train-period result.
  * `tau` and `tau_j` use q = 33.33 exactly, as registered (not 100/3).
  * Every strategy's initial allocation is free, as buy-and-hold's is. A switch cost multiplies wealth
    by (1 - 0.001) at the start of the first session of the new holding.
  * `MIX_w` and the vol-managed rules drift within the month. They pay 0.001 x |weight traded| at each
    month-end rebalance.
  * The metrics follow `backtest_rating.strategy_section`: missing daily returns count as 0, and the
    drawdown runs over W_1..W_N. Sharpe uses ddof = 1. E4 restricts the full-period daily series to
    each half.
  * The stationary and moving-block bootstrap algorithms, with seed 20260927, are written out step by
    step.
  * The expanding-window composite includes month-end t and needs 60 values per member.
  * The partial Spearman is the partial correlation of average ranks. Its p uses n_eff - 5.
  * Holm ties keep the order C01..C12, COMP.
  * The legacy reproduction uses the pinned CSV's `score` column at every month-end that has a
    Y(t,126), including the two D5 origins, as the README did.
* **N15. The synthetic-fund check (section 13.6) moves to the Test phase.** It uses no predictor, but it
  needs daily data from 2009 on, and the code guard keeps the daily frame sealed until the Test phase.
* **N16. Scope of the train-side diagnostics.**
  * The descriptives, the collinearity matrix and the TA diagnostic use the 210 parameter-window
    month-ends, the set that fixes the spec's numbers.
  * The TA diagnostic correlates each signed candidate `s_j * x_j` with the trailing k-month S&P 500
    price return. That return is measured from `gspc_tm` (the `^GSPC` close at t-) of the origin k
    month-ends earlier to `gspc_tm` at t. So the first k month-ends of 1990 have no value.
  * The target reproduction (13.4) ran again at train origins, through the seal. It passed (maximum
    absolute difference 0).
* **N17. Candidate definitions in the spec.** The `definition` fields of `frozen_spec.json` are copied
  from `prereg.json` word for word. The panels were built with D1, D3, D4 and D5 applied, and those
  deviations still hold. The Test phase reads the predictors from the panel `build` wrote. It does not
  rebuild them from these text definitions.
* **N18. What the Train phase looked at** (in addition to N11-N12).
  * Every statistic in `research/train_results.md`: train-period distributions, collinearity, the TA
    diagnostic, and the replication of the 13 predictors against all seven targets at 1990-2007
    origins.
  * The hashes of the pinned inputs, `predictor_inputs.pkl`, `panel_train.pkl` and `panel_meta.json`.
  * The train rows of the pinned CSV's outcome columns, through the seal. These were compared for
    equality only.
  * Before the real run, the renderers ran once on the synthetic panel of `tests/test_research_stats.py`,
    in the session scratch directory. That run used no real data.
  * `panel_test.pkl` was neither opened nor hashed. No test-period outcome was computed, and no
    test-period predictor value was summarised. The spec did not change after the train statistics
    were seen.

## Test phase notes (2026-09-27, written before unsealing)

None of these notes changes the plan or the frozen spec. They record how the Test phase implements
what the frozen spec leaves to code, and what was done before the one run.

* **N19. File names.**
  * The Test-phase request names `research/test_results.md` and `output/research/test_results.json`.
    The plan (section 14) names `research/RESULTS.md` and `output/research/test/*`.
  * All are written. `RESULTS.md` holds the same bytes as `test_results.md`. `output/research/test/`
    holds `test_results.json` (the same bytes), `tests.csv`, `economic_daily.csv`,
    `economic_holdings.csv`, `panel_unsealed.pkl` and `economic_test.png`.
  * `py scripts/research_signals.py report` re-renders the two `.md` files from the JSON. It reads
    no data and recomputes nothing.
* **N20. Checks before unsealing**, in addition to the frozen preconditions.
  * The token is checked first. A wrong token returns before any other file is read.
  * The frozen spec's registered choices (candidates, signs, members, clip, percentile, primary test,
    families, costs, holdings, sessions, halves, episodes, seeds, verdict labels) are compared with
    the code. Any difference stops the run.
  * The section 8 quantities are recomputed from `panel_train.pkl` and must equal the frozen numbers.
    A check on train-side data only, made before the run, found them equal (maximum absolute
    difference 0). The test itself uses the frozen numbers.
  * `panel_test.pkl` was not hashed by the Train phase. The run checks that it is sealed (targets all
    missing), that its month-ends, positions and flags follow the session calendar, and that its
    missing counts are those in `panel_meta.json`. It also checks that every stored predictor and
    component, train and test side, equals a fresh rebuild from the pinned inputs and
    `predictor_inputs.pkl`. That comparison is for equality only; no value is summarised. Any failure
    stops the run before unsealing.
* **N21. Run once.**
  * The run writes `output/research/test/STARTED` before unsealing and `COMPLETED` at the end.
  * `test` refuses to run when `test_results.json` or `COMPLETED` exists. It also refuses when
    `output/research/test/` is left from an unfinished run, unless `--after-error` is given. That flag
    is only for a bug fix recorded here, with the original results reported.
  * `train` already refuses to run once `output/research/test/` exists.
* **N22. Details the frozen spec leaves open, fixed now.**
  * Partial Spearman: p = 1 when n_eff <= 6. This is section 9's "n_eff <= 4" rule with n_eff - 5 in
    place of n_eff - 3.
  * Vol-managed rule: if IVAR is missing at the first origin, the target weight is 0 (T-bills), as
    the switching rule does for a missing first signal. C10 has no missing test-side values.
  * E1-E4 use strict inequalities, as written. A missing (NaN) Sharpe fails its comparison.
  * The cost variants (0 and 0.25%) compare the composite rule with BH and with MIX_w at the same
    cost. The median-threshold rule and each candidate's rule are compared with their own MIX_w
    (w = that rule's share of the 223 holdings in SPXL), as the primary rule is.
  * Descriptives:
    * "switches" counts holdings whose target differs from the previous one. For MIX_w and the
      vol-managed rules it counts rebalances with a nonzero trade.
    * "turnover" is the sum of |weight traded|.
    * "total costs" is reported as the sum of the cost rates and as dollars paid per initial dollar.
  * The stationary bootstrap reports the point estimates beside the intervals. Both come from the same
    monthly holding-period series (monthly Sharpe x sqrt(12), CAGR over 223 months).
  * The synthetic-spread check counts an origin as changed when Y(t,126) moves by more than 1e-9.
    Fund levels are cumulative products, so windows after 2009 move by rounding only.
  * Verdict: "no primary pass" means that no predictor is rejected by Holm in the primary family.
    * A Holm pass with an economic pass is "Timing signal confirmed", with the section 12 qualifier.
    * A Holm pass without an economic pass is "Predictive, not usable by the registered rule".
    * The robustness labels are computed for all 13 predictors but attached to the verdict of primary
      passes only.
  * Test-period predictor descriptives (section 13.9, after unsealing): n, mean, sd, minimum, median,
    maximum at the 224 test-side month-ends, and the share of the 223 economic-test origins at which
    each predictor is at or above its threshold.
* **N23. Code changes for the Test phase.** None changes a build or train result.
  * `check_pinned` and `check_candidate_registry` take optional paths.
  * `outcome_frame` takes an optional `Config`, used unsealed only, for the synthetic-spread variants.
  * `one_sided` takes the degrees of freedom lost (default 3).
  * `replication` now calls the shared `family_tests`.
  * `jsonable` accepts numpy arrays.
  * Unit tests: `tests/test_research_phase3.py` covers a synthetic world, including the whole `test`
    command end to end. The seal test now checks that a wrong token is refused before anything is read.
* **N24. What was looked at before the run** (in addition to N11-N18).
  * The frozen spec, the plan, `research/train_results.md` and `panel_meta.json`.
  * The column names and dtypes of `panel_train.pkl`.
  * The train-side preconditions in N20.
  * `panel_test.pkl` was not opened before the run. The dry runs of the `test` command used synthetic
    data only, in the session scratch directory.

## Test phase run (2026-09-27, after unsealing)

* **N25. The run.**
  * `py scripts/research_signals.py test --unseal <token>` ran once. The look log records
    2026-09-27T18:14:47Z; the results record 18:14:50Z. It exited 0 after 22 seconds.
  * Every precondition passed before unsealing. The rebuild matched the stored panels in all 14,520
    cells, and the section 8 quantities matched with a maximum absolute difference of 0.
  * Every Test-phase sanity check passed:
    * target reproduction at all origins, with a maximum absolute difference of 0;
    * legacy reproduction, r = -0.0293 against the registered -0.03 +/- 0.01;
    * the synthetic fund's correlation with SPXL, 0.9981;
    * the test calendar;
    * coverage;
    * the train replication equals `train_stats.json`.
  * The verdict is "No timing signal found".
  * No bug affecting a registered number was found, and nothing was rerun. The results are in
    `research/test_results.md` (the same bytes as `research/RESULTS.md`),
    `output/research/test_results.json` and `output/research/test/`.
* **N26. Changes after unsealing.** None changes a registered number or the verdict.
  * Figure fix (cosmetic). `plot_test` shaded each T-bill holding from its origin to that origin's
    calendar month-end. That left unshaded gaps where an origin is not the last calendar day, such as
    2022-12-30. It now shades to the next origin. `output/research/test/economic_test.png` was
    redrawn from the saved CSVs; nothing was recomputed.
  * `report` now keeps a hand-written section headed "## EXPLORATORY" when it re-renders.
  * An EXPLORATORY section was added to `research/test_results.md` and `research/RESULTS.md`. It is
    computed from the saved outputs only and is labelled as not a result. It covers where the
    composite rule lost against buy and hold, the member z-scores during its 2022-10..2024-08 T-bill
    spell, and descriptive remarks on the largest correlations.
  * A unit test covers the preserved section.

## After the reviews (2026-09-27, after unsealing)

Three independent reviews of the run (look-ahead and data, statistics, practical value) found no error that
changes a registered number or the verdict. Their challenges, and how each was handled, are listed in
`research/RESULTS.md`, Appendix B. The notes below record every change made in response. No registered step
(build, train, test) was re-run to change a result, and nothing was tuned on test-period outcomes.

* **N27. `research/RESULTS.md` becomes the owner's report; presentation of `research/test_results.md`.**
  * Plan section 14 requires `research/RESULTS.md` to contain every registered item. Until now it held the
    same bytes as `test_results.md` (N19). `report` now writes it as the owner's report: a plain-language
    summary and recommendation, the confirmatory tables (section 2), exploratory notes (section 3),
    limitations, how to reproduce, and, as Appendix A, the registered document in full, rendered from the
    same JSON by the same function. Every item of section 14 is therefore still in `RESULTS.md`.
    `test_results.md` remains the registered document.
  * Presentation changes to `test_results.md` (the renderer, not the numbers; `output/research/test_results.json`
    is unchanged, SHA-256 `82c4cd0cb5360ef9f3a7ee9b4076973186e62392a589625922d459d0a3146056`):
    * p-values to four decimals in the verdict, section 1 and section 3, with the Holm bars stated;
    * a note in the verdict that the registered n_eff counts windows and ignores persistence, so the bar is
      conservative;
    * section 3: Holm across all 91 registered tests, labelled as not registered;
    * section 2: the figures are before tax, and 2008 uses the synthetic fund;
    * section 5.1: the rules that never left SPXL (C02, C09) are named;
    * section 5.3: the intervals rest on the rule's 25 T-bill months;
    * section 6: the block-bootstrap intervals are labelled unadjusted, not a test;
    * section 9: the exposure-ledger line (N30) and the code line (N28); the statement "None was made after
      unsealing" is replaced by a pointer to N26-N30.
  * The hand-written EXPLORATORY section of `test_results.md` (N26) was rewritten after the reviews. The
    phrase "consistent with the published short-horizon VRP effect" was removed, and notes on the
    conservative n_eff, the three episodes and the 2008 dependence of the economic results were added. It
    points to `RESULTS.md` section 3.
  * Test fix. The check in `tests/test_research_phase3.py` that no rendered cell reads `nan` used a pattern
    with literal backspace characters, so it could never match. It is now `r"\bnan\b"`.

* **N28. Process safeguards added after the run** (for any future run; none changes a number).
  * `verify_unseal` writes every successful unseal to the look log (phase `unseal`, the process's own command
    line and the calling function, `[outcome_frame]` or `[read_csv_outcomes]`). The log goes next to
    `train_params.json` unless a path is given. Before this, only `main()` wrote the log, so scripts that
    imported the module and unsealed (the point-in-time audit, N12, and the reviews) left no row. A script
    that calls `backtest_rating.build_daily` on the pinned inputs directly still cannot be logged; the seal is a
    guard against accidents, not against a determined reader (the token is published in `PREREG_HASH.txt`).
  * `test` records the SHA-256 of `scripts/research_signals.py`, `scripts/backtest_rating.py`,
    `spxlcast/evaluation.py` and `spxlcast/config.py` in `test_results.json`, and copies them to
    `output/research/test/code/`. The 2026-09-27 run predates this, so its JSON has no code hashes (see N29).
  * `check_spxl_closes`: the unsealed frame stops if an SPXL close is missing after 2009-01-01.
    `backtest_rating.build_daily` fills a missing daily return with 0, and pandas 3 changes `pct_change`'s
    fill rule; either would silently change returns. The pinned data have no missing SPXL close (the N29
    re-run passed the check). `scripts/backtest_rating.py` is outside this work's files and was not edited;
    passing `fill_method=None` there is for the owner.
  * `report` writes `research/OUTPUT_SHA256.txt`, the SHA-256 of every git-ignored file the hash chain points
    to (the pinned inputs, the raw downloads, `predictor_inputs.pkl`, the panels, `train_params.json`, the
    train outputs, `test_results.json` and `output/research/test/*`), so that an archived copy of `output/` can
    be checked with `sha256sum -c`.
  * Not done, and recommended for any future pre-registration: `DEVIATIONS.md` was not hashed at the freeze,
    and hashing it now would not prove its earlier content. Commit `research/` (plan, frozen spec, both hash
    files, `DEVIATIONS.md`) before the Test phase and record that commit in the test output, or publish the
    spec hash outside the repository.

* **N29. The Test phase reproduced with the current code** (a check, not a new test).
  * The code that ran the test at 18:14Z was not kept: `scripts/research_signals.py` was edited at 18:17Z (the
    plot fix and `report`, N26) and again for N27-N28. `scripts/backtest_rating.py` (last modified 16:03Z, in
    the working tree), `spxlcast/evaluation.py` and `spxlcast/config.py` (both equal to HEAD `9b55b30`) have not
    changed since before the run.
  * `cmd_test` was run with the final code into `output/research/repro_check/` (the official outputs were not
    touched): `scripts/research_signals.py` SHA-256
    `10173ab9b54674422787249b94dcb8114032db962b68b49027c45d94d6789c76`, `scripts/backtest_rating.py`
    `fcda578de013c51a3e9ec83bcd563fa0ad9ac751961bd396ac392b8ced3bb7eb`, `spxlcast/evaluation.py`
    `141f3e431ec58d4ae21df0dc2c6edffc75b89c4ec1f86a7fa300130abd78d02d`, `spxlcast/config.py`
    `62101dc5ced6ae8253c1959c96e3cfb3a58c9b170a373ca3ec89f371d1fa461b` (byte copies in
    `output/research/repro_check/test/code/`).
  * Every value in `test_results.json` is identical, apart from run metadata (run time, look-log snapshot,
    output paths, `git describe`, and the new code hashes). `tests.csv`, `economic_daily.csv` and
    `economic_holdings.csv` are byte-identical. The comparison is in `output/research/repro_check/comparison.json`.
  * The look log records the unseals: four rows at 19:11:01Z (a first re-run with an intermediate version of the
    code, also identical) and four at 19:13:16Z (this one). The look-ahead review had re-run the test with the
    code of 18:17Z and found the same.

* **N30. The exposure ledger, and what was looked at after unsealing.**
  * Missing from the ledger (PREREGISTRATION.md section 2), found by the look-ahead review: before the plan
    was frozen (16:12Z, a scratch script `recheck_split.py`), six-month SPXL excess returns were computed by
    hurdle tercile on the SPXL-only months since 2009, besides the full-sample hurdle split and the SPXL-only
    drawdown levels that the ledger lists. It bears on C11 only, already labelled "not a clean out-of-sample
    test", and C11 failed. The line is now in `test_results.md` section 9 and `RESULTS.md` section 4.
  * After unsealing, the three reviews recomputed every registered number with their own scripts (in the
    session scratch directory; they imported `scripts/research_signals.py` or called `build_daily`, so they left
    no look-log rows) and ran the analyses that `RESULTS.md` section 3 reports: a persistence-aware Monte Carlo
    null, the episodes left out together, the secondary pass in context, feedback bias, crisis-neutral and
    real-SPXL economics, a joint placebo, bootstrap intervals from 2009, the vol-managed rule against constant
    weights, a US tax simulation, the ICNSA vintage and SPXL-splice checks, and an independent rebuild of the
    predictors and targets.
  * Those analyses are now computed by `report` from the saved outputs (`output/research/exploratory.json`;
    every draw seeded) and reproduce the reviewers' figures. Two are the reviewers' own and are not regenerated:
    the tax simulation (its script was re-run in the scratch directory and reproduced its figures exactly; it
    builds the unsealed daily frame with `build_daily`) and the independent rebuild.
  * None of this changed a registered number, the spec, a sign, a threshold, a rule or the verdict. The
    exploratory results are labelled as such and must not be used to choose anything for 2008-2026.
