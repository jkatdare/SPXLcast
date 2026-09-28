# SPXL timing-signal search: Train phase results

Phase 2 of `research/PREREGISTRATION.md`, run 2026-09-27T17:49:43Z by `py scripts/research_signals.py train`, **before unsealing**. The test period (outcome windows from 2008-01 on) is still sealed.

* **Data read:** `output/research/panel_train.pkl` only for every statistic (month-ends 1990-01-31..2007-12-31; targets only at train origins, windows ending on or before 2007-12-31). `panel_test.pkl` was not opened. Also read: `output/research/panel_meta.json` (the build's checks and the SLOOS QA, dates only), the train rows of the pinned `rating_backtest.csv` through the seal (target reproduction), and file hashes.
* **Frozen spec:** `research/frozen_spec.json`, SHA-256 `c02bbd667e56782a2b4a0e6dcaad93de77a72de4338e048ce32a351047226fed` (`research/FROZEN_HASH.txt`, frozen 2026-09-27T17:48:48Z); byte-identical copy `output/research/train_params.json`; hashes for the code guard in `research/TRAIN_REPORT.md`.
* **What the spec contains from the train period:** only quantities computed from predictor values at the 210 parameter-window month-ends (1990-01-31..2007-06-29). No outcome enters it. The replication statistics below are information only and changed nothing (section 8).

## 1. Checks

| check | result |
|---|---|
| plan files match `PREREG_HASH.txt` | pass |
| pinned inputs (3 files) match their registered SHA-256 | pass |
| `predictor_inputs.pkl` SHA-256 = `133e9d869262...` (DEVIATIONS.md D2) | pass |
| `panel_meta.json` was built from the same plan and `predictor_inputs.pkl` | pass |
| seal on the train panel: no month-end after 2007-12-31; targets only at train origins; every window ends on or before 2007-12-31 | pass |
| train origins per horizon (first, last, count, n_eff) equal the plan's calendar figures | pass |
| target reproduction at train origins against the pinned CSV (tolerance 1e-9) | pass |
| standardisation: no zero sd_j; COMP available at every parameter-window month-end | pass |
| SLOOS lag QA branch (release dates only): 67 observations checked, 0 violations | shift 0 month(s) |
| synthetic fund vs SPXL daily correlation (13.6) | deferred to the Test phase: it needs post-2008 daily data, which the seal withholds |

Train origins: h = 21: 1990-01-31..2007-10-31, 214 origins, n_eff 209.2; h = 63: 1990-01-31..2007-09-28, 213 origins, n_eff 71.7; h = 126: 1990-01-31..2007-06-29, 210 origins, n_eff 35.8.

Provenance at train time (not part of the spec; the script changes again in the Test phase): `output/research/panel_meta.json` `75e3e4dd695e79a96b043cf0b07f50910166cf75820b48c381e3a750d330e550`; `scripts/research_signals.py` `3fdac4879b143a053af9eca0302aaf12b3e8ce70ccc78b9be008c39ef3748692`; `tests/test_research_stats.py` `67eed5783177ed2be75f845a63f11f96b682fc335b0874034aeaa73afba6c2d5`; `git describe` 9b55b30-dirty.

## 2. Train-set quantities (frozen)

Computed from predictor values at the 210 parameter-window month-ends only. `tau` and `tau_j` are the 33.33rd percentile (numpy, linear) of COMP and of `s_j * x_j`.

| id | signal | sign | in COMP | n | mu_j | sd_j | tau_j (on s_j x_j) |
|---|---|---|---|---|---|---|---|
| C01 | VRP | +1 | yes | 210 | 12.4608 | 14.2464 | 6.87869 |
| C02 | GAP | -1 | yes | 210 | 0.0473395 | 0.0413312 | -0.0589448 |
| C03 | SRATE | -1 | yes | 210 | -0.19481 | 1.44246 | -0.296806 |
| C04 | TERM | +1 | yes | 210 | 1.60181 | 1.22669 | 0.829791 |
| C05 | DEF | +1 | yes | 210 | 0.836095 | 0.206895 | 0.7 |
| C06 | INFL | -1 | yes | 210 | 0.028571 | 0.00987566 | -0.0302848 |
| C07 | SLOOS | -1 | yes | 204 | 6.16029 | 21.6643 | -9 |
| C08 | CLAIMS | -1 | yes | 210 | -0.00172127 | 0.135139 | -0.0314394 |
| C09 | ECY | +1 | yes | 210 | 0.0136739 | 0.0103016 | 0.0100264 |
| C10 | IVAR | -1 | no | 210 | 0.0395624 | 0.0278117 | -0.0450867 |
| C11 | HURDLE | -1 | no | 210 | 0.0775734 | 0.0192649 | -0.0850636 |
| C12 | SCORE | +1 | no | 209 | 0.35254 | 0.360534 | 0.180552 |

* **COMP threshold** `tau` = -0.166167 (hold SPXL when COMP >= tau); train median -0.070340 (secondary analysis only). COMP exists at 210 of 210 parameter-window month-ends (members available: 8: 6, 9: 204).
* **Vol-managed rule:** `c` = median IVAR = 0.030836 (VIX 17.56); `w_bar` = 0.797242.
* **SLOOS lag branch:** shift 0 month(s).

## 3. Train-period replication (information only)

Published-sample replication (section 10): the registered statistic on train origins, one-sided in the registered direction, overlap-aware (n_eff = union of the windows / h; Fisher z with Bonett-Wright variance), Holm within each family of 13. The 1990-2007 window is published evidence for several candidates, so this is a replication, not new evidence. COMP's standardisation comes from the same years (no outcome used). **Nothing here may change the spec.** Power: at h = 126 the train n_eff is about 36, so an unadjusted one-sided 5% test needs r >= 0.28 and the first Holm step r >= 0.45.

### 6 months: SPXL excess return Y(t,126) (the registered primary target)

| predictor | signal | n | n_eff | r (signed) | 90% CI | p (1-sided) | Holm p | Holm |
|---|---|---|---|---|---|---|---|---|
| C01 | VRP | 210 | 35.8 | +0.15 | -0.13 to +0.42 | 0.188 | 1.000 | - |
| C02 | GAP | 210 | 35.8 | +0.02 | -0.27 to +0.29 | 0.465 | 1.000 | - |
| C03 | SRATE | 210 | 35.8 | -0.14 | -0.40 to +0.15 | 0.785 | 1.000 | - |
| C04 | TERM | 210 | 35.8 | +0.03 | -0.26 to +0.30 | 0.443 | 1.000 | - |
| C05 | DEF | 210 | 35.8 | -0.16 | -0.42 to +0.13 | 0.821 | 1.000 | - |
| C06 | INFL | 210 | 35.8 | +0.01 | -0.27 to +0.29 | 0.481 | 1.000 | - |
| C07 | SLOOS | 204 | 34.8 | +0.25 | -0.04 to +0.50 | 0.079 | 1.000 | - |
| C08 | CLAIMS | 210 | 35.8 | +0.12 | -0.16 to +0.39 | 0.240 | 1.000 | - |
| C09 | ECY | 210 | 35.8 | +0.15 | -0.13 to +0.42 | 0.190 | 1.000 | - |
| C10 | IVAR | 210 | 35.8 | +0.03 | -0.25 to +0.31 | 0.430 | 1.000 | - |
| C11 | HURDLE | 210 | 35.8 | -0.05 | -0.32 to +0.24 | 0.602 | 1.000 | - |
| C12 | SCORE | 209 | 35.8 | +0.11 | -0.17 to +0.38 | 0.255 | 1.000 | - |
| COMP | COMP | 210 | 35.8 | +0.08 | -0.21 to +0.35 | 0.326 | 1.000 | - |

### 3 months: Y(t,63)

| predictor | signal | n | n_eff | r (signed) | 90% CI | p (1-sided) | Holm p | Holm |
|---|---|---|---|---|---|---|---|---|
| C01 | VRP | 213 | 71.7 | +0.17 | -0.03 to +0.35 | 0.082 | 0.900 | - |
| C02 | GAP | 213 | 71.7 | +0.03 | -0.16 to +0.23 | 0.386 | 1.000 | - |
| C03 | SRATE | 213 | 71.7 | -0.06 | -0.25 to +0.14 | 0.691 | 1.000 | - |
| C04 | TERM | 213 | 71.7 | -0.03 | -0.22 to +0.17 | 0.587 | 1.000 | - |
| C05 | DEF | 213 | 71.7 | -0.05 | -0.24 to +0.15 | 0.650 | 1.000 | - |
| C06 | INFL | 213 | 71.7 | -0.01 | -0.20 to +0.19 | 0.527 | 1.000 | - |
| C07 | SLOOS | 207 | 69.7 | +0.19 | -0.01 to +0.38 | 0.059 | 0.706 | - |
| C08 | CLAIMS | 213 | 71.7 | +0.05 | -0.15 to +0.24 | 0.349 | 1.000 | - |
| C09 | ECY | 213 | 71.7 | +0.20 | +0.00 to +0.38 | 0.049 | 0.641 | - |
| C10 | IVAR | 213 | 71.7 | -0.03 | -0.23 to +0.17 | 0.600 | 1.000 | - |
| C11 | HURDLE | 213 | 71.7 | -0.11 | -0.30 to +0.09 | 0.808 | 1.000 | - |
| C12 | SCORE | 212 | 71.7 | +0.04 | -0.16 to +0.24 | 0.363 | 1.000 | - |
| COMP | COMP | 213 | 71.7 | +0.09 | -0.11 to +0.28 | 0.232 | 1.000 | - |

### 1 month: Y(t,21)

| predictor | signal | n | n_eff | r (signed) | 90% CI | p (1-sided) | Holm p | Holm |
|---|---|---|---|---|---|---|---|---|
| C01 | VRP | 214 | 209.2 | +0.03 | -0.08 to +0.14 | 0.335 | 1.000 | - |
| C02 | GAP | 214 | 209.2 | +0.00 | -0.11 to +0.12 | 0.484 | 1.000 | - |
| C03 | SRATE | 214 | 209.2 | -0.03 | -0.14 to +0.08 | 0.671 | 1.000 | - |
| C04 | TERM | 214 | 209.2 | -0.00 | -0.12 to +0.11 | 0.519 | 1.000 | - |
| C05 | DEF | 214 | 209.2 | -0.05 | -0.16 to +0.07 | 0.758 | 1.000 | - |
| C06 | INFL | 214 | 209.2 | +0.08 | -0.04 to +0.19 | 0.131 | 1.000 | - |
| C07 | SLOOS | 208 | 203.4 | +0.14 | +0.03 to +0.26 | 0.021 | 0.276 | - |
| C08 | CLAIMS | 214 | 209.2 | +0.06 | -0.06 to +0.17 | 0.207 | 1.000 | - |
| C09 | ECY | 214 | 209.2 | +0.09 | -0.02 to +0.20 | 0.099 | 1.000 | - |
| C10 | IVAR | 214 | 209.2 | -0.03 | -0.14 to +0.08 | 0.664 | 1.000 | - |
| C11 | HURDLE | 214 | 209.2 | -0.07 | -0.19 to +0.04 | 0.856 | 1.000 | - |
| C12 | SCORE | 213 | 208.3 | +0.01 | -0.11 to +0.12 | 0.452 | 1.000 | - |
| COMP | COMP | 214 | 209.2 | +0.07 | -0.04 to +0.18 | 0.150 | 1.000 | - |

### Other targets (r, with the one-sided p in brackets; * = Holm pass within the family)

| predictor | Y 6m | Y 3m | Y 1m | S&P 6m | S&P 3m | S&P 1m | fewer 20% dips (-D20) |
|---|---|---|---|---|---|---|---|
| C01 | +0.15 (0.19) | +0.17 (0.08) | +0.03 (0.34) | +0.19 (0.14) | +0.19 (0.06) | +0.04 (0.29) | -0.08 (0.74) |
| C02 | +0.02 (0.47) | +0.03 (0.39) | +0.00 (0.48) | -0.02 (0.55) | +0.00 (0.50) | -0.00 (0.52) | +0.25 (0.02) |
| C03 | -0.14 (0.78) | -0.06 (0.69) | -0.03 (0.67) | -0.13 (0.77) | -0.05 (0.65) | -0.02 (0.63) | -0.11 (0.83) |
| C04 | +0.03 (0.44) | -0.03 (0.59) | -0.00 (0.52) | +0.00 (0.49) | -0.04 (0.64) | -0.01 (0.53) | +0.16 (0.09) |
| C05 | -0.16 (0.82) | -0.05 (0.65) | -0.05 (0.76) | -0.16 (0.82) | -0.04 (0.62) | -0.04 (0.72) | -0.07 (0.72) |
| C06 | +0.01 (0.48) | -0.01 (0.53) | +0.08 (0.13) | +0.05 (0.39) | +0.02 (0.44) | +0.09 (0.11) | -0.02 (0.57) |
| C07 | +0.25 (0.08) | +0.19 (0.06) | +0.14 (0.02) | +0.21 (0.11) | +0.16 (0.10) | +0.13 (0.03) | +0.37 (0.00)* |
| C08 | +0.12 (0.24) | +0.05 (0.35) | +0.06 (0.21) | +0.12 (0.25) | +0.03 (0.39) | +0.05 (0.23) | +0.19 (0.06) |
| C09 | +0.15 (0.19) | +0.20 (0.05) | +0.09 (0.10) | +0.12 (0.25) | +0.17 (0.08) | +0.09 (0.11) | +0.27 (0.01) |
| C10 | +0.03 (0.43) | -0.03 (0.60) | -0.03 (0.66) | -0.02 (0.55) | -0.09 (0.76) | -0.05 (0.76) | +0.31 (0.00)* |
| C11 | -0.05 (0.60) | -0.11 (0.81) | -0.07 (0.86) | -0.09 (0.70) | -0.15 (0.89) | -0.09 (0.89) | +0.26 (0.01) |
| C12 | +0.11 (0.26) | +0.04 (0.36) | +0.01 (0.45) | +0.06 (0.37) | -0.01 (0.54) | -0.01 (0.55) | +0.38 (0.00)* |
| COMP | +0.08 (0.33) | +0.09 (0.23) | +0.07 (0.15) | +0.06 (0.36) | +0.07 (0.27) | +0.07 (0.14) | +0.24 (0.02) |

n_eff by family (all origins): Y 6m 35.8; Y 3m 71.7; Y 1m 209.2; S&P 6m 35.8; S&P 3m 71.7; S&P 1m 209.2; fewer 20% dips (-D20) 71.7.

Holm passes on the train period: C07 (fewer 20% dips (-D20)), C10 (fewer 20% dips (-D20)), C12 (fewer 20% dips (-D20)).

## 4. Descriptives at the parameter-window month-ends (13.9)

| id | signal | n | mean | sd | min | p25 | median | p75 | max | AR(1) |
|---|---|---|---|---|---|---|---|---|---|---|
| C01 | VRP | 210 | 12.46 | 14.25 | -66.01 | 6.046 | 9.63 | 17.66 | 78.73 | 0.19 |
| C02 | GAP | 210 | 0.04734 | 0.04133 | -0.03477 | 0.02283 | 0.04483 | 0.07051 | 0.1358 | 0.99 |
| C03 | SRATE | 210 | -0.1948 | 1.442 | -4.44 | -0.8375 | -0.22 | 0.9325 | 2.97 | 0.98 |
| C04 | TERM | 210 | 1.602 | 1.227 | -0.74 | 0.57 | 1.445 | 2.742 | 3.8 | 0.97 |
| C05 | DEF | 210 | 0.8361 | 0.2069 | 0.54 | 0.6725 | 0.815 | 0.93 | 1.44 | 0.94 |
| C06 | INFL | 210 | 0.02857 | 0.009876 | 0.01062 | 0.02206 | 0.02765 | 0.03204 | 0.061 | 0.95 |
| C07 | SLOOS | 204 | 6.16 | 21.66 | -24.1 | -7.825 | 0 | 16.62 | 59.7 | 0.97 |
| C08 | CLAIMS | 210 | -0.001721 | 0.1351 | -0.3436 | -0.08316 | -0.02332 | 0.06835 | 0.492 | 0.87 |
| C09 | ECY | 210 | 0.01367 | 0.0103 | -0.01373 | 0.007217 | 0.01572 | 0.02041 | 0.03499 | 0.96 |
| C10 | IVAR | 210 | 0.03956 | 0.02781 | 0.00988 | 0.01747 | 0.03084 | 0.05177 | 0.1568 | 0.81 |
| C11 | HURDLE | 210 | 0.07757 | 0.01926 | 0.04489 | 0.06239 | 0.07579 | 0.0901 | 0.1549 | 0.89 |
| C12 | SCORE | 209 | 0.3525 | 0.3605 | -0.4804 | 0.09681 | 0.3205 | 0.6245 | 1 | 0.95 |
| COMP | COMP | 210 | 0.0006572 | 0.4274 | -1.102 | -0.2086 | -0.07034 | 0.3982 | 0.8553 | 0.94 |

Time-series plots: `output/research/train/predictors_train.png` (git-ignored).

## 5. Coverage at train origins (13.7)

| id | signal | first value | param window | train h=126 | train h=63 | train h=21 |
|---|---|---|---|---|---|---|
| C01 | VRP | 1990-01-31 | 210/210 | 210/210 | 213/213 | 214/214 |
| C02 | GAP | 1990-01-31 | 210/210 | 210/210 | 213/213 | 214/214 |
| C03 | SRATE | 1990-01-31 | 210/210 | 210/210 | 213/213 | 214/214 |
| C04 | TERM | 1990-01-31 | 210/210 | 210/210 | 213/213 | 214/214 |
| C05 | DEF | 1990-01-31 | 210/210 | 210/210 | 213/213 | 214/214 |
| C06 | INFL | 1990-01-31 | 210/210 | 210/210 | 213/213 | 214/214 |
| C07 | SLOOS | 1990-07-31 | 204/210 | 204/210 | 207/213 | 208/214 |
| C08 | CLAIMS | 1990-01-31 | 210/210 | 210/210 | 213/213 | 214/214 |
| C09 | ECY | 1990-01-31 | 210/210 | 210/210 | 213/213 | 214/214 |
| C10 | IVAR | 1990-01-31 | 210/210 | 210/210 | 213/213 | 214/214 |
| C11 | HURDLE | 1990-01-31 | 210/210 | 210/210 | 213/213 | 214/214 |
| C12 | SCORE | 1990-01-31 | 209/210 | 209/210 | 212/213 | 213/214 |
| COMP | COMP | 1990-01-31 | 210/210 | 210/210 | 213/213 | 214/214 |

C07 starts at 1990-07-31 (first SLOOS observation usable); C12 is missing at 1996-01-31 (DEVIATIONS.md D5).

## 6. Collinearity: Spearman matrix of the raw values at the parameter-window month-ends (13.11)

|  | C01 | C02 | C03 | C04 | C05 | C06 | C07 | C08 | C09 | C10 | C11 | C12 | COMP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | +1.00 | +0.13 | -0.26 | -0.00 | -0.00 | -0.19 | +0.28 | +0.14 | -0.17 | +0.54 | +0.41 | -0.39 | +0.11 |
| C02 | +0.13 | +1.00 | +0.47 | -0.70 | -0.38 | -0.20 | -0.07 | -0.04 | -0.75 | +0.31 | +0.47 | -0.57 | -0.71 |
| C03 | -0.26 | +0.47 | +1.00 | -0.46 | -0.43 | +0.07 | -0.70 | -0.31 | -0.21 | -0.39 | -0.11 | +0.14 | -0.47 |
| C04 | -0.00 | -0.70 | -0.46 | +1.00 | +0.14 | -0.13 | +0.11 | -0.22 | +0.41 | -0.08 | -0.54 | +0.56 | +0.77 |
| C05 | -0.00 | -0.38 | -0.43 | +0.14 | +1.00 | +0.10 | +0.49 | +0.29 | +0.42 | +0.23 | -0.02 | -0.18 | +0.34 |
| C06 | -0.19 | -0.20 | +0.07 | -0.13 | +0.10 | +1.00 | +0.02 | +0.19 | +0.21 | -0.23 | +0.08 | +0.09 | -0.28 |
| C07 | +0.28 | -0.07 | -0.70 | +0.11 | +0.49 | +0.02 | +1.00 | +0.45 | -0.13 | +0.65 | +0.46 | -0.57 | +0.01 |
| C08 | +0.14 | -0.04 | -0.31 | -0.22 | +0.29 | +0.19 | +0.45 | +1.00 | -0.03 | +0.24 | +0.37 | -0.36 | -0.31 |
| C09 | -0.17 | -0.75 | -0.21 | +0.41 | +0.42 | +0.21 | -0.13 | -0.03 | +1.00 | -0.28 | -0.46 | +0.52 | +0.63 |
| C10 | +0.54 | +0.31 | -0.39 | -0.08 | +0.23 | -0.23 | +0.65 | +0.24 | -0.28 | +1.00 | +0.71 | -0.80 | -0.06 |
| C11 | +0.41 | +0.47 | -0.11 | -0.54 | -0.02 | +0.08 | +0.46 | +0.37 | -0.46 | +0.71 | +1.00 | -0.89 | -0.51 |
| C12 | -0.39 | -0.57 | +0.14 | +0.56 | -0.18 | +0.09 | -0.57 | -0.36 | +0.52 | -0.80 | -0.89 | +1.00 | +0.47 |
| COMP | +0.11 | -0.71 | -0.47 | +0.77 | +0.34 | -0.28 | +0.01 | -0.31 | +0.63 | -0.06 | -0.51 | +0.47 | +1.00 |

## 7. TA diagnostic (13.10; for the owner's judgement only, never used)

Spearman of each signed candidate `s_j * x_j` (and COMP) with the trailing S&P 500 price return up to t- (from `gspc_tm`), parameter-window month-ends. A large positive value would mean the 'favourable' reading tends to follow a rising market.

| id | signal | trailing 1m | trailing 3m | trailing 6m | trailing 12m |
|---|---|---|---|---|---|
| C01 | VRP | -0.17 (n 209) | +0.07 (n 207) | +0.03 (n 204) | +0.20 (n 198) |
| C02 | GAP | -0.01 (n 209) | +0.00 (n 207) | -0.05 (n 204) | -0.22 (n 198) |
| C03 | SRATE | -0.02 (n 209) | -0.04 (n 207) | -0.09 (n 204) | -0.05 (n 198) |
| C04 | TERM | -0.09 (n 209) | -0.14 (n 207) | -0.16 (n 204) | -0.27 (n 198) |
| C05 | DEF | -0.07 (n 209) | -0.12 (n 207) | -0.33 (n 204) | -0.39 (n 198) |
| C06 | INFL | +0.04 (n 209) | +0.06 (n 207) | +0.13 (n 204) | +0.18 (n 198) |
| C07 | SLOOS | +0.09 (n 204) | +0.11 (n 204) | +0.19 (n 204) | +0.21 (n 198) |
| C08 | CLAIMS | +0.04 (n 209) | +0.06 (n 207) | +0.12 (n 204) | +0.22 (n 198) |
| C09 | ECY | -0.03 (n 209) | -0.09 (n 207) | -0.22 (n 204) | -0.28 (n 198) |
| C10 | IVAR | +0.17 (n 209) | +0.23 (n 207) | +0.23 (n 204) | +0.07 (n 198) |
| C11 | HURDLE | +0.05 (n 209) | +0.08 (n 207) | +0.05 (n 204) | -0.14 (n 198) |
| C12 | SCORE | +0.06 (n 208) | +0.11 (n 206) | +0.11 (n 203) | -0.03 (n 197) |
| COMP | COMP | -0.06 (n 209) | -0.07 (n 207) | -0.14 (n 204) | -0.14 (n 198) |

## 8. What has now been seen

* Everything in the exposure ledger (PREREGISTRATION.md section 2) and the Build-phase notes (DEVIATIONS.md N11-N12).
* Now also: every train-period statistic above (predictor distributions, collinearity, the TA diagnostic and the replication of all 13 predictors against all seven targets on 1990-2007 origins). These are train-period results; the spec had been fully determined by the plan before they were computed and none of them changed it.
* Not seen: any test-period outcome, any statistic using an outcome window that ends after 2007-12-31, any summary of a test-period predictor value (panel_test.pkl was not opened).

## 9. Look log

| UTC | phase | command | git |
|---|---|---|---|
| 2026-09-27T17:03:57Z | build | `py scripts/research_signals.py build` | 9b55b30-dirty |
| 2026-09-27T17:05:23Z | build | `py scripts/research_signals.py build --offline` | 9b55b30-dirty |
| 2026-09-27T17:10:05Z | build | `py scripts/research_signals.py build --offline` | 9b55b30-dirty |
| 2026-09-27T17:31:52Z | build | `py scripts/research_signals.py build` | 9b55b30-dirty |
| 2026-09-27T17:48:48Z | train | `py scripts/research_signals.py train` | 9b55b30-dirty |
| 2026-09-27T17:49:43Z | train | `py scripts/research_signals.py train` | 9b55b30-dirty |

Build runs computed no predictive statistic. Train runs computed train-period statistics only.
