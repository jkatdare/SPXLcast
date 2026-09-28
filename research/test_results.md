# SPXL timing-signal search: Test phase results

Phase 3 of `research/PREREGISTRATION.md`, run once at 2026-09-27T18:14:50Z by `py scripts/research_signals.py test --unseal <SHA-256 of research/PREREGISTRATION.md>`. Every number the test uses was taken from the frozen spec `research/frozen_spec.json` (SHA-256 `c02bbd667e56782a2b4a0e6dcaad93de77a72de4338e048ce32a351047226fed`, frozen 2026-09-27T17:48:48Z, before unsealing). Nothing in the spec, the candidates, the thresholds or the decision rule was changed after the outcomes were computed. Every registered test is reported below, whatever it shows.

## Verdict

**No timing signal found**

* **Primary test** (6-month SPXL excess return, 13 predictors, Holm at 5%): 0 of 13 passed. The largest signed rank correlation was C06 INFL at r = +0.37 (one-sided p = 0.0152, Holm p = 0.197); the first Holm step needs p <= 0.0038, about r >= 0.45 at the registered n_eff = 37.1. That n_eff counts non-overlapping outcome windows and ignores how persistent each predictor is, so the bar is conservative (a persistence-aware check is in `research/RESULTS.md`, section 3.1; it is exploratory).
* **Economic test** (composite rule vs buy-and-hold SPXL and MIX_w): FAIL (E1 FAIL, E2 FAIL, E3 FAIL, E4 FAIL). CAGR +11.8% vs +17.9% buy-and-hold and +17.6% MIX_w; Sharpe +0.46 vs +0.55 and +0.55; max drawdown 93.5% vs 93.5%.
* Section 12 reads: primary Holm pass and economic pass: 'Timing signal confirmed'; primary pass, economic fail: 'Predictive, not usable by the registered rule'; composite economic pass with no primary pass: 'Not confirmed (gain could be luck); candidate for a new pre-registration on future data only'; neither: 'No timing signal found'.

## 1. Primary test: 13 predictors against Y(t,126), Holm at 5% (decisive)

Test origins 2008-01-31..2026-02-27 (218 month-ends), r = Spearman of the signed predictor with the SPXL excess return over T-bills in the next 126 sessions; n_eff = the overlap-aware count of independent windows; z = atanh(r) sqrt(n_eff - 3) / sqrt(1 + r^2/2); p one-sided (favourable direction registered before 2008); 90% interval as `backtest_rating._corr_ci`. Train (1990-2007) values are the published-sample replication from the Train phase. Holm: the smallest p must be at most 0.05/13 = 0.0038, the next at most 0.05/12 = 0.0042, and so on; testing stops at the first failure.

| predictor | signal | n | n_eff | r | 90% CI | z | p (1-sided) | Holm step (alpha) | Holm p | result | train r (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | VRP | 218 | 37.1 | +0.15 | -0.13 to +0.41 | +0.88 | 0.1896 | 5 (0.0056) | 1.000 | fail | +0.15 (0.19) |
| C02 | GAP | 218 | 37.1 | +0.36 | +0.08 to +0.58 | +2.11 | 0.0174 | 2 (0.0042) | 0.209 | fail | +0.02 (0.47) |
| C03 | SRATE | 218 | 37.1 | +0.01 | -0.26 to +0.29 | +0.07 | 0.4733 | 7 (0.0071) | 1.000 | fail | -0.14 (0.78) |
| C04 | TERM | 218 | 37.1 | -0.08 | -0.35 to +0.20 | -0.47 | 0.6822 | 8 (0.0083) | 1.000 | fail | +0.03 (0.44) |
| C05 | DEF | 218 | 37.1 | +0.10 | -0.18 to +0.37 | +0.60 | 0.2749 | 6 (0.0063) | 1.000 | fail | -0.16 (0.82) |
| C06 | INFL | 218 | 37.1 | +0.37 | +0.09 to +0.59 | +2.17 | 0.0152 | 1 (0.0038) | 0.197 | fail | +0.01 (0.48) |
| C07 | SLOOS | 218 | 37.1 | -0.21 | -0.46 to +0.07 | -1.23 | 0.8913 | 12 (0.0250) | 1.000 | fail | +0.25 (0.08) |
| C08 | CLAIMS | 218 | 37.1 | -0.23 | -0.48 to +0.05 | -1.33 | 0.9087 | 13 (0.0500) | 1.000 | fail | +0.12 (0.24) |
| C09 | ECY | 218 | 37.1 | +0.26 | -0.02 to +0.50 | +1.50 | 0.0663 | 3 (0.0045) | 0.730 | fail | +0.15 (0.19) |
| C10 | IVAR | 218 | 37.1 | -0.16 | -0.42 to +0.12 | -0.93 | 0.8249 | 10 (0.0125) | 1.000 | fail | +0.03 (0.43) |
| C11 | HURDLE | 218 | 37.1 | -0.18 | -0.44 to +0.10 | -1.06 | 0.8559 | 11 (0.0167) | 1.000 | fail | -0.05 (0.60) |
| C12 | SCORE | 217 | 37.1 | -0.14 | -0.40 to +0.14 | -0.83 | 0.7970 | 9 (0.0100) | 1.000 | fail | +0.11 (0.26) |
| COMP | COMP | 218 | 37.1 | +0.15 | -0.13 to +0.41 | +0.90 | 0.1843 | 4 (0.0050) | 1.000 | fail | +0.08 (0.33) |

## 2. Economic test: the composite rule (decisive)

At each origin 2008-01-31..2026-07-31 (223 monthly holdings, sessions 2008-02-01..2026-08-31, N = 4674): SPXL (the realised fund) if COMP >= tau = -0.166167, else T-bills; 0.10% per switch on the first session of the new holding. MIX_w holds w = 0.8879 (the rule's share of months in SPXL) in SPXL, rebalanced monthly at 0.10% x |weight traded|. Missing signals: 0. All figures are before tax (as in a tax-deferred account). Before 2009-01-02 the realised fund is the synthetic 3x fund (SPXL itself started trading on 2008-11-05).

| strategy | CAGR | Sharpe | max drawdown | vol | $1 becomes | Sharpe 2008-02..2016-12 | Sharpe 2017-01..2026-08 | SPXL weight (avg) | switches / rebalances | turnover | costs paid ($ per $1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| composite rule | +11.8% | +0.46 | 93.5% | 57% | 7.94 | +0.42 | +0.51 | 89% | 6 | 6.00 | 0.0330 |
| buy and hold SPXL | +17.9% | +0.55 | 93.5% | 59% | 21.33 | +0.42 | +0.70 | 100% | 0 | 0.00 | 0.0000 |
| MIX_w | +17.6% | +0.55 | 90.6% | 51% | 20.07 | +0.41 | +0.70 | 89% | 222 | 2.49 | 0.0089 |

| criterion | rule | values | result |
|---|---|---|---|
| E1 | CAGR(rule) > CAGR(BH) and > CAGR(MIX_w) | +11.8% vs +17.9% / +17.6% | FAIL |
| E2 | Sharpe(rule) > Sharpe(BH) and > Sharpe(MIX_w) | +0.46 vs +0.55 / +0.55 | FAIL |
| E3 | max drawdown(rule) < max drawdown(BH) | 93.5% vs 93.5% | FAIL |
| E4 | Sharpe(rule) > Sharpe(BH) in both halves | +0.42 vs +0.42; +0.51 vs +0.70 | FAIL |

**Economic pass (all of E1-E4): no.**

Months in T-bills under the composite rule (origin months): 2022-10..2024-08 (23); 2025-04 (1); 2026-06 (1). Plot: `output/research/test/economic_test.png`.

## 3. Secondary tests F2-F7 (reported, never decisive)

Same predictors, statistic and one-sided direction; Holm within each family of 13. Cells: r (one-sided p); `*` = Holm pass within the family. Within a family the smallest p must be at most 0.0038 (0.05/13), the next at most 0.0042, and so on.

| predictor | signal | Y 1m | Y 3m | S&P 6m | S&P 3m | S&P 1m | fewer 20% dips (-D20, 3m) |
|---|---|---|---|---|---|---|---|
| C01 | VRP | +0.18 (0.0045) | +0.22 (0.0294) | +0.16 (0.1741) | +0.24 (0.0208) | +0.19 (0.0032)* | +0.04 (0.3613) |
| C02 | GAP | +0.12 (0.0351) | +0.29 (0.0065) | +0.36 (0.0172) | +0.29 (0.0065) | +0.12 (0.0336) | +0.11 (0.1774) |
| C03 | SRATE | +0.02 (0.3934) | +0.05 (0.3424) | +0.06 (0.3723) | +0.07 (0.2725) | +0.03 (0.3180) | -0.13 (0.8627) |
| C04 | TERM | -0.02 (0.5883) | -0.04 (0.6394) | -0.09 (0.7090) | -0.05 (0.6678) | -0.01 (0.5390) | -0.10 (0.7933) |
| C05 | DEF | +0.05 (0.2165) | +0.08 (0.2594) | +0.11 (0.2596) | +0.09 (0.2147) | +0.07 (0.1561) | -0.12 (0.8534) |
| C06 | INFL | +0.17 (0.0069) | +0.30 (0.0048) | +0.35 (0.0197) | +0.29 (0.0064) | +0.16 (0.0078) | +0.29 (0.0071) |
| C07 | SLOOS | -0.04 (0.7239) | -0.13 (0.8557) | -0.22 (0.9054) | -0.13 (0.8675) | -0.05 (0.7597) | -0.02 (0.5748) |
| C08 | CLAIMS | -0.06 (0.8200) | -0.17 (0.9211) | -0.25 (0.9274) | -0.18 (0.9339) | -0.07 (0.8470) | -0.02 (0.5672) |
| C09 | ECY | +0.10 (0.0639) | +0.20 (0.0459) | +0.27 (0.0550) | +0.21 (0.0347) | +0.11 (0.0492) | +0.06 (0.2995) |
| C10 | IVAR | -0.14 (0.9786) | -0.20 (0.9544) | -0.20 (0.8759) | -0.24 (0.9809) | -0.16 (0.9916) | +0.19 (0.0559) |
| C11 | HURDLE | -0.07 (0.8633) | -0.17 (0.9247) | -0.21 (0.8943) | -0.20 (0.9583) | -0.09 (0.9109) | +0.15 (0.1064) |
| C12 | SCORE | -0.06 (0.8133) | -0.13 (0.8705) | -0.17 (0.8404) | -0.16 (0.9141) | -0.07 (0.8615) | +0.12 (0.1609) |
| COMP | COMP | +0.14 (0.0224) | +0.17 (0.0714) | +0.15 (0.1891) | +0.18 (0.0664) | +0.14 (0.0181) | +0.07 (0.2642) |

Origins and n_eff: Y 1m 223 origins, n_eff 218.2; Y 3m 221 origins, n_eff 74.2; S&P 6m 218 origins, n_eff 37.1; S&P 3m 221 origins, n_eff 74.2; S&P 1m 223 origins, n_eff 218.2; fewer 20% dips (-D20, 3m) 221 origins, n_eff 74.2.

Holm passes in the secondary families: C01 (S&P 1m).

With one Holm correction across all 91 registered tests (the seven families together): 0 passes; the smallest adjusted p is 0.29 (C01, S&P 1m). This pooled view is not registered; it is shown for scale.

## 4. Train replication beside the test period

Signed Spearman r, train (1990-2007 origins, published-sample replication) -> test (2008-2026 origins). `*` = Holm pass within that family and period.

| predictor | Y 6m | Y 1m | Y 3m | S&P 6m | S&P 3m | S&P 1m | fewer 20% dips (-D20, 3m) |
|---|---|---|---|---|---|---|---|
| C01 | +0.15 -> +0.15 | +0.03 -> +0.18 | +0.17 -> +0.22 | +0.19 -> +0.16 | +0.19 -> +0.24 | +0.04 -> +0.19* | -0.08 -> +0.04 |
| C02 | +0.02 -> +0.36 | +0.00 -> +0.12 | +0.03 -> +0.29 | -0.02 -> +0.36 | +0.00 -> +0.29 | -0.00 -> +0.12 | +0.25 -> +0.11 |
| C03 | -0.14 -> +0.01 | -0.03 -> +0.02 | -0.06 -> +0.05 | -0.13 -> +0.06 | -0.05 -> +0.07 | -0.02 -> +0.03 | -0.11 -> -0.13 |
| C04 | +0.03 -> -0.08 | -0.00 -> -0.02 | -0.03 -> -0.04 | +0.00 -> -0.09 | -0.04 -> -0.05 | -0.01 -> -0.01 | +0.16 -> -0.10 |
| C05 | -0.16 -> +0.10 | -0.05 -> +0.05 | -0.05 -> +0.08 | -0.16 -> +0.11 | -0.04 -> +0.09 | -0.04 -> +0.07 | -0.07 -> -0.12 |
| C06 | +0.01 -> +0.37 | +0.08 -> +0.17 | -0.01 -> +0.30 | +0.05 -> +0.35 | +0.02 -> +0.29 | +0.09 -> +0.16 | -0.02 -> +0.29 |
| C07 | +0.25 -> -0.21 | +0.14 -> -0.04 | +0.19 -> -0.13 | +0.21 -> -0.22 | +0.16 -> -0.13 | +0.13 -> -0.05 | +0.37* -> -0.02 |
| C08 | +0.12 -> -0.23 | +0.06 -> -0.06 | +0.05 -> -0.17 | +0.12 -> -0.25 | +0.03 -> -0.18 | +0.05 -> -0.07 | +0.19 -> -0.02 |
| C09 | +0.15 -> +0.26 | +0.09 -> +0.10 | +0.20 -> +0.20 | +0.12 -> +0.27 | +0.17 -> +0.21 | +0.09 -> +0.11 | +0.27 -> +0.06 |
| C10 | +0.03 -> -0.16 | -0.03 -> -0.14 | -0.03 -> -0.20 | -0.02 -> -0.20 | -0.09 -> -0.24 | -0.05 -> -0.16 | +0.31* -> +0.19 |
| C11 | -0.05 -> -0.18 | -0.07 -> -0.07 | -0.11 -> -0.17 | -0.09 -> -0.21 | -0.15 -> -0.20 | -0.09 -> -0.09 | +0.26 -> +0.15 |
| C12 | +0.11 -> -0.14 | +0.01 -> -0.06 | +0.04 -> -0.13 | +0.06 -> -0.17 | -0.01 -> -0.16 | -0.01 -> -0.07 | +0.38* -> +0.12 |
| COMP | +0.08 -> +0.15 | +0.07 -> +0.14 | +0.09 -> +0.17 | +0.06 -> +0.15 | +0.07 -> +0.18 | +0.07 -> +0.14 | +0.24 -> +0.07 |

## 5. Secondary economic analyses (descriptive, never decisive)

### 5.1 The same rule for every candidate, with its own tau_j

Hold SPXL when s_j x_j >= tau_j (the 33.33rd percentile of s_j x_j over 1990-01..2007-06), else T-bills; each against BH and its own MIX_w. Decisive only for a candidate that passed the primary test.

| rule | signal | threshold | in SPXL | switches | CAGR | MIX_w CAGR | Sharpe | MIX_w Sharpe | max DD | E1-E4 | all four | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C01 | VRP | 6.879 | 61% | 110 | +19.8% | +14.7% | +0.60 | +0.53 | 77% | E1+ E2+ E3+ E4- | no | 0 |
| C02 | GAP | -0.05894 | 100% | 0 | +17.9% | +17.9% | +0.55 | +0.55 | 93% | E1- E2- E3- E4- | no | 0 |
| C03 | SRATE | -0.2968 | 75% | 10 | +13.9% | +16.5% | +0.49 | +0.54 | 93% | E1- E2- E3- E4- | no | 0 |
| C04 | TERM | 0.8298 | 66% | 6 | +8.6% | +15.5% | +0.38 | +0.54 | 93% | E1- E2- E3- E4- | no | 0 |
| C05 | DEF | 0.7 | 78% | 21 | +13.3% | +16.8% | +0.48 | +0.54 | 93% | E1- E2- E3- E4- | no | 0 |
| C06 | INFL | -0.03028 | 73% | 8 | +27.6% | +16.3% | +0.74 | +0.54 | 77% | E1+ E2+ E3+ E4- | no | 0 |
| C07 | SLOOS | -9 | 69% | 9 | +14.4% | +15.8% | +0.50 | +0.54 | 77% | E1- E2- E3+ E4- | no | 0 |
| C08 | CLAIMS | -0.03144 | 78% | 19 | +19.2% | +16.7% | +0.59 | +0.54 | 79% | E1+ E2+ E3+ E4- | no | 0 |
| C09 | ECY | 0.01003 | 100% | 0 | +17.9% | +17.9% | +0.55 | +0.55 | 93% | E1- E2- E3- E4- | no | 0 |
| C10 | IVAR | -0.04509 | 70% | 43 | +10.1% | +15.9% | +0.41 | +0.54 | 59% | E1- E2- E3+ E4- | no | 0 |
| C11 | HURDLE | -0.08506 | 91% | 22 | +7.1% | +17.6% | +0.37 | +0.55 | 84% | E1- E2- E3+ E4- | no | 0 |
| C12 | SCORE | 0.1806 | 69% | 20 | +3.9% | +15.8% | +0.28 | +0.54 | 77% | E1- E2- E3+ E4- | no | 1 |
| COMP | COMP | -0.1662 | 89% | 6 | +11.8% | +17.6% | +0.46 | +0.55 | 93% | E1- E2- E3- E4- | no | 0 |

Buy and hold SPXL over the same sessions: CAGR +17.9%, Sharpe +0.55, max drawdown 93%.

Rules that never left SPXL, because the predictor never crossed its 1990-2007 cut-off in 2008-2026 (so the rule equals buy-and-hold and fails E1 by construction): C02 GAP, C09 ECY.

### 5.2 Composite rule: other switching costs and the median threshold

| variant | in SPXL | CAGR | MIX_w CAGR | Sharpe | MIX_w Sharpe | max DD | E1-E4 |
|---|---|---|---|---|---|---|---|
| cost 0.00% | 89% | +11.9% | +17.6% | +0.46 | +0.55 | 93% | E1- E2- E3- E4- |
| cost 0.25% | 89% | +11.8% | +17.5% | +0.46 | +0.55 | 93% | E1- E2- E3- E4- |
| threshold = train median -0.0703 (cost 0.10%) | 84% | +17.1% | +17.3% | +0.54 | +0.55 | 87% | E1- E2- E3+ E4- |

### 5.3 Stationary bootstrap of the composite rule against buy and hold

Monthly holding-period returns of the rule and of BH (costs included) with the monthly T-bill returns, resampled as aligned triples (Politis-Romano, circular, mean block 6 months, 10000 resamples, seed 20260927); monthly Sharpe x sqrt(12), CAGR over 223 months.

| difference (rule - BH) | estimate | 90% interval |
|---|---|---|
| Sharpe (monthly, annualised) | -0.11 | -0.24 to -0.02 |
| CAGR | -6.1% | -13.2% to -1.0% |

The rule differs from buy-and-hold only in its 25 T-bill months (of 223), so these intervals rest on those months.

### 5.4 Volatility-managed exposure (not a clean test: uses the VIX level)

SPXL weight min(1, c / IVAR_t) with c = 0.030836 (VIX 17.56), rest T-bills, monthly, 0.10% x |weight traded|; against BH and a constant w_bar = 0.7972. Average target weight 0.80; missing IVAR: 0.

| strategy | CAGR | Sharpe | max drawdown | vol | Sharpe 1st half | Sharpe 2nd half | turnover |
|---|---|---|---|---|---|---|---|
| vol-managed | +16.6% | +0.57 | 56.4% | 36% | +0.50 | +0.64 | 24.11 |
| constant w_bar | +16.9% | +0.54 | 87.4% | 45% | +0.40 | +0.70 | 4.02 |
| buy and hold SPXL | +17.9% | +0.55 | 93.5% | 59% | +0.42 | +0.70 | 0.00 |

## 6. Robustness of the primary results (reported; labels apply to primary passes)

Halves: origins 2008-01..2016-12 and 2017-01..2026-02. Episodes left out in turn: 2008-01..2009-06 (financial crisis), 2019-09..2020-06 (2020 crash), 2021-07..2022-12 (inflation and rate shock). Block bootstrap: 90% interval for r from blocks of 12 origins (10,000 resamples, seed 20260927). Partial r: controls ln VIX(t-) and DGS3MO(t-), p with n_eff - 5. Labels: 'episode-dependent' if p > 0.10 with any one episode left out; 'one half only' if r is not positive in both halves. The block-bootstrap intervals are descriptive: they are not corrected for testing 13 predictors and are not a significance test (the test is section 1).

| predictor | r | r halves | r (p) without crisis / 2020 / 2021-22 | block bootstrap 90% (unadjusted) | partial r (p) | labels (they count only for a primary pass) |
|---|---|---|---|---|---|---|
| C01 | +0.15 | +0.28 / +0.01 | +0.07 (0.35) / +0.17 (0.16) / +0.18 (0.15) | -0.02 to +0.31 | +0.11 (0.27) | episode-dependent |
| C02 | +0.36 | +0.39 / +0.31 | +0.25 (0.08) / +0.32 (0.03) / +0.39 (0.01) | +0.13 to +0.48 | +0.33 (0.03) | - |
| C03 | +0.01 | -0.23 / +0.20 | +0.11 (0.27) / +0.01 (0.48) / -0.02 (0.55) | -0.17 to +0.31 | -0.07 (0.66) | episode-dependent, one half only |
| C04 | -0.08 | -0.11 / -0.16 | -0.08 (0.67) / -0.10 (0.72) / -0.09 (0.69) | -0.24 to +0.11 | -0.22 (0.89) | episode-dependent, one half only |
| C05 | +0.10 | +0.05 / +0.17 | +0.19 (0.15) / +0.06 (0.36) / +0.07 (0.34) | -0.02 to +0.36 | +0.02 (0.46) | episode-dependent |
| C06 | +0.37 | +0.37 / +0.34 | +0.25 (0.08) / +0.37 (0.01) / +0.29 (0.05) | +0.07 to +0.54 | +0.40 (0.01) | - |
| C07 | -0.21 | +0.07 / -0.52 | -0.30 (0.95) / -0.22 (0.90) / -0.13 (0.77) | -0.48 to -0.03 | -0.24 (0.91) | episode-dependent, one half only |
| C08 | -0.23 | -0.05 / -0.42 | -0.32 (0.96) / -0.23 (0.91) / -0.13 (0.77) | -0.50 to -0.05 | -0.25 (0.92) | episode-dependent, one half only |
| C09 | +0.26 | +0.58 / +0.07 | +0.27 (0.07) / +0.24 (0.09) / +0.27 (0.06) | +0.08 to +0.48 | +0.26 (0.07) | - |
| C10 | -0.16 | -0.12 / -0.21 | -0.21 (0.88) / -0.10 (0.72) / -0.24 (0.91) | -0.43 to +0.01 | -0.00 (0.50) | episode-dependent, one half only |
| C11 | -0.18 | -0.10 / -0.31 | -0.20 (0.86) / -0.14 (0.79) / -0.24 (0.92) | -0.43 to +0.03 | -0.21 (0.89) | episode-dependent, one half only |
| C12 | -0.14 | -0.05 / -0.29 | -0.15 (0.80) / -0.11 (0.73) / -0.19 (0.85) | -0.38 to +0.08 | -0.15 (0.80) | episode-dependent, one half only |
| COMP | +0.15 | +0.38 / -0.05 | +0.08 (0.34) / +0.14 (0.21) / +0.15 (0.20) | -0.04 to +0.34 | +0.12 (0.24) | episode-dependent, one half only |

* **Composite with expanding-window standardisation** (1990-01 to t, at least 60 values per member): r = +0.07 (n 218, n_eff 37.1, 90% CI -0.21 to +0.34, one-sided p 0.338), against r = +0.15 with the train-fixed standardisation.
* **Synthetic fund spread 0.25%** (registered 0.75%; changes the 11 origins 2008-01-31..2008-11-28 whose window starts before 2009-01-02): C01 +0.15 (0.19), C02 +0.36 (0.02), C03 +0.01 (0.47), C04 -0.08 (0.68), C05 +0.10 (0.27), C06 +0.37 (0.02), C07 -0.21 (0.89), C08 -0.23 (0.91), C09 +0.26 (0.07), C10 -0.16 (0.82), C11 -0.18 (0.86), C12 -0.14 (0.80), COMP +0.15 (0.18); Holm passes: none.
* **Synthetic fund spread 1.25%** (registered 0.75%; changes the 11 origins 2008-01-31..2008-11-28 whose window starts before 2009-01-02): C01 +0.15 (0.19), C02 +0.36 (0.02), C03 +0.01 (0.47), C04 -0.08 (0.68), C05 +0.10 (0.27), C06 +0.37 (0.02), C07 -0.21 (0.89), C08 -0.23 (0.91), C09 +0.26 (0.07), C10 -0.16 (0.82), C11 -0.18 (0.86), C12 -0.14 (0.80), COMP +0.15 (0.18); Holm passes: none.

## 7. Sanity checks (Test phase)

| check | result |
|---|---|
| plan files match `PREREG_HASH.txt`; unseal token = SHA-256 of `PREREGISTRATION.md` | pass |
| pinned inputs (3 files) match their registered SHA-256 | pass |
| `predictor_inputs.pkl` = `133e9d869262...` (DEVIATIONS.md D2, TRAIN_REPORT.md) | pass |
| `frozen_spec.json` = `c02bbd667e56...` (FROZEN_HASH.txt), byte-identical to `train_params.json`, whose hash TRAIN_REPORT.md records | pass |
| `panel_train.pkl` = the hash in the frozen spec; `panel_meta.json` built from this plan and these inputs | pass |
| the frozen spec's registered choices are the ones the code implements | pass |
| section 8 quantities recomputed from `panel_train.pkl` equal the frozen ones (max abs diff 0.0e+00) | pass |
| `panel_test.pkl` sealed (targets all missing), on the calendar, flags and missing counts as built (224 month-ends 2008-01-31..2026-08-31) | pass |
| every stored predictor and component equals a fresh rebuild from the pinned inputs before unsealing (14520 cells, 0 mismatches) | pass |
| train-origin targets unchanged by unsealing (sealed frame = prefix of the full frame) | pass |
| target reproduction at all origins: Y, S, D20 = pinned CSV excess_h, sp_h - tbill_h, real_dd20_63 (tol 1e-9; Y21 439, Y63 437, Y126 434, S21 439, S63 437, S126 434, D20 437 origins; max abs diff 0.0e+00) | pass |
| legacy reproduction: Spearman(score, Y(t,126)) over 434 month-ends 1990-01-31..2026-02-27 = -0.0293 (registered -0.03 +/- 0.01) | pass |
| synthetic fund vs SPXL daily correlation from 2009-01-02 = 0.9981 over 4460 sessions (>= 0.99); synthetic growth +30.9% vs SPXL +31.3% a year | pass |
| test origins per horizon (first, last, count, n_eff) equal the plan's calendar figures: h = 126: 2008-01-31, 2026-02-27, 218, 37.1; h = 63: 2008-01-31, 2026-05-29, 221, 74.2; h = 21: 2008-01-31, 2026-07-31, 223, 218.2 | pass |
| coverage at test origins (h = 126) at least 90% for every predictor | pass |
| standardisation: mu_j and sd_j from the frozen spec, no sd_j is zero | pass |
| train replication recomputed = `output/research/train/train_stats.json` | pass |

Coverage (non-missing / test origins): C01 218/218; C02 218/218; C03 218/218; C04 218/218; C05 218/218; C06 218/218; C07 218/218; C08 218/218; C09 218/218; C10 218/218; C11 218/218; C12 217/218; COMP 218/218 at h = 126. Composite members available at the 224 test-side month-ends: 9: 224.

## 8. Test-period predictor values (descriptive, after unsealing)

| id | signal | n | mean | sd | min | median | max | train mean (mu_j) | favourable (>= threshold) at the 223 economic origins |
|---|---|---|---|---|---|---|---|---|---|
| C01 | VRP | 224 | 5.367 | 42.63 | -473.3 | 8.736 | 82.2 | 12.46 | 61% |
| C02 | GAP | 224 | -0.02579 | 0.03706 | -0.1698 | -0.01977 | 0.04094 | 0.04734 | 100% |
| C03 | SRATE | 224 | -0.03121 | 1.417 | -3.72 | -0.025 | 4.56 | -0.1948 | 75% |
| C04 | TERM | 224 | 1.318 | 1.312 | -1.86 | 1.52 | 3.75 | 1.602 | 66% |
| C05 | DEF | 224 | 1.021 | 0.4677 | 0.41 | 0.93 | 3.33 | 0.8361 | 78% |
| C06 | INFL | 224 | 0.02481 | 0.01913 | -0.02119 | 0.02165 | 0.08673 | 0.02857 | 73% |
| C07 | SLOOS | 224 | 7.358 | 24 | -32.4 | 0 | 83.6 | 6.16 | 69% |
| C08 | CLAIMS | 224 | -0.01964 | 0.5454 | -2.021 | -0.06767 | 3.143 | -0.001721 | 78% |
| C09 | ECY | 224 | 0.03302 | 0.0131 | 0.01019 | 0.03274 | 0.07137 | 0.01367 | 100% |
| C10 | IVAR | 224 | 0.04613 | 0.04941 | 0.00912 | 0.03005 | 0.3956 | 0.03956 | 70% |
| C11 | HURDLE | 224 | 0.06082 | 0.02299 | 0.03283 | 0.05513 | 0.1734 | 0.07757 | 91% |
| C12 | SCORE | 223 | 0.514 | 0.4404 | -0.3882 | 0.6204 | 1 | 0.3525 | 69% |
| COMP | COMP | 224 | 0.4618 | 0.4925 | -0.7154 | 0.4943 | 1.627 | - | 89% |

## 9. What was seen, deviations, and the look log

* Before this run: the exposure ledger (PREREGISTRATION.md section 2), the Build and Train notes (DEVIATIONS.md N11-N18) and the train-period statistics in `research/train_results.md`. The Test-phase implementation (DEVIATIONS.md, Test phase notes) was written and unit-tested on synthetic data before unsealing; before the run no test-period outcome was computed and no test-period predictor value was summarised (the rebuild check compares values for equality only).
* Missing from the exposure ledger (found by the look-ahead review after the run; DEVIATIONS.md N30): before the plan was frozen (2026-09-27 16:12Z), a working script also computed 6-month SPXL excess returns by hurdle tercile on the SPXL-only months since 2009. The ledger lists the full-sample hurdle split and the SPXL-only drawdown levels, not this split. It bears on C11 only, already labelled 'not a clean out-of-sample test'.
* Deviations from the plan: DEVIATIONS.md D1-D5 (all made before unsealing, for data reasons) and the Test-phase notes there. Changes after unsealing (DEVIATIONS.md N26-N30) are presentation, process safeguards and exploratory analysis only; none changes a registered number or the verdict.
* Code: this run did not record the SHA-256 of its code (added afterwards, DEVIATIONS.md N28; see N29 for the check that the current code reproduces every number).
* This run unsealed 2008-2026. From now on the test period is spent: a new idea can be tested only on data that do not exist yet (PREREGISTRATION.md, Part 1).

| UTC | phase | command | git |
|---|---|---|---|
| 2026-09-27T17:03:57Z | build | `py scripts/research_signals.py build` | 9b55b30-dirty |
| 2026-09-27T17:05:23Z | build | `py scripts/research_signals.py build --offline` | 9b55b30-dirty |
| 2026-09-27T17:10:05Z | build | `py scripts/research_signals.py build --offline` | 9b55b30-dirty |
| 2026-09-27T17:31:52Z | build | `py scripts/research_signals.py build` | 9b55b30-dirty |
| 2026-09-27T17:48:48Z | train | `py scripts/research_signals.py train` | 9b55b30-dirty |
| 2026-09-27T17:49:43Z | train | `py scripts/research_signals.py train` | 9b55b30-dirty |
| 2026-09-27T18:14:47Z | test | `py scripts/research_signals.py test --unseal 6452feeec9c4372...` | 9b55b30-dirty |

Outputs: `output/research/test_results.json` (every number above), `output/research/test/` (tests.csv, economic_daily.csv, economic_holdings.csv, panel_unsealed.pkl, economic_test.png).

## EXPLORATORY (not a result)

Added after the run and revised after three independent reviews (research/DEVIATIONS.md N26-N30). Nothing
here is a registered test or changes the verdict above, and none of it may be used to choose a new signal,
sign, threshold or rule for this period. The full exploratory analysis is `research/RESULTS.md`, section 3,
regenerated by `py scripts/research_signals.py report` (numbers in `output/research/exploratory.json`). In
short:

* **Where the composite rule lost.** COMP stayed at or above tau at every origin from 2008-01 to 2022-09, so
  the rule equalled buy-and-hold until 2022-10-31 and shared its 93.5% drawdown (bottom 2009-03-09) and its
  first-half Sharpe: E3 and the first half of E4 could not pass. The gap comes from three T-bill spells:
  holdings decided at 2022-10..2024-08 (23 months, 480 sessions: SPXL +153.6% against +10.4% for the rule),
  2025-04 (SPXL +18.2% against +0.3%) and 2026-06 (SPXL -0.9% against +0.2%). Over 2022-10..2024-08 the
  mean clipped signed z of TERM was -2.26 (inverted curve), SRATE -1.63, SLOOS -1.30, INFL -1.21 and VRP
  -0.52, against GAP +1.62, DEF +0.56, ECY +0.50 and CLAIMS +0.09.
* **The largest primary correlations** (C06 INFL r = +0.37, p = 0.0152; C02 GAP r = +0.36, p = 0.0174) are
  below the first Holm step and had train r of about zero (+0.01, +0.02). Leaving out the three episodes of
  section 6 together (not one at a time) leaves r = +0.11 and +0.23.
* **The registered n_eff is conservative.** It counts non-overlapping windows and ignores predictor
  persistence. Under a persistence-aware simulation (predictor paths fixed, outcomes rebuilt from
  block-resampled daily returns) C06 is borderline at the first Holm step: it passes at some block lengths and
  fails at others. The registered verdict stands (switching the method after unsealing would be a forking
  path); had C06 passed, its label would have been 'Predictive, not usable by the registered rule', because
  its own rule fails E4.
* **The one secondary Holm pass** (C01 VRP, S&P 500 1 month, p = 0.0032 against 0.0038) does not survive one
  Holm correction across all 91 registered tests (adjusted p 0.29). F2 and F6 are nearly the same test (the
  SPXL and S&P 500 one-month returns have rank correlation 0.998). The pass fails when any one of eight
  calendar years, or the 2008-09 crisis, is left out, and it comes from 2008-16 only (r +0.34, against +0.01
  in 2017-26; +0.04 in 1990-2007). Its switching rule lost to buy-and-hold on real SPXL. It is not evidence of
  a usable effect.
* **The economic 'wins'** of 5.1 (C06, C01, C08) and the vol-managed drawdown cut of 5.4 come from holdings in
  autumn 2008, which use the synthetic fund (SPXL started trading on 2008-11-05). On real SPXL (holdings from
  2009-01-02) no rule beat buy-and-hold, and a constant 55-60% SPXL weight matched the vol-managed rule on
  growth and drawdown. All economic figures are before tax.
