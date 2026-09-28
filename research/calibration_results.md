# Recalibrating the 20% dip odds: results

Plan: `research/CALIBRATION.md`, frozen 2026-09-28T19:42:31Z before any new number was computed
(SHA-256 `0eef9e3e...` for the plan, `1a0ba7d9...` for `research/calibration_grid.json`; full hashes
in `research/CALIBRATION_HASH.txt`, still matching when this was written). Runs finished 2026-09-28 at
19:52 (train), 19:55 (check) and 20:06 UTC (exploratory).

## Part 1. In plain English

**Decision: nothing changes.** The engine keeps its VIX haircuts of 3/5/7 points and its
stochastic-vol size of 0.35, and `MODEL_VERSION` stays 0.3.0. The rule written down in advance picked
today's setting, so there was nothing to adopt.

**Why.** Every one of the 14 alternatives failed a safety rule on 1990 to mid-2007: its 1-month 90%
ranges held 94.3% of outcomes (198 of 210 month-ends), just over the 94% limit. The settings tried
barely move that number (by one outcome at most). Today's setting breaks the limit too, but the rule keeps it as the fallback in
any case. With twice as many simulated paths the answer is the same.

**What the alternatives did on 1990-2007.** Bigger haircuts did bring the fall chances closer to what
happened. The best one (haircuts 4/7/9, stochastic-vol size 0.45) cut the 3-month chance from 25.5% to
20.4%, where 17.6% happened, and improved the score by about 5% on those same years. But it was the
best of 15 tried there, and the 90% range of its gain includes zero at 1, 3 and 6 months, so that
gain is within what chance alone could produce. It also made the 1-week ranges too narrow (85% of
outcomes inside), so it would have failed even without the 1-month limit.

**The check period tells a milder story.** On 2008-2026 today's setting ran close on average: a 20%
fall within 3 months was predicted 26.8% of the time and happened 25.3% of the time; within 6 months
38.3% against 35.8%; within 1 month 11.0% against 13.0% (it ran low there). Most of the full-sample
gap (26% predicted, 22% happened) comes from 1990-2007, when the outcome is a 3x fund rebuilt from the
S&P 500. On SPXL itself since 2009 the gap is still about 4 to 5 points (3 months: 25.8% against 22.0%;
6 months: 37.4% against 32.0%), but 2008, when falls came thick and fast, pulls the whole check period
level. A cut sized to fix 1990-2007, about 5 points at 3 months, would probably have come out below
what happened over 2008-2026 as a whole, though about right on SPXL since 2009. This was deliberately not tested: scoring a setting on 2008-2026 after
seeing these results would use up the check period.

**A flaw in the plan, stated plainly.** The coverage safety rule was a fixed range (87-94%) rather
than "no further from 90% than today's setting". Today's setting was already outside it in 1990-2007
at 1 month (94.3%) and 6 months (94.8%), so any alternative had to fix something these settings barely
touch. In hindsight the relative version was the one that made sense. The rule was not changed after the
fact.

**What this means for the page.** The drawdown-risk level stays as it is. The page already shows,
next to each level, how often a 20% fall actually followed in the backtest (8.8%, 22.4% and 35.4%
against 11.7%, 24.5% and 42.5% predicted), so the reader sees the honest frequencies. The clean test
from here on is the live track record (`python -m spxlcast score`, "Drawdown risk check"). Another
attempt on 2008-2026 would be only partly out of sample now that today's setting's numbers there are
known.

## Part 2. Numbers

### Train window, the registered run (10,000 paths)

`py scripts/backtest_rating.py --engine-grid research/calibration_grid.json --start 1990-01 --end 2007-06 --paths 10000 --tag train`
(`output/backtest/engine_train.txt` and `.csv`)

```
SPXLcast engine settings compared, run 2026-09-28 19:52 UTC
210 month-ends 1990-01 .. 2007-06 (1990-01-31 .. 2007-06-29); 10000 paths per simulation, seed 42; model simulation only; VIX term structure before 2008: impute
settings: research/calibration_grid.json (SHA-256 1a0ba7d96aae7cc6...), 15 run; inputs: rating_inputs.pkl (SHA-256 d21e55055f35d91e...)

Reference setting (the guardrails compare with it): vrp 3/5/7, sv 0.35

Brier score of the 20% dip (lower is better; the objective is the mean over 1M/3M/6M), and the chance predicted / how often it happened
           setting Brier 1M Brier 3M Brier 6M    mean dip 1M pred/real dip 3M pred/real dip 6M pred/real
vrp 3/5/7, sv 0.25   0.0637   0.1390   0.1743 0.12564      9.9% / 7.1%    26.2% / 17.6%    38.9% / 25.7%
vrp 3/5/7, sv 0.35   0.0634   0.1373   0.1727 0.12446      9.8% / 7.1%    25.5% / 17.6%    38.0% / 25.7%
vrp 3/5/7, sv 0.45   0.0631   0.1358   0.1706 0.12317      9.6% / 7.1%    24.3% / 17.6%    36.7% / 25.7%
vrp 3/6/8, sv 0.25   0.0637   0.1353   0.1663 0.12178      9.9% / 7.1%    23.9% / 17.6%    35.8% / 25.7%
vrp 3/6/8, sv 0.35   0.0634   0.1342   0.1656 0.12109      9.8% / 7.1%    23.3% / 17.6%    35.1% / 25.7%
vrp 3/6/8, sv 0.45   0.0631   0.1332   0.1645 0.12028      9.6% / 7.1%    22.3% / 17.6%    34.0% / 25.7%
vrp 3/7/9, sv 0.25   0.0637   0.1331   0.1615 0.11943      9.9% / 7.1%    21.9% / 17.6%    33.1% / 25.7%
vrp 3/7/9, sv 0.35   0.0634   0.1324   0.1613 0.11905      9.8% / 7.1%    21.4% / 17.6%    32.5% / 25.7%
vrp 3/7/9, sv 0.45   0.0631   0.1320   0.1612 0.11876      9.6% / 7.1%    20.6% / 17.6%    31.5% / 25.7%
vrp 4/6/8, sv 0.25   0.0632   0.1351   0.1660 0.12145      8.8% / 7.1%    23.8% / 17.6%    35.8% / 25.7%
vrp 4/6/8, sv 0.35   0.0630   0.1342   0.1654 0.12085      8.8% / 7.1%    23.2% / 17.6%    35.0% / 25.7%
vrp 4/6/8, sv 0.45   0.0628   0.1332   0.1642 0.12008      8.6% / 7.1%    22.2% / 17.6%    33.9% / 25.7%
vrp 4/7/9, sv 0.25   0.0632   0.1329   0.1610 0.11904      8.8% / 7.1%    21.7% / 17.6%    33.0% / 25.7%
vrp 4/7/9, sv 0.35   0.0630   0.1323   0.1609 0.11871      8.8% / 7.1%    21.2% / 17.6%    32.4% / 25.7%
vrp 4/7/9, sv 0.45   0.0628   0.1318   0.1607 0.11842      8.6% / 7.1%    20.4% / 17.6%    31.4% / 25.7%

Brier score minus the reference's, same month-ends [90% CI, overlap-aware; information only]
           setting                         1M                         3M                         6M
vrp 3/5/7, sv 0.25 +0.0003 [-0.0002, +0.0008] +0.0016 [-0.0007, +0.0040] +0.0016 [-0.0017, +0.0049]
vrp 3/5/7, sv 0.35              0 (reference)              0 (reference)              0 (reference)
vrp 3/5/7, sv 0.45 -0.0003 [-0.0009, +0.0003] -0.0015 [-0.0044, +0.0014] -0.0020 [-0.0061, +0.0020]
vrp 3/6/8, sv 0.25 +0.0003 [-0.0002, +0.0008] -0.0020 [-0.0038, -0.0002] -0.0064 [-0.0102, -0.0025]
vrp 3/6/8, sv 0.35 +0.0000 [+0.0000, +0.0000] -0.0031 [-0.0061, -0.0000] -0.0070 [-0.0128, -0.0013]
vrp 3/6/8, sv 0.45 -0.0003 [-0.0009, +0.0003] -0.0041 [-0.0095, +0.0014] -0.0082 [-0.0174, +0.0010]
vrp 3/7/9, sv 0.25 +0.0003 [-0.0002, +0.0008] -0.0042 [-0.0089, +0.0005] -0.0112 [-0.0208, -0.0016]
vrp 3/7/9, sv 0.35 +0.0000 [+0.0000, +0.0000] -0.0049 [-0.0110, +0.0012] -0.0114 [-0.0230, +0.0003]
vrp 3/7/9, sv 0.45 -0.0003 [-0.0009, +0.0003] -0.0053 [-0.0136, +0.0030] -0.0115 [-0.0263, +0.0033]
vrp 4/6/8, sv 0.25 -0.0001 [-0.0007, +0.0005] -0.0022 [-0.0041, -0.0003] -0.0067 [-0.0106, -0.0027]
vrp 4/6/8, sv 0.35 -0.0004 [-0.0012, +0.0004] -0.0032 [-0.0063, -0.0000] -0.0073 [-0.0131, -0.0015]
vrp 4/6/8, sv 0.45 -0.0006 [-0.0019, +0.0007] -0.0041 [-0.0096, +0.0014] -0.0084 [-0.0176, +0.0008]
vrp 4/7/9, sv 0.25 -0.0001 [-0.0007, +0.0005] -0.0044 [-0.0092, +0.0004] -0.0117 [-0.0214, -0.0020]
vrp 4/7/9, sv 0.35 -0.0004 [-0.0012, +0.0004] -0.0050 [-0.0113, +0.0012] -0.0118 [-0.0235, -0.0001]
vrp 4/7/9, sv 0.45 -0.0006 [-0.0019, +0.0007] -0.0055 [-0.0139, +0.0029] -0.0120 [-0.0269, +0.0029]

CRPS of the fund's log return (lower is better) and its change from the reference's (guardrail G1: at most +0.5% at each of 1M/3M/6M)
           setting              1M              3M              6M
vrp 3/5/7, sv 0.25 0.0641 (+0.29%) 0.1107 (+0.62%) 0.1570 (+0.55%)
vrp 3/5/7, sv 0.35 0.0639 (+0.00%) 0.1100 (+0.00%) 0.1562 (+0.00%)
vrp 3/5/7, sv 0.45 0.0638 (-0.12%) 0.1094 (-0.55%) 0.1554 (-0.51%)
vrp 3/6/8, sv 0.25 0.0641 (+0.29%) 0.1100 (-0.02%) 0.1558 (-0.24%)
vrp 3/6/8, sv 0.35 0.0639 (+0.00%) 0.1095 (-0.51%) 0.1551 (-0.67%)
vrp 3/6/8, sv 0.45 0.0638 (-0.12%) 0.1090 (-0.91%) 0.1545 (-1.04%)
vrp 3/7/9, sv 0.25 0.0641 (+0.29%) 0.1095 (-0.46%) 0.1550 (-0.73%)
vrp 3/7/9, sv 0.35 0.0639 (+0.00%) 0.1091 (-0.84%) 0.1545 (-1.06%)
vrp 3/7/9, sv 0.45 0.0638 (-0.12%) 0.1088 (-1.12%) 0.1541 (-1.32%)
vrp 4/6/8, sv 0.25 0.0639 (-0.04%) 0.1101 (+0.01%) 0.1557 (-0.26%)
vrp 4/6/8, sv 0.35 0.0638 (-0.18%) 0.1095 (-0.48%) 0.1551 (-0.69%)
vrp 4/6/8, sv 0.45 0.0638 (-0.13%) 0.1091 (-0.88%) 0.1545 (-1.06%)
vrp 4/7/9, sv 0.25 0.0639 (-0.04%) 0.1095 (-0.47%) 0.1550 (-0.76%)
vrp 4/7/9, sv 0.35 0.0638 (-0.18%) 0.1091 (-0.85%) 0.1544 (-1.09%)
vrp 4/7/9, sv 0.45 0.0638 (-0.13%) 0.1088 (-1.12%) 0.1540 (-1.35%)

90% band coverage (guardrail G2: 87% to 94% at each of 1W/2W/1M/3M/6M); 50% band and mean PIT at 3M for information
           setting    1W    2W    1M    3M    6M 50% band 3M mean PIT 3M
vrp 3/5/7, sv 0.25 91.0% 92.9% 94.3% 92.9% 94.8%       60.0%       0.540
vrp 3/5/7, sv 0.35 90.5% 92.9% 94.3% 93.3% 94.8%       59.0%       0.537
vrp 3/5/7, sv 0.45 90.0% 92.9% 94.3% 92.9% 93.3%       54.8%       0.535
vrp 3/6/8, sv 0.25 91.0% 92.9% 94.3% 92.9% 92.4%       57.1%       0.541
vrp 3/6/8, sv 0.35 90.5% 92.9% 94.3% 91.9% 92.4%       54.8%       0.538
vrp 3/6/8, sv 0.45 90.0% 92.9% 94.3% 89.0% 91.9%       51.9%       0.536
vrp 3/7/9, sv 0.25 91.0% 92.9% 94.3% 90.5% 91.0%       54.3%       0.542
vrp 3/7/9, sv 0.35 90.5% 92.9% 94.3% 90.0% 91.0%       52.4%       0.539
vrp 3/7/9, sv 0.45 90.0% 92.9% 94.3% 88.1% 91.0%       49.5%       0.538
vrp 4/6/8, sv 0.25 87.6% 90.5% 94.3% 92.4% 92.4%       57.1%       0.541
vrp 4/6/8, sv 0.35 85.7% 90.0% 94.3% 91.0% 91.9%       54.3%       0.538
vrp 4/6/8, sv 0.45 85.2% 90.0% 94.3% 88.6% 91.9%       51.9%       0.536
vrp 4/7/9, sv 0.25 87.6% 90.5% 94.3% 90.0% 91.0%       54.3%       0.542
vrp 4/7/9, sv 0.35 85.7% 90.0% 94.3% 89.0% 91.0%       51.0%       0.539
vrp 4/7/9, sv 0.45 85.2% 90.0% 94.3% 87.6% 90.5%       49.5%       0.538

Guardrails and rule (the reference always counts; ranked by the mean Brier score)
           setting G1 CRPS G2 coverage counts rank
vrp 3/5/7, sv 0.25   fails       fails     no     
vrp 3/5/7, sv 0.35      ok       fails    yes    1
vrp 3/5/7, sv 0.45      ok       fails     no     
vrp 3/6/8, sv 0.25      ok       fails     no     
vrp 3/6/8, sv 0.35      ok       fails     no     
vrp 3/6/8, sv 0.45      ok       fails     no     
vrp 3/7/9, sv 0.25      ok       fails     no     
vrp 3/7/9, sv 0.35      ok       fails     no     
vrp 3/7/9, sv 0.45      ok       fails     no     
vrp 4/6/8, sv 0.25      ok       fails     no     
vrp 4/6/8, sv 0.35      ok       fails     no     
vrp 4/6/8, sv 0.45      ok       fails     no     
vrp 4/7/9, sv 0.25      ok       fails     no     
vrp 4/7/9, sv 0.35      ok       fails     no     
vrp 4/7/9, sv 0.45      ok       fails     no     
(month-ends with an outcome: 1W 210, 2W 210, 1M 210, 3M 210, 6M 210)

The rule picks: vrp 3/5/7, sv 0.35 (the reference: no change)
```

Selection (section 5): no setting other than the reference passes G2. Nine fail only because the
1-month coverage is 94.3%; the 4/6/8 and 4/7/9 settings at stochastic-vol sizes 0.35 and 0.45 also
break the 1-week floor (85.2-85.7%); 3/5/7 at 0.25 and 0.35 also break the 6-month ceiling (94.8%),
and 3/5/7 at 0.25 also fails G1 (CRPS +0.62% at 3 months, +0.55% at 6 months). **Selected: vrp 3/5/7,
sv 0.35, the current setting.** The 20,000-path confirmation (section 5, step 5) does not apply.

### Check window, run once (20,000 paths)

The selected setting is the current one, so the check was run once for that setting alone. No
decision rests on it (A1-A3 compare the selected setting with the current one, and they are the same).

`py scripts/backtest_rating.py --engine-grid research/calibration_grid.json --only "vrp 3/5/7, sv 0.35" --start 2008-01 --paths 20000 --tag check`
(`output/backtest/engine_check.txt` and `.csv`)

```
SPXLcast engine settings compared, run 2026-09-28 19:55 UTC
224 month-ends 2008-01 .. 2026-08 (2008-01-31 .. 2026-08-31); 20000 paths per simulation, seed 42; model simulation only; VIX term structure before 2008: impute
settings: research/calibration_grid.json (SHA-256 1a0ba7d96aae7cc6...), 1 run; inputs: rating_inputs.pkl (SHA-256 d21e55055f35d91e...)

Reference setting (the guardrails compare with it): vrp 3/5/7, sv 0.35

Brier score of the 20% dip (lower is better; the objective is the mean over 1M/3M/6M), and the chance predicted / how often it happened
           setting Brier 1M Brier 3M Brier 6M    mean dip 1M pred/real dip 3M pred/real dip 6M pred/real
vrp 3/5/7, sv 0.35   0.1021   0.1852   0.2347 0.17400    11.0% / 13.0%    26.8% / 25.3%    38.3% / 35.8%

Brier score minus the reference's, same month-ends [90% CI, overlap-aware; information only]
           setting            1M            3M            6M
vrp 3/5/7, sv 0.35 0 (reference) 0 (reference) 0 (reference)

CRPS of the fund's log return (lower is better) and its change from the reference's (guardrail G1: at most +0.5% at each of 1M/3M/6M)
           setting              1M              3M              6M
vrp 3/5/7, sv 0.35 0.0789 (+0.00%) 0.1382 (+0.00%) 0.2083 (+0.00%)

90% band coverage (guardrail G2: 87% to 94% at each of 1W/2W/1M/3M/6M); 50% band and mean PIT at 3M for information
           setting    1W    2W    1M    3M    6M 50% band 3M mean PIT 3M
vrp 3/5/7, sv 0.35 91.5% 88.8% 91.9% 92.8% 90.8%       46.2%       0.576

Guardrails and rule (the reference always counts; ranked by the mean Brier score)
           setting G1 CRPS G2 coverage counts rank
vrp 3/5/7, sv 0.35      ok          ok    yes    1
(month-ends with an outcome: 1W 224, 2W 224, 1M 223, 3M 221, 6M 218)

The rule picks: vrp 3/5/7, sv 0.35 (the reference: no change)
```

These rows are identical (largest difference 0) to the matching columns of
`output/backtest/rating_backtest.csv`, the 0.3.0 full backtest of 2026-09-27: the same engine, seed
and paths. So the check table restates numbers that already existed; the full-sample aggregates had
been seen, the 2008-2026 split had not been tabulated.

### Decision

| rule | result |
|---|---|
| Train: lowest mean Brier score within G1 and G2 (reference always counts) | vrp 3/5/7, sv 0.35 (current) |
| Confirmation at 20,000 paths | not applicable |
| Check: A1 Brier lower, A2 CRPS within +0.5%, A3 coverage 87-94% | not applicable (selected = current) |
| **Adopt?** | **No. Defaults unchanged, MODEL_VERSION 0.3.0** |

## Part 3. Exploratory (after the decision; not part of the rule)

**1. The train grid at 20,000 paths**, to see whether the 1-month coverage failure is Monte Carlo
noise (one outcome at 1 month sat at PIT 0.051 with 10,000 paths, just inside the band). Same pick:
13 of the 15 settings still hold 198 of 210 1-month outcomes (94.3%); the other two (4/6/8 and 4/7/9
at 0.45) hold 197 (93.8%) but break the 1-week floor (85.2%). The reference rows reproduce
`rating_backtest.csv` exactly.
(`output/backtest/engine_train_20k_exploratory.txt` and `.csv`)

```
SPXLcast engine settings compared, run 2026-09-28 20:06 UTC
210 month-ends 1990-01 .. 2007-06 (1990-01-31 .. 2007-06-29); 20000 paths per simulation, seed 42; model simulation only; VIX term structure before 2008: impute
settings: research/calibration_grid.json (SHA-256 1a0ba7d96aae7cc6...), 15 run; inputs: rating_inputs.pkl (SHA-256 d21e55055f35d91e...)

Reference setting (the guardrails compare with it): vrp 3/5/7, sv 0.35

Brier score of the 20% dip (lower is better; the objective is the mean over 1M/3M/6M), and the chance predicted / how often it happened
           setting Brier 1M Brier 3M Brier 6M    mean dip 1M pred/real dip 3M pred/real dip 6M pred/real
vrp 3/5/7, sv 0.25   0.0639   0.1392   0.1738 0.12564     10.1% / 7.1%    26.5% / 17.6%    38.8% / 25.7%
vrp 3/5/7, sv 0.35   0.0635   0.1376   0.1720 0.12434      9.9% / 7.1%    25.6% / 17.6%    37.8% / 25.7%
vrp 3/5/7, sv 0.45   0.0631   0.1358   0.1697 0.12286      9.6% / 7.1%    24.5% / 17.6%    36.4% / 25.7%
vrp 3/6/8, sv 0.25   0.0639   0.1355   0.1659 0.12178     10.1% / 7.1%    24.2% / 17.6%    35.8% / 25.7%
vrp 3/6/8, sv 0.35   0.0635   0.1344   0.1649 0.12091      9.9% / 7.1%    23.5% / 17.6%    34.9% / 25.7%
vrp 3/6/8, sv 0.45   0.0631   0.1333   0.1637 0.12004      9.6% / 7.1%    22.5% / 17.6%    33.7% / 25.7%
vrp 3/7/9, sv 0.25   0.0639   0.1332   0.1609 0.11934     10.1% / 7.1%    22.3% / 17.6%    33.1% / 25.7%
vrp 3/7/9, sv 0.35   0.0635   0.1325   0.1606 0.11887      9.9% / 7.1%    21.7% / 17.6%    32.4% / 25.7%
vrp 3/7/9, sv 0.45   0.0631   0.1320   0.1603 0.11847      9.6% / 7.1%    20.9% / 17.6%    31.3% / 25.7%
vrp 4/6/8, sv 0.25   0.0634   0.1354   0.1657 0.12151      8.9% / 7.1%    24.1% / 17.6%    35.7% / 25.7%
vrp 4/6/8, sv 0.35   0.0631   0.1343   0.1648 0.12074      8.8% / 7.1%    23.4% / 17.6%    34.8% / 25.7%
vrp 4/6/8, sv 0.45   0.0628   0.1332   0.1635 0.11985      8.6% / 7.1%    22.4% / 17.6%    33.6% / 25.7%
vrp 4/7/9, sv 0.25   0.0634   0.1330   0.1606 0.11902      8.9% / 7.1%    22.0% / 17.6%    33.0% / 25.7%
vrp 4/7/9, sv 0.35   0.0631   0.1324   0.1603 0.11857      8.8% / 7.1%    21.5% / 17.6%    32.2% / 25.7%
vrp 4/7/9, sv 0.45   0.0628   0.1318   0.1599 0.11818      8.6% / 7.1%    20.6% / 17.6%    31.2% / 25.7%

Brier score minus the reference's, same month-ends [90% CI, overlap-aware; information only]
           setting                         1M                         3M                         6M
vrp 3/5/7, sv 0.25 +0.0004 [-0.0001, +0.0010] +0.0016 [-0.0007, +0.0039] +0.0019 [-0.0016, +0.0053]
vrp 3/5/7, sv 0.35              0 (reference)              0 (reference)              0 (reference)
vrp 3/5/7, sv 0.45 -0.0003 [-0.0010, +0.0003] -0.0018 [-0.0046, +0.0010] -0.0023 [-0.0066, +0.0020]
vrp 3/6/8, sv 0.25 +0.0004 [-0.0001, +0.0010] -0.0021 [-0.0038, -0.0003] -0.0061 [-0.0097, -0.0024]
vrp 3/6/8, sv 0.35 +0.0000 [+0.0000, +0.0000] -0.0032 [-0.0061, -0.0003] -0.0071 [-0.0127, -0.0015]
vrp 3/6/8, sv 0.45 -0.0003 [-0.0010, +0.0003] -0.0043 [-0.0096, +0.0011] -0.0083 [-0.0176, +0.0010]
vrp 3/7/9, sv 0.25 +0.0004 [-0.0001, +0.0010] -0.0044 [-0.0089, +0.0001] -0.0111 [-0.0203, -0.0018]
vrp 3/7/9, sv 0.35 +0.0000 [+0.0000, +0.0000] -0.0051 [-0.0110, +0.0009] -0.0113 [-0.0228, +0.0001]
vrp 3/7/9, sv 0.45 -0.0003 [-0.0010, +0.0003] -0.0056 [-0.0137, +0.0026] -0.0117 [-0.0265, +0.0031]
vrp 4/6/8, sv 0.25 -0.0001 [-0.0007, +0.0006] -0.0022 [-0.0040, -0.0003] -0.0062 [-0.0100, -0.0025]
vrp 4/6/8, sv 0.35 -0.0004 [-0.0012, +0.0004] -0.0033 [-0.0063, -0.0002] -0.0072 [-0.0130, -0.0014]
vrp 4/6/8, sv 0.45 -0.0007 [-0.0020, +0.0006] -0.0043 [-0.0098, +0.0011] -0.0085 [-0.0179, +0.0010]
vrp 4/7/9, sv 0.25 -0.0001 [-0.0007, +0.0006] -0.0045 [-0.0092, +0.0001] -0.0114 [-0.0208, -0.0019]
vrp 4/7/9, sv 0.35 -0.0004 [-0.0012, +0.0004] -0.0052 [-0.0113, +0.0008] -0.0117 [-0.0233, -0.0001]
vrp 4/7/9, sv 0.45 -0.0007 [-0.0020, +0.0006] -0.0058 [-0.0140, +0.0025] -0.0120 [-0.0270, +0.0029]

CRPS of the fund's log return (lower is better) and its change from the reference's (guardrail G1: at most +0.5% at each of 1M/3M/6M)
           setting              1M              3M              6M
vrp 3/5/7, sv 0.25 0.0641 (+0.32%) 0.1111 (+0.66%) 0.1573 (+0.60%)
vrp 3/5/7, sv 0.35 0.0639 (+0.00%) 0.1103 (+0.00%) 0.1564 (+0.00%)
vrp 3/5/7, sv 0.45 0.0638 (-0.15%) 0.1097 (-0.58%) 0.1556 (-0.54%)
vrp 3/6/8, sv 0.25 0.0641 (+0.32%) 0.1103 (-0.02%) 0.1561 (-0.17%)
vrp 3/6/8, sv 0.35 0.0639 (+0.00%) 0.1097 (-0.54%) 0.1554 (-0.63%)
vrp 3/6/8, sv 0.45 0.0638 (-0.15%) 0.1093 (-0.97%) 0.1548 (-1.02%)
vrp 3/7/9, sv 0.25 0.0641 (+0.32%) 0.1098 (-0.49%) 0.1554 (-0.63%)
vrp 3/7/9, sv 0.35 0.0639 (+0.00%) 0.1093 (-0.90%) 0.1548 (-1.00%)
vrp 3/7/9, sv 0.45 0.0638 (-0.15%) 0.1090 (-1.20%) 0.1544 (-1.26%)
vrp 4/6/8, sv 0.25 0.0639 (-0.01%) 0.1103 (+0.00%) 0.1561 (-0.18%)
vrp 4/6/8, sv 0.35 0.0638 (-0.18%) 0.1098 (-0.52%) 0.1554 (-0.65%)
vrp 4/6/8, sv 0.45 0.0638 (-0.15%) 0.1093 (-0.94%) 0.1548 (-1.04%)
vrp 4/7/9, sv 0.25 0.0639 (-0.01%) 0.1098 (-0.52%) 0.1554 (-0.67%)
vrp 4/7/9, sv 0.35 0.0638 (-0.18%) 0.1093 (-0.92%) 0.1548 (-1.02%)
vrp 4/7/9, sv 0.45 0.0638 (-0.15%) 0.1090 (-1.20%) 0.1544 (-1.29%)

90% band coverage (guardrail G2: 87% to 94% at each of 1W/2W/1M/3M/6M); 50% band and mean PIT at 3M for information
           setting    1W    2W    1M    3M    6M 50% band 3M mean PIT 3M
vrp 3/5/7, sv 0.25 91.0% 93.3% 94.3% 93.3% 94.3%       61.0%       0.544
vrp 3/5/7, sv 0.35 90.5% 92.9% 94.3% 93.3% 94.3%       58.6%       0.541
vrp 3/5/7, sv 0.45 89.0% 92.9% 94.3% 92.9% 93.3%       55.7%       0.538
vrp 3/6/8, sv 0.25 91.0% 93.3% 94.3% 92.9% 92.4%       57.6%       0.545
vrp 3/6/8, sv 0.35 90.5% 92.9% 94.3% 91.0% 91.9%       56.2%       0.542
vrp 3/6/8, sv 0.45 89.0% 92.9% 94.3% 88.6% 91.4%       51.9%       0.540
vrp 3/7/9, sv 0.25 91.0% 93.3% 94.3% 90.5% 91.0%       55.2%       0.545
vrp 3/7/9, sv 0.35 90.5% 92.9% 94.3% 89.5% 91.0%       52.4%       0.543
vrp 3/7/9, sv 0.45 89.0% 92.9% 94.3% 87.6% 89.5%       49.5%       0.541
vrp 4/6/8, sv 0.25 87.6% 91.4% 94.3% 92.4% 92.4%       57.6%       0.545
vrp 4/6/8, sv 0.35 86.7% 90.0% 94.3% 90.0% 91.9%       56.2%       0.542
vrp 4/6/8, sv 0.45 85.2% 89.5% 93.8% 88.1% 91.4%       52.4%       0.539
vrp 4/7/9, sv 0.25 87.6% 91.4% 94.3% 90.0% 91.0%       54.8%       0.545
vrp 4/7/9, sv 0.35 86.7% 90.0% 94.3% 88.1% 91.0%       51.4%       0.543
vrp 4/7/9, sv 0.45 85.2% 89.5% 93.8% 87.6% 89.5%       49.5%       0.541

Guardrails and rule (the reference always counts; ranked by the mean Brier score)
           setting G1 CRPS G2 coverage counts rank
vrp 3/5/7, sv 0.25   fails       fails     no     
vrp 3/5/7, sv 0.35      ok       fails    yes    1
vrp 3/5/7, sv 0.45      ok       fails     no     
vrp 3/6/8, sv 0.25      ok       fails     no     
vrp 3/6/8, sv 0.35      ok       fails     no     
vrp 3/6/8, sv 0.45      ok       fails     no     
vrp 3/7/9, sv 0.25      ok       fails     no     
vrp 3/7/9, sv 0.35      ok       fails     no     
vrp 3/7/9, sv 0.45      ok       fails     no     
vrp 4/6/8, sv 0.25      ok       fails     no     
vrp 4/6/8, sv 0.35      ok       fails     no     
vrp 4/6/8, sv 0.45      ok       fails     no     
vrp 4/7/9, sv 0.25      ok       fails     no     
vrp 4/7/9, sv 0.35      ok       fails     no     
vrp 4/7/9, sv 0.45      ok       fails     no     
(month-ends with an outcome: 1W 210, 2W 210, 1M 210, 3M 210, 6M 210)

The rule picks: vrp 3/5/7, sv 0.35 (the reference: no change)
```

**2. Hindsight only.** Setting the 1-month limit aside, the lowest mean Brier score within the other
limits would have been vrp 3/7/9, sv 0.45 (0.11876 against 0.12446 at 10,000 paths; 3-month chance
20.6% against 25.5%, 17.6% happened; coverage 90.0 / 92.9 / 94.3 / 88.1 / 91.0%). It was not scored on
the check window.

**3. The current setting's 20% dip odds by era**, from the existing `rating_backtest.csv` (0.3.0,
20,000 paths), predicted / happened:

| origins | 1 month | 3 months | 6 months |
|---|---|---|---|
| 1990-01 .. 2007-06 (train, 210) | 9.9% / 7.1% | 25.6% / 17.6% | 37.8% / 25.7% |
| 2008-01 .. 2026-08 (check, 223 / 221 / 218) | 11.0% / 13.0% | 26.8% / 25.3% | 38.3% / 35.8% |
| SPXL itself, 2009-01 .. 2026-08 (211 / 209 / 206) | 10.2% / 10.9% | 25.8% / 22.0% | 37.4% / 32.0% |

## Part 4. Process notes

* **Deviations from the plan:** none in the registered steps. The check window was run once, for the
  selected (= current) setting only. The two exploratory items above were run after the decision
  and changed nothing.
* **2008-on data in the 1990-2007 inputs:** VIX3M and VIX6M did not exist before 2008, so at every
  train month-end they are estimated from the VIX with a fit on 2008-01-02 to 2026-09-25 (4,713 days;
  `fit_term_structure` in `scripts/backtest_rating.py`). The plan's "nothing from 2008 on is used"
  (CALIBRATION.md, Part 1) holds for the outcomes, not for these inputs, and the 3- and 6-month
  haircuts tried act on them. Fitting on 2008-2012 or 2013-2026 alone moves the 1990-2007 average by
  at most 0.9 points (VIX6M) and 0.3 points (VIX3M), less than the haircut steps tried. The decision
  rests on the 1-month ranges, which use the VIX itself, so it does not change.
* **Code** (written after the freeze): `scripts/backtest_rating.py` gained `--end` (last month-end
  origin), `--engine-grid FILE` (only the model simulation per setting, then Brier, CRPS and coverage
  per horizon, the guardrails and the rule's pick, written to `output/backtest/engine_<tag>.txt/.csv`
  without touching the backtest's own outputs), `--only` and `--tag`. `run_origin` now shares the
  simulation and scoring code with the new `engine_origin`; on four origins across 1990-2026 its
  output was checked identical, value for value and in column order, to the committed version.
  Tests: `tests/test_fix_backtest.py` (the grid run and its `--start`/`--end` window, agreement with
  `run_origin`, the reference requirement, and the rule: guardrails, the reference as fallback, ties).
  After the runs, two fixes: each setting's inputs are now built with that setting's own values (only
  an earnings-growth setting reads them; this grid has none, so its results are unchanged), and a
  window with no outcome yet at a horizon the rule uses stops with a message before simulating.
* **Inputs:** the cached `output/backtest/rating_inputs.pkl` (SHA-256 `d21e5505...`), never refreshed;
  local Python 3.13 (numpy 2.2.6, pandas 2.3.3); the runs reproduce the 0.3.0 backtest's rows exactly.
