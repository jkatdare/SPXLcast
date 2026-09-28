"""The seal of scripts/research_signals.py (PREREGISTRATION.md sections 13.1 and 15): no outcome
window ending after 2007-12-31 before a verified unseal. Offline, synthetic data."""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_signals as rs  # noqa: E402

T = pd.Timestamp


def _inputs(start="2006-06-01", end="2008-12-31", seed=0):
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    r = 0.0003 + 0.01 * rng.standard_normal(len(idx))
    tr = 1000.0 * np.cumprod(1.0 + r)
    closes = pd.DataFrame({"^SP500TR": tr, "^GSPC": tr / 2.0, "^VIX": 18.0 + rng.random(len(idx)),
                           "^VIX3M": np.nan, "^VIX6M": np.nan, "SPXL": np.nan}, index=idx)
    daily = pd.bdate_range("2000-01-03", end)
    fred = {"DGS3MO": pd.Series(3.0 + 0.001 * np.arange(len(daily)), index=daily),
            "DGS10": pd.Series(4.5, index=daily), "T10YIE": pd.Series(2.2, index=daily),
            "BAMLH0A0HYM2": pd.Series(4.0, index=pd.bdate_range("2008-03-03", end))}   # starts after the seal
    return {"closes": closes, "fred": fred, "shiller": pd.DataFrame(), "fetched": "test"}


def test_sealed_frame_stops_at_2007_and_is_the_prefix_of_the_full_frame():
    inputs = _inputs()
    sealed = rs.outcome_frame(inputs)
    full = rs.br.build_daily(inputs, rs.Config())
    assert sealed.index.max() == T("2007-12-31")
    assert sealed.index.equals(full.index[: len(sealed)])
    for c in ("tr", "fund", "tbill_ret", "syn_ret"):
        pd.testing.assert_series_equal(sealed[c], full[c].iloc[: len(sealed)], check_exact=True)
    assert inputs["closes"].index.max() == T("2008-12-31")            # the caller's inputs are untouched


def test_sealed_targets_cannot_reach_past_2007():
    inputs = _inputs()
    cal = inputs["closes"].index
    pos = rs.br.month_end_positions(cal, "2006-07")
    sealed = rs.compute_targets(rs.outcome_frame(inputs), pos, cal)
    full = rs.compute_targets(rs.br.build_daily(inputs, rs.Config()), pos, cal)
    n_seal = int(cal.searchsorted(rs.SEAL_END, side="right"))
    for p, t in zip(pos, sealed.index):
        for h in rs.HORIZONS:
            if p + h < n_seal:
                assert sealed.at[t, f"Y{h}"] == full.at[t, f"Y{h}"]
                assert sealed.at[t, f"S{h}"] == full.at[t, f"S{h}"]
            else:
                assert np.isnan(sealed.at[t, f"Y{h}"]) and np.isnan(sealed.at[t, f"S{h}"])
        assert np.isnan(sealed.at[t, "D20"]) == (p + rs.DIP_H >= n_seal)
    assert sealed.loc[sealed.index >= rs.TEST_START].isna().all().all()
    assert full.loc[full.index >= rs.TEST_START, "Y21"].notna().any()     # the full frame would have them


def _spec(tmp_path):
    md, js = tmp_path / "PREREGISTRATION.md", tmp_path / "prereg.json"
    md.write_bytes(b"# plan\nfrozen\n")
    js.write_bytes(b'{"version": "1.0"}\n')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    hashes = tmp_path / "PREREG_HASH.txt"
    hashes.write_text(f"header\n\n{sha(md)}  research/PREREGISTRATION.md\n{sha(js)}  research/prereg.json\n",
                      encoding="utf-8")
    params, report = tmp_path / "train_params.json", tmp_path / "TRAIN_REPORT.md"
    return {"md": md, "js": js, "hashes": hashes, "params": params, "report": report}, sha(md)


def test_unseal_needs_the_token_the_hashes_and_the_train_phase(tmp_path):
    kw, token = _spec(tmp_path)
    inputs = _inputs()
    with pytest.raises(rs.SealError, match="token"):
        rs.outcome_frame(inputs, unseal_token="0" * 64, **kw)
    with pytest.raises(rs.SealError, match="Train phase"):                  # no train_params.json yet
        rs.outcome_frame(inputs, unseal_token=token, **kw)
    kw["params"].write_text('{"tau": 0.1}\n', encoding="utf-8")
    with pytest.raises(rs.SealError, match="Train phase"):                  # no TRAIN_REPORT.md yet
        rs.outcome_frame(inputs, unseal_token=token, **kw)
    kw["report"].write_text("train_params.json " + "f" * 64 + "\n", encoding="utf-8")
    with pytest.raises(rs.SealError, match="does not match"):
        rs.outcome_frame(inputs, unseal_token=token, **kw)
    good = hashlib.sha256(kw["params"].read_bytes()).hexdigest()
    kw["report"].write_text(f"| output/research/train_params.json | {good} |\n", encoding="utf-8")
    kw["js"].write_bytes(b'{"version": "1.1"}\n')                          # the plan changed after the freeze
    with pytest.raises(rs.SealError, match="PREREG_HASH"):
        rs.outcome_frame(inputs, unseal_token=token, **kw)
    kw["js"].write_bytes(b'{"version": "1.0"}\n')
    log = tmp_path / "look_log.csv"                                        # next to train_params.json
    assert not log.exists()                                                # refused unseals are not logged
    d = rs.outcome_frame(inputs, unseal_token=token, **kw)                 # every check passes
    assert d.index.max() == T("2008-12-31")
    rows = list(csv.DictReader(open(log, encoding="utf-8")))               # ... and the unseal is logged
    assert [r["phase"] for r in rows] == ["unseal"] and rows[0]["command"].endswith("[outcome_frame]")
    other = tmp_path / "elsewhere.csv"
    rs.outcome_frame(inputs, unseal_token=token, log=other, **kw)
    assert len(list(csv.DictReader(open(other, encoding="utf-8")))) == 1


def test_a_missing_spxl_close_stops_the_unsealed_frame():
    idx = pd.bdate_range("2008-12-01", "2009-03-31")
    d = pd.DataFrame({"spxl": np.linspace(10.0, 12.0, len(idx))}, index=idx)
    rs.check_spxl_closes(d)                                                 # complete: fine
    rs.check_spxl_closes(d.assign(spxl=np.nan))                             # no SPXL at all: nothing to check
    gap = d.copy()
    gap.loc[T("2009-02-10"), "spxl"] = np.nan
    with pytest.raises(ValueError, match="2009-02-10"):
        rs.check_spxl_closes(gap)
    late = d.copy()
    late.loc[late.index < T("2009-01-06"), "spxl"] = np.nan                  # SPXL must exist from SPXL_FROM
    with pytest.raises(ValueError, match="missing"):
        rs.check_spxl_closes(late)


def _csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_legacy_loader_reads_four_columns_and_checks_the_pinned_hash(tmp_path):
    p = tmp_path / "rating_backtest.csv"
    _csv(p, [["1990-01-31", 520, 0.9, "x", 0.11, 0.3, "garbage"]],
         ["date", "pos", "drift", "label", "hurdle", "score", "excess_126"])
    df = rs._read_legacy_columns(p)
    assert list(df.columns) == ["pos", "hurdle", "score"] and df.index[0] == T("1990-01-31")
    with pytest.raises(SystemExit, match="pinned"):
        rs.load_legacy(p)


def test_outcome_fields_of_sealed_rows_are_never_parsed(tmp_path):
    header = ["date", "pos"] + [f"{k}_{h}" for h in rs.HORIZONS for k in ("excess", "sp", "tbill")] + ["real_dd20_63"]
    train_row = ["2007-06-29", 1, 0.1, 0.05, 0.01, 0.2, 0.06, 0.02, 0.3, 0.07, 0.03, 0.0]
    boundary = ["2007-10-31", 2, 0.1, 0.05, 0.01, "SEALED", "SEALED", "SEALED", "SEALED", "SEALED", "SEALED", "SEALED"]
    sealed = ["2008-01-31", 3] + ["SEALED"] * 10
    p = tmp_path / "rating_backtest.csv"
    _csv(p, [train_row, boundary, sealed], header)
    flags = pd.DataFrame({"train_h21": [True, True, False], "train_h63": [True, False, False],
                          "train_h126": [True, False, False]},
                         index=pd.DatetimeIndex([T("2007-06-29"), T("2007-10-31"), T("2008-01-31")]))
    got = rs.read_train_outcomes(p, flags)                                 # no ValueError on "SEALED"
    assert got["Y126"] == {T("2007-06-29"): 0.3}
    assert got["Y21"] == {T("2007-06-29"): 0.1, T("2007-10-31"): 0.1}
    assert got["S21"][T("2007-10-31")] == pytest.approx(0.04)
    assert got["D20"] == {T("2007-06-29"): 0.0}
    with pytest.raises(SystemExit, match="pinned"):
        rs.check_target_reproduction(p, pd.DataFrame(), flags)


def test_every_run_is_logged_and_a_wrong_token_is_refused_before_anything_is_read(tmp_path, monkeypatch):
    log = tmp_path / "look_log.csv"
    monkeypatch.setattr(rs, "LOOK_LOG", log)
    monkeypatch.setattr(rs, "TEST_DIR", tmp_path / "test")
    monkeypatch.setattr(rs, "TEST_RESULTS_JSON", tmp_path / "test_results.json")
    monkeypatch.setattr(rs, "phase3_preconditions", lambda *a, **k: pytest.fail("read before the token check"))
    assert rs.main(["test", "--unseal", "0" * 64]) == 2
    assert not (tmp_path / "test").exists() and not (tmp_path / "test_results.json").exists()
    rows = list(csv.DictReader(open(log, encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["phase"] == "test"
    assert set(rows[0]) == {"utc", "phase", "command", "prereg_md_sha256", "prereg_json_sha256", "git_describe"}
    assert rows[0]["prereg_md_sha256"] == rs.sha256_file(rs.PREREG_MD)


def test_spec_check_stops_on_a_changed_plan(tmp_path):
    kw, _ = _spec(tmp_path)
    assert set(rs.check_spec(kw["md"], kw["js"], kw["hashes"])) == {"research/PREREGISTRATION.md", "research/prereg.json"}
    kw["md"].write_bytes(b"# plan\nedited\n")
    with pytest.raises(SystemExit, match="STOP"):
        rs.check_spec(kw["md"], kw["js"], kw["hashes"])
