"""The post-Test `report` helpers of scripts/research_signals.py: the daily stationary bootstrap, the
window returns rebuilt from cumulative logs, rank correlations through standardised ranks, the code
provenance, the appendix headings and the output manifest. Synthetic data only; nothing is unsealed.
(The whole `report` command runs end to end in tests/test_research_phase3.py.)"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import research_signals as rs  # noqa: E402


def test_stationary_indices_are_circular_blocks_of_the_mean_length():
    rng = np.random.default_rng(0)
    idx = rs.stationary_indices(rng, 400, 500, 20)
    assert idx.shape == (400, 500) and idx.min() >= 0 and idx.max() < 500
    cont = (np.diff(idx, axis=1) % 500 == 1).mean()
    assert 0.93 < cont < 0.97                                       # a new block starts with chance 1/20
    one = rs.stationary_indices(np.random.default_rng(1), 200, 500, 1)
    assert (np.diff(one, axis=1) % 500 == 1).mean() < 0.01          # mean block 1: every index drawn afresh
    a = rs.stationary_indices(np.random.default_rng(7), 5, 50, 6)
    assert (a == rs.stationary_indices(np.random.default_rng(7), 5, 50, 6)).all()


def test_window_excess_equals_the_product_of_daily_returns():
    rng = np.random.default_rng(2)
    f, b = 0.03 * rng.standard_normal(300), np.full(300, 0.0002)
    i0 = np.array([-1, 0, 17, 150])
    got = rs.window_excess(rs._cumlog(f), rs._cumlog(b), i0, 126)
    want = [np.prod(1 + f[i + 1: i + 127]) - np.prod(1 + b[i + 1: i + 127]) for i in i0]
    assert got == pytest.approx(want, rel=1e-12, abs=1e-14)
    two = rs.window_excess(rs._cumlog(np.vstack([f, f])), rs._cumlog(np.vstack([b, b])), i0, 126)
    assert two.shape == (2, 4) and two[1] == pytest.approx(want, rel=1e-12, abs=1e-14)


def test_standardised_ranks_give_spearman_with_ties():
    from spxlcast.evaluation import spearman
    rng = np.random.default_rng(3)
    x = np.round(rng.standard_normal(80), 1)                         # ties
    y = rng.standard_normal(80)
    assert np.mean(rs._zrank(x) * rs._zrank(y)) == pytest.approx(spearman(x, y), abs=1e-12)
    ys = np.vstack([y, -y])
    r = rs._zrank(ys, axis=1) @ rs._zrank(x) / 80
    assert r == pytest.approx([spearman(x, y), spearman(x, -y)], abs=1e-12)


def test_code_provenance_hashes_and_copies(tmp_path):
    root = tmp_path / "repo"
    for rel in rs.CODE_FILES[:2]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(rel.encode("utf-8"))
    got = rs.code_provenance(copy_to=tmp_path / "copy", root=root)
    assert got[rs.CODE_FILES[0]] == hashlib.sha256(rs.CODE_FILES[0].encode("utf-8")).hexdigest()
    assert got[rs.CODE_FILES[2]] == "missing"
    assert (tmp_path / "copy" / rs.CODE_FILES[1]).read_bytes() == rs.CODE_FILES[1].encode("utf-8")


def test_the_registered_document_becomes_appendix_a():
    text = ("# SPXL timing-signal search: Test phase results\n\nintro\n\n## Verdict\n\nv\n\n## 1. Primary test\n\n"
            "t\n\n### 5.1 The same rule\n\nx\n\n## EXPLORATORY (not a result)\n\n* a note\n")
    got = rs._demote(text)
    assert got[0] == "intro"
    assert "### Verdict" in got and "### A1. Primary test" in got and "#### A5.1 The same rule" in got
    assert not any("EXPLORATORY" in line or "a note" in line for line in got)
    assert got[-1] == "x"


def test_family_and_candidate_names_cover_every_predictor():
    assert set(rs.PLAIN) == set(rs.PREDICTORS)
    assert set(rs.FAMILY_PLAIN) == {f for f, _, _ in rs.FAMILIES}
    assert rs._n(0) == "no" and rs._n(3) == "three" and rs._n(40) == "40"
    assert len(rs.REVIEW_RECORD) == 24 and all(len(r) == 4 for r in rs.REVIEW_RECORD)
