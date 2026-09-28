"""SPXL timing-signal search, run exactly as pre-registered (research/PREREGISTRATION.md).

The plan (research/PREREGISTRATION.md, which governs; research/prereg.json is its machine-readable
copy of Part 2; hashes in research/PREREG_HASH.txt) fixes the candidates, their point-in-time
construction, the targets, the train / test split, the tests and the pass bar before the test period
(outcome windows ending after 2007-12-31) is looked at. This script is the only code that touches it:

  build    the monthly research panel: every candidate C01-C12 at every month-end origin
           1990-01..2026-08, point in time (market data dated on or before the session before the
           origin, macro data only once released, ALFRED vintages where the plan says so), plus the
           targets at train origins through the seal. Writes output/research/panel_train.pkl,
           panel_test.pkl (predictors only: its targets stay sealed) and panel_meta.json, after
           data-quality checks only (coverage, plausibility, release lags, point-in-time
           truncation invariance, target reproduction against the pinned backtest at train origins).
  train    Phase 2. Reads output/research/panel_train.pkl only (never panel_test.pkl). Sets the
           section 8 quantities from predictor values at the parameter-window month-ends
           (1990-01-31..2007-06-29; no outcomes): mu_j, sd_j, the composite threshold tau, tau_j,
           the train median of COMP, the vol-managed c and w_bar, the SLOOS lag branch. Writes the
           frozen test specification research/frozen_spec.json (deterministic; byte-identical copy
           at output/research/train_params.json, the plan's name), research/FROZEN_HASH.txt,
           research/TRAIN_REPORT.md (the hashes the code guard checks) and research/train_results.md.
           Also computes, for information only, the train-period replication (section 10: the 13
           predictors against every target on train origins, overlap-aware) and the train-side
           sanity checks (coverage, standardisation, descriptives, collinearity, TA diagnostic,
           target reproduction). None of those statistics feeds the spec.
  test     Phase 3, run once: ``test --unseal <SHA-256 of research/PREREGISTRATION.md>``. Refuses a
           wrong token before reading anything; checks every hash and the frozen spec against the code,
           recomputes the section 8 quantities from panel_train.pkl and rebuilds every stored predictor
           (equality only) before unsealing; then computes, with every number taken from
           research/frozen_spec.json, the 13 primary tests (Holm), F2-F7, the train replication, the
           economic test and its secondary analyses, the robustness checks and the Test-phase sanity
           checks, and the verdict in the registered words. Writes output/research/test_results.json,
           output/research/test/*, research/test_results.md and research/RESULTS.md (at first the same
           bytes; `report` then turns RESULTS.md into the owner's report).
  report   re-renders research/test_results.md from test_results.json, writes research/RESULTS.md (the
           owner's report: plain-language summary, the registered tables, exploratory analyses computed
           after unsealing from the saved outputs and cached in output/research/exploratory.json,
           limitations, how to reproduce) and research/OUTPUT_SHA256.txt (hashes of the git-ignored files).

Seal (section 15 code guard): the outcome loader truncates the backtest's inputs at 2007-12-31
before the daily frame is built, so no outcome window ending later can be computed, unless it is
given the unseal token and every hash check passes. ``build`` computes no statistic of a
candidate against a target; ``train`` computes them on train origins only (windows ending on or
before 2007-12-31), from panel_train.pkl, whose seal it checks before use.

Every run appends to output/research/look_log.csv (UTC time, phase, command line, spec hashes,
``git describe --always --dirty``), and so does every successful unseal, whatever script calls it.

Usage:  py scripts/research_signals.py build [--refresh] [--offline] [--pit-checks 20]
        py scripts/research_signals.py train [--refreeze]
        py scripts/research_signals.py test --unseal <token>
        py scripts/research_signals.py report [--recompute]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_rating as br                                  # noqa: E402
from spxlcast.config import Config                            # noqa: E402
from spxlcast.evaluation import effective_n                   # noqa: E402

# ---------------------------------------------------------------------------------------
# Paths and registered constants (PREREGISTRATION.md sections 3-7)
# ---------------------------------------------------------------------------------------
RESEARCH = ROOT / "research"
PREREG_MD = RESEARCH / "PREREGISTRATION.md"
PREREG_JSON = RESEARCH / "prereg.json"
PREREG_HASH = RESEARCH / "PREREG_HASH.txt"
TRAIN_REPORT = RESEARCH / "TRAIN_REPORT.md"
BACKTEST = ROOT / "output" / "backtest"
OUT = ROOT / "output" / "research"
RAW = OUT / "raw"
PREDICTOR_INPUTS = OUT / "predictor_inputs.pkl"
TRAIN_PARAMS = OUT / "train_params.json"
LOOK_LOG = OUT / "look_log.csv"

PINNED = {   # section 3: if a hash does not match, the script stops
    "rating_inputs.pkl": "d21e55055f35d91e5084f336b115b6dedcd99e8819598a1a15544851084d868e",
    "rating_backtest.csv": "a272f0cadabefc75f1c88d4026444c3961cd9ce490f75ee65a763968c2fb4be4",
    "ie_data.xls": "044196dafe44c3030b2facbdea023975b3f6aa68b4e52f8f9bafc403e19589c1",
}
LEGACY_COLUMNS = ("date", "pos", "hurdle", "score")      # the only columns the legacy loader reads

FIRST_ORIGIN = pd.Timestamp("1990-01-31")
LAST_ORIGIN = pd.Timestamp("2026-08-31")
SEAL_END = pd.Timestamp("2007-12-31")        # last session an outcome window may reach before unsealing
TEST_START = pd.Timestamp("2008-01-01")
DATA_CUTOFF = pd.Timestamp("2026-09-25")     # nothing dated later is used
PARAM_WINDOW = (pd.Timestamp("1990-01-31"), pd.Timestamp("2007-06-29"))
HORIZONS = (21, 63, 126)
DIP_H = 63
DIP_LEVEL = 0.8
STALE_SESSIONS = 5                           # a daily value older than this at t- is missing
GAP_FIT_START = pd.Timestamp("1948-01-01")
CLAIMS_RELEASE_DAYS = 5                      # week ending Saturday w is released Thursday w + 5
SLOOS_QA_DAYS = 120
ECY_MIN_MONTHS = 114                         # see research/DEVIATIONS.md (unpublished CPI months)
COMPOSITE_MIN_MEMBERS = 7

# id, short name, registered sign, member of the composite
CANDIDATES: Tuple[Tuple[str, str, int, bool], ...] = (
    ("C01", "VRP", +1, True), ("C02", "GAP", -1, True), ("C03", "SRATE", -1, True),
    ("C04", "TERM", +1, True), ("C05", "DEF", +1, True), ("C06", "INFL", -1, True),
    ("C07", "SLOOS", -1, True), ("C08", "CLAIMS", -1, True), ("C09", "ECY", +1, True),
    ("C10", "IVAR", -1, False), ("C11", "HURDLE", -1, False), ("C12", "SCORE", +1, False),
)
CAND_IDS = [c[0] for c in CANDIDATES]
TARGETS = ["Y21", "Y63", "Y126", "S21", "S63", "S126", "D20"]

# Section 13.7-style plausibility bounds (data-quality only; applied to train-side rows)
PLAUSIBLE = {
    "C01": (-1500.0, 1000.0), "C02": (-0.35, 0.35), "C03": (-7.0, 7.0), "C04": (-4.0, 6.0),
    "C05": (0.2, 4.0), "C06": (-0.05, 0.16), "C07": (-100.0, 100.0), "C08": (-1.0, 3.0),
    "C09": (-0.10, 0.20), "C10": (0.004, 0.80), "C11": (-0.10, 0.60), "C12": (-1.0, 1.0),
}

# FRED / ALFRED downloads (cached under output/research/raw, assembled once into predictor_inputs.pkl)
CUTOFF_S = DATA_CUTOFF.strftime("%Y-%m-%d")
FETCH: Dict[str, Dict[str, str]] = {
    # full vintage histories: each origin reads the vintage in effect on its date
    "INDPRO": {"realtime_start": "1990-01-01", "realtime_end": CUTOFF_S},
    "CPIAUCNS": {"realtime_start": "1990-01-01", "realtime_end": CUTOFF_S},
    # values from the vintage at the cutoff; the full history gives each observation's first release
    "DRTSCILM": {"realtime_start": "1776-07-04", "realtime_end": CUTOFF_S},
    "ICNSA": {"realtime_start": "1776-07-04", "realtime_end": CUTOFF_S},
    # daily market series, not revised: the vintage at the cutoff
    "DGS3MO": {"realtime_start": CUTOFF_S, "realtime_end": CUTOFF_S, "observation_start": "1981-01-01"},
    "DGS10": {"realtime_start": CUTOFF_S, "realtime_end": CUTOFF_S, "observation_start": "1981-01-01"},
    "DBAA": {"realtime_start": CUTOFF_S, "realtime_end": CUTOFF_S, "observation_start": "1985-01-01"},
    "DAAA": {"realtime_start": CUTOFF_S, "realtime_end": CUTOFF_S, "observation_start": "1985-01-01"},
    # cross-checks only
    "VIXCLS": {"realtime_start": CUTOFF_S, "realtime_end": CUTOFF_S, "observation_start": "1990-01-01"},
    "T10Y3M": {"realtime_start": CUTOFF_S, "realtime_end": CUTOFF_S, "observation_start": "1982-01-01"},
    # release dates only (research/DEVIATIONS.md D5): when each unemployment rate the backtest's C12
    # used was first published; the values are never read
    "UNRATE": {"realtime_start": "1776-07-04", "realtime_end": CUTOFF_S},
}
# direct ALFRED queries (realtime_start = realtime_end = t) that cross-check the vintage reconstruction
VINTAGE_CHECK_ORIGINS = ("1990-01-31", "1996-07-31", "2001-10-31", "2007-06-29")
VINTAGE_CHECK_SERIES = ("INDPRO", "CPIAUCNS")


class SealError(RuntimeError):
    """Raised when the test period would be opened without a valid unseal."""


# ---------------------------------------------------------------------------------------
# Hashes, spec check, look log
# ---------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


CODE_FILES = ("scripts/research_signals.py", "scripts/backtest_rating.py", "spxlcast/evaluation.py",
              "spxlcast/config.py")


def code_provenance(copy_to: Optional[Path] = None, root: Path = ROOT) -> Dict[str, str]:
    """SHA-256 of the code the numbers come from; with ``copy_to``, also a byte copy of each file there."""
    out = {}
    for rel in CODE_FILES:
        p = root / rel
        if not p.exists():
            out[rel] = "missing"
            continue
        out[rel] = sha256_file(p)
        if copy_to is not None:
            dst = copy_to / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(p.read_bytes())
    return out


def read_prereg_hashes(path: Path = PREREG_HASH) -> Dict[str, str]:
    """{relative path: sha256} from PREREG_HASH.txt."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+\*?(\S+)\s*$", line.strip())
        if m:
            out[m.group(2)] = m.group(1)
    return out


def check_spec(md: Path = PREREG_MD, js: Path = PREREG_JSON, hashes: Path = PREREG_HASH) -> Dict[str, str]:
    """The frozen plan must be byte-identical to the hashes recorded at the freeze."""
    rec = read_prereg_hashes(hashes)
    got = {"research/PREREGISTRATION.md": sha256_file(md), "research/prereg.json": sha256_file(js)}
    for name, h in got.items():
        if rec.get(name) != h:
            raise SystemExit(f"STOP: {name} does not match research/PREREG_HASH.txt (the plan changed after the freeze)")
    return got


def check_pinned(backtest_dir: Path = BACKTEST, pinned: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    out = {}
    for name, want in (PINNED if pinned is None else pinned).items():
        got = sha256_file(backtest_dir / name)
        if got != want:
            raise SystemExit(f"STOP: pinned input output/backtest/{name} has SHA-256 {got}, registered {want}")
        out[f"output/backtest/{name}"] = got
    return out


def _recorded_hash(report_text: str, name: str) -> Optional[str]:
    for line in report_text.splitlines():
        if name in line:
            m = re.search(r"\b[0-9a-f]{64}\b", line)
            if m:
                return m.group(0)
    return None


def git_describe() -> str:
    try:
        r = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT, capture_output=True,
                           text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def append_look_log(phase: str, argv: Sequence[str], path: Optional[Path] = None,
                    command: Optional[str] = None) -> None:
    """Section 13.13: every run is logged, whatever it does. ``command`` replaces the default
    ``py scripts/research_signals.py <argv>`` (the unseal log below records the process's own
    command line, whatever script imported this module)."""
    path = LOOK_LOG if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        md, js = sha256_file(PREREG_MD), sha256_file(PREREG_JSON)
    except OSError:
        md = js = "missing"
    row = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "phase": phase,
           "command": command if command is not None else " ".join(["py", "scripts/research_signals.py", *argv]),
           "prereg_md_sha256": md, "prereg_json_sha256": js, "git_describe": git_describe()}
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------------------
# The seal (section 15 code guard) and the targets (section 4)
# ---------------------------------------------------------------------------------------
def verify_unseal(token: str, md: Path = PREREG_MD, js: Path = PREREG_JSON, hashes: Path = PREREG_HASH,
                  params: Path = TRAIN_PARAMS, report: Path = TRAIN_REPORT, log: Optional[Path] = None,
                  caller: str = "verify_unseal") -> None:
    """Unsealing needs: the token equals the SHA-256 of PREREGISTRATION.md; both spec files match
    PREREG_HASH.txt; train_params.json exists and matches the hash recorded in TRAIN_REPORT.md.

    Every successful unseal is written to the look log (``log``; by default the look_log.csv next to
    ``params``, i.e. output/research/look_log.csv in the real layout) with the process's own command
    line, so an unsealing is logged whatever the entry point (added after the Test phase run,
    research/DEVIATIONS.md N28; before it only ``main()`` wrote the log)."""
    _check_unseal(token, md, js, hashes, params, report)
    append_look_log("unseal", [], params.parent / LOOK_LOG.name if log is None else log,
                    command=f"py {' '.join(sys.argv)} [{caller}]")


def _check_unseal(token: str, md: Path, js: Path, hashes: Path, params: Path, report: Path) -> None:
    md_sha = sha256_file(md)
    if not isinstance(token, str) or token.strip().lower() != md_sha:
        raise SealError("unseal token is not the SHA-256 of research/PREREGISTRATION.md")
    rec = read_prereg_hashes(hashes)
    if rec.get("research/PREREGISTRATION.md") != md_sha or rec.get("research/prereg.json") != sha256_file(js):
        raise SealError("the plan does not match research/PREREG_HASH.txt")
    if not params.exists():
        raise SealError(f"{params.name} does not exist: the Train phase has not run")
    if not report.exists():
        raise SealError(f"{report.name} does not exist: the Train phase has not run")
    recorded = _recorded_hash(report.read_text(encoding="utf-8"), params.name)
    if recorded is None or recorded != sha256_file(params):
        raise SealError(f"{params.name} does not match the hash recorded in {report.name}")


def sealed_inputs(inputs: Dict, end: pd.Timestamp = SEAL_END) -> Dict:
    """The backtest inputs with every daily and FRED observation after ``end`` removed, so that the
    daily frame (and hence any outcome) cannot reach past it."""
    closes = inputs["closes"].loc[inputs["closes"].index <= end]
    fred = {k: s.loc[s.index <= end] for k, s in inputs["fred"].items()}
    return {**inputs, "closes": closes, "fred": fred}


def outcome_frame(inputs: Dict, unseal_token: Optional[str] = None, cfg: Optional[Config] = None,
                  **verify_kwargs) -> pd.DataFrame:
    """The daily frame outcomes are computed from. Sealed (the default): built from inputs
    truncated at 2007-12-31. Unsealed only through ``verify_unseal``. ``cfg`` (unsealed only) is the
    section 13.12 synthetic-spread variant; the default is Config()."""
    if unseal_token is None:
        d = br.build_daily(sealed_inputs(inputs), Config())
        if len(d) and d.index.max() > SEAL_END:            # belt and braces
            raise SealError("sealed outcome frame reaches past 2007-12-31")
        return d
    verify_unseal(unseal_token, caller="outcome_frame", **verify_kwargs)
    d = br.build_daily(inputs, Config() if cfg is None else cfg)
    check_spxl_closes(d)
    return d.loc[d.index <= DATA_CUTOFF]


def check_spxl_closes(d: pd.DataFrame) -> None:
    """The realised fund is SPXL from backtest_rating.SPXL_FROM. build_daily takes pct_change() of the
    SPXL closes and fills a missing return with 0, so a missing close inside the SPXL series would
    silently turn returns into zeros (and, under pandas 3's fill_method=None, lose two days).
    Stop instead. The pinned data have no such gap (added after the Test phase run,
    research/DEVIATIONS.md N28; it changes no number)."""
    s = d.loc[d.index >= br.SPXL_FROM, "spxl"] if "spxl" in d.columns else pd.Series(dtype=float)
    if s.notna().any():
        inner = s.loc[s.first_valid_index(): s.last_valid_index()]
        if s.first_valid_index() != s.index[0] or inner.isna().any():
            gaps = list(s.index[s.isna()][:5].strftime("%Y-%m-%d"))
            raise ValueError(f"SPXL closes missing on sessions after {br.SPXL_FROM:%Y-%m-%d}: {gaps}")


def compute_targets(d: pd.DataFrame, positions: Sequence[int], cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Y(t,h), S(t,h) and D20(t) at session positions of ``cal`` (section 4), computed exactly as the
    backtest's excess_h, sp_h - tbill_h and real_dd20_63. A window that runs past the end of ``d``
    (the seal, or the data) is missing."""
    if not d.index.equals(cal[:len(d)]):
        raise ValueError("outcome frame is not a prefix of the session calendar")
    fund, tr, tb = d["fund"].values, d["tr"].values, d["tbill_ret"].values
    n = len(d)
    rows = []
    for pos in positions:
        rec = {"date": cal[pos]}
        for h in HORIZONS:
            y = s = np.nan
            if pos + h < n:
                tbill = float(np.prod(1.0 + tb[pos + 1: pos + h + 1]) - 1.0)
                window_end = fund[pos + h] / fund[pos]
                y = float(window_end - 1.0 - tbill)
                s = float(tr[pos + h] / tr[pos] - 1.0) - tbill
            rec[f"Y{h}"], rec[f"S{h}"] = y, s
        dd = np.nan
        if pos + DIP_H < n:
            window = fund[pos + 1: pos + DIP_H + 1] / fund[pos]
            dd = float(window.min() <= DIP_LEVEL)
        rec["D20"] = dd
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------------------
# FRED / ALFRED download (key never printed or put in a message) and the raw cache
# ---------------------------------------------------------------------------------------
FRED_BASE = "https://api.stlouisfed.org/fred/"
MAX_DATE = "2262-01-01"      # below pandas' Timestamp ceiling; stands in for FRED's 9999-12-31


def _fred_get(endpoint: str, params: Dict[str, str], key: str, timeout: float = 90.0) -> Dict:
    import requests
    try:
        r = requests.get(FRED_BASE + endpoint, params={**params, "api_key": key, "file_type": "json"},
                         timeout=timeout, headers={"User-Agent": "Mozilla/5.0 spxlcast-research"})
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        # requests puts the full URL, key included, in its messages: report the failure without it
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise RuntimeError(f"FRED {endpoint} {params.get('series_id', '')} request failed: {type(exc).__name__}"
                           + (f" (HTTP {status})" if status else "")) from None
    except ValueError:
        raise RuntimeError(f"FRED {endpoint} {params.get('series_id', '')}: response is not JSON") from None


def fred_rows(series_id: str, key: str, **params: str) -> pd.DataFrame:
    """Every observation row (paged), values kept as FRED's strings ('.' = missing)."""
    rows: List[Dict] = []
    offset = 0
    while True:
        j = _fred_get("series/observations", {"series_id": series_id, "limit": "100000",
                                              "offset": str(offset), **params}, key)
        obs = j.get("observations", [])
        rows += obs
        offset += len(obs)
        if not obs or offset >= int(j.get("count", offset)):
            break
        time.sleep(0.6)
    return pd.DataFrame(rows, columns=["date", "realtime_start", "realtime_end", "value"])


def parse_rows(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for c in ("date", "realtime_start", "realtime_end"):
        df[c] = pd.to_datetime(df[c].astype(str).str.replace("9999-12-31", MAX_DATE, regex=False))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")        # '.' -> NaN (a missing observation)
    return df.sort_values(["date", "realtime_start"]).reset_index(drop=True)


def _raw_path(name: str) -> Path:
    return RAW / f"{name}.csv"


def fetch_raw(refresh: bool, offline: bool) -> Dict[str, Dict]:
    """Download what is not cached (or everything with ``refresh``); returns the manifest."""
    RAW.mkdir(parents=True, exist_ok=True)
    mpath = RAW / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    wanted = [(sid, params) for sid, params in FETCH.items()]
    wanted += [(f"{sid}__vintage_{t}", {"series_id": sid, "realtime_start": t, "realtime_end": t})
               for sid in VINTAGE_CHECK_SERIES for t in VINTAGE_CHECK_ORIGINS]
    missing = [(n, p) for n, p in wanted if refresh or not _raw_path(n).exists() or n not in manifest]
    if missing and offline:
        raise SystemExit(f"--offline but not cached: {', '.join(n for n, _ in missing)}")
    if missing:
        from spxlcast.env import fred_api_key
        key = fred_api_key()
        if not key:
            raise SystemExit("FRED_API_KEY is needed (put it in .env) to download the predictor inputs")
        for name, params in missing:
            p = dict(params)
            sid = p.pop("series_id", name)
            df = fred_rows(sid, key, **p)
            if df.empty:
                raise SystemExit(f"FRED returned no observations for {name}")
            df.to_csv(_raw_path(name), index=False)
            manifest[name] = {"series_id": sid, "params": p, "rows": int(len(df)),
                              "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "sha256": sha256_file(_raw_path(name))}
            print(f"  fetched {name}: {len(df)} rows", flush=True)
            time.sleep(0.6)
        mpath.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    for name, _ in wanted:
        if sha256_file(_raw_path(name)) != manifest[name]["sha256"]:
            raise SystemExit(f"STOP: raw cache {name}.csv changed since it was downloaded")
    return manifest


def load_predictor_inputs(refresh: bool = False, offline: bool = False) -> Dict:
    """predictor_inputs.pkl: every FRED / ALFRED series the candidates need beyond the pinned files.
    Assembled once from the raw cache; later runs (and the Train and Test phases) read that same
    file, whose SHA-256 the Train phase records in research/TRAIN_REPORT.md."""
    if PREDICTOR_INPUTS.exists() and not refresh:
        pi = pd.read_pickle(PREDICTOR_INPUTS)
        if set(FETCH) <= set(pi["series"]):
            return pi
        # a series was added to FETCH (the point-in-time audit, DEVIATIONS.md D5): allowed only until the
        # Train phase records the file's hash; the cached raw files are reused, the new one is fetched
        if TRAIN_REPORT.exists() and _recorded_hash(TRAIN_REPORT.read_text(encoding="utf-8"), PREDICTOR_INPUTS.name):
            raise SystemExit(f"STOP: {PREDICTOR_INPUTS.name} lacks {sorted(set(FETCH) - set(pi['series']))} but its "
                             f"hash is already recorded in {TRAIN_REPORT.name}")
    manifest = fetch_raw(refresh, offline)
    series = {name: parse_rows(pd.read_csv(_raw_path(name), dtype=str)) for name in manifest if name in FETCH
              or "__vintage_" in name}
    pi = {"series": series, "manifest": manifest,
          "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
          "note": "FRED/ALFRED rows: date, realtime_start, realtime_end, value (NaN where FRED has '.'); "
                  f"realtime periods are inclusive; {MAX_DATE} stands for 9999-12-31"}
    PREDICTOR_INPUTS.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(pi, PREDICTOR_INPUTS)
    return pd.read_pickle(PREDICTOR_INPUTS)


def check_predictor_inputs_hash(report: Path = TRAIN_REPORT, path: Path = PREDICTOR_INPUTS) -> Optional[str]:
    """Once the Train phase has recorded the file's hash, a different file stops the script."""
    if not report.exists():
        return None
    recorded = _recorded_hash(report.read_text(encoding="utf-8"), path.name)
    got = sha256_file(path)
    if recorded is not None and recorded != got:
        raise SystemExit(f"STOP: {path.name} does not match the hash recorded in {report.name}")
    return recorded


# ---------------------------------------------------------------------------------------
# Point-in-time helpers
# ---------------------------------------------------------------------------------------
def add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    """The first day of the month ``n`` months after ``ts``'s month."""
    return (pd.Timestamp(ts).to_period("M") + n).to_timestamp()


def last_session_in_month(cal: pd.DatetimeIndex, period: pd.Period) -> Optional[int]:
    i = int(cal.searchsorted((period + 1).start_time, side="left")) - 1
    if i < 0 or cal[i].to_period("M") != period:
        return None
    return i


def asof(series: pd.Series, cal: pd.DatetimeIndex, pos: int, max_age: int = STALE_SESSIONS) -> float:
    """The last value dated on or before session ``cal[pos]``, if it is at most ``max_age``
    sessions old, else NaN. ``series`` must be sorted and free of NaN."""
    if pos < 0:
        return float("nan")
    i = int(series.index.searchsorted(cal[pos], side="right")) - 1
    if i < 0:
        return float("nan")
    j = int(cal.searchsorted(series.index[i], side="right")) - 1       # the last session on or before it
    if j < 0 or pos - j > max_age:
        return float("nan")
    return float(series.iat[i])


def realised_var21(g: np.ndarray, pos: int) -> float:
    """Sum over the 21 sessions ending at ``pos`` of (100 ln(G_i / G_{i-1}))^2 (percent squared per
    month): a sign-blind risk measure, used only inside C01."""
    if pos < 21:
        return float("nan")
    w = g[pos - 21: pos + 1]
    if len(w) != 22 or not np.all(np.isfinite(w)) or np.any(w <= 0):
        return float("nan")
    r = 100.0 * np.diff(np.log(w))
    return float(np.sum(r * r))


class Realtime:
    """ALFRED rows (date, realtime_start, realtime_end, value) and the vintage in effect on a date."""

    def __init__(self, rows: pd.DataFrame):
        self.rows = rows.sort_values(["date", "realtime_start"]).reset_index(drop=True)
        rs = self.rows["realtime_start"].values
        re_ = self.rows["realtime_end"].values
        self._rs, self._re = rs, re_
        change = np.union1d(rs, re_ + np.timedelta64(1, "D"))
        self._change = np.sort(change)
        self._cache: Dict[np.datetime64, pd.Series] = {}

    def at(self, t: pd.Timestamp) -> pd.Series:
        """Observations (date -> value, NaN where missing) valid on date ``t``
        (realtime_start <= t <= realtime_end), i.e. the vintage FRED serves for
        realtime_start = realtime_end = t."""
        t64 = np.datetime64(pd.Timestamp(t), "ns")
        k = int(np.searchsorted(self._change, t64, side="right")) - 1
        key = self._change[k] if k >= 0 else np.datetime64("NaT")
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        m = (self._rs <= t64) & (self._re >= t64)
        v = self.rows.loc[m, ["date", "value"]].set_index("date")["value"].sort_index()
        if v.index.has_duplicates:
            raise ValueError("two values for one observation in one vintage")
        self._cache[key] = v
        return v

    def vintage_date(self, t: pd.Timestamp) -> Optional[pd.Timestamp]:
        """The latest release on or before ``t`` (the start of the newest row valid at t)."""
        t64 = np.datetime64(pd.Timestamp(t), "ns")
        m = (self._rs <= t64) & (self._re >= t64)
        return pd.Timestamp(self._rs[m].max()) if m.any() else None

    def truncated(self, t: pd.Timestamp) -> "Realtime":
        """Drop every row released after ``t``."""
        return Realtime(self.rows.loc[self.rows["realtime_start"] <= pd.Timestamp(t)])

    def first_release(self) -> pd.Series:
        """Observation date -> the first vintage that contains it."""
        return self.rows.groupby("date")["realtime_start"].min()


def cpi_yoy(vintage: pd.Series) -> Tuple[float, Optional[pd.Timestamp]]:
    """(ln(CPI_m / CPI_{m-12}), m) with m the vintage's latest month with a value; NaN if the month a
    year earlier has no value (the caller then carries the previous origin's value forward)."""
    v = vintage.dropna()
    if v.empty:
        return float("nan"), None
    m = v.index[-1]
    base = v.get(add_months(m, -12))
    if base is None or not np.isfinite(base) or base <= 0 or v.iat[-1] <= 0:
        return float("nan"), m
    return float(math.log(v.iat[-1] / base)), m


def output_gap(vintage: pd.Series) -> Tuple[float, Optional[pd.Timestamp]]:
    """Residual at the latest month m of an OLS fit of ln IP_k = a + b k + c k^2 over months from
    1948-01 (or the vintage's first observation, if later) through m."""
    v = vintage.dropna()
    v = v[v > 0]
    if v.empty:
        return float("nan"), None
    start = max(GAP_FIT_START, v.index[0])
    v = v[v.index >= start]
    if len(v) < 4:
        return float("nan"), v.index[-1] if len(v) else None
    k = ((v.index.year - start.year) * 12 + (v.index.month - start.month)).to_numpy(float)
    x = k / k[-1]                               # rescaled for conditioning; the fit is unchanged
    X = np.column_stack([np.ones_like(x), x, x * x])
    y = np.log(v.to_numpy(float))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(y[-1] - X[-1] @ beta), v.index[-1]


def sloos_usable_dates(obs_dates: Iterable[pd.Timestamp], cal: pd.DatetimeIndex, shift: int = 0) -> pd.Series:
    """An observation dated at the start of quarter q is usable from the last session of the first
    month after q ends (plus ``shift`` months, the QA branch). A month past the calendar uses its
    last calendar day."""
    out = {}
    for d in obs_dates:
        d = pd.Timestamp(d)
        month = pd.Period(d, "Q").asfreq("M", "end") + 1 + shift
        i = last_session_in_month(cal, month)
        out[d] = cal[i] if i is not None else month.end_time.normalize()
    return pd.Series(out, dtype="datetime64[ns]").sort_index()


def sloos_at(values: pd.Series, usable: pd.Series, t: pd.Timestamp) -> Tuple[float, Optional[pd.Timestamp]]:
    ok = usable[usable <= t]
    if ok.empty:
        return float("nan"), None
    obs = ok.index.max()
    v = values.get(obs)
    return (float(v) if v is not None and np.isfinite(v) else float("nan")), obs


def sloos_values(rt: Realtime, t: pd.Timestamp) -> pd.Series:
    """DRTSCILM observation -> value in the vintage in effect at t (research/DEVIATIONS.md D3). ALFRED
    has vintages only from 2010-04-20; before that, the earliest vintage stands in for the values
    published at the time (the Board revised 1990-1997 and a few later values on 2019-11-04; the
    2010 vintage matches what the Board reported in 1990-91, the revised values do not)."""
    first = pd.Timestamp(rt.rows["realtime_start"].min())
    return rt.at(max(pd.Timestamp(t), first))


def sloos_qa(first_release: pd.Series, usable: pd.Series, max_days: int = SLOOS_QA_DAYS) -> Dict:
    """For observations whose first ALFRED vintage lies within ``max_days`` of the observation
    date, the rule's usable date must be on or after that vintage date."""
    fr = first_release.reindex(usable.index).dropna()
    lag = (fr - fr.index.to_series()).dt.days
    checked = fr[lag <= max_days]
    bad = checked[usable.reindex(checked.index) < checked]
    return {"n_checked": int(len(checked)), "n_violations": int(len(bad)),
            "violations": [{"obs": f"{o:%Y-%m-%d}", "first_vintage": f"{v:%Y-%m-%d}",
                            "usable": f"{usable[o]:%Y-%m-%d}"} for o, v in bad.items()],
            "first_checked_obs": f"{checked.index.min():%Y-%m-%d}" if len(checked) else None}


def claims_release_dates(weeks: Iterable[pd.Timestamp], first_release: Optional[pd.Series] = None) -> pd.Series:
    """Week-ending Saturday w -> the date its figure was first published: the Thursday rule w + 5 days,
    or ALFRED's first-release date where that is later (research/DEVIATIONS.md D4: the October-November
    2025 shutdown, when no national claims were published from late September until 2025-11-20).
    ALFRED records claims vintages only from its first vintage (2009-05-28); a week first seen on that
    date was published earlier, so it keeps the Thursday rule."""
    weeks = pd.DatetimeIndex(weeks)
    out = pd.Series(weeks + pd.Timedelta(days=CLAIMS_RELEASE_DAYS), index=weeks)
    if first_release is not None and len(first_release):
        start = first_release.min()
        fr = first_release.reindex(weeks)
        m = (fr.notna() & (fr > start)).values
        out[m] = np.maximum(out[m].values, fr[m].values)
    return out


def claims_week(t: pd.Timestamp, release: Optional[pd.Series] = None) -> pd.Timestamp:
    """The latest week-ending Saturday w with w + 5 days <= t (released on the Thursday after) and,
    given ``release`` (claims_release_dates), actually published by t."""
    t = pd.Timestamp(t).normalize()
    sat = t - pd.Timedelta(days=(t.weekday() - 5) % 7)
    if sat + pd.Timedelta(days=CLAIMS_RELEASE_DAYS) > t:
        sat -= pd.Timedelta(days=7)
    if release is not None:
        ok = release.index[(release.values <= np.datetime64(t)) & (release.index <= sat)]
        if len(ok):
            sat = ok.max()
    return sat


def claims_yoy(icnsa: pd.Series, t: pd.Timestamp, release: Optional[pd.Series] = None) -> Tuple[float, pd.Timestamp]:
    """ln(sum of ICNSA over weeks w-3..w) - ln(sum over weeks w-55..w-52)."""
    w = claims_week(t, release)
    now = icnsa.reindex([w - pd.Timedelta(weeks=j) for j in range(4)])
    ago = icnsa.reindex([w - pd.Timedelta(weeks=j) for j in range(52, 56)])
    if now.isna().any() or ago.isna().any() or now.sum() <= 0 or ago.sum() <= 0:
        return float("nan"), w
    return float(math.log(now.sum()) - math.log(ago.sum())), w


def excess_cape_yield(t: pd.Timestamp, p_tm: float, y10_tm: float, cpi_vintage: pd.Series,
                      earnings: pd.Series) -> Dict[str, float]:
    """C09: CAPE = (P(t-) / CPI_m) / mean over k = q-119..q of (E_k / CPI_k), q the last reported
    quarter (backtest_rating.reported_quarter), m the latest CPI month in the vintage at t;
    RR10 = DGS10(t-)/100 - ((CPI_m / CPI_{m-120})^(1/10) - 1); ECY = 1/CAPE - RR10.
    Months whose CPI was never published are left out of the mean (never interpolated)."""
    nan = float("nan")
    out = {"C09": nan, "cape": nan, "rr10": nan, "ecy_q": pd.NaT, "ecy_months": 0}
    q = br.reported_quarter(pd.Timestamp(t))
    out["ecy_q"] = q
    cpi = cpi_vintage.dropna()
    if cpi.empty or not np.isfinite(p_tm) or not np.isfinite(y10_tm):
        return out
    m = cpi.index[-1]
    ks = pd.date_range(add_months(q, -119), q, freq="MS")
    e = earnings.reindex(ks)
    if e.isna().any():
        return out
    ratio = (e / cpi.reindex(ks)).dropna()
    out["ecy_months"] = int(len(ratio))
    if len(ratio) < ECY_MIN_MONTHS:
        return out
    cape = (p_tm / cpi.iat[-1]) / float(ratio.mean())
    base = cpi.get(add_months(m, -120))
    if base is None or not np.isfinite(base) or base <= 0 or cape <= 0:
        out["cape"] = cape
        return out
    rr10 = y10_tm / 100.0 - ((cpi.iat[-1] / base) ** 0.1 - 1.0)
    out.update({"C09": 1.0 / cape - rr10, "cape": cape, "rr10": rr10})
    return out


# ---------------------------------------------------------------------------------------
# The inputs of every candidate, and the panel
# ---------------------------------------------------------------------------------------
@dataclasses.dataclass
class PredictorData:
    cal: pd.DatetimeIndex            # session calendar (build_daily's ^SP500TR days)
    gspc: pd.Series                  # ^GSPC closes (pinned pickle)
    vix: pd.Series                   # ^VIX closes (pinned pickle)
    dgs3mo: pd.Series                # FRED, percent
    dgs10: pd.Series
    dbaa: pd.Series
    daaa: pd.Series
    indpro: Realtime                 # ALFRED INDPRO
    cpi: Realtime                    # ALFRED CPIAUCNS
    sloos: pd.Series                 # DRTSCILM, observation date -> value (vintage at the cutoff)
    sloos_shift: int                 # 0, or 1 if the QA branch moved every usable date a month later
    icnsa: pd.Series                 # week-ending Saturday -> claims
    earnings: pd.Series              # Shiller E, month start -> trailing 12-month earnings
    legacy: pd.DataFrame             # date -> hurdle, score (pinned rating_backtest.csv)
    # point-in-time audit (research/DEVIATIONS.md D3-D5); None keeps the Build-phase behaviour
    sloos_rt: Optional[Realtime] = None          # DRTSCILM vintages: values as published at t
    icnsa_release: Optional[pd.Series] = None    # week -> first publication date (claims_release_dates)
    legacy_blocked: Optional[pd.Series] = None   # origin -> why C12 used an input not yet published


def _clean(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.loc[s.index <= DATA_CUTOFF].astype(float)


def session_calendar(inputs: Dict) -> pd.DatetimeIndex:
    """build_daily's calendar: its frame is indexed by the ^SP500TR closes (read directly here, so
    the full-period fund is never built; compute_targets checks that the sealed frame's index is a
    prefix of this calendar)."""
    cal = inputs["closes"]["^SP500TR"].dropna().index
    return cal[cal <= DATA_CUTOFF]


def origin_positions(cal: pd.DatetimeIndex) -> List[int]:
    pos = [p for p in br.month_end_positions(cal, "1990-01") if cal[p] <= LAST_ORIGIN]
    return pos


def load_legacy(path: Path = BACKTEST / "rating_backtest.csv") -> pd.DataFrame:
    """C11 and C12 from the pinned CSV: reads only the columns date, pos, hurdle and score (every
    other field is skipped unparsed) after checking the pinned hash."""
    got = sha256_file(path)
    if got != PINNED["rating_backtest.csv"]:
        raise SystemExit(f"STOP: {path.name} does not match its pinned hash")
    return _read_legacy_columns(path)


def _read_legacy_columns(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        ix = {c: header.index(c) for c in LEGACY_COLUMNS}
        for row in r:
            rows.append({c: row[ix[c]] for c in LEGACY_COLUMNS})
    df = pd.DataFrame(rows, columns=list(LEGACY_COLUMNS))
    df["date"] = pd.to_datetime(df["date"])
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    for c in ("hurdle", "score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("date").sort_index()


def first_published(rows: pd.DataFrame) -> pd.Series:
    """Observation date -> the first vintage in which it has a value (a '.' row is not a release)."""
    r = rows.dropna(subset=["value"])
    return r.groupby("date")["realtime_start"].min()


def legacy_unreleased_inputs(origins: Iterable[pd.Timestamp], monthly: Dict[str, pd.Series],
                             first_release: Dict[str, pd.Series]) -> pd.Series:
    """Origins at which the backtest behind C12 (backtest_rating.released: every monthly observation up
    to the month before the origin) used a CPI or unemployment-rate month that was first published
    after the origin (research/DEVIATIONS.md D5). Returns origin -> reason."""
    out = {}
    for t in origins:
        t = pd.Timestamp(t)
        why = []
        for name, s in monthly.items():
            used = br.released(s, t)
            if used.empty:
                continue
            k = used.index[-1]                   # earlier months were published before this one
            fr = first_release[name].get(k)
            if fr is not None and pd.Timestamp(fr) > t:
                why.append(f"{name} {k:%Y-%m} first published {pd.Timestamp(fr):%Y-%m-%d}")
        if why:
            out[t] = "; ".join(why)
    return pd.Series(out, dtype=object)


def assemble(inputs: Dict, pi: Dict, legacy: pd.DataFrame, cal: pd.DatetimeIndex) -> Tuple[PredictorData, Dict]:
    """PredictorData from the pinned inputs and predictor_inputs.pkl; also returns the SLOOS QA."""
    s = pi["series"]
    closes = inputs["closes"]

    def latest(name: str) -> pd.Series:
        rows = s[name]
        v = Realtime(rows).at(DATA_CUTOFF)
        return _clean(v)

    sloos_rt = Realtime(s["DRTSCILM"])
    sloos = _clean(sloos_rt.at(DATA_CUTOFF))
    usable0 = sloos_usable_dates(sloos.index, cal, 0)
    qa0 = sloos_qa(sloos_rt.first_release(), usable0)
    shift = 1 if qa0["n_violations"] else 0
    qa = {"rule": "obs dated at quarter start usable from the last session of the first month after the quarter",
          "shift0": qa0, "branch_shift_months": shift}
    if shift:
        qa["shift1"] = sloos_qa(sloos_rt.first_release(), sloos_usable_dates(sloos.index, cal, 1))
    icnsa = latest("ICNSA")
    if len(icnsa) and not (icnsa.index.weekday == 5).all():
        raise SystemExit("STOP: ICNSA dates are not all Saturdays")
    icnsa_release = claims_release_dates(icnsa.index, first_published(s["ICNSA"]))
    # C12: the backtest's CPI and unemployment inputs against their first publication dates (CPIAUCSL
    # is published in the same BLS release as CPIAUCNS)
    blocked = legacy_unreleased_inputs(
        legacy.index, {"CPI": inputs["fred"]["CPIAUCSL"], "UNRATE": inputs["fred"]["UNRATE"]},
        {"CPI": first_published(s["CPIAUCNS"]), "UNRATE": first_published(s["UNRATE"])})
    data = PredictorData(
        cal=cal, gspc=_clean(closes["^GSPC"]), vix=_clean(closes["^VIX"]),
        dgs3mo=latest("DGS3MO"), dgs10=latest("DGS10"), dbaa=latest("DBAA"), daaa=latest("DAAA"),
        indpro=Realtime(s["INDPRO"]), cpi=Realtime(s["CPIAUCNS"]), sloos=sloos, sloos_shift=shift,
        icnsa=icnsa, earnings=inputs["shiller"]["E"].dropna().astype(float), legacy=legacy,
        sloos_rt=sloos_rt, icnsa_release=icnsa_release, legacy_blocked=blocked)
    return data, qa


def compute_panel(data: PredictorData, positions: Sequence[int]) -> pd.DataFrame:
    """C01-C12 and their components at each origin position (increasing), point in time."""
    cal = data.cal
    g = data.gspc.reindex(cal).to_numpy(float)
    usable = sloos_usable_dates(data.sloos.index, cal, data.sloos_shift)
    rows = []
    prev_infl = float("nan")
    for pos in positions:
        t = cal[pos]
        tm = pos - 1
        rec: Dict[str, object] = {"date": t, "pos": int(pos), "t_minus": cal[tm] if tm >= 0 else pd.NaT}
        # C01 VRP and C10 IVAR: market closes of t-
        vix = asof(data.vix, cal, tm)
        rv = realised_var21(g, tm)
        rec["vix_tm"], rec["rv21_tm"] = vix, rv
        rec["C01"] = vix * vix / 12.0 - rv
        rec["C10"] = (vix / 100.0) ** 2
        # C02 GAP: INDPRO vintage in effect at t
        ip = data.indpro.at(t)
        rec["C02"], rec["indpro_m"] = output_gap(ip)
        # C03 SRATE and C04 TERM: yields at t- (and at t'-, the origin 12 month-ends earlier)
        y3 = asof(data.dgs3mo, cal, tm)
        y10 = asof(data.dgs10, cal, tm)
        tp = last_session_in_month(cal, t.to_period("M") - 12)
        y3p = asof(data.dgs3mo, cal, tp - 1) if tp is not None else float("nan")
        rec["y3_tm"], rec["y3_tprev"], rec["y10_tm"] = y3, y3p, y10
        rec["C03"] = y3 - y3p
        rec["C04"] = y10 - y3
        # C05 DEF
        baa, aaa = asof(data.dbaa, cal, tm), asof(data.daaa, cal, tm)
        rec["baa_tm"], rec["aaa_tm"] = baa, aaa
        rec["C05"] = baa - aaa
        # C06 INFL: CPIAUCNS vintage at t; carried forward if the month a year earlier is missing
        cv = data.cpi.at(t)
        infl, m = cpi_yoy(cv)
        carried = False
        if m is not None and not np.isfinite(infl):
            infl, carried = prev_infl, True
        rec["C06"], rec["cpi_m"], rec["infl_carried"] = infl, m, carried
        prev_infl = infl
        # C07 SLOOS: latest observation usable at t, valued as published at t
        vals = data.sloos if data.sloos_rt is None else sloos_values(data.sloos_rt, t)
        rec["C07"], rec["sloos_obs"] = sloos_at(vals, usable, t)
        # C08 CLAIMS: the latest week published by t
        rec["C08"], rec["claims_week"] = claims_yoy(data.icnsa, t, data.icnsa_release)
        # C09 ECY
        p_tm = asof(data.gspc, cal, tm)
        rec["gspc_tm"] = p_tm
        rec.update(excess_cape_yield(t, p_tm, y10, cv, data.earnings))
        # C11 HURDLE and C12 SCORE: the pinned backtest at t
        if t in data.legacy.index:
            lg = data.legacy.loc[t]
            rec["C11"], rec["C12"], rec["legacy_pos"] = float(lg["hurdle"]), float(lg["score"]), lg["pos"]
        else:
            rec["C11"] = rec["C12"] = rec["legacy_pos"] = float("nan")
        # C12 is missing where the backtest used a CPI or unemployment month not yet published (D5)
        blocked = data.legacy_blocked is not None and t in data.legacy_blocked.index
        rec["c12_unreleased_input"] = bool(blocked)
        if blocked:
            rec["C12"] = float("nan")
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("date")
    for c in ("indpro_m", "cpi_m", "sloos_obs", "claims_week", "ecy_q", "t_minus"):
        df[c] = pd.to_datetime(df[c])
    return df


def truncate(data: PredictorData, pos: int) -> PredictorData:
    """Everything the origin at ``pos`` could not have known deleted: market observations dated
    after t-, ALFRED rows released after t, SLOOS observations not yet usable, claims weeks not yet
    released, Shiller months after the last reported quarter, backtest rows after t."""
    cal = data.cal[: pos + 1]
    t, tm = data.cal[pos], data.cal[pos - 1]
    usable = sloos_usable_dates(data.sloos.index, data.cal, data.sloos_shift)
    q = br.reported_quarter(t)
    cut = lambda s: s.loc[s.index <= tm]  # noqa: E731
    release = (data.icnsa_release.reindex(data.icnsa.index) if data.icnsa_release is not None
               else pd.Series(data.icnsa.index + pd.Timedelta(days=CLAIMS_RELEASE_DAYS), index=data.icnsa.index))
    published = (release <= t).values
    sloos_rt = None
    if data.sloos_rt is not None:          # before ALFRED's first vintage, that vintage stands in (D3)
        sloos_rt = data.sloos_rt.truncated(max(t, pd.Timestamp(data.sloos_rt.rows["realtime_start"].min())))
    return dataclasses.replace(
        data, cal=cal, gspc=cut(data.gspc), vix=cut(data.vix), dgs3mo=cut(data.dgs3mo), dgs10=cut(data.dgs10),
        dbaa=cut(data.dbaa), daaa=cut(data.daaa), indpro=data.indpro.truncated(t), cpi=data.cpi.truncated(t),
        sloos=data.sloos.loc[usable[usable <= t].index], sloos_rt=sloos_rt,
        icnsa=data.icnsa.loc[published],
        icnsa_release=None if data.icnsa_release is None else data.icnsa_release.loc[release.index[published]],
        earnings=data.earnings.loc[data.earnings.index <= q], legacy=data.legacy.loc[data.legacy.index <= t])


def split_flags(cal: pd.DatetimeIndex, positions: Sequence[int]) -> pd.DataFrame:
    """Train / test membership per horizon (section 5), from the calendar alone."""
    n_seal = int(cal.searchsorted(SEAL_END, side="right"))
    n_all = int(cal.searchsorted(DATA_CUTOFF, side="right"))
    rows = []
    for pos in positions:
        t = cal[pos]
        rec = {"date": t, "in_param_window": bool(PARAM_WINDOW[0] <= t <= PARAM_WINDOW[1])}
        for h in HORIZONS:
            rec[f"train_h{h}"] = bool(t >= FIRST_ORIGIN and pos + h < n_seal)
            rec[f"test_h{h}"] = bool(t >= TEST_START and pos + h < n_all)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------------------
# Data-quality checks (no statistic of a candidate against a target)
# ---------------------------------------------------------------------------------------
def _same(a, b) -> bool:
    """Exact equality, with missing (NaN, NaT, None) equal to missing."""
    na, nb = pd.isna(a), pd.isna(b)
    if na or nb:
        return bool(na and nb)
    if isinstance(a, (pd.Timestamp, np.datetime64)) or isinstance(b, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(a) == pd.Timestamp(b)
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return (math.isnan(fa) and math.isnan(fb)) or fa == fb


def check_calendar(cal: pd.DatetimeIndex, positions: Sequence[int], flags: pd.DataFrame) -> Dict:
    exp = json.loads(PREREG_JSON.read_text(encoding="utf-8"))["split"]["expected_from_calendar"]
    pos_of = dict(zip(flags.index, positions))
    out, ok = {}, True
    for h in HORIZONS:
        for side in ("train", "test"):
            sel = flags.index[flags[f"{side}_h{h}"]]
            p = [pos_of[t] for t in sel]
            got = [f"{sel[0]:%Y-%m-%d}", f"{sel[-1]:%Y-%m-%d}", int(len(sel)), round(effective_n(p, h), 1)]
            want = exp[f"h{h}"][side]
            match = got == want
            ok &= match
            out[f"h{h}_{side}"] = {"got": got, "registered": want, "match": match}
    out["origins"] = {"first": f"{cal[positions[0]]:%Y-%m-%d}", "last": f"{cal[positions[-1]]:%Y-%m-%d}",
                      "count": len(positions)}
    out["param_window_count"] = int(flags["in_param_window"].sum())
    out["pass"] = bool(ok and out["origins"]["first"] == f"{FIRST_ORIGIN:%Y-%m-%d}"
                       and out["origins"]["last"] == f"{LAST_ORIGIN:%Y-%m-%d}")
    return out


def check_pit_truncation(data: PredictorData, panel: pd.DataFrame, positions: Sequence[int],
                         train_positions: Sequence[int], n: int) -> Dict:
    """Section 13.3: at ``n`` train month-ends, the value computed on all data equals the value
    computed after deleting everything the origin could not have known."""
    if n <= 0 or not train_positions:
        return {"n": 0, "pass": None}
    pick = sorted({train_positions[int(round(i))] for i in np.linspace(0, len(train_positions) - 1, n)})
    cols = [c for c in panel.columns if c not in ("legacy_pos",)]
    mismatches = []
    for pos in pick:
        sub = [p for p in positions if p <= pos]
        tp = compute_panel(truncate(data, pos), sub).iloc[-1]
        full = panel.loc[data.cal[pos]]
        for c in cols:
            if not _same(full[c], tp[c]):
                mismatches.append({"origin": f"{data.cal[pos]:%Y-%m-%d}", "column": c})
    return {"n": len(pick), "origins": [f"{data.cal[p]:%Y-%m-%d}" for p in pick],
            "mismatches": mismatches, "pass": not mismatches}


def read_train_outcomes(path: Path, flags: pd.DataFrame) -> Dict[str, Dict[pd.Timestamp, float]]:
    """The backtest CSV's outcome fields at train origins only, streamed row by row: for a date that
    is not a train origin of horizon h, the horizon's fields are never parsed (the seal)."""
    train = {h: set(flags.index[flags[f"train_h{h}"]]) for h in HORIZONS}
    got: Dict[str, Dict[pd.Timestamp, float]] = {k: {} for k in TARGETS}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        ix = {c: header.index(c) for c in header}
        for row in r:
            t = pd.Timestamp(row[ix["date"]])
            for h in HORIZONS:
                if t not in train[h]:
                    continue                                   # sealed or embargoed: never parsed
                ex, sp, tb = (float(row[ix[f"{k}_{h}"]]) for k in ("excess", "sp", "tbill"))
                got[f"Y{h}"][t] = ex
                got[f"S{h}"][t] = sp - tb
                if h == DIP_H:
                    got["D20"][t] = float(row[ix["real_dd20_63"]])
    return got


def check_target_reproduction(path: Path, targets: pd.DataFrame, flags: pd.DataFrame, tol: float = 1e-9) -> Dict:
    """Section 13.4 at train origins only: rebuilt Y, S and D20 against the pinned CSV's excess_h,
    sp_h - tbill_h and real_dd20_63. The CSV's outcome fields are dropped at load for every row that
    is not a train origin of that horizon (the seal)."""
    if sha256_file(path) != PINNED["rating_backtest.csv"]:
        raise SystemExit("STOP: rating_backtest.csv does not match its pinned hash")
    got = read_train_outcomes(path, flags)
    out, ok = {}, True
    for k in TARGETS:
        h = DIP_H if k == "D20" else int(k[1:])
        ref = pd.Series(got[k], dtype=float)
        mine = targets.loc[flags[f"train_h{h}"], k]
        both = mine.index.intersection(ref.index)
        diff = (mine.reindex(both) - ref.reindex(both)).abs()
        missing_mine = int(mine.isna().sum())
        n_bad = int((diff > tol).sum() + diff.isna().sum())
        out[k] = {"train_origins": int(len(mine)), "in_csv": int(len(both)), "missing_rebuilt": missing_mine,
                  "max_abs_diff": float(diff.max()) if len(diff) else None, "n_over_tol": n_bad}
        ok &= n_bad == 0 and missing_mine == 0 and len(both) == len(mine)
    out["tolerance"] = tol
    out["pass"] = bool(ok)
    return out


def check_release_lags(data: PredictorData, panel: pd.DataFrame, pi: Dict, flags: pd.DataFrame) -> Dict:
    """Release-lag verification (dates only, no values): months between each origin and the latest
    INDPRO / CPI month it used, the claims Thursday rule against ALFRED first-release dates, the
    Shiller earnings lag, and the vintage reconstruction against direct ALFRED queries."""
    out: Dict[str, object] = {}
    months = lambda a, b: (a.year - b.year) * 12 + a.month - b.month  # noqa: E731
    for col, name in (("indpro_m", "INDPRO"), ("cpi_m", "CPIAUCNS")):
        lag = pd.Series([months(t, m) if pd.notna(m) else np.nan for t, m in panel[col].items()], index=panel.index)
        side = np.where(panel.index < TEST_START, "train_side", "test_side")
        out[f"{name}_lag_months"] = {s: {str(int(k)): int(v) for k, v in lag[side == s].value_counts().sort_index().items()}
                                     for s in ("train_side", "test_side")}
        out[f"{name}_no_same_month_data"] = bool((lag.dropna() >= 1).all())
    # claims: every week whose first publication ALFRED records (weeks first seen at ALFRED's first
    # vintage, 2009-05-28, were published earlier and are not checked). The Build phase kept only weeks
    # published within 14 days, which hid the 2025 shutdown weeks (published 2025-11-20): audit fix D4.
    fr = first_published(pi["series"]["ICNSA"])
    known = fr[fr > fr.min()]
    late = known[known > known.index.to_series() + pd.Timedelta(days=CLAIMS_RELEASE_DAYS)]
    affected, moved = [], []
    for t, w in panel["claims_week"].items():
        if pd.isna(w):
            continue
        used = [w - pd.Timedelta(weeks=j) for j in range(4)]
        if any(u in known.index and known[u] > t for u in used):
            affected.append(f"{t:%Y-%m-%d}")
        if w != claims_week(t):
            moved.append(f"{t:%Y-%m-%d}: week {claims_week(t):%Y-%m-%d} not yet published, used {w:%Y-%m-%d}")
    out["claims_release"] = {"weeks_with_alfred_first_release": int(len(known)),
                             "first_week": f"{known.index.min():%Y-%m-%d}" if len(known) else None,
                             "released_after_thursday_rule": int(len(late)),
                             "late_weeks": [f"{w:%Y-%m-%d} released {v:%Y-%m-%d}" for w, v in late.items()],
                             "origins_moved_to_an_earlier_published_week": moved,
                             "origins_that_used_an_unreleased_week": affected}
    ql = [months(t, q) for t, q in panel["ecy_q"].items()]
    out["shiller_quarter_lag_months"] = {str(k): int(v) for k, v in pd.Series(ql).value_counts().sort_index().items()}
    # direct ALFRED queries (train origins) against the reconstruction
    vc = {}
    for sid in VINTAGE_CHECK_SERIES:
        rt = data.indpro if sid == "INDPRO" else data.cpi
        for t in VINTAGE_CHECK_ORIGINS:
            direct = pi["series"].get(f"{sid}__vintage_{t}")
            if direct is None:
                vc[f"{sid} {t}"] = "not fetched"
                continue
            a = direct.set_index("date")["value"].sort_index()
            b = rt.at(pd.Timestamp(t))
            same = a.index.equals(b.index) and bool(((a - b).abs().fillna(0) == 0).all()) \
                and bool((a.isna() == b.isna()).all())
            vc[f"{sid} {t}"] = {"n_obs": int(len(a)), "identical": same}
    out["vintage_reconstruction_vs_direct_query"] = vc
    # C12: origins where the backtest used a CPI or unemployment month not yet published (set missing)
    bl = data.legacy_blocked if data.legacy_blocked is not None else pd.Series(dtype=object)
    out["c12_unreleased_inputs"] = {f"{t:%Y-%m-%d}": why for t, why in bl.items()
                                    if FIRST_ORIGIN <= t <= LAST_ORIGIN}
    out["pass"] = bool(out["INDPRO_no_same_month_data"] and out["CPIAUCNS_no_same_month_data"]
                       and all(isinstance(v, dict) and v["identical"] for v in vc.values())
                       and set(out["shiller_quarter_lag_months"]) <= {"2", "3", "4"}
                       and not affected
                       and panel.loc[panel["c12_unreleased_input"].astype(bool), "C12"].isna().all())
    return out


def check_sloos_vintages(data: PredictorData, panel: pd.DataFrame) -> Dict:
    """D3: DRTSCILM revisions, and at how many origins the value as published differs from the value
    in the cutoff vintage (train side: also the largest difference; test side: a count only)."""
    rows = data.sloos_rt.rows if data.sloos_rt is not None else None
    if rows is None:
        return {"used": "cutoff vintage"}
    per_obs = rows.groupby("date")["value"].agg(lambda v: v.dropna().nunique())
    revised = per_obs[per_obs > 1].index
    usable = sloos_usable_dates(data.sloos.index, data.cal, data.sloos_shift)
    old = pd.Series({t: sloos_at(data.sloos, usable, t)[0] for t in panel.index})
    diff = (panel["C07"] - old).abs()
    changed = diff > 1e-12
    pre = panel.index < TEST_START
    first = rows["realtime_start"].min()
    rev_rows = rows.loc[rows["date"].isin(revised) & (rows["realtime_start"] > first)]
    return {"rule": "value in the vintage in effect at t; before ALFRED's first vintage "
                    f"({first:%Y-%m-%d}) that vintage",
            "observations_revised": int(len(revised)),
            "revision_vintages": {f"{d:%Y-%m-%d}": int(n) for d, n in rev_rows["realtime_start"].value_counts().sort_index().items()},
            "train_side_origins_changed_vs_cutoff_vintage": int(changed[pre].sum()),
            "train_side_max_abs_change": float(diff[pre].max()) if pre.any() else None,
            "test_side_origins_changed_vs_cutoff_vintage": int(changed[~pre].sum())}


def check_cross_sources(data: PredictorData, panel: pd.DataFrame, pi: Dict, inputs: Dict) -> Dict:
    """Train side only (dates before 2008): the test period's predictor values are not summarised."""
    tr = panel.loc[panel.index < TEST_START]
    out: Dict[str, object] = {}
    vixcls = _clean(Realtime(pi["series"]["VIXCLS"]).at(DATA_CUTOFF))
    d = (tr["vix_tm"] - vixcls.reindex(tr["t_minus"]).values).abs()
    out["VIX_vs_VIXCLS_at_t-"] = {"n": int(d.notna().sum()), "max_abs_diff": float(d.max()),
                                  "n_over_0.05": int((d > 0.05).sum())}
    t10y3m = _clean(Realtime(pi["series"]["T10Y3M"]).at(DATA_CUTOFF))
    d = (tr["C04"] - t10y3m.reindex(tr["t_minus"]).values).abs()
    out["TERM_vs_T10Y3M_at_t-"] = {"n": int(d.notna().sum()), "max_abs_diff": float(d.max()),
                                   "n_over_0.005": int((d > 0.005).sum())}
    for sid, mine in (("DGS3MO", data.dgs3mo), ("DGS10", data.dgs10)):
        pinned = _clean(inputs["fred"][sid])
        start = max(mine.index.min(), pinned.index.min())          # the download starts in 1981
        a = mine.loc[(mine.index >= start) & (mine.index <= SEAL_END)]
        b = pinned.loc[(pinned.index >= start) & (pinned.index <= SEAL_END)]
        both = a.index.intersection(b.index)
        out[f"{sid}_download_vs_pinned_pickle_to_2007"] = {
            "n_common": int(len(both)), "only_download": int(len(a.index.difference(b.index))),
            "only_pinned": int(len(b.index.difference(a.index))),
            "max_abs_diff": float((a.reindex(both) - b.reindex(both)).abs().max()) if len(both) else None}
    return out


def plausibility(panel: pd.DataFrame) -> Dict:
    """Train side only: min, max and count outside a loose plausible range."""
    tr = panel.loc[panel.index < TEST_START]
    out, ok = {}, True
    for c in CAND_IDS:
        x = tr[c].dropna()
        lo, hi = PLAUSIBLE[c]
        bad = int(((x < lo) | (x > hi)).sum())
        ok &= bad == 0
        out[c] = {"range": [lo, hi], "min": float(x.min()) if len(x) else None,
                  "max": float(x.max()) if len(x) else None, "n_outside": bad}
    out["pass"] = bool(ok)
    return out


def coverage(panel: pd.DataFrame, flags: pd.DataFrame) -> Dict:
    """Non-missing counts per candidate. Test side: missing counts only."""
    pre = panel.index < TEST_START
    out = {}
    for c in CAND_IDS:
        x = panel[c]
        rec = {}
        for name, sel in (("train_side_rows", pre), ("param_window", flags["in_param_window"].values),
                          ("train_h126", flags["train_h126"].values)):
            n = int(sel.sum())
            k = int(x[sel].notna().sum())
            rec[name] = {"n": n, "non_missing": k, "share": round(k / n, 4) if n else None}
        first = x[pre].first_valid_index()
        rec["first_valid_origin"] = f"{first:%Y-%m-%d}" if first is not None else None
        for name, sel in (("test_side_rows", ~pre), ("test_h126", flags["test_h126"].values),
                          ("test_h21", flags["test_h21"].values)):
            n = int(sel.sum())
            miss = int(x[sel].isna().sum())
            rec[name] = {"n": n, "missing": miss, "share_non_missing": round(1 - miss / n, 4) if n else None}
        rec["test_coverage_below_90pct"] = bool(rec["test_h126"]["share_non_missing"] is not None
                                                and rec["test_h126"]["share_non_missing"] < 0.90)
        out[c] = rec
    carried = panel["infl_carried"].astype(bool).values
    short_window = (panel["ecy_months"] < 120).values & panel["C09"].notna().values
    out["special_rules_used"] = {
        "C06_carried_forward": {"train_side": int(carried[pre].sum()), "test_side": int(carried[~pre].sum())},
        "C09_cpi_months_skipped": {"train_side": int(short_window[pre].sum()), "test_side": int(short_window[~pre].sum())},
    }
    members = [c for c, _, _, comp in CANDIDATES if comp]
    avail = panel[members].notna().sum(axis=1)
    out["composite_members_available"] = {
        "train_side_months_with_at_least_7": int((avail[pre] >= COMPOSITE_MIN_MEMBERS).sum()),
        "train_side_months": int(pre.sum()),
        "param_window_months_with_at_least_7": int((avail[flags["in_param_window"].values] >= COMPOSITE_MIN_MEMBERS).sum()),
        "test_side_months_with_fewer_than_7": int((avail[~pre] < COMPOSITE_MIN_MEMBERS).sum()),
        "test_side_months": int((~pre).sum()),
        "train_side_count_by_members": {str(int(k)): int(v) for k, v in avail[pre].value_counts().sort_index().items()},
    }
    return out


def jsonable(o):
    """NaN / NaT -> None, numpy scalars -> Python, Timestamps -> ISO dates (for panel_meta.json)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if o is None or o is pd.NaT:
        return None
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, pd.Timestamp):
        return f"{o:%Y-%m-%d}"
    return o


def check_candidate_registry(path: Optional[Path] = None) -> None:
    reg = json.loads((PREREG_JSON if path is None else path).read_text(encoding="utf-8"))
    got = [(c["id"], c["short"], int(c["sign"]), bool(c["in_composite"])) for c in reg["candidates"]]
    if tuple(got) != CANDIDATES:
        raise SystemExit("STOP: the candidate table in this script differs from research/prereg.json")
    if reg["composite"]["members"] != [c for c, _, _, comp in CANDIDATES if comp] \
            or reg["composite"]["min_members"] != COMPOSITE_MIN_MEMBERS:
        raise SystemExit("STOP: composite members differ from research/prereg.json")


# ---------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------
COMPONENT_DOC = {
    "pos": "session index of the origin in build_daily's calendar", "t_minus": "session before the origin",
    "vix_tm": "^VIX close at t- (<= 5 sessions stale)", "rv21_tm": "RV21 at t-, percent^2 per month (C01 only)",
    "indpro_m": "latest INDPRO month in the vintage at t", "y3_tm": "DGS3MO at t-", "y3_tprev": "DGS3MO at t'- (12 month-ends earlier)",
    "y10_tm": "DGS10 at t-", "baa_tm": "DBAA at t-", "aaa_tm": "DAAA at t-", "cpi_m": "latest CPIAUCNS month in the vintage at t",
    "infl_carried": "C06 carried from the previous origin (CPI_{m-12} missing)", "sloos_obs": "DRTSCILM observation used",
    "c12_unreleased_input": "C12 set missing: the backtest used a CPI or unemployment month first published after t (D5)",
    "claims_week": "week-ending Saturday w used by C08", "gspc_tm": "^GSPC close at t-", "cape": "C09 CAPE",
    "rr10": "C09 real 10-year yield", "ecy_q": "last reported earnings quarter (Shiller E month)",
    "ecy_months": "months with a published CPI in the 120-month earnings window", "legacy_pos": "pos column of the pinned CSV",
    "in_param_window": "origin in 1990-01-31..2007-06-29", "train_h21": "train origin at h=21 (window ends <= 2007-12-31)",
    "train_h63": "train origin at h=63 (also D20)", "train_h126": "train origin at h=126",
    "test_h21": "test origin at h=21 (window ends <= 2026-09-25)", "test_h63": "test origin at h=63 (also D20)",
    "test_h126": "test origin at h=126",
    "Y21": "SPXL (realised fund) excess return over T-bills, 21 sessions", "Y63": "same, 63 sessions", "Y126": "same, 126 sessions (primary)",
    "S21": "S&P 500 TR excess return, 21 sessions", "S63": "same, 63", "S126": "same, 126",
    "D20": "1 if the fund closed <= 80% of its origin level within 63 sessions",
}


def cmd_build(args) -> int:
    t0 = time.time()
    spec = check_spec()
    check_candidate_registry()
    pinned = check_pinned()
    OUT.mkdir(parents=True, exist_ok=True)
    print("spec and pinned hashes match", flush=True)

    if args.refresh and TRAIN_REPORT.exists() and _recorded_hash(TRAIN_REPORT.read_text(encoding="utf-8"),
                                                                 PREDICTOR_INPUTS.name):
        raise SystemExit("STOP: --refresh would replace predictor_inputs.pkl after the Train phase recorded its hash")
    inputs = pd.read_pickle(BACKTEST / "rating_inputs.pkl")
    cal = session_calendar(inputs)
    pi = load_predictor_inputs(refresh=args.refresh, offline=args.offline)
    recorded = check_predictor_inputs_hash()
    legacy = load_legacy()
    data, sloos_qa_result = assemble(inputs, pi, legacy, cal)
    positions = origin_positions(cal)
    flags = split_flags(cal, positions)
    cal_check = check_calendar(cal, positions, flags)
    print(f"{len(positions)} origins {cal[positions[0]]:%Y-%m-%d}..{cal[positions[-1]]:%Y-%m-%d}; "
          f"calendar matches the plan: {cal_check['pass']}", flush=True)

    panel = compute_panel(data, positions)
    legacy_pos_ok = bool((panel["legacy_pos"].dropna().astype(int) == panel.loc[panel["legacy_pos"].notna(), "pos"]).all())

    # targets through the seal: windows ending after 2007-12-31 cannot be computed
    d_out = outcome_frame(inputs)
    targets = compute_targets(d_out, positions, cal)
    sealed_ok = bool(targets.loc[targets.index >= TEST_START].isna().all().all())

    train_pos = [p for p, t in zip(positions, flags.index) if flags.at[t, "train_h126"]]
    pit = check_pit_truncation(data, panel, positions, train_pos, args.pit_checks)
    repro = check_target_reproduction(BACKTEST / "rating_backtest.csv", targets, flags)
    lags = check_release_lags(data, panel, pi, flags)
    sloos_vint = check_sloos_vintages(data, panel)
    cross = check_cross_sources(data, panel, pi, inputs)
    plaus = plausibility(panel)
    cov = coverage(panel, flags)

    full = panel.join(flags).join(targets)
    cand_cols = CAND_IDS
    comp_cols = [c for c in panel.columns if c not in cand_cols]
    order = cand_cols + comp_cols + list(flags.columns) + TARGETS
    full = full[order]
    train_df = full.loc[full.index < TEST_START].copy()
    test_df = full.loc[full.index >= TEST_START].copy()
    test_df[TARGETS] = np.nan                                  # sealed until the Test phase
    train_df.to_pickle(OUT / "panel_train.pkl")
    test_df.to_pickle(OUT / "panel_test.pkl")

    reg = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    cmeta = {c["id"]: {"short": c["short"], "name": c["name"], "sign": c["sign"], "in_composite": c["in_composite"],
                       "definition": c["definition"], "sources": c["sources"], "lag": c["lag"], "ta_flag": c["ta_flag"],
                       "exposure": c["exposure"]} for c in reg["candidates"]}
    impl = {
        "C01": "VIX(t-) and RV21 from pinned ^VIX/^GSPC closes on the session calendar; VIX <= 5 sessions stale",
        "C02": "ALFRED INDPRO rows valid at t (realtime_start <= t <= realtime_end), OLS in months on [1, k, k^2]",
        "C03": "FRED DGS3MO (predictor_inputs.pkl, vintage at the cutoff) at t- minus at t'- (t' = last session of t's month minus 12)",
        "C04": "FRED DGS10 minus DGS3MO at t- (each <= 5 sessions stale)",
        "C05": "FRED DBAA minus DAAA at t- (each <= 5 sessions stale)",
        "C06": "ALFRED CPIAUCNS vintage at t; m = latest month with a value; carry-forward if CPI_{m-12} missing",
        "C07": f"DRTSCILM value in the vintage in effect at t (before ALFRED's first vintage 2010-04-20, that vintage; D3); "
               f"usable-date rule with QA branch shift = {data.sloos_shift} month(s)",
        "C08": "ICNSA vintage at the cutoff; w = latest Saturday with w + 5 days <= t that was published by t "
               "(ALFRED first-release dates where recorded, from 2009; D4)",
        "C09": f"Shiller E (pinned) k = q-119..q over CPIAUCNS vintage at t; months without a published CPI skipped (>= {ECY_MIN_MONTHS} of 120 required)",
        "C10": "(VIX(t-)/100)^2",
        "C11": "pinned rating_backtest.csv column hurdle at date t (columns date, pos, hurdle, score only)",
        "C12": "pinned rating_backtest.csv column score at date t; missing where the backtest used a CPI or "
               "unemployment month first published after t (D5)",
    }
    for c in CAND_IDS:
        cmeta[c]["implementation"] = impl[c]
    checks = {
        "calendar": cal_check, "legacy_pos_matches_calendar": legacy_pos_ok,
        "seal_test_targets_all_missing": sealed_ok, "outcome_frame_last_session": f"{d_out.index.max():%Y-%m-%d}",
        "pit_truncation_invariance": pit, "target_reproduction_train": repro, "release_lags": lags,
        "sloos_qa": sloos_qa_result, "sloos_vintages": sloos_vint, "cross_sources_train_side": cross, "plausibility_train_side": plaus,
    }
    meta = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": "build", "git_describe": git_describe(),
        "spec_sha256": spec, "pinned_sha256": pinned,
        "predictor_inputs": {"path": "output/research/predictor_inputs.pkl", "sha256": sha256_file(PREDICTOR_INPUTS),
                             "recorded_in_train_report": recorded,
                             "series": {k: {"series_id": v["series_id"], "params": v["params"], "rows": v["rows"],
                                            "fetched_utc": v["fetched_utc"]} for k, v in pi["manifest"].items()}},
        "files": {
            "panel_train.pkl": "origins before 2008-01 (1990-01-31..2007-12-31): train origins for at least one horizon "
                               "(train_h* flags) plus the embargo month-ends 2007-11-30 and 2007-12-31 (in no train set, "
                               "kept for the Test phase's expanding-window standardisation); targets only where the "
                               "origin is a train origin of that horizon",
            "panel_test.pkl": "origins 2008-01-31..2026-08-31, predictors only: target columns are present and all "
                              "missing (sealed until `test --unseal`)",
        },
        "seal": {"sealed": True, "last_outcome_session": f"{SEAL_END:%Y-%m-%d}",
                 "how": "outcome_frame() truncates the backtest inputs at 2007-12-31 before build_daily"},
        "columns": {"candidates": CAND_IDS, "components": comp_cols, "flags": list(flags.columns), "targets": TARGETS,
                    "doc": COMPONENT_DOC},
        "candidates": cmeta,
        "coverage": cov,
        "missing_counts": {"panel_train": {c: int(v) for c, v in train_df.isna().sum().items()},
                           "panel_test": {c: int(v) for c, v in test_df.isna().sum().items()}},
        "checks": checks,
        "not_computed": "no correlation, return statistic or any other summary of candidates against targets; "
                        "no summary of test-side predictor values beyond missing counts",
    }
    (OUT / "panel_meta.json").write_text(json.dumps(jsonable(meta), indent=1, default=str, allow_nan=False) + "\n",
                                         encoding="utf-8")

    # console summary: coverage and pass / fail only
    print(f"\n{'cand':5} {'short':7} {'first':10} {'train':>9} {'param':>9} {'train126':>9} {'test miss':>10} {'test126 miss':>12}")
    for c, short, _, _ in CANDIDATES:
        r = cov[c]
        print(f"{c:5} {short:7} {r['first_valid_origin'] or '-':10} "
              f"{r['train_side_rows']['non_missing']:>4}/{r['train_side_rows']['n']:<4} "
              f"{r['param_window']['non_missing']:>4}/{r['param_window']['n']:<4} "
              f"{r['train_h126']['non_missing']:>4}/{r['train_h126']['n']:<4} "
              f"{r['test_side_rows']['missing']:>5}/{r['test_side_rows']['n']:<4} "
              f"{r['test_h126']['missing']:>6}/{r['test_h126']['n']:<4}"
              + ("  TEST COVERAGE < 90%" if r["test_coverage_below_90pct"] else ""))
    cm = cov["composite_members_available"]
    print(f"composite: >= 7 members at {cm['train_side_months_with_at_least_7']}/{cm['train_side_months']} train-side "
          f"months; fewer than 7 at {cm['test_side_months_with_fewer_than_7']}/{cm['test_side_months']} test-side months")
    print("\nchecks:")
    print(f"  calendar vs plan                  {cal_check['pass']}")
    print(f"  legacy CSV pos = calendar pos     {legacy_pos_ok}")
    print(f"  seal (test targets all missing)   {sealed_ok} (outcome frame ends {d_out.index.max():%Y-%m-%d})")
    print(f"  PIT truncation invariance         {pit['pass']} ({pit['n']} origins, {len(pit.get('mismatches', []))} mismatches)")
    print(f"  target reproduction (train)       {repro['pass']}")
    print(f"  release lags                      {lags['pass']}; claims weeks released after the Thursday rule: "
          f"{lags['claims_release']['released_after_thursday_rule']}, origins affected: "
          f"{len(lags['claims_release']['origins_that_used_an_unreleased_week'])}, moved to an earlier week: "
          f"{len(lags['claims_release']['origins_moved_to_an_earlier_published_week'])}")
    print(f"  C12 unreleased backtest inputs    {len(lags['c12_unreleased_inputs'])} origin(s) set missing: "
          f"{', '.join(lags['c12_unreleased_inputs']) or '-'}")
    print(f"  SLOOS vintages                    {sloos_vint.get('observations_revised')} observations revised; "
          f"train-side origins changed vs the cutoff vintage: {sloos_vint.get('train_side_origins_changed_vs_cutoff_vintage')}, "
          f"test-side: {sloos_vint.get('test_side_origins_changed_vs_cutoff_vintage')}")
    print(f"  SLOOS QA                          checked {sloos_qa_result['shift0']['n_checked']}, violations "
          f"{sloos_qa_result['shift0']['n_violations']}, branch shift {sloos_qa_result['branch_shift_months']} month(s)")
    print(f"  plausibility (train side)         {plaus['pass']}")
    print(f"  cross-sources (train side)        VIX max diff {cross['VIX_vs_VIXCLS_at_t-']['max_abs_diff']:.3f}, "
          f"TERM max diff {cross['TERM_vs_T10Y3M_at_t-']['max_abs_diff']:.3f}")
    print(f"\nwrote {OUT / 'panel_train.pkl'} ({len(train_df)} rows), panel_test.pkl ({len(test_df)} rows), "
          f"panel_meta.json ({time.time() - t0:.0f}s)")
    return 0


# ---------------------------------------------------------------------------------------
# train (Phase 2): the section 8 quantities, the frozen test specification, and the
# train-period replication (information only)
# ---------------------------------------------------------------------------------------
PANEL_TRAIN = OUT / "panel_train.pkl"
PANEL_META = OUT / "panel_meta.json"
TRAIN_DIR = OUT / "train"
TEST_DIR = OUT / "test"
FROZEN_SPEC = RESEARCH / "frozen_spec.json"
FROZEN_HASH = RESEARCH / "FROZEN_HASH.txt"
TRAIN_RESULTS = RESEARCH / "train_results.md"
PREDICTOR_INPUTS_SHA256 = "133e9d8692620fdc992a31c529bdc456c1dca5fde907f9955734793cf55b5560"   # DEVIATIONS.md D2

TAU_Q = 33.33                  # section 8: "the 33.33rd percentile (numpy percentile, linear interpolation)"
Z_CLIP = 3.0                   # section 7
ALPHA = 0.05                   # section 9: Holm, family-wise
SEED = 20260927                # sections 11 and 13.12
EXPANDING_MIN = 60             # section 13.12
MEMBERS = [c for c, _, _, comp in CANDIDATES if comp]
PREDICTORS = CAND_IDS + ["COMP"]
SIGNS = {**{c: s for c, _, s, _ in CANDIDATES}, "COMP": 1}
SHORTS = {**{c: sh for c, sh, _, _ in CANDIDATES}, "COMP": "COMP"}
# (family, target, horizon for n_eff and for the train / test origin flags); F7 is r against -D20
FAMILIES = (("primary", "Y126", 126), ("F2", "Y21", 21), ("F3", "Y63", 63), ("F4", "S126", 126),
            ("F5", "S63", 63), ("F6", "S21", 21), ("F7", "-D20", DIP_H))
QUALIFIERS = {"C01": "partly exposed", "C09": "partly exposed", "C10": "not a clean out-of-sample test",
              "C11": "not a clean out-of-sample test", "C12": "not a clean out-of-sample test"}
TA_LOOKBACKS = (1, 3, 6, 12)
EPISODES = (("financial crisis", "2008-01", "2009-06"), ("2020 crash", "2019-09", "2020-06"),
            ("inflation and rate shock", "2021-07", "2022-12"))
HALVES = (("2008-01", "2016-12"), ("2017-01", "2026-02"))
E4_HALVES = (("2008-02-01", "2016-12-30"), ("2017-01-03", "2026-08-31"))


def load_train_panel(path: Optional[Path] = None) -> pd.DataFrame:
    """The Train phase's only data. Refuses any file but panel_train.pkl (checked before it is
    opened) and any panel that reaches the test period (check_train_panel)."""
    path = PANEL_TRAIN if path is None else Path(path)
    if path.name != PANEL_TRAIN.name:
        raise SealError(f"the Train phase reads only {PANEL_TRAIN.name}, not {path.name}")
    p = pd.read_pickle(path)
    check_train_panel(p)
    return p


def check_train_panel(p: pd.DataFrame) -> Dict[str, int]:
    """The seal, checked on the panel itself: every month-end is before 2008; the panel ends at the
    embargo month-end 2007-12-31 (the last session of 2007, whose ``pos`` bounds every window);
    a target is present only at a train origin of its horizon, and every train origin's window ends
    on or before that session. Returns the count of present values per target."""
    if p.empty or p.index.max() >= TEST_START:
        raise SealError("the train panel holds a month-end from the test period")
    if p.index.max() != SEAL_END:
        raise SealError("the train panel must end at the embargo month-end 2007-12-31")
    months = p.index.to_period("M")
    if not (np.diff(months.asi8) == 1).all():
        raise ValueError("the train panel's month-ends are not consecutive")
    seal_pos = int(p.at[SEAL_END, "pos"])
    out = {}
    for k in TARGETS:
        h = DIP_H if k == "D20" else int(k[1:])
        has = p[k].notna().values
        flag = p[f"train_h{h}"].astype(bool).values
        if (has & ~flag).any():
            raise SealError(f"{k} is present at a month-end that is not a train origin of h = {h}")
        if (p["pos"].values[flag] + h > seal_pos).any():
            raise SealError(f"a train origin's {h}-session window ends after 2007-12-31")
        out[k] = int(has.sum())
    if any(p[f"test_h{h}"].astype(bool).any() for h in HORIZONS):
        raise SealError("the train panel flags a test origin")
    return out


def signed(p: pd.DataFrame, cid: str) -> pd.Series:
    """s_j * x_j: favourable is high."""
    return SIGNS[cid] * p[cid].astype(float)


def standardisation(pw: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """mu_j, sd_j (ddof = 1) over the parameter-window month-ends at which x_j is available."""
    out = {}
    for c in CAND_IDS:
        x = pw[c].astype(float).dropna()
        out[c] = {"n": int(len(x)), "mu": float(x.mean()), "sd": float(x.std(ddof=1))}
    return out


def composite(p: pd.DataFrame, std: Dict[str, Dict[str, float]], members: Sequence[str] = MEMBERS,
              clip: float = Z_CLIP, min_members: int = COMPOSITE_MIN_MEMBERS) -> Tuple[pd.Series, pd.Series]:
    """Section 7: COMP_t = mean over the available members of clip(s_j * (x_jt - mu_j) / sd_j, -3, 3);
    missing below ``min_members``. Returns (COMP, members available)."""
    z = pd.DataFrame({c: (SIGNS[c] * (p[c].astype(float) - std[c]["mu"]) / std[c]["sd"]).clip(-clip, clip)
                      for c in members}, index=p.index)
    n = z.notna().sum(axis=1)
    return z.mean(axis=1, skipna=True).where(n >= min_members), n


def train_quantities(p: pd.DataFrame) -> Dict:
    """Everything section 8 lets the Train phase set, from predictor values at the parameter-window
    month-ends only. No target column is read."""
    pw = p.loc[p["in_param_window"].astype(bool), CAND_IDS]
    first, last = pw.index.min(), pw.index.max()
    if (first, last) != PARAM_WINDOW or len(pw) != 210:
        raise SystemExit(f"STOP: parameter window is {first:%Y-%m-%d}..{last:%Y-%m-%d} ({len(pw)} month-ends), "
                         "registered 1990-01-31..2007-06-29 (210)")
    std = standardisation(pw)
    bad = [c for c in CAND_IDS if not (np.isfinite(std[c]["sd"]) and std[c]["sd"] > 0)]
    if bad:
        raise SystemExit(f"STOP: zero or undefined sd_j for {bad}")
    comp, n_mem = composite(pw, std)
    comp_pw = comp.dropna().to_numpy(float)
    ivar = pw["C10"].astype(float).dropna().to_numpy(float)
    c_vm = float(np.median(ivar))
    return {
        "standardisation": std,
        "composite": {"n_param_window": int(len(comp_pw)),
                      "members_available_in_param_window": {str(int(k)): int(v) for k, v in n_mem.value_counts().sort_index().items()},
                      "tau": float(np.percentile(comp_pw, TAU_Q)),
                      "tau_median": float(np.median(comp_pw))},
        "tau_j": {c: float(np.percentile(signed(pw, c).dropna().to_numpy(float), TAU_Q)) for c in CAND_IDS},
        "vol_managed": {"c": c_vm, "w_bar": float(np.mean(np.minimum(1.0, c_vm / ivar))), "n": int(len(ivar))},
    }


def one_sided(r: float, n_eff: float, dof_loss: int = 3) -> Tuple[float, float]:
    """Section 9 step 4: z = atanh(r) sqrt(n_eff - 3) / sqrt(1 + r^2 / 2), p = 1 - Phi(z); p = 1 if r
    is undefined or n_eff <= 4. ``dof_loss`` = 5 is the partial Spearman of section 13.12 ("n_eff - 5
    in place of n_eff - 3"; then p = 1 if n_eff <= 6)."""
    from scipy.stats import norm
    if r is None or not np.isfinite(r) or not n_eff > dof_loss + 1:
        return float("nan"), 1.0
    if r >= 1.0:
        return float("inf"), 0.0
    if r <= -1.0:
        return float("-inf"), 1.0
    z = math.atanh(r) * math.sqrt(n_eff - float(dof_loss)) / math.sqrt(1.0 + r * r / 2.0)
    return float(z), float(norm.sf(z))


def holm(pvals: Dict[str, float], alpha: float = ALPHA) -> Dict[str, Dict]:
    """Holm step-down over the given p-values (ties keep the given order): reject H_(i) while
    p_(i) <= alpha / (K - i + 1). Holm-adjusted p = running max of min(1, (K - i + 1) p_(i)); a
    hypothesis is rejected exactly when its adjusted p <= alpha."""
    keys = list(pvals)
    k_tests = len(keys)
    order = sorted(range(k_tests), key=lambda i: pvals[keys[i]])          # stable
    out: Dict[str, Dict] = {}
    running, stopped = 0.0, False
    for step, i in enumerate(order):
        p = pvals[keys[i]]
        thr = alpha / (k_tests - step)
        running = max(running, min(1.0, (k_tests - step) * p))
        reject = (not stopped) and p <= thr
        stopped = stopped or not reject
        out[keys[i]] = {"p_holm": running, "reject": bool(reject), "step": step + 1, "threshold": thr}
    return {k: out[k] for k in keys}


def corr_test(x: np.ndarray, y: np.ndarray, pos: np.ndarray, h: int) -> Dict:
    """Spearman r of x with y over the rows where both exist, with the overlap-aware n_eff of their
    session positions, the one-sided p (section 9) and the backtest's 90% interval."""
    from spxlcast.evaluation import spearman
    x, y, pos = np.asarray(x, float), np.asarray(y, float), np.asarray(pos)
    m = np.isfinite(x) & np.isfinite(y)
    x, y, pos = x[m], y[m], pos[m]
    n = int(len(x))
    r = spearman(x, y) if n >= 3 else float("nan")
    n_eff = float(effective_n(pos, h)) if n else 0.0
    z, p = one_sided(r, n_eff)
    lo, hi = br._corr_ci(r, n_eff) if np.isfinite(r) else (float("nan"), float("nan"))
    return {"r": r, "n": n, "n_eff": n_eff, "ci90": [lo, hi], "z": z, "p": p}


def target_values(p: pd.DataFrame, target: str) -> pd.Series:
    return -p["D20"].astype(float) if target == "-D20" else p[target].astype(float)


def family_tests(p: pd.DataFrame, comp: pd.Series, side: str) -> Dict[str, Dict]:
    """Sections 9-10: the 13 predictors (signed, COMP as is) against each family's target on the
    ``side`` ('train' or 'test') origins of its horizon, one-sided, overlap-aware, Holm within each
    family of 13 (ties in the order C01..C12, COMP)."""
    comp = comp.reindex(p.index)
    out = {}
    for fam, target, h in FAMILIES:
        sel = p[f"{side}_h{h}"].astype(bool).values
        y = target_values(p, target).values[sel]
        pos = p["pos"].values[sel]
        rows = {}
        for pid in PREDICTORS:
            x = (comp if pid == "COMP" else signed(p, pid)).values[sel]
            rows[pid] = corr_test(x, y, pos, h)
        adj = holm({k: v["p"] for k, v in rows.items()})
        for k in rows:
            rows[k].update(adj[k])
        idx = p.index[sel]
        out[fam] = {"target": target, "h": h, "origins": int(sel.sum()),
                    "first": f"{idx[0]:%Y-%m-%d}" if len(idx) else None,
                    "last": f"{idx[-1]:%Y-%m-%d}" if len(idx) else None, "rows": rows}
    return out


def replication(p: pd.DataFrame, comp: pd.Series) -> Dict[str, Dict]:
    """Section 10: the 13 predictors against each target on train origins of its horizon, Holm within
    each family of 13. Information only (published-sample replication): nothing here feeds the spec."""
    return family_tests(p, comp, "train")


def descriptives(p: pd.DataFrame, comp: pd.Series) -> Dict[str, Dict]:
    """Section 13.9 at the parameter-window month-ends: summary statistics and the first-order
    autocorrelation of the level (persistence)."""
    pw = p["in_param_window"].astype(bool)
    out = {}
    for pid in PREDICTORS:
        x = (comp if pid == "COMP" else p[pid].astype(float)).loc[pw]
        v = x.dropna()
        q = np.percentile(v.to_numpy(float), [5, 25, 50, 75, 95]) if len(v) else [np.nan] * 5
        out[pid] = {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()),
                    "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]),
                    "p95": float(q[4]), "max": float(v.max()), "ac1": float(x.autocorr(1))}
    return out


def collinearity(p: pd.DataFrame, comp: pd.Series) -> pd.DataFrame:
    """Section 13.11: Spearman matrix (pairwise complete) at the parameter-window month-ends, raw
    candidate values (not signed) and COMP."""
    pw = p["in_param_window"].astype(bool)
    x = p.loc[pw, CAND_IDS].astype(float).copy()
    x["COMP"] = comp.loc[pw]
    return x.corr(method="spearman")


def ta_diagnostic(p: pd.DataFrame, comp: pd.Series) -> Dict[str, Dict[str, Dict]]:
    """Section 13.10 (reported for the owner's judgement only; never used to add, drop or adjust a
    candidate): Spearman of each signed candidate s_j * x_j and of COMP with the trailing k-month
    S&P 500 price return, measured from ^GSPC at t- of the origin k month-ends earlier to ^GSPC at
    t- (panel column gspc_tm), at the parameter-window month-ends where it exists."""
    from spxlcast.evaluation import spearman
    pw = p["in_param_window"].astype(bool).values
    g = p["gspc_tm"].astype(float)
    out = {}
    for pid in PREDICTORS:
        x = (comp if pid == "COMP" else signed(p, pid))
        rec = {}
        for k in TA_LOOKBACKS:
            ret = (g / g.shift(k) - 1.0).values
            m = pw & np.isfinite(ret) & np.isfinite(x.values)
            rec[f"{k}m"] = {"r": spearman(x.values[m], ret[m]) if m.sum() >= 3 else float("nan"), "n": int(m.sum())}
        out[pid] = rec
    return out


def train_calendar_check(p: pd.DataFrame) -> Dict:
    """Train origins per horizon (first, last, count, n_eff) against the plan's calendar figures."""
    exp = json.loads(PREREG_JSON.read_text(encoding="utf-8"))["split"]["expected_from_calendar"]
    out, ok = {}, True
    for h in HORIZONS:
        sel = p.index[p[f"train_h{h}"].astype(bool)]
        pos = p.loc[sel, "pos"].values
        got = [f"{sel[0]:%Y-%m-%d}", f"{sel[-1]:%Y-%m-%d}", int(len(sel)), round(effective_n(pos, h), 1)]
        match = got == exp[f"h{h}"]["train"]
        ok &= match
        out[f"h{h}"] = {"got": got, "registered": exp[f"h{h}"]["train"], "match": match}
    out["pass"] = bool(ok)
    return out


def train_coverage(p: pd.DataFrame, comp: pd.Series) -> Dict[str, Dict]:
    """Section 13.7, train side: non-missing share of every predictor at the train origins of each
    horizon and in the parameter window."""
    out = {}
    for pid in PREDICTORS:
        x = comp if pid == "COMP" else p[pid]
        rec = {}
        for name, sel in [("param_window", p["in_param_window"].astype(bool))] + \
                         [(f"train_h{h}", p[f"train_h{h}"].astype(bool)) for h in HORIZONS]:
            n = int(sel.sum())
            k = int(x[sel].notna().sum())
            rec[name] = {"n": n, "non_missing": k, "share": k / n if n else None}
        rec["first_valid"] = f"{x.first_valid_index():%Y-%m-%d}" if x.first_valid_index() is not None else None
        out[pid] = rec
    return out


def prereg_frozen_utc(path: Path = PREREG_HASH) -> Optional[str]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("frozen_utc:"):
            return line.split(":", 1)[1].strip()
    return None


def spec_bytes(spec: Dict) -> bytes:
    """The frozen spec as written: JSON, ASCII, one-space indent, LF, trailing newline."""
    return (json.dumps(spec, indent=1, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def frozen_spec(q: Dict, provenance: Dict, sloos: Dict) -> Dict:
    """Everything the Test phase needs, deterministic (no clock time, no train-period outcome
    statistic). ``q`` is train_quantities(); ``provenance`` the file hashes; ``sloos`` the SLOOS QA."""
    reg = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    std, tau_j = q["standardisation"], q["tau_j"]
    cands = []
    for c in reg["candidates"]:
        cid = c["id"]
        cands.append({
            "id": cid, "short": c["short"], "name": c["name"], "sign": c["sign"], "in_composite": c["in_composite"],
            "definition": c["definition"], "exposure": c["exposure"], "ta_flag": c["ta_flag"],
            "verdict_qualifier": QUALIFIERS.get(cid),
            "mu": std[cid]["mu"], "sd": std[cid]["sd"], "n_param_window": std[cid]["n"],
            "tau_j": tau_j[cid],
            "secondary_rule": f"hold SPXL at origin t if {c['sign']:+d} * {cid}_t >= tau_j, else T-bills",
        })
    return {
        "title": "SPXL timing-signal search: frozen test specification",
        "version": "1.0",
        "status": "Frozen at the end of the Train phase, before unsealing. The Test phase takes every number it needs "
                  "from this file and changes none of them.",
        "governing_document": "research/PREREGISTRATION.md governs (research/prereg.json is its machine-readable copy). "
                              "This file adds the quantities section 8 lets the Train phase set and fixes, before "
                              "unsealing, the implementation details the plan leaves open ('interpretations').",
        "hash_file": "research/FROZEN_HASH.txt (SHA-256 of this file's bytes); the same bytes are "
                     "output/research/train_params.json, whose SHA-256 research/TRAIN_REPORT.md records",
        "prereg": {"research/PREREGISTRATION.md": provenance["prereg"]["research/PREREGISTRATION.md"],
                   "research/prereg.json": provenance["prereg"]["research/prereg.json"],
                   "frozen_utc": provenance["prereg_frozen_utc"],
                   "unseal_token": "the SHA-256 of research/PREREGISTRATION.md above"},
        "inputs": {
            "pinned_sha256": provenance["pinned"],
            "predictor_inputs_sha256": {"output/research/predictor_inputs.pkl": provenance["predictor_inputs"]},
            "train_panel_sha256": {"output/research/panel_train.pkl": provenance["panel_train"]},
            "note": "on a mismatch of a pinned or predictor-input hash the Test phase stops. The train quantities were "
                    "computed from panel_train.pkl above. panel_test.pkl was not opened or hashed by the Train phase; "
                    "the Test phase uses the panel built by `build` from the inputs above (if it rebuilds it with "
                    "`build --offline`, the rebuilt panel_train.pkl must match the hash above). The code and "
                    "panel_meta.json hashes at train time are in research/train_results.md (provenance only)",
        },
        "calendar": {**reg["calendar"],
                     "split": reg["split"],
                     "test_origin_rule": "month-end origins from 2008-01-31 whose h-session window's last session is "
                                         "on or before 2026-09-25 (h = 126 for the primary test and F4; 63 for F3, F5, "
                                         "F7; 21 for F2, F6)"},
        "targets": reg["targets"],
        "train_set": {
            "parameter_window": {"first": f"{PARAM_WINDOW[0]:%Y-%m-%d}", "last": f"{PARAM_WINDOW[1]:%Y-%m-%d}",
                                 "month_ends": 210, "source": "predictor values only (no outcome)"},
            "percentile": {"q": TAU_Q, "function": "numpy.percentile(values, 33.33), method 'linear'"},
            "standardisation": {"formula": "z_jt = (x_jt - mu_j) / sd_j; mu_j, sd_j (ddof = 1) over the parameter-window "
                                           "month-ends at which x_j is available",
                                "mu_sd": {c: std[c] for c in CAND_IDS},
                                "note": "C10-C12 are reported only: they are not composite members"},
            "composite": {
                "id": "COMP", "sign": 1, "members": MEMBERS, "weights": "equal", "clip": [-Z_CLIP, Z_CLIP],
                "min_members": COMPOSITE_MIN_MEMBERS,
                "formula": "COMP_t = mean over members j available at t of clip(s_j * (x_jt - mu_j) / sd_j, -3, 3); "
                           "missing if fewer than 7 of the 9 members are available",
                "tau": q["composite"]["tau"],
                "tau_rule": "hold SPXL at origin t if COMP_t >= tau (the 33.33rd percentile of COMP over the "
                            "parameter window), else T-bills",
                "tau_median": q["composite"]["tau_median"],
                "tau_median_use": "secondary economic analysis only: the composite rule with the threshold at the "
                                  "train median (section 11)",
                "n_param_window": q["composite"]["n_param_window"],
                "members_available_in_param_window": q["composite"]["members_available_in_param_window"],
            },
            "tau_j": {"on": "s_j * x_j (favourable is high)", "values": tau_j,
                      "use": "the economic rule of any candidate that passes the primary test, and the secondary "
                             "table of all 12 candidates"},
            "vol_managed": {"c": q["vol_managed"]["c"], "w_bar": q["vol_managed"]["w_bar"],
                            "n_param_window": q["vol_managed"]["n"],
                            "c_definition": "median of IVAR_t = C10 over the parameter window",
                            "w_bar_definition": "mean of min(1, c / IVAR_t) over the parameter window",
                            "rule": "SPXL weight w_t = min(1, c / IVAR_t) at each test origin, the rest T-bills; against "
                                    "BH and against the constant weight w_bar (secondary, not clean: uses the VIX level)"},
            "sloos_lag_branch": {"shift_months": sloos["branch_shift_months"],
                                 "qa": {k: sloos["shift0"][k] for k in ("n_checked", "n_violations", "first_checked_obs")},
                                 "source": "ALFRED first-release dates of DRTSCILM only (build phase, panel_meta.json)"},
        },
        "candidates": cands,
        "primary_test": {
            "predictors": PREDICTORS, "K": len(PREDICTORS), "target": "Y", "h": 126,
            "sample": "test origins for h = 126 (2008-01-31..2026-02-27, 218) at which the predictor and Y(t,126) both exist",
            "statistic": "r = spxlcast.evaluation.spearman(s_P * P_t, Y(t,126)); s_COMP = +1",
            "n_eff": "spxlcast.evaluation.effective_n(session positions (panel column pos) of the sample's origins, 126)",
            "z": "atanh(r) * sqrt(n_eff - 3) / sqrt(1 + r^2 / 2)",
            "p_one_sided": "1 - Phi(z) (scipy.stats.norm.sf(z)); p = 1 if r is undefined or n_eff <= 4; p = 0 if r = 1",
            "interval": "backtest_rating._corr_ci(r, n_eff), 90%",
            "correction": {"method": "Holm step-down", "alpha": ALPHA, "K": len(PREDICTORS),
                           "rule": "sort p ascending (ties in the order C01..C12, COMP); reject H_(i) while "
                                   "p_(i) <= 0.05 / (13 - i + 1); stop at the first failure",
                           "adjusted_p": "running maximum of min(1, (13 - i + 1) * p_(i))"},
            "pass": "a hypothesis rejected by Holm is a primary statistical pass for that predictor",
            "not_tested_rule": "a candidate whose data cannot be built is 'not tested': K stays 13 and its p is 1",
        },
        "secondary_tests": {
            "families": [{"id": fam, "target": tgt, "h": h, "direction": "r > 0" + (" against -D20" if tgt == "-D20" else "")}
                         for fam, tgt, h in FAMILIES if fam != "primary"],
            "same_as_primary": "predictors, statistic, n_eff (with the family's h), one-sided p, 90% interval; Holm "
                               "within each family of 13",
            "status": "reported, never decisive",
            "replication": "the primary test and F2-F7 on train origins, reported beside the test period as "
                           "published-sample replication (research/train_results.md holds the train side)",
        },
        "economic_test": {
            "rule": "at each test origin t from 2008-01-31 through 2026-07-31 hold the realised fund (backtest_rating."
                    "build_daily column fund_ret: the synthetic 3x fund through 2008-12-31, SPXL from 2009-01-02) from the "
                    "close of t to the close of the next month-end if COMP_t >= tau, else T-bills (tbill_ret)",
            "holding_periods": 223, "sessions": ["2008-02-01", "2026-08-31"],
            "holding_sessions": "the holding decided at the origin with session position a, followed by the month-end "
                                "at position b, earns the daily returns of sessions a+1..b",
            "switch_cost": 0.001,
            "switch_cost_application": "a switch between SPXL and T-bills multiplies wealth by (1 - 0.001) at the start of "
                                       "the first session of the new holding: that session's return is "
                                       "(1 - 0.001) * (1 + r) - 1",
            "initial_allocation": "free for every strategy (rule, MIX_w, vol-managed, constant w_bar), as for BH: costs "
                                  "apply only to changes between consecutive holdings",
            "missing_signal": "keep the previous holding (the count is reported); T-bills if the signal is missing at "
                              "the first origin",
            "benchmarks": {
                "BH": "buy-and-hold the realised fund over the same sessions, no cost",
                "MIX_w": "w in the realised fund and 1 - w in T-bills, w = (number of the 223 holdings in SPXL) / 223; the "
                         "two sleeves drift within each month and are rebalanced to w at each month-end close; cost "
                         "0.001 * |w - w_pre| on the first session of the new month (w_pre = the fund's weight just "
                         "before rebalancing), applied as the switch cost",
            },
            "metrics": {
                "daily_returns": "fund_ret and tbill_ret with a missing value treated as 0, as backtest_rating.strategy_section",
                "CAGR": "W_N^(252 / N) - 1, W the wealth after the N sessions 2008-02-01..2026-08-31 (W_0 = 1)",
                "Sharpe": "mean / sd (ddof = 1) of the daily (return - tbill_ret) * sqrt(252)",
                "max_drawdown": "max over sessions of 1 - W_t / max_{s <= t} W_s over the wealth path W_1..W_N, as "
                                "backtest_rating.strategy_section",
            },
            "pass_all_of": reg["economic_test"]["pass_all_of"],
            "E4_halves": [list(h) for h in E4_HALVES],
            "E4_method": "Sharpe of the full-period daily return series restricted to each half's sessions (the strategies "
                         "are not restarted)",
            "also_applied_to": "every candidate that passes the primary test, with its own tau_j (hold SPXL if "
                               "s_j * x_jt >= tau_j) and the same criteria; for all 12 candidates as a secondary table",
            "secondary_descriptive": {
                "costs": "the composite rule with 0 and 0.0025 switching costs",
                "median_threshold": "the composite rule with COMP_t >= tau_median",
                "stationary_bootstrap": {
                    "what": "90% intervals for Sharpe(rule) - Sharpe(BH) and CAGR(rule) - CAGR(BH)",
                    "data": "the 223 monthly holding-period returns of the rule and of BH (costs included) and the "
                            "monthly T-bill holding-period returns, resampled as aligned triples",
                    "algorithm": "rng = numpy.random.default_rng(20260927); U = rng.random((10000, 223)); "
                                 "S = rng.integers(0, 223, size=(10000, 223)); idx[:, 0] = S[:, 0]; for i >= 1: "
                                 "idx[:, i] = S[:, i] where U[:, i] < 1/6 else (idx[:, i-1] + 1) mod 223 "
                                 "(Politis-Romano, circular, mean block 6 months)",
                    "statistics": "monthly Sharpe = mean / sd (ddof = 1) of (r - r_bill) * sqrt(12); CAGR = "
                                  "prod(1 + r)^(12 / 223) - 1",
                    "interval": "numpy.percentile of the 10000 differences at 5 and 95 (linear)",
                },
                "vol_managed": "w_t = min(1, c / IVAR_t) at each test origin (IVAR = C10), rest T-bills; sleeves drift "
                               "within the month; cost 0.001 * |w_t - w_pre|; if IVAR_t is missing the previous target "
                               "weight is kept; compared with BH and with a constant w_bar under the same mechanics",
                "descriptives": "time in SPXL, number of switches, turnover (sum of |weight traded|), total costs paid",
            },
        },
        "robustness": {
            "halves": {"origins": [list(h) for h in HALVES], "label": "'one half only' if r is not positive in both"},
            "episodes_left_out": [{"name": n, "origins": [a, b]} for n, a, b in EPISODES],
            "episode_method": "drop the origins whose month lies in the episode (inclusive); r, n_eff (of the remaining "
                              "positions) and one-sided p recomputed; 'episode-dependent' if p > 0.10 for any one episode",
            "moving_block_bootstrap": {
                "what": "90% interval for the primary r of each predictor",
                "algorithm": "for each predictor a fresh rng = numpy.random.default_rng(20260927); the predictor's primary "
                             "sample in origin order (n rows); k = ceil(n / 12) blocks per resample; "
                             "starts = rng.integers(0, n - 12 + 1, size=(10000, k)); each resample is the concatenation "
                             "of rows start..start+11 of its k blocks, cut to n rows; r_b = spearman on the resample",
                "interval": "numpy.percentile of the finite r_b at 5 and 95 (linear)",
            },
            "expanding_window_composite": "at each origin t, mu_j(t) and sd_j(t) (ddof = 1) over the month-ends from "
                                          "1990-01-31 through t inclusive at which x_j is available; member j is "
                                          "available at t only if x_jt exists and it has at least 60 such values; then "
                                          "the section 7 formula (clip 3, at least 7 members); its primary r is reported",
            "partial_spearman": "controls ln VIX(t-) (ln of panel column vix_tm) and DGS3MO(t-) (panel column y3_tm): "
                                "average ranks of s_P * P, Y(t,126) and both controls over the rows where all four exist; "
                                "OLS of rank(s_P * P) and of rank(Y) on [1, rank(ln VIX), rank(DGS3MO)]; the partial r is "
                                "the Pearson correlation of the two residual vectors; one-sided p as section 9 with "
                                "n_eff - 5 in place of n_eff - 3; for primary-passing predictors (reported for all)",
            "synthetic_spread": "rebuild the daily frame with dataclasses.replace(Config(), swap_spread = 0.0025) and "
                                "0.0125 (only fund returns before 2009-01-02 change); recompute Y(t,126) and the 13 "
                                "primary r and p (reported only)",
        },
        "verdict": reg["verdict"],
        "robustness_labels": reg["robustness_labels"],
        "sanity_checks_test_phase": [
            "pinned inputs, predictor_inputs.pkl, prereg files, this file and train_params.json match their recorded hashes",
            "target reproduction at all origins: rebuilt Y, S and D20 equal the pinned CSV's excess_h, sp_h - tbill_h and "
            "real_dd20_63 (tolerance 1e-9)",
            "legacy reproduction: spearman(score, Y(t,126)) over every month-end 1990-01-31..2026-02-27 with both (the "
            "pinned CSV's score column, as the README computed it, so the two D5 origins are included) is -0.03 +/- 0.01, "
            "else fixed as a deviation before anything is interpreted",
            "synthetic fund: daily correlation of syn_ret with spxl_ret from 2009-01-02 where both exist "
            "(backtest_rating.validate_synthetic) is at least 0.99",
            "coverage: non-missing share of every candidate and of COMP at test origins; below 90% at h = 126 flags "
            "that predictor's test",
            "look log summarised in the results",
        ],
        "test_phase_preconditions": [
            "py scripts/research_signals.py test --unseal <SHA-256 of research/PREREGISTRATION.md>",
            "verify_unseal: the token, research/PREREG_HASH.txt, and output/research/train_params.json against "
            "research/TRAIN_REPORT.md",
            "research/frozen_spec.json against research/FROZEN_HASH.txt, and byte-identical to output/research/train_params.json",
            "run once; afterwards only bug fixes, reporting original and corrected results (DEVIATIONS.md)",
        ],
        "interpretations": [
            "I1 tau and tau_j use q = 33.33 exactly as registered (not 100/3)",
            "I2 COMP averages the available clipped signed z-scores with equal weight; it is missing below 7 members",
            "I3 holdings, switch costs, initial allocation and missing signals as in economic_test above",
            "I4 MIX_w and the vol-managed rules drift within the month and pay 0.001 x |weight traded| at each "
            "month-end rebalance",
            "I5 metrics follow backtest_rating.strategy_section (daily NaN returns as 0; drawdown over W_1..W_N); "
            "Sharpe uses ddof = 1; E4 restricts the full-period daily series to each half",
            "I6 the stationary and moving-block bootstrap algorithms and seeds above",
            "I7 the expanding-window composite includes month-end t and needs 60 values per member",
            "I8 the partial Spearman is the partial correlation of average ranks, p with n_eff - 5",
            "I9 Holm ties keep the order C01..C12, COMP",
            "I10 the legacy reproduction uses the pinned CSV's score column at every month-end with Y(t,126)",
            "These were fixed before unsealing and before any train-period outcome statistic was looked at; none depends "
            "on a train-period result",
        ],
        "outputs": reg["always_reported"],
    }


def _fmt(x, spec="{:+.2f}", na="-") -> str:
    try:
        return na if x is None or not np.isfinite(float(x)) else spec.format(x)
    except (TypeError, ValueError):
        return na


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def plot_train(p: pd.DataFrame, comp: pd.Series, q: Dict, path: Path) -> Optional[str]:
    """Section 13.9: every predictor at the train-side month-ends, with mu_j (candidates) and tau (COMP)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:                                       # pragma: no cover
        return f"not drawn ({type(exc).__name__})"
    fig, axes = plt.subplots(5, 3, figsize=(13, 14), sharex=True)
    for ax, pid in zip(axes.flat, PREDICTORS):
        x = comp if pid == "COMP" else p[pid]
        ax.plot(x.index, x.values, lw=1.0, color="#3a6ea5")
        if pid == "COMP":
            ax.axhline(q["composite"]["tau"], color="#b5523b", lw=0.9, ls="--", label="tau")
            ax.axhline(0.0, color="#888", lw=0.6)
            ax.legend(loc="upper left", fontsize=7, frameon=False)
        else:
            ax.axhline(q["standardisation"][pid]["mu"], color="#888", lw=0.8, ls=":")
        ax.axvspan(PARAM_WINDOW[1] + pd.Timedelta(days=1), SEAL_END, color="#ddd", lw=0)
        ax.set_title(f"{pid} {SHORTS[pid]} (sign {SIGNS[pid]:+d})", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(PREDICTORS):]:
        ax.axis("off")
    fig.suptitle("Predictors at train-side month-ends 1990-01..2007-12 (grey: after the parameter window)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def look_log_summary(path: Path = LOOK_LOG) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_train_results(res: Dict) -> str:
    q, spec_sha = res["q"], res["spec_sha256"]
    L: List[str] = []
    a = L.append
    a("# SPXL timing-signal search: Train phase results")
    a("")
    a(f"Phase 2 of `research/PREREGISTRATION.md`, run {res['run_utc']} by `py scripts/research_signals.py train`, "
      "**before unsealing**. The test period (outcome windows from 2008-01 on) is still sealed.")
    a("")
    a("* **Data read:** `output/research/panel_train.pkl` only for every statistic (month-ends 1990-01-31..2007-12-31; "
      "targets only at train origins, windows ending on or before 2007-12-31). `panel_test.pkl` was not opened. Also "
      "read: `output/research/panel_meta.json` (the build's checks and the SLOOS QA, dates only), the train rows of "
      "the pinned `rating_backtest.csv` through the seal (target reproduction), and file hashes.")
    a(f"* **Frozen spec:** `research/frozen_spec.json`, SHA-256 `{spec_sha}` (`research/FROZEN_HASH.txt`, frozen "
      f"{res['frozen_utc']}); byte-identical copy `output/research/train_params.json`; hashes for the code guard in "
      "`research/TRAIN_REPORT.md`.")
    a("* **What the spec contains from the train period:** only quantities computed from predictor values at the 210 "
      "parameter-window month-ends (1990-01-31..2007-06-29). No outcome enters it. The replication statistics below "
      "are information only and changed nothing (section 8).")
    a("")
    a("## 1. Checks")
    a("")
    ck = res["checks"]
    rows = [
        ["plan files match `PREREG_HASH.txt`", "pass"],
        ["pinned inputs (3 files) match their registered SHA-256", "pass"],
        [f"`predictor_inputs.pkl` SHA-256 = `{PREDICTOR_INPUTS_SHA256[:12]}...` (DEVIATIONS.md D2)", "pass"],
        ["`panel_meta.json` was built from the same plan and `predictor_inputs.pkl`", "pass"],
        ["seal on the train panel: no month-end after 2007-12-31; targets only at train origins; every window ends "
         "on or before 2007-12-31", "pass"],
        ["train origins per horizon (first, last, count, n_eff) equal the plan's calendar figures",
         "pass" if ck["calendar"]["pass"] else "FAIL"],
        ["target reproduction at train origins against the pinned CSV (tolerance 1e-9)",
         "pass" if ck["target_reproduction"]["pass"] else "FAIL"],
        ["standardisation: no zero sd_j; COMP available at every parameter-window month-end",
         "pass" if ck["standardisation_ok"] else "FAIL"],
        [f"SLOOS lag QA branch (release dates only): {ck['sloos']['shift0']['n_checked']} observations checked, "
         f"{ck['sloos']['shift0']['n_violations']} violations", f"shift {ck['sloos']['branch_shift_months']} month(s)"],
        ["synthetic fund vs SPXL daily correlation (13.6)", "deferred to the Test phase: it needs post-2008 daily "
                                                           "data, which the seal withholds"],
    ]
    L += md_table(["check", "result"], rows)
    a("")
    a("Train origins: " + "; ".join(f"h = {h[1:]}: {v['got'][0]}..{v['got'][1]}, {v['got'][2]} origins, n_eff "
                                     f"{v['got'][3]}" for h, v in ck["calendar"].items() if h != "pass") + ".")
    a("")
    prov = res.get("provenance_only", {})
    if prov:
        a("Provenance at train time (not part of the spec; the script changes again in the Test phase): "
          + "; ".join(f"`{k}` `{v}`" for k, v in prov.items()) + f"; `git describe` {res.get('git', 'unknown')}.")
        a("")
    a("## 2. Train-set quantities (frozen)")
    a("")
    a("Computed from predictor values at the 210 parameter-window month-ends only. `tau` and `tau_j` are the 33.33rd "
      "percentile (numpy, linear) of COMP and of `s_j * x_j`.")
    a("")
    rows = []
    for c, short, s, comp in CANDIDATES:
        st = q["standardisation"][c]
        rows.append([c, short, f"{s:+d}", "yes" if comp else "no", st["n"], f"{st['mu']:.6g}", f"{st['sd']:.6g}",
                     f"{q['tau_j'][c]:.6g}"])
    L += md_table(["id", "signal", "sign", "in COMP", "n", "mu_j", "sd_j", "tau_j (on s_j x_j)"], rows)
    a("")
    cq = q["composite"]
    a(f"* **COMP threshold** `tau` = {cq['tau']:.6f} (hold SPXL when COMP >= tau); train median {cq['tau_median']:.6f} "
      f"(secondary analysis only). COMP exists at {cq['n_param_window']} of 210 parameter-window month-ends "
      f"(members available: {', '.join(f'{k}: {v}' for k, v in cq['members_available_in_param_window'].items())}).")
    a(f"* **Vol-managed rule:** `c` = median IVAR = {q['vol_managed']['c']:.6f} (VIX {100 * math.sqrt(q['vol_managed']['c']):.2f}); "
      f"`w_bar` = {q['vol_managed']['w_bar']:.6f}.")
    a(f"* **SLOOS lag branch:** shift {ck['sloos']['branch_shift_months']} month(s).")
    a("")
    a("## 3. Train-period replication (information only)")
    a("")
    a("Published-sample replication (section 10): the registered statistic on train origins, one-sided in the "
      "registered direction, overlap-aware (n_eff = union of the windows / h; Fisher z with Bonett-Wright variance), Holm "
      "within each family of 13. The 1990-2007 window is published evidence for several candidates, so this is a "
      "replication, not new evidence. COMP's standardisation comes from the same years (no outcome used). **Nothing "
      "here may change the spec.** Power: at h = 126 the train n_eff is about 36, so an unadjusted one-sided 5% test "
      "needs r >= 0.28 and the first Holm step r >= 0.45.")
    a("")
    rep = res["replication"]
    for fam, label in (("primary", "6 months: SPXL excess return Y(t,126) (the registered primary target)"),
                       ("F3", "3 months: Y(t,63)"), ("F2", "1 month: Y(t,21)")):
        f = rep[fam]
        a(f"### {label}")
        a("")
        rows = []
        for pid in PREDICTORS:
            v = f["rows"][pid]
            rows.append([pid, SHORTS[pid], v["n"], f"{v['n_eff']:.1f}", _fmt(v["r"]),
                         f"{_fmt(v['ci90'][0])} to {_fmt(v['ci90'][1])}", _fmt(v["p"], "{:.3f}"),
                         _fmt(v["p_holm"], "{:.3f}"), "pass" if v["reject"] else "-"])
        L += md_table(["predictor", "signal", "n", "n_eff", "r (signed)", "90% CI", "p (1-sided)", "Holm p", "Holm"], rows)
        a("")
    a("### Other targets (r, with the one-sided p in brackets; * = Holm pass within the family)")
    a("")
    fams = ["primary", "F3", "F2", "F4", "F5", "F6", "F7"]
    heads = {"primary": "Y 6m", "F3": "Y 3m", "F2": "Y 1m", "F4": "S&P 6m", "F5": "S&P 3m", "F6": "S&P 1m", "F7": "fewer 20% dips (-D20)"}
    rows = []
    for pid in PREDICTORS:
        row = [pid]
        for fam in fams:
            v = rep[fam]["rows"][pid]
            row.append(f"{_fmt(v['r'])} ({_fmt(v['p'], '{:.2f}')}){'*' if v['reject'] else ''}")
        rows.append(row)
    L += md_table(["predictor"] + [heads[f] for f in fams], rows)
    a("")
    a("n_eff by family (all origins): " + "; ".join(f"{heads[f]} {rep[f]['rows']['C01']['n_eff']:.1f}" for f in fams) + ".")
    a("")
    passes = [(fam, pid) for fam in fams for pid in PREDICTORS if rep[fam]["rows"][pid]["reject"]]
    a("Holm passes on the train period: " + (", ".join(f"{pid} ({heads[fam]})" for fam, pid in passes) if passes else "none") + ".")
    a("")
    a("## 4. Descriptives at the parameter-window month-ends (13.9)")
    a("")
    rows = []
    for pid in PREDICTORS:
        d = res["descriptives"][pid]
        rows.append([pid, SHORTS[pid], d["n"], f"{d['mean']:.4g}", f"{d['sd']:.4g}", f"{d['min']:.4g}", f"{d['p25']:.4g}",
                     f"{d['median']:.4g}", f"{d['p75']:.4g}", f"{d['max']:.4g}", f"{d['ac1']:.2f}"])
    L += md_table(["id", "signal", "n", "mean", "sd", "min", "p25", "median", "p75", "max", "AR(1)"], rows)
    a("")
    a(f"Time-series plots: `{res['plot']}` (git-ignored).")
    a("")
    a("## 5. Coverage at train origins (13.7)")
    a("")
    rows = []
    for pid in PREDICTORS:
        c = res["coverage"][pid]
        rows.append([pid, SHORTS[pid], c["first_valid"]] + [f"{c[k]['non_missing']}/{c[k]['n']}" for k in
                                                          ("param_window", "train_h126", "train_h63", "train_h21")])
    L += md_table(["id", "signal", "first value", "param window", "train h=126", "train h=63", "train h=21"], rows)
    a("")
    a("C07 starts at 1990-07-31 (first SLOOS observation usable); C12 is missing at 1996-01-31 (DEVIATIONS.md D5).")
    a("")
    a("## 6. Collinearity: Spearman matrix of the raw values at the parameter-window month-ends (13.11)")
    a("")
    cm = res["collinearity"]
    L += md_table([""] + list(cm.columns), [[i] + [_fmt(cm.at[i, j]) for j in cm.columns] for i in cm.index])
    a("")
    a("## 7. TA diagnostic (13.10; for the owner's judgement only, never used)")
    a("")
    a("Spearman of each signed candidate `s_j * x_j` (and COMP) with the trailing S&P 500 price return up to t- "
      "(from `gspc_tm`), parameter-window month-ends. A large positive value would mean the 'favourable' reading "
      "tends to follow a rising market.")
    a("")
    rows = []
    for pid in PREDICTORS:
        t = res["ta"][pid]
        rows.append([pid, SHORTS[pid]] + [f"{_fmt(t[f'{k}m']['r'])} (n {t[f'{k}m']['n']})" for k in TA_LOOKBACKS])
    L += md_table(["id", "signal"] + [f"trailing {k}m" for k in TA_LOOKBACKS], rows)
    a("")
    a("## 8. What has now been seen")
    a("")
    a("* Everything in the exposure ledger (PREREGISTRATION.md section 2) and the Build-phase notes (DEVIATIONS.md N11-N12).")
    a("* Now also: every train-period statistic above (predictor distributions, collinearity, the TA diagnostic and the "
      "replication of all 13 predictors against all seven targets on 1990-2007 origins). These are train-period "
      "results; the spec had been fully determined by the plan before they were computed and none of them changed it.")
    a("* Not seen: any test-period outcome, any statistic using an outcome window that ends after 2007-12-31, any "
      "summary of a test-period predictor value (panel_test.pkl was not opened).")
    a("")
    a("## 9. Look log")
    a("")
    log = res["look_log"]
    L += md_table(["UTC", "phase", "command", "git"], [[r["utc"], r["phase"], f"`{r['command']}`", r["git_describe"]] for r in log])
    a("")
    a("Build runs computed no predictive statistic. Train runs computed train-period statistics only.")
    a("")
    return "\n".join(L)


def render_train_report(pi_sha: str, spec_sha: str, frozen_utc: str) -> str:
    """research/TRAIN_REPORT.md: the hashes the code guard reads (one file per line)."""
    return "\n".join([
        "# Train phase report",
        "",
        "Phase 2 of `research/PREREGISTRATION.md` (section 15). Full results: `research/train_results.md`.",
        "",
        "Hashes recorded for the Test phase (section 3 and the code guard in section 15):",
        "",
        "| file | SHA-256 |",
        "|---|---|",
        f"| output/research/predictor_inputs.pkl | {pi_sha} |",
        f"| output/research/train_params.json | {spec_sha} |",
        f"| research/frozen_spec.json (the same bytes as train_params.json) | {spec_sha} |",
        "",
        f"Frozen {frozen_utc} (research/FROZEN_HASH.txt).",
        "",
    ])


def render_frozen_hash(spec_sha: str, frozen_utc: str) -> str:
    return "\n".join([
        "SPXL timing-signal search: frozen test specification hash",
        f"frozen_utc: {frozen_utc}",
        "algorithm: SHA-256 of the file's bytes as written (UTF-8, LF line endings; research/.gitattributes keeps LF in every checkout)",
        "",
        f"{spec_sha}  research/frozen_spec.json",
        "",
        "output/research/train_params.json holds the same bytes (the plan's name; its hash is in research/TRAIN_REPORT.md)",
        "verify: sha256sum research/frozen_spec.json output/research/train_params.json",
        "",
    ])


def write_lf(path: Path, text_or_bytes) -> None:
    """Write bytes as given (no newline translation on Windows)."""
    b = text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else text_or_bytes
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b)


def train_compute(p: pd.DataFrame) -> Dict:
    """The Train phase's computations on the train panel, spec quantities first (from predictor
    values only), then the information-only statistics."""
    check_train_panel(p)
    q = train_quantities(p)                                # the spec: predictor values, parameter window
    comp, n_mem = composite(p, q["standardisation"])
    pw = p["in_param_window"].astype(bool)
    std_ok = bool(all(q["standardisation"][c]["sd"] > 0 for c in CAND_IDS) and comp[pw].notna().all())
    return {"q": q, "comp": comp, "members": n_mem, "standardisation_ok": std_ok,
            "calendar": train_calendar_check(p), "coverage": train_coverage(p, comp),
            "descriptives": descriptives(p, comp), "collinearity": collinearity(p, comp),
            "ta": ta_diagnostic(p, comp), "replication": replication(p, comp)}


def cmd_train(args) -> int:
    t0 = time.time()
    spec_h = check_spec()
    check_candidate_registry()
    pinned = check_pinned()
    pi_sha = sha256_file(PREDICTOR_INPUTS)
    if pi_sha != PREDICTOR_INPUTS_SHA256:
        raise SystemExit(f"STOP: predictor_inputs.pkl has SHA-256 {pi_sha}, DEVIATIONS.md D2 records {PREDICTOR_INPUTS_SHA256}")
    check_predictor_inputs_hash()
    if TEST_DIR.exists():
        raise SystemExit("STOP: output/research/test exists: the Test phase has run, the Train phase is closed")
    meta = json.loads(PANEL_META.read_text(encoding="utf-8"))
    if meta.get("spec_sha256") != spec_h or meta.get("predictor_inputs", {}).get("sha256") != pi_sha \
            or meta.get("pinned_sha256") != pinned:
        raise SystemExit("STOP: panel_meta.json was not built from this plan, these pinned inputs and predictor_inputs.pkl")
    sloos = meta["checks"]["sloos_qa"]
    print("spec, pinned inputs, predictor_inputs.pkl and panel_meta.json match", flush=True)

    p = load_train_panel()                                  # the only data file (never panel_test.pkl)
    res = train_compute(p)
    repro = check_target_reproduction(BACKTEST / "rating_backtest.csv", p[TARGETS],
                                      p[[f"train_h{h}" for h in HORIZONS]])
    res["checks"] = {"calendar": res["calendar"], "target_reproduction": repro,
                     "standardisation_ok": res["standardisation_ok"], "sloos": sloos}
    if not (res["calendar"]["pass"] and repro["pass"] and res["standardisation_ok"]):
        raise SystemExit("STOP: a train-side check failed; nothing was frozen")

    provenance = {"prereg": spec_h, "prereg_frozen_utc": prereg_frozen_utc(), "pinned": pinned,
                  "predictor_inputs": pi_sha, "panel_train": sha256_file(PANEL_TRAIN)}
    res["provenance_only"] = {"output/research/panel_meta.json": sha256_file(PANEL_META),
                              "scripts/research_signals.py": sha256_file(Path(__file__).resolve()),
                              "tests/test_research_stats.py": sha256_file(ROOT / "tests" / "test_research_stats.py")}
    spec = frozen_spec(res["q"], provenance, sloos)
    b = spec_bytes(spec)
    spec_sha = hashlib.sha256(b).hexdigest()

    # freeze: once FROZEN_HASH.txt exists the spec may not change (unless --refreeze, before unsealing,
    # with the reason recorded in research/DEVIATIONS.md)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    frozen_utc, refrozen = now, False
    if FROZEN_HASH.exists():
        rec = read_prereg_hashes(FROZEN_HASH).get("research/frozen_spec.json")
        same = rec == spec_sha and FROZEN_SPEC.exists() and sha256_file(FROZEN_SPEC) == spec_sha
        if same:
            frozen_utc = next((ln.split(":", 1)[1].strip() for ln in FROZEN_HASH.read_text(encoding="utf-8").splitlines()
                               if ln.startswith("frozen_utc:")), now)
        elif not args.refreeze:
            raise SystemExit(f"STOP: the spec would change (recorded {rec}, now {spec_sha}). Re-freezing needs "
                             "--refreeze and a data or code reason in research/DEVIATIONS.md, never a train result")
        else:
            refrozen = True
    write_lf(FROZEN_SPEC, b)
    write_lf(TRAIN_PARAMS, b)
    if frozen_utc == now:
        write_lf(FROZEN_HASH, render_frozen_hash(spec_sha, frozen_utc))
    write_lf(TRAIN_REPORT, render_train_report(pi_sha, spec_sha, frozen_utc))
    if sha256_file(FROZEN_SPEC) != spec_sha or sha256_file(TRAIN_PARAMS) != spec_sha \
            or read_prereg_hashes(FROZEN_HASH).get("research/frozen_spec.json") != spec_sha \
            or _recorded_hash(TRAIN_REPORT.read_text(encoding="utf-8"), TRAIN_PARAMS.name) != spec_sha \
            or check_predictor_inputs_hash() != pi_sha:
        raise SystemExit("STOP: the written spec does not match its recorded hashes")

    # information-only outputs (git-ignored) and the results document
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    comp_df = pd.DataFrame({"COMP": res["comp"], "members": res["members"],
                            "in_param_window": p["in_param_window"].astype(bool)})
    comp_df.to_csv(TRAIN_DIR / "composite_train.csv", date_format="%Y-%m-%d")
    rep_rows = [{"family": fam, "target": f["target"], "h": f["h"], "predictor": pid, "signal": SHORTS[pid],
                 "n": v["n"], "n_eff": v["n_eff"], "r": v["r"], "ci90_lo": v["ci90"][0], "ci90_hi": v["ci90"][1],
                 "z": v["z"], "p": v["p"], "p_holm": v["p_holm"], "holm_reject": v["reject"]}
                for fam, f in res["replication"].items() for pid, v in f["rows"].items()]
    pd.DataFrame(rep_rows).to_csv(TRAIN_DIR / "replication_train.csv", index=False)
    res["collinearity"].to_csv(TRAIN_DIR / "collinearity_train.csv")
    res["plot"] = plot_train(p, res["comp"], res["q"], TRAIN_DIR / "predictors_train.png")
    res.update({"spec_sha256": spec_sha, "frozen_utc": frozen_utc, "run_utc": now, "look_log": look_log_summary(),
                "git": git_describe()})
    stats = {k: res[k] for k in ("q", "checks", "coverage", "descriptives", "ta", "replication", "provenance_only")}
    stats["collinearity"] = res["collinearity"].to_dict()
    stats.update({"spec_sha256": spec_sha, "frozen_utc": frozen_utc, "run_utc": now})
    (TRAIN_DIR / "train_stats.json").write_text(json.dumps(jsonable(stats), indent=1, allow_nan=False) + "\n",
                                                encoding="utf-8")
    write_lf(TRAIN_RESULTS, render_train_results(res))

    # console: the spec and the primary replication
    q = res["q"]
    print(f"\nfrozen spec research/frozen_spec.json  SHA-256 {spec_sha}  ({'re-frozen' if refrozen else 'frozen'} {frozen_utc})")
    print(f"tau = {q['composite']['tau']:.6f}  median = {q['composite']['tau_median']:.6f}  "
          f"c = {q['vol_managed']['c']:.6f}  w_bar = {q['vol_managed']['w_bar']:.6f}  SLOOS shift = {sloos['branch_shift_months']}")
    print(f"\n{'id':5} {'short':7} {'mu':>11} {'sd':>11} {'tau_j':>11}")
    for c in CAND_IDS:
        s = q["standardisation"][c]
        print(f"{c:5} {SHORTS[c]:7} {s['mu']:>11.5g} {s['sd']:>11.5g} {q['tau_j'][c]:>11.5g}")
    print("\ntrain replication (information only): signed Spearman r [one-sided p], * = Holm pass")
    print(f"{'id':5} {'short':7} {'Y 1m':>16} {'Y 3m':>16} {'Y 6m':>16}")
    rep = res["replication"]
    for pid in PREDICTORS:
        cells = []
        for fam in ("F2", "F3", "primary"):
            v = rep[fam]["rows"][pid]
            cells.append(f"{_fmt(v['r'])} [{_fmt(v['p'], '{:.3f}')}]{'*' if v['reject'] else ' '}")
        print(f"{pid:5} {SHORTS[pid]:7} " + " ".join(f"{c:>16}" for c in cells))
    print(f"\nwrote research/frozen_spec.json, research/FROZEN_HASH.txt, research/TRAIN_REPORT.md, "
          f"research/train_results.md, output/research/train_params.json, output/research/train/ ({time.time() - t0:.0f}s)")
    return 0


# ---------------------------------------------------------------------------------------
# test (Phase 3): the one run on the test period, with every number taken from
# research/frozen_spec.json
# ---------------------------------------------------------------------------------------
PANEL_TEST = OUT / "panel_test.pkl"
TRAIN_STATS = TRAIN_DIR / "train_stats.json"
TEST_RESULTS_JSON = OUT / "test_results.json"
TEST_RESULTS_MD = RESEARCH / "test_results.md"
RESULTS_MD = RESEARCH / "RESULTS.md"          # the plan's name (section 14): the owner's report (`report`)
STARTED, COMPLETED = "STARTED", "COMPLETED"   # run-once markers in output/research/test/
ECON_ORIGINS = (pd.Timestamp("2008-01-31"), pd.Timestamp("2026-07-31"))
ECON_SESSIONS = (pd.Timestamp("2008-02-01"), pd.Timestamp("2026-08-31"))
ECON_HOLDINGS = 223
SWITCH_COST = 0.001
ALT_COSTS = (0.0, 0.0025)
BOOT_N = 10000
MBB_BLOCK = 12                                 # moving-block bootstrap: blocks of 12 origins
SB_MEAN_BLOCK = 6                              # stationary bootstrap: mean block of 6 months
EPISODE_P_MAX = 0.10
LEGACY_R, LEGACY_TOL = -0.03, 0.01
LEGACY_RANGE = (pd.Timestamp("1990-01-31"), pd.Timestamp("2026-02-27"))
SYNTHETIC_CORR_MIN = 0.99
COVERAGE_MIN = 0.90
SPREADS = (0.0025, 0.0125)
REPRO_TOL = 1e-9
EXPLORATORY_HEADING = "## EXPLORATORY"      # a section `report` keeps: extra analysis, not a result


@dataclasses.dataclass
class Phase3Paths:
    """Every file the Test phase reads or writes and the hashes it checks. The real run uses
    ``from_globals()``; the unit tests point these at a synthetic world."""
    prereg_md: Path
    prereg_json: Path
    prereg_hash: Path
    train_report: Path
    train_params: Path
    frozen_spec: Path
    frozen_hash: Path
    backtest: Path
    predictor_inputs: Path
    panel_train: Path
    panel_test: Path
    panel_meta: Path
    train_stats: Path
    test_dir: Path
    results_json: Path
    results_md: Path
    results_md_plan: Path
    look_log: Path
    pinned: Dict[str, str]
    predictor_inputs_sha256: str

    @classmethod
    def from_globals(cls) -> "Phase3Paths":
        return cls(prereg_md=PREREG_MD, prereg_json=PREREG_JSON, prereg_hash=PREREG_HASH, train_report=TRAIN_REPORT,
                   train_params=TRAIN_PARAMS, frozen_spec=FROZEN_SPEC, frozen_hash=FROZEN_HASH, backtest=BACKTEST,
                   predictor_inputs=PREDICTOR_INPUTS, panel_train=PANEL_TRAIN, panel_test=PANEL_TEST,
                   panel_meta=PANEL_META, train_stats=TRAIN_STATS, test_dir=TEST_DIR, results_json=TEST_RESULTS_JSON,
                   results_md=TEST_RESULTS_MD, results_md_plan=RESULTS_MD, look_log=LOOK_LOG, pinned=dict(PINNED),
                   predictor_inputs_sha256=PREDICTOR_INPUTS_SHA256)

    def verify_kwargs(self) -> Dict[str, Path]:
        return {"md": self.prereg_md, "js": self.prereg_json, "hashes": self.prereg_hash,
                "params": self.train_params, "report": self.train_report, "log": self.look_log}


# ---- before unsealing: the preconditions (nothing here reads an outcome) ---------------------
def check_spec_against_code(spec: Dict) -> Dict:
    """The frozen spec's registered choices must be the ones this code implements."""
    problems = []
    got = tuple((c["id"], c["short"], int(c["sign"]), bool(c["in_composite"])) for c in spec["candidates"])
    if got != CANDIDATES:
        problems.append("candidates, signs or composite membership")
    comp = spec["train_set"]["composite"]
    if comp["members"] != MEMBERS or comp["min_members"] != COMPOSITE_MIN_MEMBERS \
            or [float(v) for v in comp["clip"]] != [-Z_CLIP, Z_CLIP] or comp["sign"] != 1 or comp["weights"] != "equal":
        problems.append("composite")
    if spec["train_set"]["percentile"]["q"] != TAU_Q:
        problems.append("percentile")
    pt = spec["primary_test"]
    if pt["predictors"] != PREDICTORS or pt["K"] != len(PREDICTORS) or pt["target"] != "Y" or pt["h"] != 126 \
            or pt["correction"]["alpha"] != ALPHA or pt["correction"]["K"] != len(PREDICTORS):
        problems.append("primary test")
    fams = [(f["id"], f["target"], f["h"]) for f in spec["secondary_tests"]["families"]]
    if fams != [tuple(f) for f in FAMILIES[1:]]:
        problems.append("secondary families")
    et = spec["economic_test"]
    if et["switch_cost"] != SWITCH_COST or et["holding_periods"] != ECON_HOLDINGS \
            or et["sessions"] != [f"{ECON_SESSIONS[0]:%Y-%m-%d}", f"{ECON_SESSIONS[1]:%Y-%m-%d}"] \
            or [tuple(h) for h in et["E4_halves"]] != list(E4_HALVES):
        problems.append("economic test")
    sd = et["secondary_descriptive"]
    if "0.0025" not in sd["costs"] or str(SEED) not in sd["stationary_bootstrap"]["algorithm"] \
            or "1/6" not in sd["stationary_bootstrap"]["algorithm"]:
        problems.append("secondary economic analyses")
    rb = spec["robustness"]
    if [tuple(h) for h in rb["halves"]["origins"]] != list(HALVES) \
            or [(e["name"], e["origins"][0], e["origins"][1]) for e in rb["episodes_left_out"]] != list(EPISODES) \
            or str(SEED) not in rb["moving_block_bootstrap"]["algorithm"] \
            or f"at least {EXPANDING_MIN}" not in rb["expanding_window_composite"] \
            or "0.0025" not in rb["synthetic_spread"] or "0.0125" not in rb["synthetic_spread"]:
        problems.append("robustness")
    if [v["label"] for v in spec["verdict"]][0] != "Timing signal confirmed" or len(spec["verdict"]) != 4:
        problems.append("verdict labels")
    if problems:
        raise SystemExit("STOP: the frozen spec differs from what this code implements: " + ", ".join(problems))
    return {"pass": True, "problems": []}


def check_spec_quantities(spec: Dict, p_train: pd.DataFrame) -> Dict:
    """The section 8 quantities recomputed from panel_train.pkl must equal the frozen ones (the Test
    phase still uses the frozen numbers)."""
    q = train_quantities(p_train)
    diffs, counts_ok = [], True
    for c in spec["candidates"]:
        st = q["standardisation"][c["id"]]
        diffs += [abs(c["mu"] - st["mu"]), abs(c["sd"] - st["sd"]), abs(c["tau_j"] - q["tau_j"][c["id"]])]
        counts_ok &= c["n_param_window"] == st["n"]
    ts = spec["train_set"]
    diffs += [abs(ts["composite"]["tau"] - q["composite"]["tau"]),
              abs(ts["composite"]["tau_median"] - q["composite"]["tau_median"]),
              abs(ts["vol_managed"]["c"] - q["vol_managed"]["c"]),
              abs(ts["vol_managed"]["w_bar"] - q["vol_managed"]["w_bar"])]
    for c in spec["candidates"]:
        diffs.append(abs(ts["tau_j"]["values"][c["id"]] - c["tau_j"]))
    mx = float(max(diffs))
    return {"max_abs_diff": mx, "counts_match": bool(counts_ok), "pass": bool(counts_ok and mx <= 1e-12)}


def check_test_panel(p_train: pd.DataFrame, p_test: pd.DataFrame, cal: pd.DatetimeIndex, meta: Dict) -> Dict:
    """panel_test.pkl as `build` wrote it: sealed (targets all missing), the plan's month-ends on the
    session calendar, the plan's flags, and the missing counts panel_meta.json recorded."""
    problems = []
    if list(p_test.columns) != list(p_train.columns):
        problems.append("columns differ from panel_train.pkl")
    if p_test.empty or p_test.index[0].to_period("M") != pd.Period("2008-01", "M") or p_test.index[-1] != LAST_ORIGIN:
        problems.append("month-ends are not 2008-01..2026-08")
    if p_test[TARGETS].notna().any().any():
        problems.append("targets present (not sealed)")
    p_all = pd.concat([p_train, p_test])
    positions = origin_positions(cal)
    if list(p_all["pos"].astype(int)) != positions:
        problems.append("positions differ from the session calendar's month-ends")
    elif not (cal[p_all["pos"].astype(int).to_numpy()] == p_all.index).all():
        problems.append("dates differ from the session calendar")
    else:
        flags = split_flags(cal, positions)
        if not (p_all[flags.columns].astype(bool).values == flags.values).all():
            problems.append("split flags differ from the calendar")
    miss = {c: int(v) for c, v in p_test.isna().sum().items()}
    if miss != meta.get("missing_counts", {}).get("panel_test"):
        problems.append("missing counts differ from panel_meta.json")
    return {"pass": not problems, "problems": problems, "rows": int(len(p_test)),
            "first": f"{p_test.index[0]:%Y-%m-%d}" if len(p_test) else None,
            "last": f"{p_test.index[-1]:%Y-%m-%d}" if len(p_test) else None}


def rebuild_predictors(inputs: Dict, pi: Dict, legacy: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Every candidate and component at every origin, rebuilt exactly as `build` builds them."""
    data, _ = assemble(inputs, pi, legacy, cal)
    return compute_panel(data, origin_positions(cal))


def compare_rebuilt(p_all: pd.DataFrame, rebuilt: pd.DataFrame) -> Dict:
    """Equality only (no value is summarised): the stored panels against a fresh rebuild. The Train
    phase did not hash panel_test.pkl, so this is its integrity check."""
    if not rebuilt.index.equals(p_all.index):
        return {"pass": False, "n_mismatches": None, "problem": "origins differ", "mismatches": []}
    mism = []
    for c in rebuilt.columns:
        if c not in p_all.columns:
            mism.append(("-", c))
            continue
        a, b = p_all[c].tolist(), rebuilt[c].tolist()
        mism += [(t, c) for t, u, v in zip(p_all.index, a, b) if not _same(u, v)]
    return {"pass": not mism, "cells": int(rebuilt.shape[0] * rebuilt.shape[1]), "columns": int(rebuilt.shape[1]),
            "n_mismatches": len(mism),
            "mismatches": [{"origin": t if isinstance(t, str) else f"{t:%Y-%m-%d}", "column": c} for t, c in mism[:50]]}


def read_legacy_checked(path: Path, pinned_sha: str) -> pd.DataFrame:
    if sha256_file(path) != pinned_sha:
        raise SystemExit(f"STOP: {path.name} does not match its pinned hash")
    return _read_legacy_columns(path)


def phase3_preconditions(paths: Phase3Paths) -> Tuple[Dict, Dict, Dict]:
    """frozen_spec.test_phase_preconditions and the section 15 code guard, all before unsealing.
    Stops on any failure. Returns (checks, spec, panel_meta)."""
    ck: Dict[str, object] = {}
    ck["prereg_sha256"] = check_spec(paths.prereg_md, paths.prereg_json, paths.prereg_hash)
    check_candidate_registry(paths.prereg_json)
    ck["pinned_sha256"] = check_pinned(paths.backtest, paths.pinned)
    pi_sha = sha256_file(paths.predictor_inputs)
    if pi_sha != paths.predictor_inputs_sha256:
        raise SystemExit(f"STOP: predictor_inputs.pkl has SHA-256 {pi_sha}, DEVIATIONS.md D2 records "
                         f"{paths.predictor_inputs_sha256}")
    report_text = paths.train_report.read_text(encoding="utf-8")
    if _recorded_hash(report_text, paths.predictor_inputs.name) != pi_sha:
        raise SystemExit("STOP: predictor_inputs.pkl does not match the hash recorded in TRAIN_REPORT.md")
    spec_b = paths.frozen_spec.read_bytes()
    spec_sha = hashlib.sha256(spec_b).hexdigest()
    if read_prereg_hashes(paths.frozen_hash).get("research/frozen_spec.json") != spec_sha:
        raise SystemExit("STOP: research/frozen_spec.json does not match research/FROZEN_HASH.txt")
    if paths.train_params.read_bytes() != spec_b:
        raise SystemExit("STOP: output/research/train_params.json is not byte-identical to research/frozen_spec.json")
    if _recorded_hash(report_text, paths.train_params.name) != spec_sha:
        raise SystemExit("STOP: train_params.json does not match the hash recorded in TRAIN_REPORT.md")
    spec = json.loads(spec_b.decode("utf-8"))
    pre = ck["prereg_sha256"]
    if spec["prereg"]["research/PREREGISTRATION.md"] != pre["research/PREREGISTRATION.md"] \
            or spec["prereg"]["research/prereg.json"] != pre["research/prereg.json"]:
        raise SystemExit("STOP: the frozen spec was not made from this plan")
    if spec["inputs"]["pinned_sha256"] != ck["pinned_sha256"]:
        raise SystemExit("STOP: the frozen spec records other pinned inputs")
    if spec["inputs"]["predictor_inputs_sha256"].get("output/research/predictor_inputs.pkl") != pi_sha:
        raise SystemExit("STOP: the frozen spec records another predictor_inputs.pkl")
    pt_sha = sha256_file(paths.panel_train)
    if spec["inputs"]["train_panel_sha256"].get("output/research/panel_train.pkl") != pt_sha:
        raise SystemExit("STOP: panel_train.pkl does not match the hash in the frozen spec")
    meta = json.loads(paths.panel_meta.read_text(encoding="utf-8"))
    if meta.get("spec_sha256") != pre or meta.get("predictor_inputs", {}).get("sha256") != pi_sha \
            or meta.get("pinned_sha256") != ck["pinned_sha256"]:
        raise SystemExit("STOP: panel_meta.json was not built from this plan, these pinned inputs and predictor_inputs.pkl")
    if not meta.get("checks", {}).get("calendar", {}).get("pass"):
        raise SystemExit("STOP: the build's calendar check did not pass")
    ck["spec_matches_code"] = check_spec_against_code(spec)
    frozen_utc = next((ln.split(":", 1)[1].strip() for ln in paths.frozen_hash.read_text(encoding="utf-8").splitlines()
                       if ln.startswith("frozen_utc:")), None)
    ck.update({"predictor_inputs_sha256": pi_sha, "frozen_spec_sha256": spec_sha, "frozen_utc": frozen_utc,
               "train_params_identical_to_frozen_spec": True, "panel_train_sha256": pt_sha,
               "panel_meta_sha256": sha256_file(paths.panel_meta)})
    return ck, spec, meta


def read_csv_outcomes(path: Path, unseal_token: str, pinned_sha: str, **verify_kwargs) -> Dict[str, Dict[pd.Timestamp, float]]:
    """The pinned CSV's outcome fields at every origin (section 13.4, Test phase), readable only
    through a verified unseal. Empty fields are missing."""
    verify_unseal(unseal_token, caller="read_csv_outcomes", **verify_kwargs)
    if sha256_file(path) != pinned_sha:
        raise SystemExit("STOP: rating_backtest.csv does not match its pinned hash")

    def num(s: str) -> float:
        try:
            return float(s)
        except ValueError:
            return float("nan")
    got: Dict[str, Dict[pd.Timestamp, float]] = {k: {} for k in TARGETS}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        ix = {c: header.index(c) for c in header}
        for row in r:
            t = pd.Timestamp(row[ix["date"]])
            for h in HORIZONS:
                ex, sp, tb = (num(row[ix[f"{k}_{h}"]]) for k in ("excess", "sp", "tbill"))
                got[f"Y{h}"][t] = ex
                got[f"S{h}"][t] = sp - tb
            got["D20"][t] = num(row[ix[f"real_dd20_{DIP_H}"]])
    return got


# ---- after unsealing: the registered computations ----------------------------------------------
def target_reproduction_all(p_all: pd.DataFrame, csv_outcomes: Dict[str, Dict], tol: float = REPRO_TOL) -> Dict:
    """Section 13.4 (Test phase): rebuilt Y, S and D20 against the pinned CSV at every origin."""
    out, ok = {}, True
    for k in TARGETS:
        ref = pd.Series(csv_outcomes[k], dtype=float).dropna()
        ref = ref.loc[ref.index.isin(p_all.index)]
        mine = p_all[k].astype(float).dropna()
        both = mine.index.intersection(ref.index)
        diff = (mine.reindex(both) - ref.reindex(both)).abs()
        n_over = int((diff > tol).sum())
        miss_csv = int(len(mine.index.difference(ref.index)))
        csv_only = int(len(ref.index.difference(mine.index)))
        test_side = both[both >= TEST_START]
        out[k] = {"compared": int(len(both)), "compared_test_side": int(len(test_side)),
                  "max_abs_diff": float(diff.max()) if len(diff) else None, "n_over_tol": n_over,
                  "rebuilt_not_in_csv": miss_csv, "csv_not_rebuilt": csv_only}
        ok &= n_over == 0 and miss_csv == 0 and csv_only == 0 and len(both) > 0
    out["tolerance"] = tol
    out["pass"] = bool(ok)
    return out


def legacy_reproduction(p_all: pd.DataFrame, legacy_score: pd.Series) -> Dict:
    """Section 13.5: spearman(score, Y(t,126)) over every month-end 1990-01-31..2026-02-27 with both,
    from the pinned CSV's score column (the two D5 origins included, as the README computed it)."""
    from spxlcast.evaluation import spearman
    y = p_all["Y126"].astype(float)
    s = legacy_score.astype(float).reindex(p_all.index)
    m = (p_all.index >= LEGACY_RANGE[0]) & (p_all.index <= LEGACY_RANGE[1]) & y.notna().values & s.notna().values
    r = spearman(s.values[m], y.values[m]) if m.sum() >= 3 else float("nan")
    return {"r": r, "n": int(m.sum()), "registered": LEGACY_R, "tolerance": LEGACY_TOL,
            "first": f"{p_all.index[m][0]:%Y-%m-%d}" if m.any() else None,
            "last": f"{p_all.index[m][-1]:%Y-%m-%d}" if m.any() else None,
            "pass": bool(np.isfinite(r) and abs(r - LEGACY_R) <= LEGACY_TOL + 1e-12)}


def test_side_coverage(p_all: pd.DataFrame, comp: pd.Series) -> Dict:
    """Section 13.7 (Test phase): non-missing share of every predictor at the test origins of each
    horizon; below 90% at h = 126 flags that predictor's test."""
    out = {}
    for pid in PREDICTORS:
        x = comp if pid == "COMP" else p_all[pid]
        rec = {}
        for h in HORIZONS:
            sel = p_all[f"test_h{h}"].astype(bool)
            n, k = int(sel.sum()), int(x[sel].notna().sum())
            rec[f"test_h{h}"] = {"n": n, "non_missing": k, "share": k / n if n else None}
        rec["flag_below_90pct"] = bool(rec["test_h126"]["share"] is not None and rec["test_h126"]["share"] < COVERAGE_MIN)
        out[pid] = rec
    return out


def test_calendar(p_all: pd.DataFrame, spec: Dict) -> Dict:
    exp = spec["calendar"]["split"]["expected_from_calendar"]
    out, ok = {}, True
    for h in HORIZONS:
        sel = p_all.index[p_all[f"test_h{h}"].astype(bool)]
        pos = p_all.loc[sel, "pos"].values
        got = [f"{sel[0]:%Y-%m-%d}", f"{sel[-1]:%Y-%m-%d}", int(len(sel)), round(effective_n(pos, h), 1)] if len(sel) else []
        match = got == exp[f"h{h}"]["test"]
        ok &= match
        out[f"h{h}"] = {"got": got, "registered": exp[f"h{h}"]["test"], "match": match}
    out["pass"] = bool(ok)
    return out


def block_bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = BOOT_N, block: int = MBB_BLOCK,
                       seed: int = SEED) -> Dict:
    """Section 13.12 moving-block bootstrap of r: a fresh rng per predictor; k = ceil(n / 12) blocks of
    12 consecutive origins per resample, starts = rng.integers(0, n - 12 + 1, size=(10000, k)), cut
    to n rows; 5th and 95th percentiles (numpy, linear) of the finite r_b."""
    from spxlcast.evaluation import spearman
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < block:
        return {"ci90": [float("nan"), float("nan")], "n_finite": 0, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    k = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    rows = (starts[:, :, None] + np.arange(block)).reshape(n_boot, k * block)[:, :n]
    rb = np.array([spearman(x[r], y[r]) for r in rows])
    fin = rb[np.isfinite(rb)]
    lo, hi = (np.percentile(fin, [5, 95]) if len(fin) else (float("nan"), float("nan")))
    return {"ci90": [float(lo), float(hi)], "n_finite": int(len(fin)), "n_boot": n_boot}


def partial_spearman(x, y, c1, c2, pos, h: int) -> Dict:
    """Section 13.12: average ranks of x, y and the two controls over the rows where all four exist;
    residuals of rank(x) and rank(y) on [1, rank(c1), rank(c2)] by OLS; r = their Pearson correlation;
    one-sided p with n_eff - 5 in place of n_eff - 3."""
    from scipy.stats import rankdata
    arr = [np.asarray(v, float) for v in (x, y, c1, c2)]
    pos = np.asarray(pos)
    m = np.all([np.isfinite(v) for v in arr], axis=0)
    n = int(m.sum())
    if n < 5:
        return {"r": float("nan"), "n": n, "n_eff": 0.0, "z": float("nan"), "p": 1.0}
    rx, ry, r1, r2 = (rankdata(v[m]) for v in arr)
    X = np.column_stack([np.ones(n), r1, r2])
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    r = float(np.corrcoef(ex, ey)[0, 1]) if ex.std() > 0 and ey.std() > 0 else float("nan")
    n_eff = float(effective_n(pos[m], h))
    z, p = one_sided(r, n_eff, dof_loss=5)
    return {"r": r, "n": n, "n_eff": n_eff, "z": z, "p": p}


def expanding_composite(p_all: pd.DataFrame, min_n: int = EXPANDING_MIN) -> Tuple[pd.Series, pd.Series]:
    """Section 13.12: at each month-end t, mu_j(t) and sd_j(t) (ddof = 1) over the month-ends from
    1990-01-31 through t where x_j exists; member j counts at t only if x_jt exists and it has at
    least 60 such values; then the section 7 formula (clip 3, at least 7 members)."""
    z = {}
    for c in MEMBERS:
        x = p_all[c].astype(float)
        mu, sd, cnt = x.expanding().mean(), x.expanding().std(ddof=1), x.expanding().count()
        zc = SIGNS[c] * (x - mu) / sd
        z[c] = zc.where(x.notna() & (cnt >= min_n)).clip(-Z_CLIP, Z_CLIP)
    zdf = pd.DataFrame(z, index=p_all.index)
    n = zdf.notna().sum(axis=1)
    return zdf.mean(axis=1, skipna=True).where(n >= COMPOSITE_MIN_MEMBERS), n


def primary_sample(p_all: pd.DataFrame, x: pd.Series, target: str = "Y126", h: int = 126) -> pd.DataFrame:
    """The primary sample: test origins for h at which the (signed) predictor and the target exist."""
    sel = p_all[f"test_h{h}"].astype(bool)
    y = p_all[target].astype(float)
    x = x.reindex(p_all.index).astype(float)
    m = sel & x.notna() & y.notna()
    return pd.DataFrame({"x": x[m], "y": y[m], "pos": p_all.loc[m, "pos"].astype(int),
                         "lnvix": np.log(p_all.loc[m, "vix_tm"].astype(float)),
                         "y3": p_all.loc[m, "y3_tm"].astype(float)})


def primary_robustness(p_all: pd.DataFrame, comp: pd.Series, d_spread: Dict[float, pd.DataFrame],
                       cal: pd.DatetimeIndex) -> Dict:
    """Section 13.12 for all 13 predictors (labels apply to primary passes): halves, episodes left
    out, moving-block bootstrap interval, partial Spearman on ln VIX(t-) and DGS3MO(t-); then the
    expanding-window composite and the synthetic-spread variants."""
    out: Dict[str, object] = {"per_predictor": {}}
    for pid in PREDICTORS:
        s = primary_sample(p_all, comp if pid == "COMP" else signed(p_all, pid))
        months = s.index.to_period("M")
        xs, ys, ps = s["x"].to_numpy(), s["y"].to_numpy(), s["pos"].to_numpy()
        halves = []
        for lo, hi in HALVES:
            k = np.asarray((months >= pd.Period(lo, "M")) & (months <= pd.Period(hi, "M")))
            halves.append({"origins": [lo, hi], **corr_test(xs[k], ys[k], ps[k], 126)})
        eps = []
        for name, lo, hi in EPISODES:
            k = ~np.asarray((months >= pd.Period(lo, "M")) & (months <= pd.Period(hi, "M")))
            eps.append({"episode": name, "left_out": [lo, hi], "n_left_out": int((~k).sum()),
                        **corr_test(xs[k], ys[k], ps[k], 126)})
        labels = []
        if any(e["p"] > EPISODE_P_MAX for e in eps):
            labels.append("episode-dependent")
        if not all(np.isfinite(hv["r"]) and hv["r"] > 0 for hv in halves):
            labels.append("one half only")
        out["per_predictor"][pid] = {
            "halves": halves, "episodes": eps, "block_bootstrap": block_bootstrap_ci(xs, ys),
            "partial_spearman": partial_spearman(xs, ys, s["lnvix"].to_numpy(), s["y3"].to_numpy(), ps, 126),
            "labels": labels}
    # expanding-window standardisation of the composite
    ecomp, en = expanding_composite(p_all)
    s = primary_sample(p_all, ecomp)
    sel = p_all["test_h126"].astype(bool)
    out["expanding_composite"] = {**corr_test(s["x"].to_numpy(), s["y"].to_numpy(), s["pos"].to_numpy(), 126),
                                  "members_available_test_h126": {str(int(k)): int(v) for k, v in
                                                                  en[sel].value_counts().sort_index().items()}}
    # synthetic fund spread at 0.25% and 1.25% (fund returns before 2009-01-02 only)
    sp = {}
    pos = p_all.loc[sel, "pos"].astype(int).to_numpy()
    base = p_all.loc[sel, "Y126"].astype(float).to_numpy()
    for spread, ds in d_spread.items():
        if not ds.index.equals(cal):
            raise ValueError("a synthetic-spread frame is not on the session calendar")
        y = compute_targets(ds, pos, cal)["Y126"].to_numpy(float)
        rows = {pid: corr_test((comp if pid == "COMP" else signed(p_all, pid)).to_numpy(float)[sel.values], y, pos, 126)
                for pid in PREDICTORS}
        adj = holm({k: v["p"] for k, v in rows.items()})
        for k in rows:
            rows[k].update(adj[k])
        # a fund level is a cumulative product, so later ratios move by rounding only: count a change
        # above the reproduction tolerance
        same = (np.abs(y - base) <= REPRO_TOL) | (np.isnan(y) & np.isnan(base))
        changed = p_all.index[sel.values][~same]
        sp[f"{spread:g}"] = {"rows": rows, "origins_changed": int(len(changed)),
                             "changed_from": f"{changed.min():%Y-%m-%d}" if len(changed) else None,
                             "changed_to": f"{changed.max():%Y-%m-%d}" if len(changed) else None}
    out["synthetic_spread"] = sp
    return out


# ---- the economic test (section 11) ------------------------------------------------------------
def econ_segments(p_all: pd.DataFrame, cal: pd.DatetimeIndex) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """The 223 monthly holdings: origin t (2008-01-31..2026-07-31, session position a) to the next
    month-end (position b), earning the returns of sessions a+1..b."""
    idx = p_all.index
    loc = np.flatnonzero((idx >= ECON_ORIGINS[0]) & (idx <= ECON_ORIGINS[1]))
    if len(loc) != ECON_HOLDINGS or loc[-1] + 1 >= len(idx):
        raise ValueError(f"expected {ECON_HOLDINGS} holdings, found {len(loc)}")
    pos = p_all["pos"].astype(int).to_numpy()
    a, b = pos[loc], pos[loc + 1]
    dates = cal[a[0] + 1: b[-1] + 1]
    if not (b[:-1] == a[1:]).all() or dates[0] != ECON_SESSIONS[0] or dates[-1] != ECON_SESSIONS[1]:
        raise ValueError("the holdings do not tile the sessions 2008-02-01..2026-08-31")
    return idx[loc], a, b, dates


def rule_weights(signal: Sequence[float], threshold: float) -> Tuple[np.ndarray, int]:
    """1 (SPXL) where signal >= threshold, else 0 (T-bills); a missing signal keeps the previous
    holding (T-bills at the first origin). Returns (weights, missing count)."""
    x = np.asarray(signal, float)
    w, prev, miss = np.empty(len(x)), 0.0, 0
    for i, v in enumerate(x):
        if np.isfinite(v):
            prev = 1.0 if v >= threshold else 0.0
        else:
            miss += 1
        w[i] = prev
    return w, miss


def simulate_weights(w: Sequence[float], a: np.ndarray, b: np.ndarray, fund: np.ndarray, bill: np.ndarray,
                     cost: float) -> Dict[str, object]:
    """Target SPXL weight w_i for holding i (sessions a_i+1..b_i), the rest T-bills. The two sleeves
    drift within the holding; at each later month-end the portfolio is rebalanced to w_i, paying
    cost * |w_i - w_pre| (w_pre = the drifted weight) as a factor (1 - cost |.|) on the first session
    of the new holding. The first allocation is free. w in {0, 1} is the switching rule (a switch
    costs ``cost``)."""
    w = np.asarray(w, float)
    rets, monthly = [], np.empty(len(w))
    traded, crate = np.zeros(len(w)), np.zeros(len(w))
    paid, wealth, w_end = 0.0, 1.0, None
    for i in range(len(w)):
        f, bl = fund[a[i] + 1: b[i] + 1], bill[a[i] + 1: b[i] + 1]
        wi = float(w[i])
        if wi == 1.0:
            r, we = np.array(f, float), 1.0
        elif wi == 0.0:
            r, we = np.array(bl, float), 0.0
        else:
            vf, vb = wi * np.cumprod(1.0 + f), (1.0 - wi) * np.cumprod(1.0 + bl)
            v = vf + vb
            r = v / np.concatenate(([1.0], v[:-1])) - 1.0
            we = float(vf[-1] / v[-1])
        if i > 0:
            traded[i] = abs(wi - w_end)
            crate[i] = cost * traded[i]
            if crate[i] > 0:
                r[0] = (1.0 - crate[i]) * (1.0 + r[0]) - 1.0
                paid += wealth * crate[i]
        g = float(np.prod(1.0 + r))
        monthly[i] = g - 1.0
        wealth *= g
        rets.append(r)
        w_end = we
    return {"r": np.concatenate(rets), "monthly": monthly, "w": w, "traded": traded, "cost_rate": crate,
            "cost_paid": paid}


def perf(r: np.ndarray, bill: np.ndarray, dates: pd.DatetimeIndex) -> Dict:
    """Section 11 metrics on daily returns (backtest_rating.strategy_section): CAGR = W_N^(252/N) - 1;
    Sharpe = mean / sd (ddof = 1) of (r - tbill_ret) x sqrt(252); max drawdown over W_1..W_N; E4's
    Sharpe within each half from the full-period series."""
    r, bill = np.asarray(r, float), np.asarray(bill, float)
    W = np.cumprod(1.0 + r)
    n = len(r)
    ex = r - bill

    def sharpe(e: np.ndarray) -> float:
        sd = float(e.std(ddof=1)) if len(e) > 1 else 0.0
        return float(e.mean() / sd * math.sqrt(252.0)) if sd > 0 else float("nan")
    halves, half_n = [], []
    for lo, hi in E4_HALVES:
        m = np.asarray((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi)))
        halves.append(sharpe(ex[m]))
        half_n.append(int(m.sum()))
    dd = 1.0 - W / np.maximum.accumulate(W)
    return {"sessions": n, "cagr": float(W[-1] ** (252.0 / n) - 1.0), "sharpe": sharpe(ex),
            "max_drawdown": float(dd.max()), "vol": float(r.std(ddof=1) * math.sqrt(252.0)),
            "final_wealth": float(W[-1]), "sharpe_halves": halves, "half_sessions": half_n}


def sim_descr(s: Dict) -> Dict:
    w, tr = np.asarray(s["w"]), np.asarray(s["traded"])
    return {"time_in_spxl": float(w.mean()), "holdings_in_spxl": int((w == 1.0).sum()),
            "switches_or_rebalances": int((tr[1:] > 0).sum()), "turnover": float(tr.sum()),
            "cost_rate_sum": float(np.sum(s["cost_rate"])), "costs_paid_per_initial_dollar": float(s["cost_paid"])}


def e_criteria(pf: Dict[str, Dict]) -> Dict:
    r, bh, mx = pf["rule"], pf["BH"], pf["MIX_w"]
    e1 = bool(r["cagr"] > bh["cagr"] and r["cagr"] > mx["cagr"])
    e2 = bool(r["sharpe"] > bh["sharpe"] and r["sharpe"] > mx["sharpe"])
    e3 = bool(r["max_drawdown"] < bh["max_drawdown"])
    e4 = bool(all(a > b for a, b in zip(r["sharpe_halves"], bh["sharpe_halves"])))
    return {"E1": e1, "E2": e2, "E3": e3, "E4": e4}


def rule_vs_benchmarks(w: np.ndarray, a: np.ndarray, b: np.ndarray, fund: np.ndarray, bill: np.ndarray,
                       dates: pd.DatetimeIndex, cost: float) -> Tuple[Dict, Dict]:
    """A switching rule against BH (no cost) and MIX_w (w = holdings in SPXL / 223, monthly
    rebalanced, cost x |weight traded|), with E1-E4. Returns (summary, simulations)."""
    n = len(w)
    bill_d = bill[a[0] + 1: b[-1] + 1]
    sims = {"rule": simulate_weights(w, a, b, fund, bill, cost),
            "BH": simulate_weights(np.ones(n), a, b, fund, bill, 0.0)}
    w_mix = float(np.sum(np.asarray(w) == 1.0)) / n
    sims["MIX_w"] = simulate_weights(np.full(n, w_mix), a, b, fund, bill, cost)
    pf = {k: perf(s["r"], bill_d, dates) for k, s in sims.items()}
    e = e_criteria(pf)
    return ({"cost": cost, "w_mix": w_mix, "strategies": {k: {**pf[k], **sim_descr(sims[k])} for k in sims},
             "E": e, "pass": bool(all(e.values()))}, sims)


def stationary_bootstrap_idx(n: int, n_boot: int = BOOT_N, mean_block: int = SB_MEAN_BLOCK, seed: int = SEED) -> np.ndarray:
    """Politis-Romano, circular, as frozen: rng = default_rng(seed); U = rng.random((B, n));
    S = rng.integers(0, n, size=(B, n)); idx[:, 0] = S[:, 0]; idx[:, i] = S[:, i] where U[:, i] <
    1/mean_block else (idx[:, i-1] + 1) mod n."""
    rng = np.random.default_rng(seed)
    U = rng.random((n_boot, n))
    S = rng.integers(0, n, size=(n_boot, n))
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = S[:, 0]
    for i in range(1, n):
        idx[:, i] = np.where(U[:, i] < 1.0 / mean_block, S[:, i], (idx[:, i - 1] + 1) % n)
    return idx


def monthly_sharpe(r: np.ndarray, rb: np.ndarray) -> np.ndarray:
    ex = np.asarray(r, float) - np.asarray(rb, float)
    sd = ex.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, ex.mean(axis=-1) / sd * math.sqrt(12.0), np.nan)


def monthly_cagr(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, float)
    return np.prod(1.0 + r, axis=-1) ** (12.0 / r.shape[-1]) - 1.0


def stationary_bootstrap(rule_m: np.ndarray, bh_m: np.ndarray, bill_m: np.ndarray) -> Dict:
    """Section 11 (secondary): 90% intervals for Sharpe(rule) - Sharpe(BH) and CAGR(rule) - CAGR(BH)
    from the monthly holding-period returns, resampled as aligned triples."""
    rule_m, bh_m, bill_m = (np.asarray(v, float) for v in (rule_m, bh_m, bill_m))
    idx = stationary_bootstrap_idx(len(rule_m))
    R, B, F = rule_m[idx], bh_m[idx], bill_m[idx]
    ds = monthly_sharpe(R, F) - monthly_sharpe(B, F)
    dc = monthly_cagr(R) - monthly_cagr(B)

    def ci(v: np.ndarray) -> List[float]:
        v = v[np.isfinite(v)]
        return [float(q) for q in np.percentile(v, [5, 95])] if len(v) else [float("nan")] * 2
    return {"sharpe_diff": float(monthly_sharpe(rule_m, bill_m) - monthly_sharpe(bh_m, bill_m)),
            "sharpe_diff_ci90": ci(ds),
            "cagr_diff": float(monthly_cagr(rule_m) - monthly_cagr(bh_m)), "cagr_diff_ci90": ci(dc),
            "n_boot": BOOT_N, "mean_block_months": SB_MEAN_BLOCK, "seed": SEED, "months": int(len(rule_m))}


def tbill_spells(origins: pd.DatetimeIndex, w: np.ndarray) -> List[Dict]:
    """Runs of consecutive holdings in T-bills (w = 0), by origin month."""
    out, start = [], None
    for i, wi in enumerate(np.append(np.asarray(w, float), 1.0)):
        if wi == 0.0 and start is None:
            start = i
        elif wi != 0.0 and start is not None:
            out.append({"from": f"{origins[start]:%Y-%m}", "to": f"{origins[i - 1]:%Y-%m}", "months": i - start})
            start = None
    return out


def economic_suite(p_all: pd.DataFrame, comp: pd.Series, d: pd.DataFrame, cal: pd.DatetimeIndex,
                   spec: Dict) -> Tuple[Dict, Dict[str, pd.DataFrame]]:
    """Section 11: the composite rule against BH and MIX_w (decisive), the same for every candidate
    with its own tau_j, and the secondary descriptive analyses."""
    origins, a, b, dates = econ_segments(p_all, cal)
    fund = d["fund_ret"].fillna(0.0).to_numpy(float)
    bill = d["tbill_ret"].fillna(0.0).to_numpy(float)
    ts, cost = spec["train_set"], float(spec["economic_test"]["switch_cost"])
    tau, tau_med, tau_j = ts["composite"]["tau"], ts["composite"]["tau_median"], ts["tau_j"]["values"]
    sig = comp.reindex(origins).to_numpy(float)
    rules, sims = {}, {}
    w_comp, miss = rule_weights(sig, tau)
    rules["COMP"], sims["COMP"] = rule_vs_benchmarks(w_comp, a, b, fund, bill, dates, cost)
    rules["COMP"].update({"threshold": tau, "missing_signal": miss, "tbill_spells": tbill_spells(origins, w_comp)})
    for c in CAND_IDS:
        wc, mc = rule_weights(signed(p_all, c).reindex(origins).to_numpy(float), tau_j[c])
        rules[c], _ = rule_vs_benchmarks(wc, a, b, fund, bill, dates, cost)
        rules[c].update({"threshold": tau_j[c], "missing_signal": mc})
    alt = {}
    for x in ALT_COSTS:
        alt[f"{x:g}"], _ = rule_vs_benchmarks(w_comp, a, b, fund, bill, dates, x)
    w_med, mm = rule_weights(sig, tau_med)
    median, _ = rule_vs_benchmarks(w_med, a, b, fund, bill, dates, cost)
    median.update({"threshold": tau_med, "missing_signal": mm})
    bill_m = simulate_weights(np.zeros(len(w_comp)), a, b, fund, bill, 0.0)["monthly"]
    boot = stationary_bootstrap(sims["COMP"]["rule"]["monthly"], sims["COMP"]["BH"]["monthly"], bill_m)
    # volatility-managed exposure (Moreira-Muir style; not clean: VIX level)
    c_vm, w_bar = ts["vol_managed"]["c"], ts["vol_managed"]["w_bar"]
    ivar = p_all["C10"].reindex(origins).to_numpy(float)
    w_vm, prev, miss_vm = [], 0.0, 0
    for v in ivar:
        if np.isfinite(v) and v > 0:
            prev = min(1.0, c_vm / v)
        else:
            miss_vm += 1
        w_vm.append(prev)
    bill_d = bill[a[0] + 1: b[-1] + 1]
    s_vm = simulate_weights(np.array(w_vm), a, b, fund, bill, cost)
    s_const = simulate_weights(np.full(len(w_vm), w_bar), a, b, fund, bill, cost)
    vol = {"c": c_vm, "w_bar": w_bar, "missing_ivar": miss_vm, "mean_target_weight": float(np.mean(w_vm)),
           "strategies": {"vol_managed": {**perf(s_vm["r"], bill_d, dates), **sim_descr(s_vm)},
                          "constant_w_bar": {**perf(s_const["r"], bill_d, dates), **sim_descr(s_const)},
                          "BH": rules["COMP"]["strategies"]["BH"]}}
    res = {"holdings": int(len(origins)), "first_origin": f"{origins[0]:%Y-%m-%d}", "last_origin": f"{origins[-1]:%Y-%m-%d}",
           "sessions": [f"{dates[0]:%Y-%m-%d}", f"{dates[-1]:%Y-%m-%d}"], "n_sessions": int(len(dates)),
           "switch_cost": cost, "tau": tau, "rules": rules, "alt_costs": alt, "median_threshold": median,
           "stationary_bootstrap": boot, "vol_managed": vol}
    s = sims["COMP"]
    daily = pd.DataFrame({"fund_ret": fund[a[0] + 1: b[-1] + 1], "tbill_ret": bill_d, "rule": s["rule"]["r"],
                          "BH": s["BH"]["r"], "MIX_w": s["MIX_w"]["r"], "vol_managed": s_vm["r"],
                          "constant_w_bar": s_const["r"]}, index=dates)
    holdings = pd.DataFrame({"next_origin": cal[b], "COMP": sig, "w_rule": w_comp, "rule_hpr": s["rule"]["monthly"],
                             "BH_hpr": s["BH"]["monthly"], "MIX_w_hpr": s["MIX_w"]["monthly"], "bill_hpr": bill_m,
                             "IVAR": ivar, "w_vol_managed": w_vm}, index=origins)
    return res, {"daily": daily, "holdings": holdings}


def test_descriptives(p_all: pd.DataFrame, comp: pd.Series, spec: Dict) -> Dict:
    """Section 13.9 after unsealing: predictor values at the 224 test-side month-ends, and the share
    of the 223 economic-test origins at which each is favourable (s_j x_j >= tau_j; COMP >= tau)."""
    rows = p_all.index >= TEST_START
    econ = (p_all.index >= ECON_ORIGINS[0]) & (p_all.index <= ECON_ORIGINS[1])
    thr = {**spec["train_set"]["tau_j"]["values"], "COMP": spec["train_set"]["composite"]["tau"]}
    out = {}
    for pid in PREDICTORS:
        x = (comp if pid == "COMP" else p_all[pid]).astype(float)
        v = x[rows].dropna().to_numpy(float)
        q = np.percentile(v, [25, 50, 75]) if len(v) else [np.nan] * 3
        sx = (comp if pid == "COMP" else signed(p_all, pid))[econ].dropna()
        out[pid] = {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None,
                    "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "min": float(v.min()) if len(v) else None,
                    "p25": float(q[0]), "median": float(q[1]), "p75": float(q[2]), "max": float(v.max()) if len(v) else None,
                    "favourable_share_econ_origins": float((sx >= thr[pid]).mean()) if len(sx) else None}
    return out


def decide(spec: Dict, primary_rows: Dict[str, Dict], rules: Dict[str, Dict], robustness: Dict[str, Dict]) -> Dict:
    """Section 12 in the registered words."""
    confirmed, predictive, luck, none_found = [v["label"] for v in spec["verdict"]]
    passing = [pid for pid in PREDICTORS if primary_rows[pid]["reject"]]
    per = {}
    for pid in passing:
        econ_ok = bool(rules[pid]["pass"])
        per[pid] = {"economic_pass": econ_ok, "label": confirmed if econ_ok else predictive,
                    "qualifier": QUALIFIERS.get(pid) if econ_ok else None,
                    "robustness_labels": robustness[pid]["labels"]}
    comp_econ = bool(rules["COMP"]["pass"])
    if passing:
        conf = [pid for pid in passing if per[pid]["economic_pass"]]
        if conf:
            headline = confirmed + ": " + ", ".join(
                f"{pid} {SHORTS[pid]}" + (f" ({per[pid]['qualifier']})" if per[pid]["qualifier"] else "") for pid in conf)
        else:
            headline = predictive + ": " + ", ".join(f"{pid} {SHORTS[pid]}" for pid in passing)
    else:
        headline = luck if comp_econ else none_found
    return {"headline": headline, "primary_passes": passing, "composite_economic_pass": comp_econ,
            "per_predictor": per, "registered_labels": [confirmed, predictive, luck, none_found]}


def phase3_compute(p_train: pd.DataFrame, p_test: pd.DataFrame, d: pd.DataFrame, cal: pd.DatetimeIndex, spec: Dict,
                   csv_outcomes: Dict[str, Dict], legacy_score: pd.Series, d_spread: Dict[float, pd.DataFrame],
                   train_stats: Optional[Dict] = None) -> Tuple[Dict, Dict[str, pd.DataFrame]]:
    """Everything the frozen spec prescribes for the Test phase, from the unsealed daily frame ``d``.
    Returns (results, frames); the results are plain data (written as JSON), the frames go to CSV."""
    if not d.index.equals(cal):
        raise ValueError("the unsealed daily frame is not on the panel's session calendar")
    p_all = pd.concat([p_train, p_test]).sort_index()
    tg = compute_targets(d, p_all["pos"].astype(int).tolist(), cal)
    if not tg.index.equals(p_all.index):
        raise ValueError("targets and panel month-ends differ")
    for k in TARGETS:
        p_all[k] = tg[k].to_numpy(float)
    std = {c["id"]: {"mu": float(c["mu"]), "sd": float(c["sd"])} for c in spec["candidates"]}
    comp, n_mem = composite(p_all, std)

    # the train side is unchanged by unsealing (the sealed frame is a prefix of the full one)
    same = {}
    for k in TARGETS:
        h = DIP_H if k == "D20" else int(k[1:])
        sel = p_train[f"train_h{h}"].astype(bool)
        a_, b_ = p_train.loc[sel, k].to_numpy(float), p_all.loc[p_train.index[sel], k].to_numpy(float)
        same[k] = bool(len(a_) and np.array_equal(a_, b_))
    fam_test = family_tests(p_all, comp, "test")
    rep_train = replication(p_train, comp.reindex(p_train.index))
    rep_match = None
    if train_stats is not None:
        old = train_stats.get("replication", {})
        rep_match = all(
            _same(old[f]["rows"][pid]["r"], rep_train[f]["rows"][pid]["r"])
            and _same(old[f]["rows"][pid]["p"], rep_train[f]["rows"][pid]["p"])
            for f in rep_train for pid in PREDICTORS)
    sanity = {"train_targets_unchanged_by_unsealing": {"per_target": same, "pass": bool(all(same.values()))},
              "target_reproduction_all_origins": target_reproduction_all(p_all, csv_outcomes),
              "legacy_reproduction": legacy_reproduction(p_all, legacy_score),
              "synthetic_fund": {**br.validate_synthetic(d)},
              "coverage": test_side_coverage(p_all, comp),
              "calendar_test_origins": test_calendar(p_all, spec),
              "composite_members_test_side": {str(int(k)): int(v) for k, v in
                                              n_mem[p_all.index >= TEST_START].value_counts().sort_index().items()},
              "standardisation": {c["id"]: {"mu": c["mu"], "sd": c["sd"], "n": c["n_param_window"]} for c in spec["candidates"]},
              "no_zero_sd": bool(all(c["sd"] > 0 for c in spec["candidates"])),
              "train_replication_matches_train_stats": rep_match}
    sanity["synthetic_fund"]["pass"] = bool(sanity["synthetic_fund"]["corr"] >= SYNTHETIC_CORR_MIN)
    robust = primary_robustness(p_all, comp, d_spread, cal)
    econ, frames = economic_suite(p_all, comp, d, cal, spec)
    verdict = decide(spec, fam_test["primary"]["rows"], econ["rules"], robust["per_predictor"])
    res = {"primary": fam_test["primary"], "families": {f: v for f, v in fam_test.items() if f != "primary"},
           "replication_train": rep_train, "sanity": sanity, "robustness": robust, "economic": econ,
           "descriptives_test": test_descriptives(p_all, comp, spec), "verdict": verdict}
    panel = p_all.copy()
    panel["COMP"], panel["COMP_members"] = comp, n_mem
    frames["panel"] = panel
    return res, frames


# ---- outputs ----------------------------------------------------------------------------------
def tests_table(res: Dict) -> pd.DataFrame:
    rows = []
    for side, fams in (("test", {"primary": res["primary"], **res["families"]}), ("train", res["replication_train"])):
        for fam, f in fams.items():
            for pid, v in f["rows"].items():
                rows.append({"side": side, "family": fam, "target": f["target"], "h": f["h"], "predictor": pid,
                             "signal": SHORTS[pid], "n": v["n"], "n_eff": v["n_eff"], "r": v["r"],
                             "ci90_lo": v["ci90"][0], "ci90_hi": v["ci90"][1], "z": v["z"], "p": v["p"],
                             "p_holm": v["p_holm"], "holm_step": v["step"], "holm_threshold": v["threshold"],
                             "holm_reject": v["reject"]})
    return pd.DataFrame(rows)


def plot_test(frames: Dict[str, pd.DataFrame], tau: float, path: Path) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:                                       # pragma: no cover
        return f"not drawn ({type(exc).__name__})"
    daily, hold = frames["daily"], frames["holdings"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for col, colour in (("BH", "#3a6ea5"), ("rule", "#b5523b"), ("MIX_w", "#6b8e23")):
        ax1.plot(daily.index, np.cumprod(1.0 + daily[col].to_numpy()), lw=1.0, color=colour,
                 label={"BH": "buy and hold SPXL", "rule": "composite rule", "MIX_w": "MIX_w"}[col])
    ax1.set_yscale("log")
    ax1.legend(loc="upper left", fontsize=8, frameon=False)
    ax1.set_title("Wealth of $1, 2008-02-01..2026-08-31 (0.10% per switch)", fontsize=10)
    ax2.step(hold.index, hold["COMP"], where="post", lw=1.0, color="#333")
    ax2.axhline(tau, color="#b5523b", lw=0.9, ls="--", label="tau")
    for t, nxt, w in zip(hold.index, pd.to_datetime(hold["next_origin"]), hold["w_rule"]):
        if w == 0.0:
            ax2.axvspan(t, nxt, color="#f2d7d0", lw=0)            # the holding runs to the next origin
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.set_title("COMP at each origin (shaded: T-bills)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def write_phase3_outputs(res: Dict, frames: Dict[str, pd.DataFrame], paths: Phase3Paths) -> Dict[str, str]:
    paths.test_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    try:
        written["plot"] = plot_test(frames, res["economic"]["tau"], paths.test_dir / "economic_test.png")
    except Exception as exc:                                         # a figure must never cost the results
        written["plot"] = f"not drawn ({type(exc).__name__})"
    tests_table(res).to_csv(paths.test_dir / "tests.csv", index=False)
    frames["daily"].to_csv(paths.test_dir / "economic_daily.csv", date_format="%Y-%m-%d")
    frames["holdings"].to_csv(paths.test_dir / "economic_holdings.csv", date_format="%Y-%m-%d")
    frames["panel"].to_pickle(paths.test_dir / "panel_unsealed.pkl")
    written.update({"tests": "output/research/test/tests.csv", "daily": "output/research/test/economic_daily.csv",
                    "holdings": "output/research/test/economic_holdings.csv",
                    "panel": "output/research/test/panel_unsealed.pkl"})
    res["outputs"] = written
    text = json.dumps(jsonable(res), indent=1, allow_nan=False) + "\n"
    write_lf(paths.results_json, text)
    write_lf(paths.test_dir / "test_results.json", text)
    return written


def _yn(b) -> str:
    return "pass" if b is True else ("FAIL" if b is False else "-")


def _pct(x, spec="{:+.1%}") -> str:
    return _fmt(x, spec)


def render_test_results(res: Dict) -> str:
    """research/test_results.md (and research/RESULTS.md) from the results JSON."""
    L: List[str] = []
    a = L.append
    v, pr, econ, san, rob = res["verdict"], res["primary"], res["economic"], res["sanity"], res["robustness"]
    main = econ["rules"]["COMP"]
    st = main["strategies"]
    rep = res["replication_train"]
    a("# SPXL timing-signal search: Test phase results")
    a("")
    a(f"Phase 3 of `research/PREREGISTRATION.md`, run once at {res['run_utc']} by "
      "`py scripts/research_signals.py test --unseal <SHA-256 of research/PREREGISTRATION.md>`. Every number the "
      f"test uses was taken from the frozen spec `research/frozen_spec.json` (SHA-256 `{res['spec_sha256']}`, frozen "
      f"{res['frozen_utc']}, before unsealing). Nothing in the spec, the candidates, the thresholds or the decision "
      "rule was changed after the outcomes were computed. Every registered test is reported below, whatever it shows.")
    a("")
    a("## Verdict")
    a("")
    a(f"**{v['headline']}**")
    a("")
    n_pass = len(v["primary_passes"])
    best = max(PREDICTORS, key=lambda k: (pr["rows"][k]["r"] if pr["rows"][k]["r"] is not None else -9))
    bv = pr["rows"][best]
    a(f"* **Primary test** (6-month SPXL excess return, 13 predictors, Holm at 5%): {n_pass} of 13 passed"
      + (f" ({', '.join(v['primary_passes'])})." if n_pass else ".")
      + f" The largest signed rank correlation was {best} {SHORTS[best]} at r = {_fmt(bv['r'])} (one-sided p = "
      f"{_fmt(bv['p'], '{:.4f}')}, Holm p = {_fmt(bv['p_holm'], '{:.3f}')}); the first Holm step needs p <= "
      f"{0.05 / 13:.4f}, about r >= 0.45 at the registered n_eff = {pr['rows']['COMP']['n_eff']:.1f}. That n_eff "
      "counts non-overlapping outcome windows and ignores how persistent each predictor is, so the bar is "
      "conservative (a persistence-aware check is in `research/RESULTS.md`, section 3.1; it is exploratory).")
    e = main["E"]
    a(f"* **Economic test** (composite rule vs buy-and-hold SPXL and MIX_w): "
      f"{'PASS' if main['pass'] else 'FAIL'} (E1 {_yn(e['E1'])}, E2 {_yn(e['E2'])}, E3 {_yn(e['E3'])}, E4 {_yn(e['E4'])}). "
      f"CAGR {_pct(st['rule']['cagr'])} vs {_pct(st['BH']['cagr'])} buy-and-hold and {_pct(st['MIX_w']['cagr'])} MIX_w; "
      f"Sharpe {_fmt(st['rule']['sharpe'])} vs {_fmt(st['BH']['sharpe'])} and {_fmt(st['MIX_w']['sharpe'])}; "
      f"max drawdown {_fmt(st['rule']['max_drawdown'], '{:.1%}')} vs {_fmt(st['BH']['max_drawdown'], '{:.1%}')}.")
    if v["per_predictor"]:
        for pid, pv in v["per_predictor"].items():
            ex = econ["rules"][pid]
            a(f"* **{pid} {SHORTS[pid]}**: {pv['label']}" + (f" ({pv['qualifier']})" if pv["qualifier"] else "")
              + f"; its own economic rule (tau_j = {_fmt(ex['threshold'], '{:.6g}')}): E1 {_yn(ex['E']['E1'])}, "
              f"E2 {_yn(ex['E']['E2'])}, E3 {_yn(ex['E']['E3'])}, E4 {_yn(ex['E']['E4'])}"
              + (f"; robustness labels: {', '.join(pv['robustness_labels'])}" if pv["robustness_labels"] else
                 "; robustness labels: none") + ".")
    a("* Section 12 reads: primary Holm pass and economic pass: 'Timing signal confirmed'; primary pass, economic "
      "fail: 'Predictive, not usable by the registered rule'; composite economic pass with no primary pass: 'Not "
      "confirmed (gain could be luck); candidate for a new pre-registration on future data only'; neither: 'No "
      "timing signal found'.")
    a("")
    a("## 1. Primary test: 13 predictors against Y(t,126), Holm at 5% (decisive)")
    a("")
    a(f"Test origins {pr['first']}..{pr['last']} ({pr['origins']} month-ends), r = Spearman of the signed predictor with "
      "the SPXL excess return over T-bills in the next 126 sessions; n_eff = the overlap-aware count of independent "
      "windows; z = atanh(r) sqrt(n_eff - 3) / sqrt(1 + r^2/2); p one-sided (favourable direction registered before "
      "2008); 90% interval as `backtest_rating._corr_ci`. Train (1990-2007) values are the published-sample "
      "replication from the Train phase. Holm: the smallest p must be at most 0.05/13 = 0.0038, the next at most "
      "0.05/12 = 0.0042, and so on; testing stops at the first failure.")
    a("")
    rows = []
    for pid in PREDICTORS:
        t, tr = pr["rows"][pid], rep["primary"]["rows"][pid]
        rows.append([pid, SHORTS[pid], t["n"], _fmt(t["n_eff"], "{:.1f}"), _fmt(t["r"]),
                     f"{_fmt(t['ci90'][0])} to {_fmt(t['ci90'][1])}", _fmt(t["z"]), _fmt(t["p"], "{:.4f}"),
                     f"{t['step']} ({t['threshold']:.4f})", _fmt(t["p_holm"], "{:.3f}"),
                     "**PASS**" if t["reject"] else "fail", f"{_fmt(tr['r'])} ({_fmt(tr['p'], '{:.2f}')})"])
    L += md_table(["predictor", "signal", "n", "n_eff", "r", "90% CI", "z", "p (1-sided)", "Holm step (alpha)",
                   "Holm p", "result", "train r (p)"], rows)
    a("")
    a("## 2. Economic test: the composite rule (decisive)")
    a("")
    a(f"At each origin {econ['first_origin']}..{econ['last_origin']} ({econ['holdings']} monthly holdings, sessions "
      f"{econ['sessions'][0]}..{econ['sessions'][1]}, N = {econ['n_sessions']}): SPXL (the realised fund) if COMP >= "
      f"tau = {econ['tau']:.6f}, else T-bills; 0.10% per switch on the first session of the new holding. MIX_w holds "
      f"w = {main['w_mix']:.4f} (the rule's share of months in SPXL) in SPXL, rebalanced monthly at 0.10% x |weight "
      f"traded|. Missing signals: {main['missing_signal']}. All figures are before tax (as in a tax-deferred "
      "account). Before 2009-01-02 the realised fund is the synthetic 3x fund (SPXL itself started trading on "
      "2008-11-05).")
    a("")
    rows = []
    for k, name in (("rule", "composite rule"), ("BH", "buy and hold SPXL"), ("MIX_w", "MIX_w")):
        s = st[k]
        rows.append([name, _pct(s["cagr"]), _fmt(s["sharpe"]), _fmt(s["max_drawdown"], "{:.1%}"), _fmt(s["vol"], "{:.0%}"),
                     _fmt(s["final_wealth"], "{:,.2f}"), _fmt(s["sharpe_halves"][0]), _fmt(s["sharpe_halves"][1]),
                     _fmt(s["time_in_spxl"], "{:.0%}"), s["switches_or_rebalances"], _fmt(s["turnover"], "{:.2f}"),
                     _fmt(s["costs_paid_per_initial_dollar"], "{:.4f}")])
    L += md_table(["strategy", "CAGR", "Sharpe", "max drawdown", "vol", "$1 becomes", "Sharpe 2008-02..2016-12",
                   "Sharpe 2017-01..2026-08", "SPXL weight (avg)", "switches / rebalances", "turnover",
                   "costs paid ($ per $1)"], rows)
    a("")
    rows = [["E1", "CAGR(rule) > CAGR(BH) and > CAGR(MIX_w)",
             f"{_pct(st['rule']['cagr'])} vs {_pct(st['BH']['cagr'])} / {_pct(st['MIX_w']['cagr'])}", _yn(e["E1"])],
            ["E2", "Sharpe(rule) > Sharpe(BH) and > Sharpe(MIX_w)",
             f"{_fmt(st['rule']['sharpe'])} vs {_fmt(st['BH']['sharpe'])} / {_fmt(st['MIX_w']['sharpe'])}", _yn(e["E2"])],
            ["E3", "max drawdown(rule) < max drawdown(BH)",
             f"{_fmt(st['rule']['max_drawdown'], '{:.1%}')} vs {_fmt(st['BH']['max_drawdown'], '{:.1%}')}", _yn(e["E3"])],
            ["E4", "Sharpe(rule) > Sharpe(BH) in both halves",
             f"{_fmt(st['rule']['sharpe_halves'][0])} vs {_fmt(st['BH']['sharpe_halves'][0])}; "
             f"{_fmt(st['rule']['sharpe_halves'][1])} vs {_fmt(st['BH']['sharpe_halves'][1])}", _yn(e["E4"])]]
    L += md_table(["criterion", "rule", "values", "result"], rows)
    a("")
    a(f"**Economic pass (all of E1-E4): {'yes' if main['pass'] else 'no'}.**")
    a("")
    spells = main.get("tbill_spells") or []
    a("Months in T-bills under the composite rule (origin months): "
      + ("; ".join(f"{s['from']}..{s['to']} ({s['months']})" if s["from"] != s["to"] else f"{s['from']} (1)" for s in spells)
         if spells else "none") + f". Plot: `{res.get('outputs', {}).get('plot', '-')}`.")
    a("")
    a("## 3. Secondary tests F2-F7 (reported, never decisive)")
    a("")
    a("Same predictors, statistic and one-sided direction; Holm within each family of 13. Cells: r (one-sided p); "
      "`*` = Holm pass within the family. Within a family the smallest p must be at most 0.0038 (0.05/13), the next "
      "at most 0.0042, and so on.")
    a("")
    fams = ["F2", "F3", "F4", "F5", "F6", "F7"]
    heads = {"primary": "Y 6m", "F2": "Y 1m", "F3": "Y 3m", "F4": "S&P 6m", "F5": "S&P 3m", "F6": "S&P 1m",
             "F7": "fewer 20% dips (-D20, 3m)"}
    allf = {"primary": pr, **res["families"]}
    rows = []
    for pid in PREDICTORS:
        rows.append([pid, SHORTS[pid]] + [f"{_fmt(allf[f]['rows'][pid]['r'])} ({_fmt(allf[f]['rows'][pid]['p'], '{:.4f}')})"
                                          f"{'*' if allf[f]['rows'][pid]['reject'] else ''}" for f in fams])
    L += md_table(["predictor", "signal"] + [heads[f] for f in fams], rows)
    a("")
    a("Origins and n_eff: " + "; ".join(f"{heads[f]} {allf[f]['origins']} origins, n_eff "
                                        f"{allf[f]['rows']['C01']['n_eff']:.1f}" for f in fams) + ".")
    passes = [(f, pid) for f in fams for pid in PREDICTORS if allf[f]["rows"][pid]["reject"]]
    a("")
    a("Holm passes in the secondary families: " + (", ".join(f"{pid} ({heads[f]})" for f, pid in passes) if passes else "none") + ".")
    pool = holm({f"{f}:{pid}": allf[f]["rows"][pid]["p"] for f in ["primary"] + fams for pid in PREDICTORS})
    low = min(pool, key=lambda k: pool[k]["p_holm"])
    n_pool = sum(v["reject"] for v in pool.values())
    a("")
    a(f"With one Holm correction across all {len(pool)} registered tests (the seven families together): {n_pool} "
      f"pass{'es' if n_pool != 1 else ''}; the smallest adjusted p is {pool[low]['p_holm']:.2f} "
      f"({low.split(':')[1]}, {heads[low.split(':')[0]]}). This pooled view is not registered; it is shown for scale.")
    a("")
    a("## 4. Train replication beside the test period")
    a("")
    a("Signed Spearman r, train (1990-2007 origins, published-sample replication) -> test (2008-2026 origins). "
      "`*` = Holm pass within that family and period.")
    a("")
    allfam = ["primary"] + fams
    rows = []
    for pid in PREDICTORS:
        cells = []
        for f in allfam:
            t, tr = allf[f]["rows"][pid], rep[f]["rows"][pid]
            cells.append(f"{_fmt(tr['r'])}{'*' if tr['reject'] else ''} -> {_fmt(t['r'])}{'*' if t['reject'] else ''}")
        rows.append([pid] + cells)
    L += md_table(["predictor"] + [heads[f] for f in allfam], rows)
    a("")
    a("## 5. Secondary economic analyses (descriptive, never decisive)")
    a("")
    a("### 5.1 The same rule for every candidate, with its own tau_j")
    a("")
    a("Hold SPXL when s_j x_j >= tau_j (the 33.33rd percentile of s_j x_j over 1990-01..2007-06), else T-bills; each "
      "against BH and its own MIX_w. Decisive only for a candidate that passed the primary test.")
    a("")
    rows = []
    for pid in CAND_IDS + ["COMP"]:
        r_ = econ["rules"][pid]
        s = r_["strategies"]
        rows.append([pid, SHORTS[pid], _fmt(r_["threshold"], "{:.4g}"), _fmt(s["rule"]["time_in_spxl"], "{:.0%}"),
                     s["rule"]["switches_or_rebalances"], _pct(s["rule"]["cagr"]), _pct(s["MIX_w"]["cagr"]),
                     _fmt(s["rule"]["sharpe"]), _fmt(s["MIX_w"]["sharpe"]), _fmt(s["rule"]["max_drawdown"], "{:.0%}"),
                     " ".join(f"{k}{'+' if r_['E'][k] else '-'}" for k in ("E1", "E2", "E3", "E4")),
                     "yes" if r_["pass"] else "no", r_["missing_signal"]])
    L += md_table(["rule", "signal", "threshold", "in SPXL", "switches", "CAGR", "MIX_w CAGR", "Sharpe", "MIX_w Sharpe",
                   "max DD", "E1-E4", "all four", "missing"], rows)
    a("")
    a(f"Buy and hold SPXL over the same sessions: CAGR {_pct(st['BH']['cagr'])}, Sharpe {_fmt(st['BH']['sharpe'])}, "
      f"max drawdown {_fmt(st['BH']['max_drawdown'], '{:.0%}')}.")
    never = [pid for pid in CAND_IDS if econ["rules"][pid]["strategies"]["rule"]["time_in_spxl"] == 1.0]
    if never:
        a("")
        a("Rules that never left SPXL, because the predictor never crossed its 1990-2007 cut-off in 2008-2026 (so the "
          "rule equals buy-and-hold and fails E1 by construction): " + ", ".join(f"{pid} {SHORTS[pid]}" for pid in never)
          + ".")
    a("")
    a("### 5.2 Composite rule: other switching costs and the median threshold")
    a("")
    rows = []
    variants = [(f"cost {float(k):.2%}", econ["alt_costs"][k]) for k in econ["alt_costs"]] + \
               [(f"threshold = train median {econ['median_threshold']['threshold']:.4f} (cost 0.10%)", econ["median_threshold"])]
    for name, r_ in variants:
        s = r_["strategies"]
        rows.append([name, _fmt(s["rule"]["time_in_spxl"], "{:.0%}"), _pct(s["rule"]["cagr"]), _pct(s["MIX_w"]["cagr"]),
                     _fmt(s["rule"]["sharpe"]), _fmt(s["MIX_w"]["sharpe"]), _fmt(s["rule"]["max_drawdown"], "{:.0%}"),
                     " ".join(f"{k}{'+' if r_['E'][k] else '-'}" for k in ("E1", "E2", "E3", "E4"))])
    L += md_table(["variant", "in SPXL", "CAGR", "MIX_w CAGR", "Sharpe", "MIX_w Sharpe", "max DD", "E1-E4"], rows)
    a("")
    a("### 5.3 Stationary bootstrap of the composite rule against buy and hold")
    a("")
    bt = econ["stationary_bootstrap"]
    a(f"Monthly holding-period returns of the rule and of BH (costs included) with the monthly T-bill returns, "
      f"resampled as aligned triples (Politis-Romano, circular, mean block {bt['mean_block_months']} months, "
      f"{bt['n_boot']} resamples, seed {bt['seed']}); monthly Sharpe x sqrt(12), CAGR over {bt['months']} months.")
    a("")
    L += md_table(["difference (rule - BH)", "estimate", "90% interval"],
                  [["Sharpe (monthly, annualised)", _fmt(bt["sharpe_diff"]),
                    f"{_fmt(bt['sharpe_diff_ci90'][0])} to {_fmt(bt['sharpe_diff_ci90'][1])}"],
                   ["CAGR", _pct(bt["cagr_diff"]), f"{_pct(bt['cagr_diff_ci90'][0])} to {_pct(bt['cagr_diff_ci90'][1])}"]])
    a("")
    k_out = econ["holdings"] - main["strategies"]["rule"]["holdings_in_spxl"]
    a(f"The rule differs from buy-and-hold only in its {k_out} T-bill months (of {econ['holdings']}), so these "
      "intervals rest on those months.")
    a("")
    a("### 5.4 Volatility-managed exposure (not a clean test: uses the VIX level)")
    a("")
    vm = econ["vol_managed"]
    a(f"SPXL weight min(1, c / IVAR_t) with c = {vm['c']:.6f} (VIX {100 * math.sqrt(vm['c']):.2f}), rest T-bills, "
      f"monthly, 0.10% x |weight traded|; against BH and a constant w_bar = {vm['w_bar']:.4f}. Average target weight "
      f"{vm['mean_target_weight']:.2f}; missing IVAR: {vm['missing_ivar']}.")
    a("")
    rows = []
    for k, name in (("vol_managed", "vol-managed"), ("constant_w_bar", "constant w_bar"), ("BH", "buy and hold SPXL")):
        s = vm["strategies"][k]
        rows.append([name, _pct(s["cagr"]), _fmt(s["sharpe"]), _fmt(s["max_drawdown"], "{:.1%}"), _fmt(s["vol"], "{:.0%}"),
                     _fmt(s["sharpe_halves"][0]), _fmt(s["sharpe_halves"][1]), _fmt(s["turnover"], "{:.2f}")])
    L += md_table(["strategy", "CAGR", "Sharpe", "max drawdown", "vol", "Sharpe 1st half", "Sharpe 2nd half", "turnover"], rows)
    a("")
    a("## 6. Robustness of the primary results (reported; labels apply to primary passes)")
    a("")
    a("Halves: origins 2008-01..2016-12 and 2017-01..2026-02. Episodes left out in turn: 2008-01..2009-06 (financial "
      "crisis), 2019-09..2020-06 (2020 crash), 2021-07..2022-12 (inflation and rate shock). Block bootstrap: 90% "
      "interval for r from blocks of 12 origins (10,000 resamples, seed 20260927). Partial r: controls ln VIX(t-) and "
      "DGS3MO(t-), p with n_eff - 5. Labels: 'episode-dependent' if p > 0.10 with any one episode left out; 'one half "
      "only' if r is not positive in both halves. The block-bootstrap intervals are descriptive: they are not "
      "corrected for testing 13 predictors and are not a significance test (the test is section 1).")
    a("")
    rows = []
    for pid in PREDICTORS:
        q = rob["per_predictor"][pid]
        h1, h2 = q["halves"]
        eps = q["episodes"]
        bb, ps = q["block_bootstrap"], q["partial_spearman"]
        rows.append([pid, _fmt(pr["rows"][pid]["r"]), f"{_fmt(h1['r'])} / {_fmt(h2['r'])}",
                     " / ".join(f"{_fmt(x['r'])} ({_fmt(x['p'], '{:.2f}')})" for x in eps),
                     f"{_fmt(bb['ci90'][0])} to {_fmt(bb['ci90'][1])}", f"{_fmt(ps['r'])} ({_fmt(ps['p'], '{:.2f}')})",
                     ", ".join(q["labels"]) or "-"])
    L += md_table(["predictor", "r", "r halves", "r (p) without crisis / 2020 / 2021-22",
                   "block bootstrap 90% (unadjusted)",
                   "partial r (p)", "labels (they count only for a primary pass)"], rows)
    a("")
    ec = rob["expanding_composite"]
    a(f"* **Composite with expanding-window standardisation** (1990-01 to t, at least 60 values per member): r = "
      f"{_fmt(ec['r'])} (n {ec['n']}, n_eff {_fmt(ec['n_eff'], '{:.1f}')}, 90% CI {_fmt(ec['ci90'][0])} to "
      f"{_fmt(ec['ci90'][1])}, one-sided p {_fmt(ec['p'], '{:.3f}')}), against r = {_fmt(pr['rows']['COMP']['r'])} with "
      "the train-fixed standardisation.")
    for k, sp in rob["synthetic_spread"].items():
        a(f"* **Synthetic fund spread {float(k):.2%}** (registered 0.75%; changes the {sp['origins_changed']} origins "
          f"{sp['changed_from']}..{sp['changed_to']} whose window starts before 2009-01-02): "
          + ", ".join(f"{pid} {_fmt(sp['rows'][pid]['r'])} ({_fmt(sp['rows'][pid]['p'], '{:.2f}')})" for pid in PREDICTORS)
          + "; Holm passes: " + (", ".join(pid for pid in PREDICTORS if sp["rows"][pid]["reject"]) or "none") + ".")
    a("")
    a("## 7. Sanity checks (Test phase)")
    a("")
    pc = res["preconditions"]
    tr_ = san["target_reproduction_all_origins"]
    lg = san["legacy_reproduction"]
    sf = san["synthetic_fund"]
    cal_ = san["calendar_test_origins"]
    cov = san["coverage"]
    flagged = [pid for pid in PREDICTORS if cov[pid]["flag_below_90pct"]]
    rows = [
        ["plan files match `PREREG_HASH.txt`; unseal token = SHA-256 of `PREREGISTRATION.md`", "pass"],
        ["pinned inputs (3 files) match their registered SHA-256", "pass"],
        [f"`predictor_inputs.pkl` = `{pc['predictor_inputs_sha256'][:12]}...` (DEVIATIONS.md D2, TRAIN_REPORT.md)", "pass"],
        [f"`frozen_spec.json` = `{pc['frozen_spec_sha256'][:12]}...` (FROZEN_HASH.txt), byte-identical to "
         "`train_params.json`, whose hash TRAIN_REPORT.md records", "pass"],
        [f"`panel_train.pkl` = the hash in the frozen spec; `panel_meta.json` built from this plan and these inputs", "pass"],
        ["the frozen spec's registered choices are the ones the code implements", _yn(pc["spec_matches_code"]["pass"])],
        [f"section 8 quantities recomputed from `panel_train.pkl` equal the frozen ones (max abs diff "
         f"{pc['spec_quantities']['max_abs_diff']:.1e})", _yn(pc["spec_quantities"]["pass"])],
        [f"`panel_test.pkl` sealed (targets all missing), on the calendar, flags and missing counts as built "
         f"({pc['test_panel']['rows']} month-ends {pc['test_panel']['first']}..{pc['test_panel']['last']})",
         _yn(pc["test_panel"]["pass"])],
        [f"every stored predictor and component equals a fresh rebuild from the pinned inputs before unsealing "
         f"({pc['rebuild']['cells']} cells, {pc['rebuild']['n_mismatches']} mismatches)", _yn(pc["rebuild"]["pass"])],
        ["train-origin targets unchanged by unsealing (sealed frame = prefix of the full frame)",
         _yn(san["train_targets_unchanged_by_unsealing"]["pass"])],
        ["target reproduction at all origins: Y, S, D20 = pinned CSV excess_h, sp_h - tbill_h, real_dd20_63 (tol 1e-9; "
         + ", ".join(f"{k} {tr_[k]['compared']}" for k in TARGETS) + " origins; max abs diff "
         + _fmt(max((tr_[k]['max_abs_diff'] or 0.0) for k in TARGETS), "{:.1e}") + ")", _yn(tr_["pass"])],
        [f"legacy reproduction: Spearman(score, Y(t,126)) over {lg['n']} month-ends {lg['first']}..{lg['last']} = "
         f"{_fmt(lg['r'], '{:+.4f}')} (registered -0.03 +/- 0.01)", _yn(lg["pass"])],
        [f"synthetic fund vs SPXL daily correlation from 2009-01-02 = {_fmt(sf['corr'], '{:.4f}')} over {sf['days']} "
         f"sessions (>= 0.99); synthetic growth {_pct(sf['syn_growth'])} vs SPXL {_pct(sf['spxl_growth'])} a year", _yn(sf["pass"])],
        ["test origins per horizon (first, last, count, n_eff) equal the plan's calendar figures: "
         + "; ".join(f"h = {h[1:]}: {', '.join(str(x) for x in cal_[h]['got'])}" for h in ("h126", "h63", "h21")),
         _yn(cal_["pass"])],
        ["coverage at test origins (h = 126) at least 90% for every predictor"
         + (f" (flagged: {', '.join(flagged)})" if flagged else ""), _yn(not flagged)],
        ["standardisation: mu_j and sd_j from the frozen spec, no sd_j is zero", _yn(san["no_zero_sd"])],
        ["train replication recomputed = `output/research/train/train_stats.json`",
         _yn(san["train_replication_matches_train_stats"])],
    ]
    L += md_table(["check", "result"], rows)
    a("")
    a("Coverage (non-missing / test origins): " + "; ".join(
        f"{pid} {cov[pid]['test_h126']['non_missing']}/{cov[pid]['test_h126']['n']}" for pid in PREDICTORS)
      + " at h = 126. Composite members available at the 224 test-side month-ends: "
      + ", ".join(f"{k}: {n}" for k, n in san["composite_members_test_side"].items()) + ".")
    a("")
    a("## 8. Test-period predictor values (descriptive, after unsealing)")
    a("")
    rows = []
    for pid in PREDICTORS:
        q = res["descriptives_test"][pid]
        rows.append([pid, SHORTS[pid], q["n"], _fmt(q["mean"], "{:.4g}"), _fmt(q["sd"], "{:.4g}"), _fmt(q["min"], "{:.4g}"),
                     _fmt(q["median"], "{:.4g}"), _fmt(q["max"], "{:.4g}"),
                     _fmt(san["standardisation"][pid]["mu"], "{:.4g}") if pid != "COMP" else "-",
                     _fmt(q["favourable_share_econ_origins"], "{:.0%}")])
    L += md_table(["id", "signal", "n", "mean", "sd", "min", "median", "max", "train mean (mu_j)",
                   "favourable (>= threshold) at the 223 economic origins"], rows)
    a("")
    a("## 9. What was seen, deviations, and the look log")
    a("")
    a("* Before this run: the exposure ledger (PREREGISTRATION.md section 2), the Build and Train notes "
      "(DEVIATIONS.md N11-N18) and the train-period statistics in `research/train_results.md`. The Test-phase "
      "implementation (DEVIATIONS.md, Test phase notes) was written and unit-tested on synthetic data before "
      "unsealing; before the run no test-period outcome was computed and no test-period predictor value was "
      "summarised (the rebuild check compares values for equality only).")
    a("* Missing from the exposure ledger (found by the look-ahead review after the run; DEVIATIONS.md N30): before "
      "the plan was frozen (2026-09-27 16:12Z), a working script also computed 6-month SPXL excess returns by "
      "hurdle tercile on the SPXL-only months since 2009. The ledger lists the full-sample hurdle split and the "
      "SPXL-only drawdown levels, not this split. It bears on C11 only, already labelled 'not a clean out-of-sample "
      "test'.")
    a("* Deviations from the plan: DEVIATIONS.md D1-D5 (all made before unsealing, for data reasons) and the "
      "Test-phase notes there. Changes after unsealing (DEVIATIONS.md N26-N30) are presentation, process "
      "safeguards and exploratory analysis only; none changes a registered number or the verdict.")
    code = res.get("code_sha256")
    a("* Code: " + ("; ".join(f"`{k}` {v[:12]}..." for k, v in code.items()) + " (copies in `output/research/test/code/`)."
                    if code else "this run did not record the SHA-256 of its code (added afterwards, DEVIATIONS.md "
                    "N28; see N29 for the check that the current code reproduces every number)."))
    a("* This run unsealed 2008-2026. From now on the test period is spent: a new idea can be tested only on data "
      "that do not exist yet (PREREGISTRATION.md, Part 1).")
    a("")
    log = res.get("look_log", [])
    L += md_table(["UTC", "phase", "command", "git"],
                  [[r["utc"], r["phase"], f"`{r['command'][:60] + '...' if len(r['command']) > 60 else r['command']}`",
                    r["git_describe"]] for r in log])
    a("")
    a(f"Outputs: `output/research/test_results.json` (every number above), `output/research/test/` (tests.csv, "
      "economic_daily.csv, economic_holdings.csv, panel_unsealed.pkl, economic_test.png).")
    a("")
    return "\n".join(L)


def print_test_summary(res: Dict) -> None:
    v, pr, main = res["verdict"], res["primary"], res["economic"]["rules"]["COMP"]
    print(f"\nVERDICT: {v['headline']}")
    print(f"\nprimary test (Y 6m, test origins {pr['origins']}): signed Spearman r [one-sided p, Holm p]")
    for pid in PREDICTORS:
        t = pr["rows"][pid]
        print(f"  {pid:5} {SHORTS[pid]:7} r {_fmt(t['r']):>6}  n_eff {t['n_eff']:.1f}  p {_fmt(t['p'], '{:.3f}')}  "
              f"Holm p {_fmt(t['p_holm'], '{:.3f}')}  {'PASS' if t['reject'] else 'fail'}")
    s = main["strategies"]
    print(f"\neconomic test (composite): {'PASS' if main['pass'] else 'FAIL'} "
          + " ".join(f"{k}={'+' if main['E'][k] else '-'}" for k in ("E1", "E2", "E3", "E4")))
    for k in ("rule", "BH", "MIX_w"):
        print(f"  {k:6} CAGR {_pct(s[k]['cagr']):>7}  Sharpe {_fmt(s[k]['sharpe']):>6}  maxDD {_fmt(s[k]['max_drawdown'], '{:.1%}'):>6}")
    san = res["sanity"]
    print("\nsanity: target reproduction " + _yn(san["target_reproduction_all_origins"]["pass"])
          + f", legacy r {_fmt(san['legacy_reproduction']['r'], '{:+.4f}')} " + _yn(san["legacy_reproduction"]["pass"])
          + f", synthetic corr {_fmt(san['synthetic_fund']['corr'], '{:.4f}')} " + _yn(san["synthetic_fund"]["pass"])
          + ", calendar " + _yn(san["calendar_test_origins"]["pass"]))


def cmd_test(args, paths: Optional[Phase3Paths] = None, rebuild=None) -> int:
    """Phase 3. The token is checked before anything is read; every precondition is checked before
    unsealing; the run happens once (output/research/test/ marks it)."""
    t0 = time.time()
    paths = Phase3Paths.from_globals() if paths is None else paths
    token = (args.unseal or "").strip().lower()
    if token != sha256_file(paths.prereg_md):
        print("REFUSED: --unseal is not the SHA-256 of research/PREREGISTRATION.md; nothing was read.", file=sys.stderr)
        return 2
    if paths.results_json.exists() or (paths.test_dir / COMPLETED).exists():
        raise SystemExit("STOP: the Test phase has already run (it runs once). `report` re-renders the results.")
    if paths.test_dir.exists() and not getattr(args, "after_error", False):
        raise SystemExit("STOP: output/research/test/ exists from an unfinished run. Rerunning needs --after-error "
                         "and the reason recorded in research/DEVIATIONS.md (bug fixes only, original results reported).")
    # 1. preconditions, before unsealing
    ck, spec, meta = phase3_preconditions(paths)
    inputs = pd.read_pickle(paths.backtest / "rating_inputs.pkl")
    cal = session_calendar(inputs)
    p_train = pd.read_pickle(paths.panel_train)
    check_train_panel(p_train)
    p_test = pd.read_pickle(paths.panel_test)
    ck["test_panel"] = check_test_panel(p_train, p_test, cal, meta)
    ck["spec_quantities"] = check_spec_quantities(spec, p_train)
    legacy = read_legacy_checked(paths.backtest / "rating_backtest.csv", paths.pinned["rating_backtest.csv"])
    print("preconditions pass; rebuilding every predictor from the pinned inputs (equality check only)...", flush=True)
    rebuilt = (rebuild_predictors if rebuild is None else rebuild)(inputs, pd.read_pickle(paths.predictor_inputs), legacy, cal)
    ck["rebuild"] = compare_rebuilt(pd.concat([p_train, p_test]), rebuilt)
    bad = [k for k in ("test_panel", "spec_quantities", "rebuild") if not ck[k]["pass"]]
    if bad:
        raise SystemExit(f"STOP before unsealing: {', '.join(bad)} failed: " + json.dumps(jsonable({k: ck[k] for k in bad}))[:2000])
    # 2. the run starts: mark it, then unseal
    paths.test_dir.mkdir(parents=True, exist_ok=True)
    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_lf(paths.test_dir / STARTED, f"{run_utc}\n")
    vk = paths.verify_kwargs()
    d = outcome_frame(inputs, unseal_token=token, **vk)
    d_spread = {s: outcome_frame(inputs, unseal_token=token, cfg=dataclasses.replace(Config(), swap_spread=s), **vk)
                for s in SPREADS}
    csv_out = read_csv_outcomes(paths.backtest / "rating_backtest.csv", token, paths.pinned["rating_backtest.csv"], **vk)
    train_stats = json.loads(paths.train_stats.read_text(encoding="utf-8")) if paths.train_stats.exists() else None
    print("unsealed; computing the registered tests...", flush=True)
    res, frames = phase3_compute(p_train, p_test, d, cal, spec, csv_out, legacy["score"], d_spread, train_stats)
    res = {"phase": "test", "run_utc": run_utc, "spec_sha256": ck["frozen_spec_sha256"], "frozen_utc": ck["frozen_utc"],
           "git_describe": git_describe(), "preconditions": ck, **res}
    res["look_log"] = look_log_summary(paths.look_log)
    # the code that produced the numbers (added after the 2026-09-27 run, DEVIATIONS.md N28): hashes in
    # the JSON and a byte copy of each file beside the outputs
    res["code_sha256"] = code_provenance(copy_to=paths.test_dir / "code")
    write_phase3_outputs(res, frames, paths)
    # 3. the document, rendered from the JSON as written (so `report` reproduces it byte for byte)
    text = render_test_results(json.loads(paths.results_json.read_text(encoding="utf-8")))
    write_lf(paths.results_md, text)
    write_lf(paths.results_md_plan, text)
    write_lf(paths.test_dir / COMPLETED, f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}\n")
    print_test_summary(res)
    print(f"\nwrote {paths.results_json}, {paths.results_md}, {paths.results_md_plan}, {paths.test_dir} "
          f"({time.time() - t0:.0f}s)")
    return 0


# ---------------------------------------------------------------------------------------
# After the Test phase: exploratory analyses for research/RESULTS.md (never a result)
# ---------------------------------------------------------------------------------------
# Everything below runs after unsealing, from the saved Test-phase outputs (panel_unsealed.pkl,
# economic_daily.csv, test_results.json, the frozen spec) and, for two data checks, from
# predictor_inputs.pkl and the pinned rating_inputs.pkl. It answers the independent reviews of the run
# (research/DEVIATIONS.md N27-N30). Nothing here is a registered test or changes the verdict, and none
# of it may be used to choose a signal, sign, threshold or rule for 2008-2026 (the period is spent).
MC_BLOCKS = (63, 126, 252, 504)               # mean block lengths (sessions) of the daily stationary bootstrap
MC_REPS = 10000
MC_CHUNK = 250
MC_FAMILIES = (("primary", "Y126", 126), ("F2", "Y21", 21), ("F3", "Y63", 63))
FEEDBACK_REPS = 4000
FEEDBACK_BURN = 200
CRISIS = EPISODES[0][1:]                       # ("2008-01", "2009-06")
REAL_SPXL_ORIGIN = pd.Timestamp("2008-12-31")  # the first holding on real SPXL (its sessions start 2009-01-02)
HIGH_COST = 0.005
CONST_WEIGHTS = (0.55, 0.60)
BOOT_RULES = ("C06", "C01", "C08", "VM", "COMP")
EXPLORATORY_NAME = "exploratory.json"
MANIFEST_NAME = "OUTPUT_SHA256.txt"


@dataclasses.dataclass
class PostHoc:
    """The saved Test-phase outputs, on local session indices (0 = 2008-02-01)."""
    res: Dict
    spec: Dict
    p: pd.DataFrame                 # panel_unsealed.pkl: predictors, targets and COMP at every month-end
    dates: pd.DatetimeIndex         # the economic sessions 2008-02-01..2026-08-31
    fund: np.ndarray                # the realised fund's daily returns on them, as the test used them
    bill: np.ndarray
    off: int                        # session position of dates[0]: local index = position - off
    origins: pd.DatetimeIndex       # the 223 economic origins
    a: np.ndarray                   # holding i earns local sessions a_i + 1..b_i
    b: np.ndarray


def load_posthoc(paths: Phase3Paths) -> PostHoc:
    res = json.loads(paths.results_json.read_text(encoding="utf-8"))
    spec = json.loads(paths.frozen_spec.read_text(encoding="utf-8"))
    p = pd.read_pickle(paths.test_dir / "panel_unsealed.pkl")
    daily = pd.read_csv(paths.test_dir / "economic_daily.csv", index_col=0, parse_dates=True)
    idx = p.index
    loc = np.flatnonzero((idx >= ECON_ORIGINS[0]) & (idx <= ECON_ORIGINS[1]))
    pos = p["pos"].astype(int).to_numpy()
    a_abs, b_abs = pos[loc], pos[loc + 1]
    off = int(a_abs[0]) + 1
    a, b = a_abs - off, b_abs - off
    if len(loc) != ECON_HOLDINGS or not (b[:-1] == a[1:]).all() or len(daily) != int(b[-1]) + 1 \
            or daily.index[0] != ECON_SESSIONS[0] or daily.index[-1] != ECON_SESSIONS[1]:
        raise SystemExit("STOP: economic_daily.csv and panel_unsealed.pkl do not tile the economic sessions")
    return PostHoc(res=res, spec=spec, p=p, dates=pd.DatetimeIndex(daily.index),
                   fund=daily["fund_ret"].to_numpy(float), bill=daily["tbill_ret"].to_numpy(float), off=off,
                   origins=idx[loc], a=a, b=b)


def _cumlog(r: np.ndarray) -> np.ndarray:
    """[0, cumsum(log1p(r))] along the last axis."""
    c = np.cumsum(np.log1p(r), axis=-1)
    return np.concatenate([np.zeros(c.shape[:-1] + (1,)), c], axis=-1)


def window_excess(lf: np.ndarray, lb: np.ndarray, i0: np.ndarray, h: int) -> np.ndarray:
    """The fund's return minus the T-bill return over local sessions i0 + 1..i0 + h, from cumulative logs."""
    s, e = np.asarray(i0) + 1, np.asarray(i0) + h + 1
    return np.expm1(lf[..., e] - lf[..., s]) - np.expm1(lb[..., e] - lb[..., s])


def _zrank(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Average ranks standardised to mean 0 and (population) sd 1 along ``axis``: the mean of the
    product of two such vectors is Spearman's r."""
    from scipy.stats import rankdata
    r = rankdata(v, axis=axis)
    return (r - r.mean(axis=axis, keepdims=True)) / r.std(axis=axis, keepdims=True)


def _signed_all(p: pd.DataFrame) -> Dict[str, pd.Series]:
    return {pid: (p["COMP"].astype(float) if pid == "COMP" else signed(p, pid)) for pid in PREDICTORS}


def stationary_indices(rng: np.random.Generator, n_rows: int, n: int, mean_block: float) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (circular), one resample of 0..n-1 per row."""
    u = rng.random((n_rows, n)) < 1.0 / mean_block
    u[:, 0] = True
    s = rng.integers(0, n, size=(n_rows, n))
    ar = np.arange(n)
    start = np.maximum.accumulate(np.where(u, ar, 0), axis=1)
    return (np.take_along_axis(s, start, axis=1) + (ar - start)) % n


def _family_rows(res: Dict, fam: str) -> Dict[str, Dict]:
    return (res["primary"] if fam == "primary" else res["families"][fam])["rows"]


def mc_calibration(ph: PostHoc, blocks: Sequence[int] = MC_BLOCKS, reps: int = MC_REPS, seed: int = SEED) -> Dict:
    """A persistence-aware null for the registered statistic. Each predictor's actual test-period path
    is held fixed; the outcomes Y(t,h) are rebuilt from the daily (fund, T-bill) returns of
    2008-02-01..2026-08-31 resampled by a stationary bootstrap (so they are independent of the
    predictor but keep their own overlap and volatility clustering). Reports, per family and mean block
    length: the Monte Carlo p of the observed r, the null sd of r (and the n_eff it implies), the size
    of the registered test (share of null draws it rejects at 5% and at the first Holm step) and the r
    a predictor would have needed at the first Holm step. Ignores feedback from returns to later
    predictor values (see feedback_null)."""
    from scipy.stats import norm
    n_sess = len(ph.fund)
    xs = _signed_all(ph.p)
    pos_all = ph.p["pos"].astype(int).to_numpy()
    lf0, lb0 = _cumlog(ph.fund), _cumlog(ph.bill)
    k_tests = len(PREDICTORS)
    fams: Dict[str, Dict] = {}
    for fam, target, h in MC_FAMILIES:
        sel = ph.p[f"test_h{h}"].astype(bool).to_numpy()
        # the plan's calendar puts every test window inside the economic sessions; keep only windows that
        # are (all of them on the real data; a synthetic calendar can differ)
        inside = (pos_all - ph.off >= -1) & (pos_all - ph.off + h + 1 <= n_sess)
        outside = int((sel & ~inside).sum())
        sel = sel & inside
        if not sel.any():
            raise SystemExit(f"STOP: no {target} window falls inside economic_daily.csv")
        pos = pos_all[sel]
        i0 = pos - ph.off
        y_reg = ph.p[target].astype(float).to_numpy()[sel]
        y_re = window_excess(lf0, lb0, i0, h)
        rows_reg = _family_rows(ph.res, fam)
        preds = {}
        for pid in PREDICTORS:
            x = xs[pid].to_numpy(float)[sel]
            m = np.isfinite(x) & np.isfinite(y_reg)
            zx = _zrank(x[m])
            preds[pid] = {"m": m, "zx": zx, "r": float(np.mean(zx * _zrank(y_re[m]))),
                          "n_eff": float(effective_n(pos[m], h)), "r_reg": rows_reg[pid]["r"], "p_reg": rows_reg[pid]["p"]}
        fams[fam] = {"target": target, "h": h, "i0": i0, "preds": preds, "outside": outside,
                     "rebuild_max_abs_diff": float(np.nanmax(np.abs(y_re - y_reg))),
                     "r_max_abs_diff": (float(max(abs(v["r"] - v["r_reg"]) for v in preds.values()))
                                        if outside == 0 else None)}
    out: Dict[str, object] = {"blocks": list(blocks), "reps": reps, "seed": seed, "families": {}}
    for fam, F in fams.items():
        out["families"][fam] = {"target": F["target"], "h": F["h"], "origins": int(len(F["i0"])),
                                "origins_outside_daily": F["outside"],
                                "rebuild_max_abs_diff": F["rebuild_max_abs_diff"],
                                "r_max_abs_diff": F["r_max_abs_diff"], "per_block": {}}
    for mb in blocks:
        rng = np.random.default_rng([seed, int(mb)])
        draws = {fam: {pid: np.empty(reps) for pid in PREDICTORS} for fam in fams}
        for s0 in range(0, reps, MC_CHUNK):
            nb = min(MC_CHUNK, reps - s0)
            ix = stationary_indices(rng, nb, n_sess, mb)
            lf, lb = _cumlog(ph.fund[ix]), _cumlog(ph.bill[ix])
            for fam, F in fams.items():
                y = window_excess(lf, lb, F["i0"], F["h"])
                cache: Dict[bytes, np.ndarray] = {}
                for pid, P in F["preds"].items():
                    key = P["m"].tobytes()
                    if key not in cache:
                        cache[key] = _zrank(y[:, P["m"]], axis=1)
                    draws[fam][pid][s0:s0 + nb] = cache[key] @ P["zx"] / len(P["zx"])
        for fam, F in fams.items():
            rows, pm = {}, {}
            for pid, P in F["preds"].items():
                r = draws[fam][pid]
                rc = np.clip(r, -0.999999, 0.999999)
                z = np.arctanh(rc) * math.sqrt(max(P["n_eff"] - 3.0, 0.0)) / np.sqrt(1.0 + rc * rc / 2.0)
                p_reg_null = norm.sf(z)
                sd = float(r.std(ddof=1))
                pm[pid] = max(float(np.mean(r >= P["r"] - 1e-12)), 1.0 / reps)
                rows[pid] = {"r": P["r"], "p_registered": P["p_reg"], "n_eff_registered": P["n_eff"], "p_mc": pm[pid],
                             "null_mean": float(r.mean()), "null_sd": sd,
                             "implied_n_eff": float(1.0 / sd ** 2 + 3.0) if sd > 0 else None,
                             "size_at_5pct": float(np.mean(p_reg_null <= ALPHA)),
                             "size_at_holm1": float(np.mean(p_reg_null <= ALPHA / k_tests)),
                             "r_needed_holm1": float(np.quantile(r, 1.0 - ALPHA / k_tests))}
            adj = holm(pm)
            for pid in rows:
                rows[pid].update({"p_mc_holm": adj[pid]["p_holm"], "mc_holm_pass": adj[pid]["reject"]})
            out["families"][fam]["per_block"][str(mb)] = {
                "rows": rows, "holm_passes": [pid for pid in PREDICTORS if adj[pid]["reject"]]}
    return out


def feedback_null(ph: PostHoc, reps: int = FEEDBACK_REPS, seed: int = SEED) -> Dict:
    """Stambaugh-type feedback. How strongly the signed predictor's change over the outcome window moves
    with the outcome (Spearman of Y(t,126) with x(t+6) - x(t)), and a parametric null that includes
    it: monthly x_t = phi x_(t-1) + v_t, monthly return u_t with corr(u_t, v_t) = rho (phi and rho
    estimated on the test-side month-ends), y_t = u_(t+1) + ... + u_(t+6); Spearman of x_t with y_t
    over the same number of origins. The null mean of r is the bias; p is the share of null draws at
    or above the observed r. Gaussian and approximate: it shows the direction and size of the bias."""
    from spxlcast.evaluation import spearman
    p = ph.p
    xs = _signed_all(p)
    test = np.asarray(p.index >= TEST_START)
    sel126 = p["test_h126"].astype(bool).to_numpy()
    sel21 = p["test_h21"].astype(bool).to_numpy()
    y126, y21 = p["Y126"].astype(float).to_numpy(), p["Y21"].astype(float).to_numpy()
    n_months = int(test.sum())
    n_orig = n_months - 6
    rng = np.random.default_rng([seed, 6])
    out = {}
    for pid in PREDICTORS:
        x = xs[pid].to_numpy(float)
        dx6 = np.full(len(x), np.nan)
        dx6[:-6] = x[6:] - x[:-6]
        dx1 = np.full(len(x), np.nan)
        dx1[:-1] = x[1:] - x[:-1]
        m6 = sel126 & np.isfinite(dx6) & np.isfinite(y126) & np.isfinite(x)
        m1 = sel21 & np.isfinite(dx1) & np.isfinite(y21)
        xt = x[test]

        def lag_corr(k: int) -> float:
            u, v = xt[k:], xt[:-k]
            ok = np.isfinite(u) & np.isfinite(v)
            return float(np.corrcoef(u[ok], v[ok])[0, 1])
        phi1, phi6 = lag_corr(1), lag_corr(6)
        rho1 = float(np.corrcoef(y21[m1], dx1[m1])[0, 1])
        e = rng.standard_normal((2, reps, n_months + FEEDBACK_BURN))
        u = e[0]
        v = rho1 * e[0] + math.sqrt(max(1.0 - rho1 * rho1, 0.0)) * e[1]
        xsim = np.zeros_like(u)
        for t in range(1, u.shape[1]):
            xsim[:, t] = phi1 * xsim[:, t - 1] + v[:, t]
        xsim, u = xsim[:, FEEDBACK_BURN:], u[:, FEEDBACK_BURN:]
        cs = np.concatenate([np.zeros((reps, 1)), np.cumsum(u, axis=1)], axis=1)
        ysim = cs[:, 7:n_orig + 7] - cs[:, 1:n_orig + 1]
        r_null = np.mean(_zrank(xsim[:, :n_orig], axis=1) * _zrank(ysim, axis=1), axis=1)
        r_obs = ph.res["primary"]["rows"][pid]["r"]
        out[pid] = {"rho_y126_dx6": float(spearman(y126[m6], dx6[m6])), "phi1": phi1, "phi6": phi6, "rho1": rho1,
                    "r": r_obs, "p_registered": ph.res["primary"]["rows"][pid]["p"],
                    "null_mean_r": float(r_null.mean()), "null_sd_r": float(r_null.std(ddof=1)),
                    "p_feedback_null": max(float(np.mean(r_null >= r_obs)), 1.0 / reps)}
    return {"reps": reps, "months": n_months, "origins": n_orig, "rows": out}


def _test_sample(p: pd.DataFrame, x: pd.Series, target: str, h: int) -> pd.DataFrame:
    sel = p[f"test_h{h}"].astype(bool)
    y = target_values(p, target)
    m = sel & x.notna() & y.notna()
    return pd.DataFrame({"x": x[m].astype(float), "y": y[m].astype(float), "pos": p.loc[m, "pos"].astype(int)})


def _drop_months(idx: pd.DatetimeIndex, spans: Sequence[Tuple[str, str]]) -> np.ndarray:
    per = idx.to_period("M")
    keep = np.ones(len(idx), bool)
    for lo, hi in spans:
        keep &= ~np.asarray((per >= pd.Period(lo, "M")) & (per <= pd.Period(hi, "M")))
    return keep


def _ct(s: pd.DataFrame, keep: np.ndarray, h: int) -> Dict:
    t = corr_test(s["x"].to_numpy()[keep], s["y"].to_numpy()[keep], s["pos"].to_numpy()[keep], h)
    return {"r": t["r"], "p": t["p"], "n": t["n"], "n_eff": t["n_eff"]}


def secondary_checks(ph: PostHoc) -> Dict:
    """The one secondary Holm pass in context: Holm across all 91 registered tests; how close F2 and F6
    are; C01 VRP at 1 and 3 months by half, without the crisis and leaving out one year at a time."""
    from spxlcast.evaluation import spearman
    res, p = ph.res, ph.p
    fams = {"primary": res["primary"], **res["families"]}
    allp = {f"{f}:{pid}": fams[f]["rows"][pid]["p"] for f in fams for pid in PREDICTORS}
    adj = holm(allp)
    best = min(adj, key=lambda k: adj[k]["p_holm"])
    sel = p["test_h21"].astype(bool)
    out: Dict[str, object] = {
        "n_tests": len(allp), "passes_all_91": [k for k, v in adj.items() if v["reject"]],
        "min_adjusted_p": adj[best]["p_holm"], "min_adjusted_p_test": best,
        "passes_within_family": [f"{f}:{pid}" for f in fams for pid in PREDICTORS if fams[f]["rows"][pid]["reject"]],
        "spearman_y21_s21": float(spearman(p.loc[sel, "Y21"].astype(float), p.loc[sel, "S21"].astype(float))),
        "vrp": {}}
    x = signed(p, "C01")
    holm1 = ALPHA / len(PREDICTORS)
    for fam, target, h in (("F6", "S21", 21), ("F2", "Y21", 21), ("F5", "S63", 63), ("F3", "Y63", 63)):
        s = _test_sample(p, x, target, h)
        allk = np.ones(len(s), bool)
        rec = {"family": fam, "all": _ct(s, allk, h),
               "halves": [{"origins": list(hv), **_ct(s, ~_drop_months(s.index, [hv]), h)} for hv in HALVES],
               "without_crisis": _ct(s, _drop_months(s.index, [CRISIS]), h),
               "train": {"r": res["replication_train"][fam]["rows"]["C01"]["r"],
                         "p": res["replication_train"][fam]["rows"]["C01"]["p"]}}
        loyo = []
        for yr in sorted(set(s.index.year)):
            loyo.append({"year": int(yr), **_ct(s, np.asarray(s.index.year != yr), h)})
        rec["leave_one_year_out"] = loyo
        rec["years_whose_removal_lifts_p_above_holm1"] = [d["year"] for d in loyo if d["p"] > holm1]
        out["vrp"][target] = rec
    return out


def episode_checks(ph: PostHoc) -> Dict:
    """The primary correlations with the three episodes of the robustness check left out together (the
    registered check leaves them out one at a time)."""
    xs = _signed_all(ph.p)
    sets = {"crisis and 2020": [e[1:] for e in EPISODES[:2]], "all three": [e[1:] for e in EPISODES]}
    out = {}
    for pid in PREDICTORS:
        s = _test_sample(ph.p, xs[pid], "Y126", 126)
        rec = {"all": _ct(s, np.ones(len(s), bool), 126)}
        for name, spans in sets.items():
            rec[name] = _ct(s, _drop_months(s.index, spans), 126)
        out[pid] = rec
    return {"sets": {k: [list(v) for v in spans] for k, spans in sets.items()}, "rows": out}


def rule_weight_sets(ph: PostHoc) -> Dict[str, np.ndarray]:
    """The registered switching rules (COMP with tau, C01-C12 with tau_j) and the vol-managed weights,
    at the 223 economic origins, exactly as economic_suite builds them."""
    ts = ph.spec["train_set"]
    w = {"COMP": rule_weights(ph.p["COMP"].reindex(ph.origins).to_numpy(float), ts["composite"]["tau"])[0]}
    for c in CAND_IDS:
        w[c] = rule_weights(signed(ph.p, c).reindex(ph.origins).to_numpy(float), ts["tau_j"]["values"][c])[0]
    c_vm, prev, vm = ts["vol_managed"]["c"], 0.0, []
    for v in ph.p["C10"].reindex(ph.origins).to_numpy(float):
        if np.isfinite(v) and v > 0:
            prev = min(1.0, c_vm / v)
        vm.append(prev)
    w["VM"] = np.array(vm)
    return w


def _sim(ph: PostHoc, w: Sequence[float], cost: float, i0: int = 0) -> Tuple[Dict, Dict]:
    """simulate_weights and perf over holdings i0.. (i0 = 0: the registered 223 holdings)."""
    a, b = ph.a[i0:], ph.b[i0:]
    s = simulate_weights(np.asarray(w, float)[i0:], a, b, ph.fund, ph.bill, cost)
    lo, hi = int(a[0]) + 1, int(b[-1]) + 1
    return s, perf(s["r"], ph.bill[lo:hi], ph.dates[lo:hi])


def _pf(d: Dict) -> Dict:
    return {k: d[k] for k in ("cagr", "sharpe", "max_drawdown", "vol")}


def economics(ph: PostHoc) -> Dict:
    """Practical checks of the switching rules (and the vol-managed weights): where the gains came from,
    the real-SPXL period alone, execution one month late, a higher switching cost, a joint placebo for
    picking the best of 12 rules, bootstrap intervals for 2009 on, the vol-managed rule against
    constant weights, and the average outcome in favourable and unfavourable months."""
    W = rule_weight_sets(ph)
    n = len(ph.origins)
    years = len(ph.dates) / 252.0
    ones, zeros = np.ones(n), np.zeros(n)
    i09 = int(np.argmax(np.asarray(ph.origins >= REAL_SPXL_ORIGIN)))
    s_bh, pf_bh = _sim(ph, ones, 0.0)
    s_bh09, pf_bh09 = _sim(ph, ones, 0.0, i09)
    crisis = ~_drop_months(ph.origins, [CRISIS])
    json_cagr = {**{k: ph.res["economic"]["rules"][k]["strategies"]["rule"]["cagr"] for k in ["COMP"] + CAND_IDS},
                 "VM": ph.res["economic"]["vol_managed"]["strategies"]["vol_managed"]["cagr"]}
    rules, check = {}, []
    for k in ["COMP"] + CAND_IDS + ["VM"]:
        w = W[k]
        s, pf = _sim(ph, w, SWITCH_COST)
        check.append(abs(pf["cagr"] - json_cagr[k]))
        gap = np.log1p(s["monthly"]) - np.log1p(s_bh["monthly"])
        order = np.argsort(gap)[::-1]
        share = float(np.mean(w)) if k == "VM" else float(np.mean(w == 1.0))
        share09 = float(np.mean(w[i09:])) if k == "VM" else float(np.mean(w[i09:] == 1.0))
        _, pf09 = _sim(ph, w, SWITCH_COST, i09)
        _, mix09 = _sim(ph, np.full(n, share09), SWITCH_COST, i09)
        _, late = _sim(ph, np.concatenate([w[:1], w[:-1]]), SWITCH_COST)
        _, hc = _sim(ph, w, HIGH_COST)
        rules[k] = {"share_in_spxl": share, "switches": int(np.sum(np.diff(w) != 0)), "full": _pf(pf),
                    "edge_vs_bh": float(np.expm1(gap.sum() / years)),
                    "edge_vs_bh_without_crisis": float(np.expm1(gap[~crisis].sum() / years)),
                    "edge_vs_bh_without_top3": float(np.expm1(gap[order[3:]].sum() / years)),
                    "top3_months": [f"{ph.origins[i]:%Y-%m}" for i in order[:3]],
                    "top3_gap_share": float(gap[order[:3]].sum() / gap.sum()) if gap.sum() > 0 else None,
                    "real_spxl": {**_pf(pf09), "share_in_spxl": share09, "mix_matched": _pf(mix09)},
                    "one_month_late": _pf(late), "cost_0p5pct": _pf(hc)}
    # joint placebo: all 12 candidate rules shifted by the same phase; the best of the 12 each time
    cag = np.empty((n, len(CAND_IDS)))
    for sh in range(n):
        for j, c in enumerate(CAND_IDS):
            cag[sh, j] = _sim(ph, np.roll(W[c], sh), SWITCH_COST)[1]["cagr"]
    best = cag.max(axis=1)
    placebo = {"shifts": n, "actual_best": CAND_IDS[int(np.argmax(cag[0]))], "actual_best_cagr": float(best[0]),
               "share_at_or_above_actual": float(np.mean(best >= best[0] - 1e-15)),
               "median_best": float(np.median(best)), "p90_best": float(np.percentile(best, 90)),
               "own_shift_percentile": {c: float(np.mean(cag[:, j] <= cag[0, j])) for j, c in enumerate(CAND_IDS)}}
    # 2009 on: stationary bootstrap of the monthly holding returns (frozen algorithm and seed)
    bill_m09 = _sim(ph, zeros, 0.0, i09)[0]["monthly"]
    bh_m09 = s_bh09["monthly"]
    boot = {}
    for k in BOOT_RULES:
        w = W[k]
        share09 = rules[k]["real_spxl"]["share_in_spxl"]
        rm = _sim(ph, w, SWITCH_COST, i09)[0]["monthly"]
        mm = _sim(ph, np.full(n, share09), SWITCH_COST, i09)[0]["monthly"]
        idx = stationary_bootstrap_idx(len(rm))
        rec = {}
        for name, other in (("BH", bh_m09), ("mix_matched", mm)):
            dc = monthly_cagr(rm[idx]) - monthly_cagr(other[idx])
            ds = monthly_sharpe(rm[idx], bill_m09[idx]) - monthly_sharpe(other[idx], bill_m09[idx])
            rec[name] = {"cagr_diff": float(monthly_cagr(rm) - monthly_cagr(other)),
                         "cagr_diff_ci90": [float(q) for q in np.percentile(dc[np.isfinite(dc)], [5, 95])],
                         "sharpe_diff": float(monthly_sharpe(rm, bill_m09) - monthly_sharpe(other, bill_m09)),
                         "sharpe_diff_ci90": [float(q) for q in np.percentile(ds[np.isfinite(ds)], [5, 95])]}
        boot[k] = rec
    # the vol-managed rule against constant weights, full period and 2009 on
    wbar = float(ph.spec["train_set"]["vol_managed"]["w_bar"])
    vm = {}
    for label, i0 in (("full", 0), ("real_spxl", i09)):
        rows = {"vol_managed": _pf(_sim(ph, W["VM"], SWITCH_COST, i0)[1]),
                "BH": _pf(_sim(ph, ones, 0.0, i0)[1])}
        for c in CONST_WEIGHTS + (wbar,):
            rows[f"constant {c:.4g}"] = _pf(_sim(ph, np.full(n, c), SWITCH_COST, i0)[1])
        vm[label] = rows
    vm["mean_weight"] = {"full": float(np.mean(W["VM"])), "real_spxl": float(np.mean(W["VM"][i09:]))}
    # average 6-month excess return in favourable and unfavourable months (primary origins)
    ts = ph.spec["train_set"]
    thr = {**ts["tau_j"]["values"], "COMP": ts["composite"]["tau"]}
    fav = {}
    xs = _signed_all(ph.p)
    for pid in PREDICTORS:
        s = _test_sample(ph.p, xs[pid], "Y126", 126)
        f = s["x"] >= thr[pid]
        rec = {}
        for label, keep in (("all", np.ones(len(s), bool)), ("from_2009", np.asarray(s.index >= pd.Timestamp("2009-01-01")))):
            yf, yu = s["y"][keep & f], s["y"][keep & ~f]
            rec[label] = {"favourable_mean": float(yf.mean()) if len(yf) else None, "n_favourable": int(len(yf)),
                          "unfavourable_mean": float(yu.mean()) if len(yu) else None, "n_unfavourable": int(len(yu))}
        fav[pid] = rec
    # where the composite rule lost: its T-bill spells, and the member z-scores in the longest one
    s_comp = _sim(ph, W["COMP"], SWITCH_COST)[0]
    spells, i, wc = [], 0, W["COMP"]
    while i < n:
        if wc[i] == 0.0:
            j = i
            while j < n and wc[j] == 0.0:
                j += 1
            spells.append({"from": f"{ph.origins[i]:%Y-%m}", "to": f"{ph.origins[j - 1]:%Y-%m}", "months": j - i,
                           "sessions": int(ph.b[j - 1] - ph.a[i]),
                           "spxl": float(np.prod(1.0 + s_bh["monthly"][i:j]) - 1.0),
                           "rule": float(np.prod(1.0 + s_comp["monthly"][i:j]) - 1.0), "i": i, "j": j})
            i = j
        else:
            i += 1
    std = {c["id"]: (float(c["mu"]), float(c["sd"])) for c in ph.spec["candidates"]}
    members = {}
    if spells:
        lng = max(spells, key=lambda d: d["months"])
        o = ph.origins[lng["i"]:lng["j"]]
        for c in MEMBERS:
            z = (SIGNS[c] * (ph.p[c].reindex(o).astype(float) - std[c][0]) / std[c][1]).clip(-Z_CLIP, Z_CLIP)
            members[c] = float(z.mean())
    first_below = next((f"{t:%Y-%m}" for t, wi in zip(ph.origins, wc) if wi == 0.0), None)
    return {"years": years, "real_spxl_first_origin": f"{ph.origins[i09]:%Y-%m-%d}",
            "real_spxl_first_session": f"{ph.dates[int(ph.a[i09]) + 1]:%Y-%m-%d}",
            "bh": {"full": _pf(pf_bh), "real_spxl": _pf(pf_bh09)}, "rules": rules,
            "cagr_check_max_abs_diff": float(max(check)), "placebo": placebo, "bootstrap_real_spxl": boot,
            "vol_managed": vm, "favourable": fav,
            "composite_spells": [{k: v for k, v in d.items() if k not in ("i", "j")} for d in spells],
            "composite_first_tbill_origin": first_below, "composite_members_in_longest_spell": members}


def claims_vintage_check(ph: PostHoc, predictor_inputs: Path) -> Dict:
    """C08 as registered reads ICNSA at its 2026-09-25 vintage. Rebuild it at every test-side origin from
    the vintage in effect on that date (ALFRED has vintages from 2009-05-28), and redo the primary
    test for C08 and COMP and the composite rule with it."""
    if not predictor_inputs.exists():
        return {"available": False, "reason": "predictor_inputs.pkl not found"}
    pi = pd.read_pickle(predictor_inputs)
    rows = pi.get("series", {}).get("ICNSA") if isinstance(pi, dict) else None
    if rows is None or not len(rows):
        return {"available": False, "reason": "no ICNSA vintages in predictor_inputs.pkl"}
    rt = Realtime(rows)
    latest = _clean(rt.at(DATA_CUTOFF))
    release = claims_release_dates(latest.index, first_published(rows))
    first_vintage = pd.Timestamp(rows["realtime_start"].min())
    p = ph.p
    new, stored_ok, missing = {}, 0, 0
    for t in p.index[p.index >= TEST_START]:
        if t < first_vintage:
            continue
        v, _ = claims_yoy(_clean(rt.at(t)), t, release)
        again, _ = claims_yoy(latest, t, release)
        stored_ok += int(_same(again, float(p.at[t, "C08"])))
        missing += int(not np.isfinite(v))
        new[t] = v
    if not new:
        return {"available": False, "reason": "no test-side origin after the first ALFRED vintage"}
    old = p.loc[list(new), "C08"].astype(float)
    diff = (pd.Series(new) - old).abs()
    p2 = p.copy()
    p2.loc[list(new), "C08"] = list(new.values())
    std = {c["id"]: {"mu": float(c["mu"]), "sd": float(c["sd"])} for c in ph.spec["candidates"]}
    comp2, _ = composite(p2, std)
    ts = ph.spec["train_set"]
    w_old = rule_weights(p["COMP"].reindex(ph.origins).to_numpy(float), ts["composite"]["tau"])[0]
    w_new = rule_weights(comp2.reindex(ph.origins).to_numpy(float), ts["composite"]["tau"])[0]
    c_old = rule_weights(signed(p, "C08").reindex(ph.origins).to_numpy(float), ts["tau_j"]["values"]["C08"])[0]
    c_new = rule_weights(signed(p2, "C08").reindex(ph.origins).to_numpy(float), ts["tau_j"]["values"]["C08"])[0]
    s_c08 = _test_sample(p2, signed(p2, "C08"), "Y126", 126)
    s_comp = _test_sample(p2, comp2, "Y126", 126)
    return {"available": True, "first_vintage": f"{first_vintage:%Y-%m-%d}", "origins_rebuilt": len(new),
            "stored_equals_latest_vintage_rebuild": stored_ok, "missing_with_vintage": missing,
            "max_abs_change": float(diff.max()), "median_abs_change": float(diff.median()),
            "max_change_origin": f"{diff.idxmax():%Y-%m-%d}",
            "primary_C08": {"registered": {k: ph.res["primary"]["rows"]["C08"][k] for k in ("r", "p")},
                            "vintage": _ct(s_c08, np.ones(len(s_c08), bool), 126)},
            "primary_COMP": {"registered": {k: ph.res["primary"]["rows"]["COMP"][k] for k in ("r", "p")},
                             "vintage": _ct(s_comp, np.ones(len(s_comp), bool), 126)},
            "composite_rule_holdings_changed": int(np.sum(w_old != w_new)),
            "c08_rule_holdings_changed": int(np.sum(c_old != c_new))}


def splice_check(ph: PostHoc, rating_inputs: Path, pinned_sha: Optional[str]) -> Dict:
    """The registered fund is the synthetic 3x fund through 2008-12-31, although SPXL traded from
    2008-11-05. Replace the synthetic returns with SPXL's from its second close to 2008-12-31 and redo
    the primary test (the other windows do not change)."""
    if not rating_inputs.exists():
        return {"available": False, "reason": "rating_inputs.pkl not found"}
    if pinned_sha is not None and sha256_file(rating_inputs) != pinned_sha:
        return {"available": False, "reason": "rating_inputs.pkl does not match its pinned hash"}
    closes = pd.read_pickle(rating_inputs)["closes"]
    spxl = closes["SPXL"].dropna() if "SPXL" in closes else pd.Series(dtype=float)
    if spxl.empty:
        return {"available": False, "reason": "no SPXL closes"}
    first = spxl.index[0]
    m = np.asarray((ph.dates > first) & (ph.dates < br.SPXL_FROM))
    if not m.any():
        return {"available": False, "reason": "SPXL starts after the synthetic period"}
    r = spxl.pct_change(fill_method=None).reindex(ph.dates[m])
    if r.isna().any():
        return {"available": False, "reason": "SPXL closes missing between its launch and 2009"}
    fund2 = ph.fund.copy()
    fund2[m] = r.to_numpy(float)
    sel = ph.p["test_h126"].astype(bool).to_numpy()
    i0 = ph.p["pos"].astype(int).to_numpy()[sel] - ph.off
    inside = (i0 >= -1) & (i0 + 127 <= len(ph.fund))      # all primary windows on the real data
    lb = _cumlog(ph.bill)
    delta = np.zeros(len(i0))
    delta[inside] = (window_excess(_cumlog(fund2), lb, i0[inside], 126)
                     - window_excess(_cumlog(ph.fund), lb, i0[inside], 126))
    p2 = ph.p.copy()
    p2.loc[p2.index[sel], "Y126"] = p2.loc[p2.index[sel], "Y126"].astype(float).to_numpy() + delta
    changed = np.abs(delta) > 1e-12
    xs = _signed_all(p2)
    rows = {}
    for pid in PREDICTORS:
        s = _test_sample(p2, xs[pid], "Y126", 126)
        rows[pid] = _ct(s, np.ones(len(s), bool), 126)
    adj = holm({k: v["p"] for k, v in rows.items()})
    return {"available": True, "spxl_first_close": f"{first:%Y-%m-%d}", "sessions_replaced": int(m.sum()),
            "synthetic_return": float(np.prod(1.0 + ph.fund[m]) - 1.0), "spxl_return": float(np.prod(1.0 + fund2[m]) - 1.0),
            "origins_changed": int(changed.sum()), "max_abs_change_y126": float(np.max(np.abs(delta))),
            "rows": rows, "holm_passes": [k for k in PREDICTORS if adj[k]["reject"]],
            "max_abs_change_r": float(max(abs(rows[k]["r"] - ph.res["primary"]["rows"][k]["r"]) for k in PREDICTORS)),
            "max_abs_change_p": float(max(abs(rows[k]["p"] - ph.res["primary"]["rows"][k]["p"]) for k in PREDICTORS))}


def _exploratory_inputs(paths: Phase3Paths) -> Dict[str, Path]:
    return {"test_results.json": paths.results_json, "frozen_spec.json": paths.frozen_spec,
            "panel_unsealed.pkl": paths.test_dir / "panel_unsealed.pkl",
            "economic_daily.csv": paths.test_dir / "economic_daily.csv",
            "predictor_inputs.pkl": paths.predictor_inputs, "rating_inputs.pkl": paths.backtest / "rating_inputs.pkl"}


def exploratory(paths: Phase3Paths, recompute: bool = False, mc_reps: int = MC_REPS,
                feedback_reps: int = FEEDBACK_REPS) -> Dict:
    """Every exploratory number in research/RESULTS.md, cached in output/research/exploratory.json and
    recomputed when an input, this script or the settings change (or with ``report --recompute``).
    Deterministic: every random draw has a fixed seed."""
    target = paths.results_json.parent / EXPLORATORY_NAME
    key = {"inputs": {k: sha256_file(v) for k, v in _exploratory_inputs(paths).items() if v.exists()},
           "code": sha256_file(Path(__file__)), "mc_reps": int(mc_reps), "mc_blocks": list(MC_BLOCKS),
           "feedback_reps": int(feedback_reps)}
    if not recompute and target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old.get("key") == key:
            return old
    ph = load_posthoc(paths)
    print("computing the exploratory analyses (under a minute)...", flush=True)
    out = {"key": key,
           "note": "Exploratory, after unsealing: not a registered test and not a result (research/RESULTS.md section 3).",
           "mc": mc_calibration(ph, reps=mc_reps), "feedback": feedback_null(ph, reps=feedback_reps),
           "secondary": secondary_checks(ph), "episodes": episode_checks(ph), "economics": economics(ph),
           "claims_vintage": claims_vintage_check(ph, paths.predictor_inputs),
           "splice": splice_check(ph, paths.backtest / "rating_inputs.pkl", paths.pinned.get("rating_inputs.pkl"))}
    out = jsonable(out)
    write_lf(target, json.dumps(out, indent=1, allow_nan=False) + "\n")
    return out


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def output_manifest(paths: Phase3Paths) -> str:
    """sha256sum-format list of the git-ignored files the hash chain points to (pinned inputs, the raw
    downloads, the panels, the frozen spec's copy, the train and test outputs), so that an archived copy
    can be checked with `sha256sum -c research/OUTPUT_SHA256.txt`. The look log and exploratory.json
    are left out: they change with every run."""
    files: List[Path] = [paths.backtest / n for n in paths.pinned]
    raw = paths.predictor_inputs.parent / "raw"
    files += [paths.predictor_inputs] + (sorted(q for q in raw.rglob("*") if q.is_file()) if raw.exists() else [])
    files += [paths.panel_train, paths.panel_test, paths.panel_meta, paths.train_params]
    tdir = paths.train_stats.parent
    files += sorted(q for q in tdir.rglob("*") if q.is_file()) if tdir.exists() else []
    files += [paths.results_json]
    files += sorted(q for q in paths.test_dir.rglob("*") if q.is_file()) if paths.test_dir.exists() else []
    seen, lines = set(), []
    for f in files:
        if f.exists() and f not in seen:
            seen.add(f)
            lines.append(f"{sha256_file(f)}  {_rel(f)}")
    return "\n".join(lines) + "\n"


PLAIN = {
    "C01": ("Variance risk premium", "the VIX squared minus the S&P 500's variance over the last month", "high"),
    "C02": ("Output gap", "industrial production against its trend, as published at the time", "low"),
    "C03": ("Short-rate change", "the 3-month Treasury yield against a year earlier", "falling"),
    "C04": ("Yield curve", "10-year minus 3-month Treasury yield", "steep"),
    "C05": ("Credit spread", "Baa minus Aaa corporate bond yields", "wide"),
    "C06": ("Inflation", "consumer prices against a year earlier", "low"),
    "C07": ("Bank lending standards", "net share of banks tightening business-loan standards", "easing"),
    "C08": ("Jobless claims", "new unemployment claims against a year earlier", "falling"),
    "C09": ("Valuation", "excess CAPE yield: long-run earnings yield minus the real 10-year yield", "high (cheap)"),
    "C10": ("VIX level", "VIX squared (option-implied variance)", "low"),
    "C11": ("Leverage-cost hurdle", "the S&P 500 return SPXL needs to break even", "low"),
    "C12": ("Old rating score", "the retired BUY / HOLD / SELL score", "high"),
    "COMP": ("Blend of C01-C09", "equal-weight average of the nine standardised readings", "high"),
}
PAGE_WORDING = ("Timing: SPXLcast gives no buy or sell signal. We tested 12 economic and market-risk indicators "
                "(valuation, inflation, interest rates, credit spreads, jobless claims, bank lending, industrial "
                "output and volatility) against SPXL's returns from 2008 to 2026, with the rules fixed in advance. "
                "None reliably picked better or worse times to hold SPXL.")

REVIEW_RECORD = [
    ["look-ahead", "The order plan, freeze, test rests on the pipeline's own records; nothing external anchors it",
     "Recorded (section 4). Fixed for the future: every unsealing is now logged whatever the entry point "
     "(`verify_unseal`, DEVIATIONS.md N28). Committing `research/` and archiving `output/` is for the owner", "none"],
    ["look-ahead", "The code that ran the test was not preserved or hashed, and was edited 3 minutes later",
     "Fixed for future runs: `test` records the SHA-256 of its code and copies it beside the outputs. For this run the "
     "current code was re-run into a scratch folder and reproduced every number (DEVIATIONS.md N29)", "none"],
    ["look-ahead", "The files the hash chain points to are git-ignored and cannot all be rebuilt",
     "Fixed: `report` writes `research/OUTPUT_SHA256.txt`; archiving `output/` is for the owner", "none"],
    ["look-ahead", "C08 uses the 2026 claims vintage (revisions)", "Recorded; effect regenerated in 3.8", "none"],
    ["look-ahead", "Synthetic fund through 2008-12-31 although SPXL traded from 2008-11-05",
     "Recorded; effect regenerated in 3.8", "none"],
    ["look-ahead", "Single bad closes in the Yahoo data", "Recorded (3.8); no outcome window starts or ends on them", "none"],
    ["look-ahead", "The exposure ledger leaves out the SPXL-only hurdle split computed before the freeze",
     "Recorded in section 4 and DEVIATIONS.md N30", "none (C11 was already not clean, and failed)"],
    ["look-ahead", "pct_change fill behaviour in build_daily (future pandas)",
     "Fixed in the research script: `check_spxl_closes` stops on any missing SPXL close (there are none). "
     "`backtest_rating.py` is the owner's file and was not edited", "none"],
    ["statistics", "n_eff ignores predictor persistence, so the test is conservative and C06 is borderline",
     "Recorded; persistence-aware simulation regenerated in 3.1; wording fixed in the registered document",
     "none (the registered method stands)"],
    ["statistics", "VRP had far too few effective observations; its effect is 2008-16 only",
     "Recorded in 3.1 and 3.3", "none"],
    ["statistics", "The one secondary Holm pass is shown without multiplicity or robustness context",
     "Recorded in 2.4 and 3.3; the registered document now shows the pooled Holm; 'consistent with the published "
     "effect' removed", "none"],
    ["statistics", "INFL and GAP come from the known episodes", "Recorded; joint exclusion regenerated in 3.2",
     "none (strengthens the null)"],
    ["statistics", "Stambaugh feedback bias", "Recorded; simulation regenerated in 3.4", "none"],
    ["statistics", "Fisher z and one-sided logic correct; Wald variance slightly conservative", "No change needed", "none"],
    ["statistics", "Two-decimal p-values, unexplained stars, unadjusted bootstrap intervals, 25-month basis of 5.3",
     "Fixed in the registered document's rendering (p to 4 decimals, Holm bars stated, intervals labelled, the basis of "
     "5.3 stated); no number changed", "none"],
    ["statistics", "The economic test is degenerate for GAP and ECY (level shift)",
     "Recorded in 3.5; the registered document now names the rules that never left SPXL", "none"],
    ["practical", "Every economic win comes from three synthetic-fund months in autumn 2008",
     "Recorded; crisis-neutral and real-SPXL figures regenerated in 3.5", "none"],
    ["practical", "Best-of-12 selection: C06's CAGR is what luck produces", "Recorded; joint placebo regenerated in 3.5",
     "none"],
    ["practical", "Taxes are not modelled", "Recorded in 3.7 (the reviewer's simulation, re-run and matched)", "none"],
    ["practical", "The VRP secondary pass has no practical value", "Recorded in 3.3 and 3.5", "none"],
    ["practical", "The vol-managed drawdown cut is a 2008 artefact; a constant weight does as well after 2008",
     "Recorded; regenerated in 3.6", "none"],
    ["practical", "The realistic edge is about zero; even 'unfavourable' months pay", "Recorded in section 1 and 3.5",
     "none"],
    ["practical", "Real-time usability holds, with minor residuals", "Recorded (3.5 execution lag, section 4)", "none"],
    ["practical", "The only positive evidence rests on a hypothetical 2008 instrument with optimistic financing",
     "Recorded in 3.5 and section 4; the registered document now labels 2008 as the synthetic fund", "none"],
]


def _demote(text: str) -> List[str]:
    """The registered document (research/test_results.md) as an appendix: its title and hand-written
    EXPLORATORY section dropped, every heading one level down and its numbered sections prefixed A
    (section 5.1 becomes A5.1)."""
    k = text.find("\n" + EXPLORATORY_HEADING)
    body = text[:k] if k >= 0 else text
    out = [re.sub(r"^(#+) (\d)", r"\1 A\2", "#" + line) if line.startswith("#") else line
           for line in body.splitlines()[1:]]
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


FAMILY_PLAIN = {"primary": "SPXL's next 6 months", "F2": "SPXL's next month", "F3": "SPXL's next 3 months",
                "F4": "the S&P 500's next 6 months", "F5": "the S&P 500's next 3 months",
                "F6": "the S&P 500's next month", "F7": "the chance of a 20% SPXL fall within 3 months"}
_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
          "thirteen")


def _n(k: int) -> str:
    return _WORDS[k] if 0 <= k < len(_WORDS) else str(k)


def _pp(x) -> str:
    return _fmt(x, "{:.1%}")


def _ids(keys: Iterable[str]) -> str:
    keys = list(keys)
    return ", ".join(keys) if keys else "none"


def render_results_report(res: Dict, ex: Dict, registered: str, code: Dict[str, str]) -> str:
    """research/RESULTS.md: the owner's report. A plain-language summary, the confirmatory (registered)
    results, exploratory notes, limitations and how to reproduce; the registered document in full as
    Appendix A and the review record as Appendix B. Every number comes from test_results.json or
    exploratory.json."""
    L: List[str] = []
    a = L.append
    v, pr, econ = res["verdict"], res["primary"], res["economic"]
    main = econ["rules"]["COMP"]
    st = main["strategies"]
    mc, fb, sec, eps, ec = ex["mc"], ex["feedback"], ex["secondary"], ex["episodes"], ex["economics"]
    cv, sp = ex["claims_vintage"], ex["splice"]
    rows_p = pr["rows"]
    rep = res["replication_train"]["primary"]["rows"]
    holm1 = ALPHA / len(PREDICTORS)

    def rkey(k: str) -> float:
        return rows_p[k]["r"] if rows_p[k]["r"] is not None else -9.0
    best = max(PREDICTORS, key=rkey)
    second = max((k for k in PREDICTORS if k != best), key=rkey)
    spells = ec["composite_spells"]
    long_spell = max(spells, key=lambda d: d["months"]) if spells else None
    bh09 = ec["bh"]["real_spxl"]
    rl = ec["rules"]
    beat09 = [k for k in CAND_IDS if rl[k]["real_spxl"]["cagr"] > bh09["cagr"]]
    top09 = max(CAND_IDS, key=lambda k: rl[k]["real_spxl"]["cagr"] if rl[k]["share_in_spxl"] < 1.0 else -9.0)
    pm = mc["families"]["primary"]["per_block"]
    f3 = mc["families"]["F3"]["per_block"]
    f2 = mc["families"]["F2"]["per_block"]
    blocks = [str(b) for b in mc["blocks"]]
    c06_mc = [pm[b]["rows"]["C06"]["p_mc"] for b in blocks]
    c06_pass = [b for b in blocks if "C06" in pm[b]["holm_passes"]]
    all_rows = [r for b in blocks for r in pm[b]["rows"].values()]
    size5 = max(r["size_at_5pct"] for r in all_rows)
    sizeh = max(r["size_at_holm1"] for r in all_rows)
    persistent = [k for k in PREDICTORS if k != "C01"]
    rbar = [pm[b]["rows"][k]["r_needed_holm1"] for b in blocks for k in persistent]
    ineff = [pm[b]["rows"][k]["implied_n_eff"] for b in blocks for k in persistent]
    vrp = sec["vrp"]["S21"]
    fav06, favc = ec["favourable"]["C06"]["from_2009"], ec["favourable"]["COMP"]["from_2009"]
    pl = ec["placebo"]
    b06 = ec["bootstrap_real_spxl"]["C06"]
    vm09 = ec["vol_managed"]["real_spxl"]
    wins = [k for k in CAND_IDS if rl[k]["edge_vs_bh"] > 0 and rl[k]["share_in_spxl"] < 1.0]

    a("# SPXL timing-signal search: results")
    a("")
    a("For the owner of SPXLcast. Section 1 is a plain-language summary with a recommendation. Section 2 gives the "
      "pre-registered (confirmatory) results, the only ones that decide the answer. Section 3 holds exploratory notes "
      "made after the test to answer three independent reviews; they are not results. Section 4 lists the limitations "
      "and section 5 how to reproduce everything. Appendix A repeats every registered table "
      "(`research/test_results.md`); Appendix B lists each challenge from the reviews and how it was handled.")
    a("")
    a(f"The test ran once, at {res['run_utc']}, against the plan in `research/PREREGISTRATION.md` and the spec frozen at "
      f"{res['frozen_utc']} (`research/frozen_spec.json`). This file is regenerated by "
      "`py scripts/research_signals.py report` from `output/research/test_results.json` and "
      "`output/research/exploratory.json`.")
    a("")
    # ------------------------------------------------------------------ 1. plain language
    a("## 1. Summary in plain language")
    a("")
    a("### What was tested")
    a("")
    a("SPXLcast's own 1990-2026 backtest found that its forecast ranges are reliable but that its old BUY / HOLD / SELL "
      "rating could not tell good times to hold SPXL from bad ones. This study asked whether anything else can, "
      "without chart-reading (no momentum, trends, moving averages or price direction, as the owner required).")
    a("")
    a("* **Twelve indicators**, taken from research published before 2008: valuation, the economy's output gap, "
      "inflation, jobless claims, bank lending standards, short-term interest rates, the yield curve, corporate credit "
      "spreads, the VIX and a measure built from it (the variance risk premium), SPXL's leverage-cost hurdle and the "
      "retired rating score. A thirteenth, the **blend**, averages the first nine.")
    a("* **Rules fixed first.** Which indicators, which direction counts as good, how to blend them and what counts as "
      "success were written down and locked (with a fingerprint of the file) before anyone looked at 2008-2026. "
      "1990-2007 was used only to set each indicator's normal level and a cut-off. Then 2008-2026 was tested once.")
    a("* **Two tests, both needed.** A statistical test: were better readings followed by better SPXL returns over the "
      "next six months, clearly enough to survive the fact that 13 things were tried? And a money test: would "
      "switching between SPXL and Treasury bills on the blend have beaten simply holding SPXL on growth, on return per "
      "unit of risk and on the worst fall, in both halves of the period, after switching costs?")
    a("")
    a("### What was found")
    a("")
    a(f"* **No timing signal.** None of the 13 passed the statistical test. The switching rule based on the blend lost "
      f"on every money criterion: it grew {_pp(st['rule']['cagr'])} a year against {_pp(st['BH']['cagr'])} for simply "
      f"holding SPXL, with the same {_pp(st['rule']['max_drawdown'])} worst fall (2008-09).")
    if long_spell:
        a(f"* Most of that shortfall is one spell: the blend sat in Treasury bills for the months decided from "
          f"{long_spell['from']} to {long_spell['to']}, when SPXL gained {long_spell['spxl']:.0%}. An inverted yield "
          "curve, rising rates, tighter bank lending and high inflation all read as bad news just before a strong rally.")
    a(f"* The two strongest indicators were {PLAIN[best][0].lower()} ({best}) and the {PLAIN[second][0].lower()} "
      f"({second}). Neither cleared the bar, neither showed any link in 1990-2007, and most of their apparent link in "
      "2008-2026 comes from three well-known episodes: the 2008-09 crisis, the 2020 crash and the 2021-22 inflation "
      "shock.")
    a((f"* On paper, {_n(len(wins))} single-indicator switching rules made more money than holding SPXL "
       f"({_ids(wins)}), but all of that came from the 2008-09 crisis, mostly from being out of the market in "
       "September-November 2008, before SPXL existed (those months use a simulated 3x fund). " if wins else
       "* No single-indicator switching rule made more money than holding SPXL. ")
      + "From 2009, when real SPXL traded, "
      f"{'none of the 12 rules' if not beat09 else _ids(beat09) + ' of the 12'} beat holding SPXL "
      f"({_pp(bh09['cagr'])} a year); the best rule that ever left SPXL, {top09}, made "
      f"{_pp(rl[top09]['real_spxl']['cagr'])}.")
    a("* Independent reviewers rebuilt the data, the timing of every release, the returns and the statistics; everything "
      "reproduced. A finer statistical check made afterwards (exploratory) suggests the registered test was cautious, "
      "and that inflation is borderline under the finer check, but even then it would not have been a usable timing "
      "rule.")
    a("")
    a("### What it means for SPXLcast")
    a("")
    a("* The page's current approach is the right one: show the forecast range, the cost of holding SPXL and the risk "
      "of a sharp fall, and give no buy or sell call. Nothing tested here would make a timing call trustworthy.")
    a("* Holding less SPXL lowers risk about as well as anything tested: from 2009 a steady 55-60% in SPXL (the rest in "
      "T-bills) matched a volatility-based rule on growth and on the worst fall. That is risk control, not timing.")
    a("")
    a("### Recommendation")
    a("")
    a("**Do not add a timing indicator to the page.** Do not show 'favourable' or 'unfavourable' readings of inflation, "
      "the output gap, the VIX or the blend: none of them earned it, and showing them would suggest a signal the test "
      "did not find. If you want to say what was tested, one line is enough, for example under the assessment, where "
      "the page already says it gives no buy or sell call:")
    a("")
    a(f"> {PAGE_WORDING}")
    a("")
    a("Do not try variations of these indicators on 2008-2026: those years are now used up as a test. A new idea can "
      "only be tested on data that do not exist yet, and six-month outcomes take years to accumulate (18 years gave "
      "only about 37 independent six-month periods). Logging indicators from now on would support a new test in "
      "a decade at the earliest.")
    a("")
    a("### How sure is this?")
    a("")
    a("'No usable timing signal' is a firm answer. 'None of these indicators has any link at all' is not: with about 37 "
      "independent six-month periods, a weak link of the size found in published research (a rank correlation of "
      "0.1-0.2) would usually go undetected. But a link that weak, used to switch between SPXL and T-bills, is worth "
      "about nothing to a holder. Even for inflation, the strongest indicator, SPXL beat T-bills by "
      f"{_pp(fav06['unfavourable_mean'])} on average over the six months after an unfavourable reading from 2009 "
      f"(against {_pp(fav06['favourable_mean'])} after a favourable one), so stepping aside in the 'bad' months gave "
      "that up.")
    a("")
    # ------------------------------------------------------------------ 2. confirmatory
    a("## 2. Confirmatory results (pre-registered)")
    a("")
    a(f"### 2.1 Verdict: **{v['headline']}**")
    a("")
    a("The plan fixed four possible answers: 'Timing signal confirmed' (an indicator passes the statistical test and its "
      "switching rule passes the money test); 'Predictive, not usable by the registered rule' (statistical pass, money "
      "fail); 'Not confirmed (gain could be luck)' (the blend's rule passes the money test without a statistical "
      f"pass); and 'No timing signal found' (neither). Here {_n(len(v['primary_passes']))} indicator passed the "
      f"statistical test and the blend's rule {'passed' if main['pass'] else 'failed'} the money test.")
    a("")
    a("### 2.2 Statistical test: 13 indicators against the next six months (decisive)")
    a("")
    a(f"At each of {pr['origins']} month-ends from {pr['first']} to {pr['last']}, each indicator was ranked against "
      "SPXL's return over T-bills in the next six months (126 trading days). **r** is the rank correlation: +1 would mean "
      "the better the reading, the better the next six months, every time; 0 means no link. **p** is how often an r at "
      "least this large would turn up by chance if there were no link, counting the overlapping six-month windows as "
      f"about {rows_p['COMP']['n_eff']:.0f} independent ones. Because 13 indicators were tried, the smallest p had to be "
      f"at most {holm1:.4f} (the Holm correction), the next at most {ALPHA / 12:.4f}, and so on.")
    a("")
    rows = []
    for pid in PREDICTORS:
        t = rows_p[pid]
        rows.append([pid, PLAIN[pid][0], PLAIN[pid][2], _fmt(t["r"]), _fmt(t["p"], "{:.4f}"),
                     _fmt(t["threshold"], "{:.4f}"), _fmt(t["p_holm"], "{:.3f}"), "**PASS**" if t["reject"] else "fail",
                     _fmt(rep[pid]["r"])])
    L += md_table(["id", "indicator", "good when", "r 2008-26", "p (one-sided)", "p needed (Holm step)",
                   "Holm-adjusted p", "result", "r 1990-2007"], rows)
    a("")
    wrong = [k for k in PREDICTORS if rows_p[k]["r"] is not None and rows_p[k]["r"] < -0.15]
    n_pass = len(v["primary_passes"])
    a(f"{'None' if n_pass == 0 else _n(n_pass).capitalize()} of the 13 passed. The strongest, {best}, had p = "
      f"{rows_p[best]['p']:.4f} against the "
      f"{rows_p[best]['threshold']:.4f} it needed. {_ids(wrong)} came out clearly the wrong way round: their good "
      "readings were followed by worse returns, the opposite of what research before 2008 said.")
    a("")
    a("### 2.3 Money test: the blend's switching rule (decisive)")
    a("")
    a(f"At each month-end from {econ['first_origin']} to {econ['last_origin']}: hold SPXL for the next month if the "
      "blend is at or above the line that marked its lowest third in 1990-2007, otherwise hold T-bills; 0.10% per switch. "
      f"Compared with buy-and-hold SPXL and with a no-timing mix that always holds the rule's average share in SPXL "
      f"({main['w_mix']:.0%}). Before tax; the 2008 months use a simulated 3x fund (SPXL started trading in November "
      "2008).")
    a("")
    rows = []
    for k, name in (("rule", "blend's switching rule"), ("BH", "buy and hold SPXL"),
                    ("MIX_w", f"mix, {main['w_mix']:.0%} SPXL")):
        s = st[k]
        rows.append([name, _pp(s["cagr"]), _fmt(s["sharpe"], "{:.2f}"), _pp(s["max_drawdown"]),
                     _fmt(s["final_wealth"], "{:,.2f}"), _fmt(s["sharpe_halves"][0], "{:.2f}"),
                     _fmt(s["sharpe_halves"][1], "{:.2f}")])
    L += md_table(["strategy", "growth a year", "return per unit of risk (Sharpe)", "worst fall", "$1 became",
                   "Sharpe 2008-16", "Sharpe 2017-26"], rows)
    a("")
    e = main["E"]
    rows = [["E1 growth", "beats both",
             f"{_pp(st['rule']['cagr'])} vs {_pp(st['BH']['cagr'])} and {_pp(st['MIX_w']['cagr'])}", _yn(e["E1"])],
            ["E2 Sharpe", "beats both",
             f"{st['rule']['sharpe']:.2f} vs {st['BH']['sharpe']:.2f} and {st['MIX_w']['sharpe']:.2f}", _yn(e["E2"])],
            ["E3 worst fall", "smaller than buy-and-hold",
             f"{_pp(st['rule']['max_drawdown'])} vs {_pp(st['BH']['max_drawdown'])}", _yn(e["E3"])],
            ["E4 both halves", "Sharpe beats buy-and-hold in 2008-16 and in 2017-26",
             f"{st['rule']['sharpe_halves'][0]:.2f} vs {st['BH']['sharpe_halves'][0]:.2f}; "
             f"{st['rule']['sharpe_halves'][1]:.2f} vs {st['BH']['sharpe_halves'][1]:.2f}", _yn(e["E4"])]]
    L += md_table(["criterion", "needed", "result", "pass?"], rows)
    a("")
    a(f"Money test: **{'PASS' if main['pass'] else 'FAIL'}** (all four were needed).")
    a("")
    a("### 2.4 Secondary tests (registered, never decisive)")
    a("")
    a(f"The same 13 indicators were also tested against SPXL over 1 and 3 months, the S&P 500 over 1, 3 and 6 months, "
      f"and the chance of a 20% fall within 3 months ({13 * (len(FAMILIES) - 1)} tests in six families of 13). "
      f"{_n(len(sec['passes_within_family'])).capitalize()} passed within its family: "
      + ("; ".join(f"{PLAIN[k.split(':')[1]][0].lower()} ({k.split(':')[1]}) against {FAMILY_PLAIN[k.split(':')[0]]}"
                   for k in sec["passes_within_family"]) or "none")
      + f". Counted "
      f"together with the primary family, none of the {sec['n_tests']} registered tests survives one Holm correction "
      f"(smallest adjusted p {sec['min_adjusted_p']:.2f}). Section 3.3 looks at that one pass.")
    a("")
    # ------------------------------------------------------------------ 3. exploratory
    a("## 3. Exploratory notes (not results)")
    a("")
    a("**Everything in this section was computed after the test period was opened**, from its saved outputs, to answer "
      "three independent reviews (Appendix B). None of it is a registered test, none of it changes the verdict, and "
      "none of it may be used to pick an indicator, direction, cut-off or rule for 2008-2026. Changing the method after "
      "seeing the data is exactly what the pre-registration exists to prevent.")
    a("")
    a("### 3.1 Was the registered test too strict?")
    a("")
    a("The registered p-values count the overlapping six-month windows as about 37 independent ones, whatever the "
      "indicator. How much independent information a correlation really carries also depends on how slowly the "
      "indicator itself changes. To check, a simulation kept each indicator's actual 2008-2026 path and rebuilt SPXL's "
      "six-month returns from its own daily returns reshuffled in blocks (so there is no real link, but the overlap "
      f"and the calm and stormy stretches are kept): {mc['reps']:,} draws for each average block length of "
      f"{', '.join(blocks)} trading days.")
    a("")
    rows = []
    for pid in PREDICTORS:
        r0 = pm[blocks[0]]["rows"][pid]
        ie = [pm[b]["rows"][pid]["implied_n_eff"] for b in blocks]
        rows.append([pid, PLAIN[pid][0], _fmt(r0["r"]), _fmt(r0["p_registered"], "{:.4f}")]
                    + [_fmt(pm[b]["rows"][pid]["p_mc"], "{:.4f}") + ("*" if pid in pm[b]["holm_passes"] else "")
                       for b in blocks]
                    + [f"{min(ie):.0f}-{max(ie):.0f}"])
    L += md_table(["id", "indicator", "r", "registered p"] + [f"simulated p, blocks of {b}" for b in blocks]
                  + ["independent windows implied"], rows)
    a("")
    a("`*` = would pass the Holm correction on the simulated p-values.")
    a("")
    a(f"* The registered test was cautious. With no real link it wrongly 'found' one in at most {size5:.1%} of draws "
      f"at the 5% level and {sizeh:.2%} at the first Holm step (the intended rates are 5% and {holm1:.2%}). For the "
      f"slow-moving indicators the simulation implies {min(ineff):.0f}-{max(ineff):.0f} independent windows rather "
      f"than 37, and a correlation of about {min(rbar):.2f}-{max(rbar):.2f}, not 0.45, would have cleared the first "
      "Holm step.")
    a(f"* **Inflation (C06) is borderline.** Its simulated p is {', '.join(f'{x:.4f}' for x in c06_mc)} for blocks of "
      f"{', '.join(blocks)} days, against {holm1:.4f}: it would pass with "
      f"{('blocks of ' + ' and '.join(c06_pass)) if c06_pass else 'no block length'} and fail with the others "
      f"(simulation error about +/-0.0006). The output gap (C02) just misses under every block length (smallest "
      f"{min(pm[b]['rows']['C02']['p_mc'] for b in blocks):.4f}).")
    c06r = econ["rules"]["C06"]["strategies"]
    a(f"* Even had inflation passed, the registered label would have been 'Predictive, not usable by the registered "
      f"rule': its own switching rule fails E4 (Sharpe {c06r['rule']['sharpe_halves'][1]:.2f} against "
      f"{c06r['BH']['sharpe_halves'][1]:.2f} for buy-and-hold in 2017-26). Sections 3.2 and 3.5 show why it would not "
      "have been usable in practice either.")
    a(f"* At three months (SPXL, family F3) the simulation would give Holm passes to "
      f"{_ids(sorted(set(k for b in blocks for k in f3[b]['holm_passes'])))} under every block length shown; at one "
      f"month (F2) to {_ids(sorted(set(k for b in blocks for k in f2[b]['holm_passes'])))} under some. These are "
      "short-horizon associations; sections 3.2 and 3.3 show where they come from.")
    a("* The registered method was fixed before the data were seen. Switching to this one afterwards, because it gives "
      "a different answer, would be a forking path, so the registered verdict stands. The simulation also leaves out "
      "feedback from returns to the indicator (3.4).")
    a("")
    a("### 3.2 The largest correlations come from three known episodes")
    a("")
    a("The registered robustness check left out one episode at a time. Leaving out the three together (2008-01..2009-06, "
      "2019-09..2020-06 and 2021-07..2022-12, all listed as common knowledge in the plan's exposure ledger):")
    a("")
    rows = []
    for pid in PREDICTORS:
        q = eps["rows"][pid]
        rows.append([pid, PLAIN[pid][0], _fmt(q["all"]["r"]),
                     f"{_fmt(q['crisis and 2020']['r'])} ({_fmt(q['crisis and 2020']['p'], '{:.3f}')})",
                     f"{_fmt(q['all three']['r'])} ({_fmt(q['all three']['p'], '{:.3f}')})", _fmt(rep[pid]["r"])])
    L += md_table(["id", "indicator", "r, all", "r (p) without crisis and 2020", "r (p) without all three",
                   "r 1990-2007"], rows)
    a("")
    q6, q2, qc = eps["rows"]["C06"], eps["rows"]["C02"], eps["rows"]["COMP"]
    a(f"Inflation falls from {q6['all']['r']:+.2f} to {q6['all three']['r']:+.2f}, the output gap from "
      f"{q2['all']['r']:+.2f} to {q2['all three']['r']:+.2f} and the blend from {qc['all']['r']:+.2f} to "
      f"{qc['all three']['r']:+.2f}. Their 2008-2026 correlations are mostly those three episodes. With their "
      "1990-2007 correlations near zero, this strengthens the null.")
    a("")
    a("### 3.3 The one secondary pass: variance risk premium against the S&P 500's next month")
    a("")
    h1, h2 = vrp["halves"]
    y63 = sec["vrp"]["Y63"]
    a(f"* It passed its family with r = {vrp['all']['r']:+.2f} and p = {vrp['all']['p']:.4f} against {holm1:.4f}. The "
      f"same indicator against SPXL's next month (family F2) had p = {sec['vrp']['Y21']['all']['p']:.4f} and missed. The "
      f"two targets rank almost identically (rank correlation {sec['spearman_y21_s21']:.3f}), so this is one borderline "
      "result, not two.")
    a(f"* Pooled with all {sec['n_tests']} registered tests under one Holm correction it does not pass (adjusted p "
      f"{sec['min_adjusted_p']:.2f}).")
    a(f"* Leaving out any one of {len(vrp['years_whose_removal_lifts_p_above_holm1'])} calendar years "
      f"({', '.join(str(y) for y in vrp['years_whose_removal_lifts_p_above_holm1'])}) lifts p above {holm1:.4f}; "
      f"without the 2008-09 crisis p is {vrp['without_crisis']['p']:.4f}.")
    a(f"* It is a first-half effect: r {h1['r']:+.2f} in 2008-16 and {h2['r']:+.2f} in 2017-26. In 1990-2007 (the "
      f"published sample) r was {vrp['train']['r']:+.2f}.")
    n_rej = int(round(max(pm[b]["rows"]["C01"]["size_at_5pct"] for b in blocks) * mc["reps"]))
    a(f"* This indicator barely persists from month to month, so the registered test had almost no power for it: in the "
      f"simulation of 3.1 its registered six-month test 'found' a link at the 5% level in at most {n_rej} of "
      f"{mc['reps']:,} no-link draws per block length, where about {int(round(ALPHA * mc['reps'])):,} were intended. "
      "Under the "
      f"simulated null it does show a 1-3 month association in 2008-2026 (three-month simulated p about "
      f"{min(f3[b]['rows']['C01']['p_mc'] for b in blocks):.4f}), but from 2008-16 only (three-month r "
      f"{y63['halves'][0]['r']:+.2f} then and {y63['halves'][1]['r']:+.2f} in 2017-26). Its switching rule needed "
      f"{rl['C01']['switches']} switches and lost to buy-and-hold on real SPXL ({_pp(rl['C01']['real_spxl']['cagr'])} "
      f"against {_pp(bh09['cagr'])} a year; 3.5). It is not a usable signal.")
    a("")
    a("### 3.4 Feedback bias (Stambaugh)")
    a("")
    a("Some indicators move with the market during the six months being predicted: a crash raises the VIX and makes "
      "stocks cheaper. With so few independent windows that feedback biases the correlation. A simple simulation with "
      "the feedback measured in 2008-2026 and no real link shows its size and direction:")
    a("")
    rows = []
    for pid in PREDICTORS:
        q = fb["rows"][pid]
        rows.append([pid, PLAIN[pid][0], _fmt(q["rho_y126_dx6"]), _fmt(q["null_mean_r"], "{:+.3f}"), _fmt(q["r"]),
                     _fmt(q["p_registered"], "{:.3f}"), _fmt(q["p_feedback_null"], "{:.3f}")])
    L += md_table(["id", "indicator", "moves with the 6-month return (r)", "bias in r with no real link", "r",
                   "registered p", "p with feedback"], rows)
    a("")
    a(f"Valuation (C09) is pushed up by about {fb['rows']['C09']['null_mean_r']:+.2f}, so part of its "
      f"{rows_p['C09']['r']:+.2f} is mechanical; the VIX-based indicators (C10-C12) are pushed down. For inflation and "
      f"the output gap the bias is negligible ({fb['rows']['C06']['null_mean_r']:+.3f} and "
      f"{fb['rows']['C02']['null_mean_r']:+.3f}). The 'p with feedback' also reflects this simple model's own "
      "assumptions (normal returns, one monthly persistence), so read it for direction, not as a better p-value.")
    a("")
    a("### 3.5 Would any rule have made money in practice?")
    a("")
    a(f"Each indicator's switching rule (SPXL when the reading is at or above its 1990-2007 cut-off, otherwise "
      f"T-bills), the blend's (COMP), and the volatility-managed weights (VM, Appendix A5.4). 'Edge' is growth a year "
      f"relative to buy-and-hold. 'Real SPXL' is {ec['real_spxl_first_session']} to {econ['sessions'][1]}, when the "
      f"fund is SPXL itself (buy-and-hold {_pp(bh09['cagr'])} a year, Sharpe {bh09['sharpe']:.2f}).")
    a("")
    rows = []
    for k in ["COMP"] + CAND_IDS + ["VM"]:
        q = rl[k]
        rows.append([k, _fmt(q["share_in_spxl"], "{:.0%}"), q["switches"] if k != "VM" else f"{q['switches']} rebal.",
                     _pp(q["full"]["cagr"]), _pct(q["edge_vs_bh"]), _pct(q["edge_vs_bh_without_crisis"]),
                     _pct(q["edge_vs_bh_without_top3"]), _pp(q["real_spxl"]["cagr"]),
                     _fmt(q["real_spxl"]["sharpe"], "{:.2f}"), _pp(q["one_month_late"]["cagr"])])
    L += md_table(["rule", "time in SPXL", "switches", "growth 2008-26", "edge", "edge, crisis months as buy-and-hold",
                   "edge without its best 3 months", "growth, real SPXL", "Sharpe, real SPXL",
                   "growth, acting a month late"], rows)
    a("")
    if wins:
        tops = sorted(set(m for k in wins for m in rl[k]["top3_months"]))
        a(f"* **The on-paper wins are one episode.** {_ids(wins)} beat buy-and-hold only because they were in T-bills "
          f"for their best months, decided in {', '.join(tops)}: the 2008-09 crisis, mostly autumn 2008, when the "
          "fund is the simulated one. With the crisis months (2008-01..2009-06) set equal to buy-and-hold, their edge "
          "becomes "
          + ", ".join(f"{_pct(rl[k]['edge_vs_bh_without_crisis'])} ({k})" for k in wins)
          + " a year. Every E3 pass in Appendix A5.1 (a smaller worst fall) likewise just means the rule was out in "
          "autumn 2008.")
    a(f"* **On real SPXL {'no rule wins' if not beat09 else 'few rules win'}.** From 2009 "
      f"{'no rule' if not beat09 else _ids(beat09)} beat buy-and-hold; "
      f"the best rule that ever left SPXL, {top09}, made {_pp(rl[top09]['real_spxl']['cagr'])} with a Sharpe of "
      f"{rl[top09]['real_spxl']['sharpe']:.3f} against {bh09['sharpe']:.3f}. A 90% resampling interval for C06 against "
      f"buy-and-hold over that period is {_pct(b06['BH']['cagr_diff_ci90'][0])} to "
      f"{_pct(b06['BH']['cagr_diff_ci90'][1])} a year (estimate {_pct(b06['BH']['cagr_diff'])}); against a mix with "
      f"the same average SPXL share, {_pct(b06['mix_matched']['cagr_diff_ci90'][0])} to "
      f"{_pct(b06['mix_matched']['cagr_diff_ci90'][1])} (estimate {_pct(b06['mix_matched']['cagr_diff'])}). The "
      "realistic edge is indistinguishable from zero.")
    a(f"* **Picking the best of 12 looks like luck.** Shifting all 12 rules by the same number of months (keeping "
      f"their on/off patterns but breaking any link to returns), the best of the 12 did at least as well as "
      f"{pl['actual_best']}'s {_pp(pl['actual_best_cagr'])} in {pl['share_at_or_above_actual']:.0%} of the "
      f"{pl['shifts']} shifts (median best {_pp(pl['median_best'])}).")
    a(f"* **Even the 'bad' months paid.** From 2009, SPXL's average six-month return over T-bills after a favourable "
      f"inflation reading was {_pp(fav06['favourable_mean'])}, and after an unfavourable one "
      f"{_pp(fav06['unfavourable_mean'])}; stepping into T-bills gave that up. For the blend it was the other way round "
      f"({_pp(favc['favourable_mean'])} after favourable readings, {_pp(favc['unfavourable_mean'])} after unfavourable "
      "ones).")
    a("* **Fixed cut-offs broke.** The output gap and valuation never crossed their 1990-2007 cut-offs in 2008-2026 "
      "(their levels shifted), so their rules equal buy-and-hold and the money test could not have confirmed them.")
    a(f"* **Timing and costs.** The macro rules barely change when acted on a month late (C06 "
      f"{_pp(rl['C06']['one_month_late']['cagr'])} against {_pp(rl['C06']['full']['cagr'])}); the VIX-based ones need "
      f"same-day action (C01 {_pp(rl['C01']['full']['cagr'])} on time, {_pp(rl['C01']['one_month_late']['cagr'])} a "
      f"month late; VM {_pp(rl['VM']['full']['cagr'])} and {_pp(rl['VM']['one_month_late']['cagr'])}). At 0.5% per "
      f"switch C01 falls to {_pp(rl['C01']['cost_0p5pct']['cagr'])}.")
    a("")
    a("### 3.6 Volatility-managed exposure against simply holding less SPXL")
    a("")
    rows = []
    wbar = float(res["economic"]["vol_managed"]["w_bar"])

    def vm_label(k: str) -> str:
        if k == "vol_managed":
            return "volatility-managed"
        if k == "BH":
            return "buy and hold SPXL"
        w = float(k.split()[1])
        return f"{w:.0%} SPXL" + (" (the rule's 1990-2007 average weight)" if abs(w - wbar) < 1e-3 else "")
    for label in ("full", "real_spxl"):
        for k, q in ec["vol_managed"][label].items():
            rows.append(["2008-26" if label == "full" else "real SPXL (2009 on)", vm_label(k), _pp(q["cagr"]),
                         _fmt(q["sharpe"], "{:.2f}"), _pp(q["max_drawdown"]), _pp(q["vol"])])
    L += md_table(["period", "strategy", "growth a year", "Sharpe", "worst fall", "volatility"], rows)
    a("")
    vf = ec["vol_managed"]["full"]
    a(f"Over 2008-26 the volatility-managed rule cut the worst fall from {_pp(vf['BH']['max_drawdown'])} to "
      f"{_pp(vf['vol_managed']['max_drawdown'])}, but that is the autumn-2008 crash again. On real SPXL a constant "
      f"55-60% in SPXL matched it on growth and worst fall, with a higher Sharpe and lower volatility (growth / worst "
      f"fall: {_pp(vm09['vol_managed']['cagr'])} / {_pp(vm09['vol_managed']['max_drawdown'])} for the rule, "
      f"{_pp(vm09['constant 0.55']['cagr'])} / {_pp(vm09['constant 0.55']['max_drawdown'])} at 55% and "
      f"{_pp(vm09['constant 0.6']['cagr'])} / {_pp(vm09['constant 0.6']['max_drawdown'])} at 60%). Holding less SPXL "
      "does as well as managing it by volatility.")
    a("")
    a("### 3.7 Taxes (from the practical review; not regenerated by `report`)")
    a("")
    a("All economic figures above and in Appendix A are before tax, as in a tax-deferred account. The practical review "
      "simulated US taxes lot by lot (short- and long-term gains, netting, loss carry-forward, T-bill interest as "
      "income, everything sold at the end); re-running its script reproduced its figures exactly. At a high bracket "
      "(40.8% short-term, 23.8% long-term) buy-and-hold made 16.3% a year (17.9% if never sold), the "
      "variance-risk-premium rule 11.9%, the jobless-claims rule 16.0%, the volatility-managed rule 12.8% and the "
      "blend's rule 9.8%. At a moderate bracket (24% / 15%) the variance-risk-premium rule made 15.2% against "
      "buy-and-hold's 17.0%. The inflation rule kept 23.8% after tax, but its edge is the single 2008 episode (3.5). "
      "In a taxable account, rules that switch often lose 1-6 points a year to tax.")
    a("")
    a("### 3.8 Data checks after the test")
    a("")
    if cv.get("available"):
        pc8, pcc = cv["primary_C08"], cv["primary_COMP"]
        a(f"* **Jobless claims as first published.** C08 used the 2026 version of the claims data, with later "
          f"revisions. Rebuilt from the version available at each date from {cv['first_vintage']} "
          f"({cv['origins_rebuilt']} month-ends), C08 changes by at most {cv['max_abs_change']:.3f} (median "
          f"{cv['median_abs_change']:.4f}); its r moves from {pc8['registered']['r']:+.3f} to "
          f"{pc8['vintage']['r']:+.3f} and the blend's from {pcc['registered']['r']:+.4f} to {pcc['vintage']['r']:+.4f}; "
          f"the blend's rule changes {cv['composite_rule_holdings_changed']} of 223 holdings (C08's own rule "
          f"{cv['c08_rule_holdings_changed']}).")
    if sp.get("available"):
        a(f"* **Real SPXL from its launch.** The plan uses the simulated fund through 2008-12-31 although SPXL traded "
          f"from {sp['spxl_first_close']} (simulated {sp['synthetic_return']:+.1%} against real {sp['spxl_return']:+.1%} "
          f"over those {sp['sessions_replaced']} sessions). Using real SPXL changes {sp['origins_changed']} six-month "
          f"outcomes by at most {sp['max_abs_change_y126']:.1%} and no r by more than {sp['max_abs_change_r']:.4f}; "
          f"Holm passes: {_ids(sp['holm_passes'])}.")
    a("* **Independent rebuild (look-ahead review; not regenerated here).** The reviewer rebuilt C01-C10 from raw sources "
      "with separate code at 22 month-ends (all equal), re-queried FRED and ALFRED for the test period (identical), "
      "confirmed that no month uses an economic figure before its first release, rebuilt the six-month outcomes "
      "without the project's code (equal), and re-ran the test with the current code (every number identical).")
    a("* **Single bad price closes** in the Yahoo data (the S&P 500 total return on 1990-01-24/25 and 2019-07-03/05, "
      "SPXL on 2026-06-26) are each reversed the next day and fall on no window's first or last day, so no outcome "
      "changes (look-ahead review).")
    a("")
    if long_spell:
        a("### 3.9 Where the blend's rule lost")
        a("")
        a(f"The blend stayed at or above its line at every month-end until {ec['composite_first_tbill_origin']}, so the "
          "rule equalled buy-and-hold until then. Its T-bill spells (months decided, and what SPXL and the rule "
          "returned over them):")
        a("")
        L += md_table(["months decided", "months", "SPXL", "rule (T-bills)"],
                      [[f"{d['from']}..{d['to']}" if d["from"] != d["to"] else d["from"], d["months"], _pct(d["spxl"]),
                        _pct(d["rule"])] for d in spells])
        a("")
        mem = ec["composite_members_in_longest_spell"]
        neg = sorted((k for k in mem if mem[k] < 0), key=lambda k: mem[k])
        pos_ = sorted((k for k in mem if mem[k] >= 0), key=lambda k: -mem[k])
        a(f"Average standardised readings over {long_spell['from']}..{long_spell['to']} (positive = favourable): "
          "unfavourable " + ", ".join(f"{PLAIN[k][0].lower()} {mem[k]:+.2f}" for k in neg)
          + "; favourable " + ", ".join(f"{PLAIN[k][0].lower()} {mem[k]:+.2f}" for k in pos_) + ".")
        a("")
    # ------------------------------------------------------------------ 4. limitations
    a("## 4. Limitations")
    a("")
    a("* **Thin data.** 2008-2026 holds about 37 independent six-month windows. A real but weak link would usually be "
      "missed (the plan put the power at a true rank correlation of 0.2 at 7% for the first Holm step). The answer is "
      "'no usable signal shown', not proof that none exists.")
    a("* **Conservative statistics.** The registered count of independent windows ignores how persistent each "
      "indicator is (3.1). That errs towards missing a signal, not towards inventing one; inflation is borderline under "
      "a finer check.")
    a("* **Feedback bias.** Valuation- and VIX-based indicators move with the returns being predicted (3.4).")
    a("* **Not all of 2008-2026 was fresh.** The plan's exposure ledger lists what was known before the test: the old "
      "rating's and the hurdle's full-sample results, VIX spikes followed by rebounds, and the big episodes of "
      "2008-2026. One item was missing from it: before the plan was frozen, a working script had also computed "
      "six-month SPXL excess returns by hurdle level on the SPXL-only months since 2009 (the ledger listed the "
      "full-sample hurdle split and the SPXL-only drawdown levels). It touches only C11, already labelled 'not a clean "
      "out-of-sample test', and C11 failed.")
    a("* **Simulated 2008.** SPXL launched in November 2008; the months before 2009 use a simulated 3x fund financed at "
      "the T-bill rate plus 0.75%. Real crisis funding cost roughly 2-3.5 points more, which would have lowered "
      "buy-and-hold further in 2008 and flattered the rules that stepped aside. The only positive money results depend "
      "on those months.")
    a("* **Before tax.** All money results assume a tax-deferred account (3.7).")
    a("* **Data residuals.** Jobless claims use their latest version, not the first release (3.8); the VIX for 1990-92 "
      "is CBOE's later back-calculation (it sets the VIX-based cut-offs); the 0.10% switching cost is conservative for "
      "SPXL itself.")
    a("* **Fixed cut-offs.** Cut-offs set on 1990-2007 levels were never crossed by the output gap and valuation in "
      "2008-2026, so their money tests were uninformative.")
    a("* **Record-keeping.** The research files are not yet committed, so the order of events (plan frozen 16:49Z, spec "
      "frozen 17:48Z, test 18:14Z, all on 2026-09-27) rests on file times, fingerprints and an editable look log; all "
      "are consistent. The unlock code is the plan's published fingerprint, a guard against accidents rather than a "
      "secret. Scripts that imported the code (the reviews and the audit) were not logged before DEVIATIONS.md N28. The "
      "exact bytes of the code that ran the test were not kept (it was edited three minutes later, for the plot); the "
      "current code reproduces every number (N29). `DEVIATIONS.md` was not fingerprinted at the freeze. The data files "
      "the fingerprints point to are git-ignored; `research/OUTPUT_SHA256.txt` lists them so an archived copy can be "
      "checked.")
    a("* **Approximate checks.** The simulations in 3.1 and 3.4 are models of the no-link case, not exact tests; "
      "they show direction and size.")
    a("* **Spent period.** 2008-2026 has now been used as the test. Any new idea needs new data.")
    a("")
    # ------------------------------------------------------------------ 5. reproduce
    a("## 5. How to reproduce")
    a("")
    a("```bash")
    a("py scripts/research_signals.py build             # the point-in-time panel, 1990-2026 (FRED/ALFRED; --offline uses the cache)")
    a("py scripts/research_signals.py train             # the 1990-2007 quantities and the frozen test spec")
    a("py scripts/research_signals.py test --unseal <SHA-256 of research/PREREGISTRATION.md>   # the Test phase, once")
    a("py scripts/research_signals.py report            # research/test_results.md, this file, research/OUTPUT_SHA256.txt")
    a("py scripts/research_signals.py report --recompute   # ... recomputing the exploratory analyses (about a minute)")
    a("py -m pytest -q")
    a("```")
    a("")
    a("* `test` has already run and refuses to run again, and `train` refuses to run once the test has. `report` "
      "re-renders the registered tables from `output/research/test_results.json` without reading new data. The "
      "exploratory numbers (section 3) come from `output/research/exploratory.json`, which `report` rebuilds from the "
      "saved test outputs whenever they, the script or the settings change; every random draw has a fixed seed.")
    a("* Every run is logged in `output/research/look_log.csv`, and so is every unsealing, whatever script does it.")
    a("* The inputs and outputs are git-ignored (`output/`). `research/OUTPUT_SHA256.txt` lists the SHA-256 of every "
      "file the fingerprint chain points to; with an archived copy of `output/`, `sha256sum -c research/OUTPUT_SHA256.txt` "
      "checks it. A fresh download cannot reproduce the pinned Yahoo prices byte for byte.")
    a("* Code used for this report: " + "; ".join(f"`{k}` {h[:16]}..." for k, h in code.items()) + ".")
    a("")
    # ------------------------------------------------------------------ appendices
    a("## Appendix A. Registered results in full")
    a("")
    a("The registered document, `research/test_results.md`, regenerated from the same JSON (its hand-written notes are "
      "left out; section 3 above replaces them).")
    a("")
    L.extend(_demote(registered))
    a("")
    a("## Appendix B. Independent reviews: every challenge and how it was handled")
    a("")
    a("Three reviews ran after the test: look-ahead and data, statistics, and practical value. None found an error that "
      "changes the verdict. 'Fixed' means the script changed without re-running any registered step or tuning anything "
      "on test results; 'recorded' means the challenge and its effect are reported in this file.")
    a("")
    L += md_table(["review", "challenge", "handled", "effect on the verdict"], REVIEW_RECORD)
    a("")
    return "\n".join(L).rstrip("\n") + "\n"


def cmd_report(args, paths: Optional[Phase3Paths] = None) -> int:
    """Regenerate the documents from the Test phase's saved outputs:
      research/test_results.md  the registered results, re-rendered from output/research/test_results.json
                                (nothing recomputed; a hand-written EXPLORATORY section is kept);
      research/RESULTS.md       the owner's report: plain-language summary, the same registered tables, the
                                exploratory analyses (output/research/exploratory.json, rebuilt from the saved
                                outputs when they or this script change, or with --recompute), limitations,
                                how to reproduce, and the registered document in full as an appendix;
      research/OUTPUT_SHA256.txt  the SHA-256 of every git-ignored file the hash chain points to.
    (Until 2026-09-27 RESULTS.md held the same bytes as test_results.md; DEVIATIONS.md N27.)"""
    paths = Phase3Paths.from_globals() if paths is None else paths
    if not paths.results_json.exists():
        print("no test results yet (run `test` first)", file=sys.stderr)
        return 2
    res = json.loads(paths.results_json.read_text(encoding="utf-8"))
    registered = render_test_results(res)
    text = registered
    if paths.results_md.exists():             # keep a hand-written EXPLORATORY section (not a result)
        old = paths.results_md.read_text(encoding="utf-8")
        k = old.find("\n" + EXPLORATORY_HEADING)
        if k >= 0:
            text = text.rstrip("\n") + "\n" + old[k:]
    write_lf(paths.results_md, text)
    ex = exploratory(paths, recompute=bool(getattr(args, "recompute", False)),
                     mc_reps=int(getattr(args, "mc_reps", None) or MC_REPS),
                     feedback_reps=int(getattr(args, "feedback_reps", None) or FEEDBACK_REPS))
    write_lf(paths.results_md_plan, render_results_report(res, ex, registered, code_provenance()))
    manifest = paths.results_md.parent / MANIFEST_NAME
    write_lf(manifest, output_manifest(paths))
    print(f"wrote {paths.results_md}, {paths.results_md_plan} and {manifest}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the point-in-time research panel (no predictive statistics)")
    b.add_argument("--refresh", action="store_true", help="download the FRED / ALFRED inputs again "
                   "(replaces predictor_inputs.pkl; only before the Train phase records its hash)")
    b.add_argument("--offline", action="store_true", help="never touch the network; fail if a series is not cached")
    b.add_argument("--pit-checks", type=int, default=20, help="train month-ends for the truncation-invariance check")
    tr = sub.add_parser("train", help="Phase 2: set the train quantities from panel_train.pkl and freeze the test spec")
    tr.add_argument("--refreeze", action="store_true", help="allow the frozen spec to change (before unsealing only, "
                    "for a data or code reason recorded in research/DEVIATIONS.md)")
    t = sub.add_parser("test", help="Phase 3: unseal the test period and run every registered test once")
    t.add_argument("--unseal", required=True, help="SHA-256 of research/PREREGISTRATION.md")
    t.add_argument("--after-error", action="store_true", help="only after a run that stopped with an error after "
                   "unsealing, for a bug fix recorded in research/DEVIATIONS.md (original results reported)")
    rp = sub.add_parser("report", help="regenerate research/test_results.md, research/RESULTS.md and "
                        "research/OUTPUT_SHA256.txt from the Test phase's saved outputs")
    rp.add_argument("--recompute", action="store_true", help="recompute the exploratory analyses even if "
                    "output/research/exploratory.json is up to date")
    args = ap.parse_args(argv)
    append_look_log(args.cmd, argv)
    if args.cmd == "build":
        return cmd_build(args)
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "test":
        return cmd_test(args)
    return cmd_report(args)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    sys.exit(main())
