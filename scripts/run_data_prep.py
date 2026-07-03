#!/usr/bin/env python3
"""
run_data_prep.py — CLI for the master calendar + CEX diagnostics
                   build (NB01 + NB02).

Factorises 01_calendar.ipynb + 02_diagnostics.ipynb into a single
reproducible script. The notebook build cells become pure functions;
the CLI is a single --out_dir flag. NB02 reads the calendar produced
by NB01, so the two notebooks are merged here in a single linear
pipeline (calendar → diagnostics). 01_02_data_prep_report.ipynb is
the reading-side companion.

CHANGELOG vs 01_calendar.ipynb + 02_diagnostics.ipynb
─────────────────────────────────────────────────────
NEW
- Single CLI for the upstream data prep; --out_dir is the only flag.
- Pure-function pipeline (compute_calendar_bounds → build_master_calendar
  → save_calendar_outputs → load_venues → build_diagnostics_panel →
  compute_spread_stats → build_diagnostics_qa → save_diagnostics_outputs),
  matching the cell order 1:1 across both notebooks.
- QA dict insertion order locked to match the legacy calendar_qa.json
  and diagnostics_qa.json bit-for-bit.

CONTRACT WITH NB03 / run_core_panel.py (do not break)
- master_calendar_1h.parquet: 1 column (`date` UTC), 41,328 rows.
- window_metadata.json: 4 top-level keys (full_window, core_window,
  series_bounds, convention) — series_bounds preserved with its
  pre-`+1h` upper bounds (verbatim).
- calendar_qa.json, cex_diagnostics_1h.parquet, diagnostics_qa.json
  are diagnostic / audit terminals (no downstream consumer); preserved
  bit-for-bit for legacy parity.
- Any drift on master_calendar / window_metadata breaks NB03 → NB04
  → NB05 / NB07 / NB08 / NB09 + the factorised scripts.

METHODOLOGICAL NOTES (known quirks, preserved verbatim)
- oi_col / fund_col picked as "first non-date column" of the upstream
  bybit_oi / bybit_funding parquets (NB02 cell 08922d96). Same fragile
  pattern as run_core_panel.py NB03; preserved verbatim.
- The legacy loader's never-used `cols_rename` parameter (NB02 cell
  08922d96, always None at all 4 call sites) was dropped in the
  src.io consolidation.
- Cosmetic display/QA threshold inconsistency at NB02 cell 344eea2a:
  display prints "Check for micro-structure differences" when corr
  ≤ 0.999, while QA `status` flips at 0.99. With the current
  corr ≈ 0.99895, display says "Check…" but QA says "PASS". No
  artefact impact. Preserved verbatim.
- full_window is anchored to bybit_klines bounds only ("broadest
  Bybit series"); core_window is the strict 3-series intersection.
  In practice the 3 series have identical bounds → full ≡ core
  on the current data. The asymmetry is a no-op but codified;
  preserved verbatim.
- Ordering constraint: save_calendar_outputs() MUST run before
  load_venues() — the latter re-reads CFG.FILES.master_calendar
  from disk. main() enforces this; no skip-diagnostics flag exposed.

Usage
-----
    python run_data_prep.py                       # default: legacy paths
    python run_data_prep.py --out_dir /tmp/smoke  # smoke test (single dir)

When --out_dir is unset (production mode), the 5 artefacts go to
their legacy locations split by destination:
    master_calendar_1h.parquet   → WINDOWS_DIR
    window_metadata.json         → WINDOWS_DIR
    calendar_qa.json             → REPORTS_DIR
    cex_diagnostics_1h.parquet   → DATASETS_DIR
    diagnostics_qa.json          → REPORTS_DIR
This is the mode that reproduces the legacy bit-for-bit.

When --out_dir is set (smoke / isolated validation), all 5 artefacts
go to that single directory. NO writes to legacy paths in this mode.

Validation (after first run)
----------------------------
    diff -q data/analysis/windows/master_calendar_1h.parquet \\
            data/analysis/windows/master_calendar_1h.legacy.parquet
    python -c "import json; \\
        a=json.load(open('data/analysis/windows/window_metadata.json')); \\
        b=json.load(open('data/analysis/windows/window_metadata.legacy.json')); \\
        assert a==b"
    # …same for calendar_qa.json, cex_diagnostics, diagnostics_qa.

Smoke test (copy-paste, ~2s)
----------------------------
    python scripts/run_data_prep.py --out_dir /tmp/prep_smoke
    ls -la /tmp/prep_smoke
    # expect: 5 files (master_calendar_1h.parquet, window_metadata.json,
    #         calendar_qa.json, cex_diagnostics_1h.parquet,
    #         diagnostics_qa.json)
    python -c "
    import pandas as pd, json
    cal = pd.read_parquet('/tmp/prep_smoke/master_calendar_1h.parquet')
    assert len(cal) == 41328 and cal['date'].is_monotonic_increasing
    cex = pd.read_parquet('/tmp/prep_smoke/cex_diagnostics_1h.parquet')
    assert cex.shape == (41328, 12)
    assert abs(cex['price_spread_bps'].abs().mean() - 1.9775) < 1e-3
    assert cex[['ret_bybit', 'ret_binance']].dropna().corr().iloc[0,1] > 0.998
    wm = json.load(open('/tmp/prep_smoke/window_metadata.json'))
    assert wm['full_window']['n_hours'] == 41328
    qa1 = json.load(open('/tmp/prep_smoke/calendar_qa.json'))
    qa2 = json.load(open('/tmp/prep_smoke/diagnostics_qa.json'))
    assert qa1['status'] == 'PASS' and qa2['status'] == 'PASS'
    print('SMOKE OK')
    "
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from config import CFG, REPORTS_DIR  # noqa: E402
from src.io import load_utc_parquet as _load_parquet  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Constants — match NB01 / NB02 cells verbatim
# ──────────────────────────────────────────────────────────────
SERIES_FOR_BOUNDS: list[tuple[str, Path]] = [
    ("bybit_klines",  CFG.FILES.bybit_klines),
    ("bybit_funding", CFG.FILES.bybit_funding),
    ("bybit_oi",      CFG.FILES.bybit_oi),
]

KLINES_OHLCV_COLS: list[str] = ["date", "open", "high", "low", "close", "volume"]
DIAG_MISSING_COLS: list[str] = [
    "close_bybit", "close_binance", "oi_bybit", "funding_bybit",
]
DIAG_PASS_THRESHOLD = 0.99   # NB02 QA `status` threshold (verbatim)


@dataclass
class _Paths:
    """Resolved output paths for the 5 artefacts produced by this script."""
    master_calendar: Path
    window_metadata: Path
    calendar_qa:     Path
    cex_diagnostics: Path
    diagnostics_qa:  Path


def resolve_paths(out_dir: Path | None) -> _Paths:
    """Return the 5 output paths.

    If `out_dir` is None → legacy split paths (master_calendar /
    window_metadata→WINDOWS_DIR, cex_diagnostics→DATASETS_DIR,
    JSONs→REPORTS_DIR). Bit-for-bit reproducible mode.
    If set → all five go to `out_dir` (smoke mode).
    """
    if out_dir is None:
        return _Paths(
            master_calendar=CFG.FILES.master_calendar,
            window_metadata=CFG.FILES.window_metadata,
            calendar_qa=REPORTS_DIR / "calendar_qa.json",
            cex_diagnostics=CFG.FILES.cex_diagnostics,
            diagnostics_qa=REPORTS_DIR / "diagnostics_qa.json",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    return _Paths(
        master_calendar=out_dir / "master_calendar_1h.parquet",
        window_metadata=out_dir / "window_metadata.json",
        calendar_qa=out_dir / "calendar_qa.json",
        cex_diagnostics=out_dir / "cex_diagnostics_1h.parquet",
        diagnostics_qa=out_dir / "diagnostics_qa.json",
    )


# ──────────────────────────────────────────────────────────────
# Stage 1 — Master calendar (NB01)
# ──────────────────────────────────────────────────────────────
def _get_date_bounds(path: Path, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Read only the date column, return (min, max) aligned to the hour.
    NB01 cell 54d050e4 helper.
    """
    dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"], utc=True)
    lo, hi = dates.min().floor("h"), dates.max().floor("h")
    print(f"  {name:20s}  {lo}  →  {hi}  ({len(dates):,} rows)", flush=True)
    return lo, hi


def compute_calendar_bounds() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Read (min, max) hourly bounds from each Bybit series.
    NB01 cell 54d050e4.
    """
    print("Series bounds:", flush=True)
    return {name: _get_date_bounds(path, name) for name, path in SERIES_FOR_BOUNDS}


def build_master_calendar(
    bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[pd.DataFrame, dict, dict]:
    """Build the hourly calendar + window metadata + QA dict.
    NB01 cells e143dac8 + 8b739e4b + ebe46c31 (in-memory parts).

    Returns (calendar, window_metadata, calendar_qa).

    The full window is anchored to klines (broadest Bybit series); the
    core window is the strict 3-series intersection. In practice the
    bounds are identical here — the asymmetry is a no-op but codified.
    """
    # 2. Compute windows
    full_start = bounds["bybit_klines"][0]
    full_end   = bounds["bybit_klines"][1] + pd.Timedelta(hours=1)
    core_start = max(b[0] for b in bounds.values())
    core_end   = min(b[1] for b in bounds.values()) + pd.Timedelta(hours=1)

    print(f"FULL window: [{full_start}, {full_end})", flush=True)
    print(f"CORE window: [{core_start}, {core_end})", flush=True)

    assert core_start >= full_start
    assert core_end   <= full_end

    # 3. Build calendar
    calendar = pd.DataFrame({
        "date": pd.date_range(
            full_start, full_end, freq="1h", tz="UTC", inclusive="left",
        )
    })
    n_expected = int((full_end - full_start) / pd.Timedelta(hours=1))
    n_actual = len(calendar)

    print(f"Calendar rows: {n_actual:,}  (expected: {n_expected:,})", flush=True)
    assert n_actual == n_expected, f"Mismatch! {n_actual} != {n_expected}"
    diffs = calendar["date"].diff().dropna()
    assert diffs.eq(pd.Timedelta(hours=1)).all(), "Non-uniform spacing detected!"
    assert calendar["date"].is_monotonic_increasing, "Not monotonic!"
    assert str(calendar["date"].dt.tz) == "UTC", "Not UTC!"

    # 4a. Window metadata — key insertion order locked
    metadata = {
        "full_window": {
            "start": full_start.isoformat(),
            "end_excl": full_end.isoformat(),
            "n_hours": n_actual,
        },
        "core_window": {
            "start": core_start.isoformat(),
            "end_excl": core_end.isoformat(),
            "n_hours": int((core_end - core_start) / pd.Timedelta(hours=1)),
        },
        "series_bounds": {
            k: {"start": v[0].isoformat(), "end": v[1].isoformat()}
            for k, v in bounds.items()
        },
        "convention": "timestamp = bucket_start, window = [start, end_excl)",
    }

    # 4b. QA dict — key insertion order locked
    qa = {
        "n_hours": n_actual,
        "expected": n_expected,
        "monotonic": True,
        "uniform_1h": True,
        "timezone": "UTC",
        "status": "PASS",
    }
    return calendar, metadata, qa


def save_calendar_outputs(
    calendar: pd.DataFrame,
    metadata: dict,
    qa: dict,
    paths: _Paths,
) -> None:
    """Write master_calendar parquet + window_metadata + calendar_qa JSON.
    NB01 cell ebe46c31. MUST run before load_venues() (which re-reads
    paths.master_calendar from disk).
    """
    calendar.to_parquet(paths.master_calendar, index=False, engine="pyarrow")
    print(f"Saved: {paths.master_calendar}", flush=True)

    with open(paths.window_metadata, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {paths.window_metadata}", flush=True)

    with open(paths.calendar_qa, "w") as f:
        json.dump(qa, f, indent=2)
    print(f"Saved: {paths.calendar_qa}", flush=True)


# ──────────────────────────────────────────────────────────────
# Stage 2 — CEX diagnostics (NB02)
# ──────────────────────────────────────────────────────────────
def load_venues(master_calendar_path: Path) -> dict[str, pd.DataFrame]:
    """Load the freshly-saved calendar + 4 venues, standardise column
    names. NB02 cells 5b51f2d0 + 08922d96.

    The OI and funding parquets contribute their *first non-date column*
    (renamed to `oi_bybit` and `funding_bybit`). Preserved verbatim.
    """
    cal = pd.read_parquet(master_calendar_path, engine="pyarrow")
    cal["date"] = pd.to_datetime(cal["date"], utc=True)
    print(f"Calendar: {len(cal):,} hours", flush=True)

    # Bybit klines — keep OHLCV, suffix non-date columns with _bybit
    bybit_k = _load_parquet(CFG.FILES.bybit_klines)
    bybit_k = bybit_k[KLINES_OHLCV_COLS].copy()
    bybit_k.columns = ["date"] + [f"{c}_bybit" for c in bybit_k.columns[1:]]

    # Bybit OI — first non-date column → oi_bybit
    bybit_oi = _load_parquet(CFG.FILES.bybit_oi)
    oi_col = [c for c in bybit_oi.columns if c != "date"][0]
    bybit_oi = bybit_oi[["date", oi_col]].rename(columns={oi_col: "oi_bybit"})

    # Bybit funding — first non-date column → funding_bybit
    bybit_f = _load_parquet(CFG.FILES.bybit_funding)
    fund_col = [c for c in bybit_f.columns if c != "date"][0]
    bybit_f = bybit_f[["date", fund_col]].rename(columns={fund_col: "funding_bybit"})

    # Binance futures — close → close_binance
    binance = _load_parquet(CFG.FILES.binance_futures)
    binance = binance[["date", "close"]].rename(columns={"close": "close_binance"})

    print("All sources loaded:", flush=True)
    for name, df in [("bybit_klines", bybit_k), ("bybit_oi", bybit_oi),
                     ("bybit_funding", bybit_f), ("binance", binance)]:
        print(f"  {name:18s}: {len(df):,} rows, "
              f"[{df['date'].min()}, {df['date'].max()}]", flush=True)

    return {"cal": cal, "bybit_k": bybit_k, "bybit_oi": bybit_oi,
            "bybit_f": bybit_f, "binance": binance}


def build_diagnostics_panel(venues: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join 4 venues onto the calendar, append derived columns.
    NB02 cell 4b1caca4.

    Final 12-column schema (order locked):
      date, open_bybit, high_bybit, low_bybit, close_bybit, volume_bybit,
      oi_bybit, funding_bybit, close_binance, price_spread_bps,
      ret_bybit, ret_binance.
    """
    panel = venues["cal"].copy()
    for df in [venues["bybit_k"], venues["bybit_oi"],
               venues["bybit_f"], venues["binance"]]:
        panel = panel.merge(df, on="date", how="left")

    panel["price_spread_bps"] = (
        1e4 * (panel["close_bybit"] - panel["close_binance"])
        / panel["close_binance"]
    )
    panel["ret_bybit"]   = np.log(panel["close_bybit"]).diff()
    panel["ret_binance"] = np.log(panel["close_binance"]).diff()

    print(f"Panel shape: {panel.shape}", flush=True)
    print("\nMissing values:", flush=True)
    for c in DIAG_MISSING_COLS:
        n_miss = panel[c].isna().sum()
        print(f"  {c:20s}: {n_miss:,}  ({100*n_miss/len(panel):.2f}%)",
              flush=True)
    return panel


def compute_spread_stats(panel: pd.DataFrame) -> tuple[float, float, float]:
    """Print the spread-diagnostics block, return (spread_mean, spread_std,
    return_correlation). NB02 cell 344eea2a.

    The display threshold (0.999) and the QA threshold (0.99) are
    intentionally different (cosmetic-vs-contract) — preserved verbatim.
    """
    spread = panel["price_spread_bps"].dropna()

    print("═" * 55, flush=True)
    print("PRICE SPREAD: BYBIT vs BINANCE (bps)", flush=True)
    print("═" * 55, flush=True)
    print(f"Mean (bias)        : {spread.mean():.4f} bps", flush=True)
    print(f"Std (volatility)   : {spread.std():.4f} bps", flush=True)
    print(f"Mean |spread|      : {spread.abs().mean():.4f} bps", flush=True)
    print(f"Median |spread|    : {spread.abs().median():.4f} bps", flush=True)
    print(f"P99 |spread|       : {np.percentile(spread.abs(), 99):.2f} bps",
          flush=True)
    print(f"P99.9 |spread|     : {np.percentile(spread.abs(), 99.9):.2f} bps",
          flush=True)

    corr = panel[["ret_bybit", "ret_binance"]].dropna().corr().iloc[0, 1]
    print(f"\nReturn correlation : {corr:.6f}", flush=True)
    print(
        "→ Confirms near-perfect market integration"
        if corr > 0.999 else "→ Check for micro-structure differences",
        flush=True,
    )
    return float(spread.mean()), float(spread.std()), float(corr)


def build_diagnostics_qa(
    panel: pd.DataFrame,
    stats: tuple[float, float, float],
) -> dict:
    """Assemble the diagnostics QA dict; key insertion order locked
    on the legacy diagnostics_qa.json. NB02 cell fd675e55.
    """
    spread_mean, spread_std, corr = stats
    return {
        "panel_rows":             len(panel),
        "missing_bybit_close":    int(panel["close_bybit"].isna().sum()),
        "missing_binance_close":  int(panel["close_binance"].isna().sum()),
        "spread_mean_bps":        spread_mean,
        "spread_std_bps":         spread_std,
        "return_correlation":     corr,
        "status":                 "PASS" if corr > DIAG_PASS_THRESHOLD else "CHECK",
    }


def save_diagnostics_outputs(
    panel: pd.DataFrame,
    qa: dict,
    paths: _Paths,
) -> None:
    """Write cex_diagnostics parquet + diagnostics_qa JSON.
    NB02 cell fd675e55.
    """
    panel.to_parquet(paths.cex_diagnostics, index=False, engine="pyarrow")
    print(f"\nSaved: {paths.cex_diagnostics}", flush=True)

    with open(paths.diagnostics_qa, "w") as f:
        json.dump(qa, f, indent=2)
    print(f"Saved: {paths.diagnostics_qa}", flush=True)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--out_dir", type=Path, default=None,
        help=("Override: write all 5 artefacts to DIR. "
              "If unset, use legacy paths (default for prod): "
              "master_calendar / window_metadata→WINDOWS_DIR, "
              "cex_diagnostics→DATASETS_DIR, JSONs→REPORTS_DIR. "
              "Legacy mode is the bit-for-bit reproducible mode."),
    )
    args = ap.parse_args()

    print(f"run_data_prep: out_dir={args.out_dir or '<legacy split>'}",
          flush=True)
    t0 = time.time()

    paths = resolve_paths(args.out_dir)

    # Stage 1 — calendar (NB01)
    bounds = compute_calendar_bounds()
    calendar, metadata, qa_cal = build_master_calendar(bounds)
    save_calendar_outputs(calendar, metadata, qa_cal, paths)

    # Stage 2 — diagnostics (NB02). Re-reads paths.master_calendar.
    venues = load_venues(paths.master_calendar)
    panel = build_diagnostics_panel(venues)
    stats = compute_spread_stats(panel)
    qa_diag = build_diagnostics_qa(panel, stats)
    save_diagnostics_outputs(panel, qa_diag, paths)

    print(f"\nDone. Total wall time: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    CFG.ensure_dirs()
    raise SystemExit(main())
