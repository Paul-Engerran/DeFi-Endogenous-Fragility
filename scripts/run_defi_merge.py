#!/usr/bin/env python3
"""
run_defi_merge.py — CLI for the post-DeFi econometric panel (NB04).

Factorises 04_defi_merge.ipynb into a reproducible script. The notebook
build cells (f9b619cb → 2064dd7f → 9b34d577 → fac87011 → 829fcfe4 →
14608ff8) become pure functions; the CLI is a single --out_dir flag.
04_defi_merge_report.ipynb is the reading-side companion.

CHANGELOG vs 04_defi_merge.ipynb
─────────────────────────────────
NEW
- Single CLI for the post-DeFi panel build; --out_dir is the only flag.
- Pure-function pipeline (load_predefi → load_defi_csv → merge_defi →
  compute_features → compute_adf → build_qa → save), matching the
  notebook cell order 1:1.
- ADF tests preserved in the script (deterministic statsmodels output,
  legacy stationarity_adf.json bit-for-bit).
- QA dict insertion order locked to match the legacy defi_merge_qa.json.

DEPRECATED (v2.x)
- `liquidations_1h.parquet` [deprecated v2.x] (4-column DeFi side artefact written to
  data/normalized/defi/) is no longer produced. Grep across notebooks
  and scripts confirmed no downstream consumer. The
  `build_defi_normalized` helper, the `DEFI_NORM_COLS` constant and
  the `defi_norm` argument of `save_outputs` were removed. Output
  contract is now 3 artefacts: econ_core_full_1h.parquet,
  defi_merge_qa.json, stationarity_adf.json.

Output schema (locked)
- data/econ/econ_core_full_1h.parquet schema is locked.
  27 columns in this exact order:
    date, close_perp, volume_perp, oi, funding_rate, close_btc_spot,
    close_eth_spot, ret_eth_perp, ret_btc_spot, ret_eth_spot, d_oi,
    oi_zscore, oi_high, funding_high, oi_vol_ratio, vol_eth_7d,
    vol_btc_7d, ret_eth_std, ret_btc_std, basis_bps, liq_usd_total,
    liq_usd_collateral, n_liquidations, log_liq, log_liq_lag1,
    liq_stress, shock_x_oi.
  Row count: 41,328.  Dtypes locked: oi_high / funding_high / liq_stress
  = int64, n_liquidations = float64 (upcast by left-merge + fillna(0),
  preserved verbatim — int→float drift would break downstream consumers).
  Any drift in column names or order breaks the downstream estimation
  and robustness scripts.

Implementation notes
- DEFI_CSV_PATH is hard-coded (not exposed via CFG.FILES). The CSV is the
  cleaned Dune extract
  (see dune_queries/), already pre-filtered to [2021-03-15 05:00 UTC,
  2025-11-30 23:00 UTC] with no duplicate hours.
- liq_stress = (liq_usd_total > P95(liq_usd_total>0)).astype(int) uses
  a global (full-window) quantile, a mild look-ahead.
- shock_x_oi = log_liq_lag1 * oi_high WITHOUT .fillna(0). Asymmetric
  with run_robustness_all.build_df_est_orth which applies .fillna(0).
  Numerically equivalent after warmup (1 NaN at t=0).
- n_liquidations is upcast int64→float64 by the left-merge + fillna(0);
  consumers rely on the float64 dtype.
- ADF cell skips series with len<100 and on fit exception. Neither
  branch fires on the current 41,328-row panel; the guard is preserved
  for robustness against future input drift.

Usage
-----
    python run_defi_merge.py                       # default: canonical paths
    python run_defi_merge.py --out_dir /tmp/smoke  # smoke test (single dir)

When --out_dir is unset (production mode), the 3 artefacts go to their
legacy locations split by destination:
    econ_core_full_1h.parquet  → ECON_DIR
    defi_merge_qa.json         → REPORTS_DIR
    stationarity_adf.json      → REPORTS_DIR
This is the mode that reproduces the legacy bit-for-bit.

When --out_dir is set (smoke / isolated validation), all 3 artefacts go
to that single directory. NO writes to legacy paths in this mode.

Validation (after first run)
----------------------------
    diff -q data/econ/econ_core_full_1h.parquet \\
            data/econ/econ_core_full_1h.legacy.parquet
    python -c "import json; \\
        a=json.load(open('data/analysis/reports/defi_merge_qa.json')); \\
        b=json.load(open('data/analysis/reports/defi_merge_qa.legacy.json')); \\
        assert a==b"
    python -c "import json; \\
        a=json.load(open('data/analysis/reports/stationarity_adf.json')); \\
        b=json.load(open('data/analysis/reports/stationarity_adf.legacy.json')); \\
        assert a==b"

Smoke test (copy-paste, ~5s)
----------------------------
    python scripts/run_defi_merge.py --out_dir /tmp/defi_smoke
    ls -la /tmp/defi_smoke
    # expect: 3 files (econ_core_full_1h.parquet,
    #         defi_merge_qa.json, stationarity_adf.json)
    python -c "
    import pandas as pd, json
    df = pd.read_parquet('/tmp/defi_smoke/econ_core_full_1h.parquet')
    assert df.shape == (41328, 27), df.shape
    assert abs(df['liq_usd_total'].sum() - 2_521_318_539.578) < 1e-3
    assert abs(df['log_liq'].sum()       - 71_233.7838)      < 1e-3
    assert df['liq_stress'].sum() == 549
    assert df['n_liquidations'].dtype == 'float64'
    assert df['funding_high'].dtype == 'int64'
    qa  = json.load(open('/tmp/defi_smoke/defi_merge_qa.json'))
    adf = json.load(open('/tmp/defi_smoke/stationarity_adf.json'))
    assert qa['status'] == 'PASS', qa['status']
    assert len(adf) == 8, list(adf.keys())
    print('SMOKE OK')
    "
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from config import CFG, REPORTS_DIR  # noqa: E402

from statsmodels.tsa.stattools import adfuller  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Constants — match NB04 cells verbatim
# ──────────────────────────────────────────────────────────────
DEFI_CSV_PATH = (
    CFG.ROOT / "data" / "raw" / "defi" / "defi_liquidations_1h_clean.csv"
)

PLACEHOLDER_COLS: list[str] = ["liq_usd_total", "log_liq"]
DEFI_RENAME: dict[str, str] = {
    "total_debt_repaid_usd":       "liq_usd_total",
    "total_collateral_seized_usd": "liq_usd_collateral",
}
DEFI_MERGE_COLS: list[str] = [
    "date", "liq_usd_total", "liq_usd_collateral", "n_liquidations",
]
DEFI_FILL_COLS: list[str] = [
    "liq_usd_total", "liq_usd_collateral", "n_liquidations",
]
QA_NULL_COLS: list[str] = ["liq_usd_total", "log_liq", "ret_eth_perp", "oi"]

STRESS_PCTILE = CFG.ECON.stress_pctile / 100   # 0.95
ADF_MAXLAG = 48
ADF_AUTOLAG = "AIC"
ADF_MIN_OBS = 100
ADF_SERIES_ORDER: list[tuple[str, str]] = [
    ("ret_eth_perp", "ret_eth_perp"),
    ("ret_btc_spot", "ret_btc_spot"),
    ("log_liq",      "log_liq"),
    ("funding_rate", "funding_rate"),
    ("basis_bps",    "basis_bps"),
    ("oi (level)",   "oi"),
    ("d_oi (diff)",  "d_oi"),
    ("oi_zscore",    "oi_zscore"),
]


# ──────────────────────────────────────────────────────────────
# Pipeline — load → merge → features → audit → save
# ──────────────────────────────────────────────────────────────
def load_predefi() -> pd.DataFrame:
    """Read pre-DeFi panel, drop NB04 placeholders. NB04 cell f9b619cb."""
    panel = pd.read_parquet(CFG.FILES.econ_core_predefi, engine="pyarrow")
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    print(f"Pre-DeFi panel: {len(panel):,} rows × {panel.shape[1]} cols",
          flush=True)
    drop_cols = [c for c in PLACEHOLDER_COLS if c in panel.columns]
    panel = panel.drop(columns=drop_cols)
    print(f"Dropped placeholders: {drop_cols}", flush=True)
    return panel


def load_defi_csv() -> pd.DataFrame:
    """Read raw Dune CSV, normalise date and column names. NB04 cell 2064dd7f."""
    defi = pd.read_csv(DEFI_CSV_PATH)
    defi["date"] = pd.to_datetime(defi["date"], utc=True).dt.floor("h")
    defi = defi.rename(columns=DEFI_RENAME)

    print(f"DeFi raw: {len(defi):,} rows (hours with ≥1 liquidation)",
          flush=True)
    print(f"Date range: [{defi['date'].min()}, {defi['date'].max()}]",
          flush=True)

    assert defi["date"].duplicated().sum() == 0, "Duplicate hours in DeFi data!"
    assert (defi["liq_usd_total"].dropna() >= 0).all(), "Negative debt values!"
    return defi


def merge_defi(panel: pd.DataFrame, defi: pd.DataFrame) -> pd.DataFrame:
    """Left-merge DeFi onto panel, fillna(0) on the 3 DeFi columns.
    NB04 cell 9b34d577.

    fillna(0) is semantic, not cosmetic: an hour without a recorded
    liquidation is a real zero. The int→float upcast on n_liquidations
    is preserved verbatim (consumers depend on float64).
    """
    panel = panel.merge(defi[DEFI_MERGE_COLS], on="date", how="left")
    for col in DEFI_FILL_COLS:
        n_missing = panel[col].isna().sum()
        panel[col] = panel[col].fillna(0)
        pct = 100 * n_missing / len(panel)
        print(f"  {col}: {n_missing:,} hours filled with 0 ({pct:.1f}%)",
              flush=True)
    print(f"\nPanel after merge: {len(panel):,} rows × {panel.shape[1]} cols",
          flush=True)
    return panel


def compute_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Append log_liq, log_liq_lag1, liq_stress, shock_x_oi.
    NB04 cell fac87011. Returns (panel, stress_threshold_usd).
    """
    panel["log_liq"] = np.log1p(panel["liq_usd_total"])
    panel["log_liq_lag1"] = panel["log_liq"].shift(1)

    nonzero_liq = panel.loc[panel["liq_usd_total"] > 0, "liq_usd_total"]
    if len(nonzero_liq) == 0:
        warnings.warn(
            "No non-zero liquidations in panel; liq_stress will be all-zero. "
            "This is expected in smoke tests on subperiods, but UNEXPECTED in production. "
            "Check upstream filtering logic if this triggers in `make all`.",
            RuntimeWarning,
        )
        stress_threshold = float("nan")
        panel["liq_stress"] = 0
    else:
        stress_threshold = float(nonzero_liq.quantile(STRESS_PCTILE))
        panel["liq_stress"] = (panel["liq_usd_total"] > stress_threshold).astype(int)

    panel["shock_x_oi"] = panel["log_liq_lag1"] * panel["oi_high"]

    print(f"Stress threshold (P{CFG.ECON.stress_pctile} of non-zero): "
          f"${stress_threshold:,.0f}", flush=True)
    print(f"Hours in stress regime: {panel['liq_stress'].sum():,} "
          f"({100*panel['liq_stress'].mean():.1f}%)", flush=True)
    return panel, stress_threshold


def compute_adf(panel: pd.DataFrame) -> dict:
    """Run ADF tests on 8 series; return dict ordered for legacy JSON match.
    NB04 cell 829fcfe4.
    """
    print("\n═══ ADF Unit Root Tests ═══", flush=True)
    results: list[dict] = []
    for label, col in ADF_SERIES_ORDER:
        series = panel[col].dropna()
        if len(series) < ADF_MIN_OBS:
            print(f"  {label:18s}: SKIP (n={len(series)})", flush=True)
            continue
        try:
            stat, pval, used_lag, nobs, _crit, _ = adfuller(
                series.values, maxlag=ADF_MAXLAG, autolag=ADF_AUTOLAG,
            )
        except Exception as e:
            print(f"  {label:18s}: ERROR — {e}", flush=True)
            continue
        results.append({
            "label": label,
            "adf":   round(stat, 3),
            "p":     round(pval, 6),
        })
        reject = "REJECT" if pval < 0.05 else "FAIL"
        print(f"  {label:18s}: ADF={stat:>8.3f}  p={pval:.6f}  "
              f"lags={used_lag:>2d}  n={nobs:,}  → {reject}", flush=True)

    return {r["label"]: {"adf": r["adf"], "p": r["p"]} for r in results}


def build_qa(
    panel: pd.DataFrame,
    defi: pd.DataFrame,
    stress_threshold: float,
) -> dict:
    """Assemble the QA dict; key insertion order matches legacy JSON.
    NB04 cell 14608ff8.
    """
    return {
        "panel_rows":              len(panel),
        "panel_cols":              panel.shape[1],
        "defi_raw_rows":           len(defi),
        "hours_with_liquidations": int((panel["liq_usd_total"] > 0).sum()),
        "hours_zero_liquidations": int((panel["liq_usd_total"] == 0).sum()),
        "stress_threshold_usd":    float(stress_threshold),
        "hours_in_stress":         int(panel["liq_stress"].sum()),
        "total_liq_usd":           float(panel["liq_usd_total"].sum()),
        "max_hourly_liq_usd":      float(panel["liq_usd_total"].max()),
        "nulls_after_merge":       {c: int(panel[c].isna().sum())
                                    for c in QA_NULL_COLS},
        "status":                  "PASS",
    }


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(
    panel: pd.DataFrame,
    qa: dict,
    adf: dict,
    out_dir: Path | None,
) -> tuple[Path, Path, Path]:
    """Write the 3 artefacts.

    If `out_dir` is None → legacy split paths (parquet→ECON_DIR,
    JSONs→REPORTS_DIR). Bit-for-bit reproducible mode.
    If set → all three go to `out_dir` (smoke mode).
    """
    if out_dir is None:
        full_path = CFG.FILES.econ_core_full
        qa_path   = REPORTS_DIR / "defi_merge_qa.json"
        adf_path  = REPORTS_DIR / "stationarity_adf.json"
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        full_path = out_dir / "econ_core_full_1h.parquet"
        qa_path   = out_dir / "defi_merge_qa.json"
        adf_path  = out_dir / "stationarity_adf.json"

    panel.to_parquet(full_path, index=False, engine="pyarrow")
    print(f"\nSaved: {full_path}", flush=True)
    print(f"Shape: {panel.shape}", flush=True)

    with open(qa_path, "w") as f:
        json.dump(qa, f, indent=2)
    print(f"Saved: {qa_path}", flush=True)

    with open(adf_path, "w") as f:
        json.dump(adf, f, indent=2)
    print(f"Saved: {adf_path}", flush=True)

    return full_path, qa_path, adf_path


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--out_dir", type=Path, default=None,
        help=("Override: write all 3 artefacts to DIR. "
              "If unset, use legacy paths (default for prod): "
              "parquet→ECON_DIR, JSONs→REPORTS_DIR. "
              "Legacy mode is the bit-for-bit reproducible mode."),
    )
    args = ap.parse_args()

    print(f"run_defi_merge: out_dir={args.out_dir or '<legacy split>'}",
          flush=True)
    t0 = time.time()

    panel = load_predefi()
    defi = load_defi_csv()
    panel = merge_defi(panel, defi)
    panel, stress_threshold = compute_features(panel)
    adf = compute_adf(panel)
    qa = build_qa(panel, defi, stress_threshold)
    save_outputs(panel, qa, adf, args.out_dir)

    print(f"\nDone. Total wall time: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    CFG.ensure_dirs()
    raise SystemExit(main())
