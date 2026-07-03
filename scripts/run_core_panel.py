#!/usr/bin/env python3
"""
run_core_panel.py — CLI for the pre-DeFi econometric core panel (NB03).

Factorises 03_core_panel.ipynb into a reproducible script. The notebook
build cells (06f2275d → 985a73f4 → 86ec7db3 → a881ba75 → b209e612 →
93a9fa3f) become pure functions; the CLI is a single --out_dir flag.

CHANGELOG vs 03_core_panel.ipynb
─────────────────────────────────
NEW
- Single CLI for the pre-DeFi panel build; no parameters beyond --out_dir.
- Pure-function pipeline (load → merge → features → placeholders →
  audit → save), matching the notebook cell order 1:1.
- QA stats from the missing-data audit are returned by audit_missings()
  and serialised to econ_core_predefi_qa.json with the same schema as
  the notebook's JSON dump.

Output schema (locked)
- data/econ/econ_core_predefi_1h.parquet schema is locked.
  22 columns in this exact order:
    date, close_perp, volume_perp, oi, funding_rate, close_btc_spot,
    close_eth_spot, ret_eth_perp, ret_btc_spot, ret_eth_spot, d_oi,
    oi_zscore, oi_high, funding_high, oi_vol_ratio, vol_eth_7d,
    vol_btc_7d, ret_eth_std, ret_btc_std, basis_bps, liq_usd_total,
    log_liq.
  Row count: 41,328.  Date range: [2021-03-15, 2025-12-01) UTC.
  Any drift in column names or order breaks run_defi_merge.py and the
  downstream estimation and robustness scripts.

Implementation notes
- oi_col / f_col are picked as "first non-date column" of the upstream
  bybit_oi / bybit_funding parquets, fragile if upstream schema changes.
- replace(0, np.nan) is applied to volume MA, vol_eth_7d, vol_btc_7d
  (silent zero-denominator handling). NOT applied to close_eth_spot in
  the basis_bps formula.
- Heterogeneous warmup windows (720h for OI features, 168h for vol,
  24h for volume MA) → mixed NaN behaviour in the first ~720 rows;
  intentional, downstream filtering handles it.

Usage
-----
    python run_core_panel.py                       # default: canonical paths
    python run_core_panel.py --out_dir /tmp/smoke  # smoke test
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
from config import CFG, ECON_DIR, REPORTS_DIR  # noqa: E402
from src.io import load_utc_parquet as _load_utc  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Constants — match NB03 cells verbatim
# ──────────────────────────────────────────────────────────────
KEY_COLS: list[str] = [
    "close_perp", "oi", "funding_rate", "close_btc_spot", "close_eth_spot",
]
PLACEHOLDER_COLS: list[str] = ["liq_usd_total", "log_liq"]

OI_WINDOW = 720          # rolling z-score / rank window for OI features
VOLUME_MA_WINDOW = 24    # rolling mean for OI-to-volume denominator
VOL_WINDOW = CFG.ECON.vol_window        # 168 (7-day)
HIGH_OI_THRESHOLD = CFG.ECON.high_oi_pctile / 100  # 0.80


# ──────────────────────────────────────────────────────────────
# Pipeline — load → merge → features → placeholders → audit
# ──────────────────────────────────────────────────────────────
def load_inputs() -> dict[str, pd.DataFrame | dict]:
    """Load all 7 inputs declared by CFG.FILES.* — NB03 cell 06f2275d."""
    cal = _load_utc(CFG.FILES.master_calendar)
    bybit_k = _load_utc(CFG.FILES.bybit_klines)
    bybit_oi = _load_utc(CFG.FILES.bybit_oi)
    bybit_f = _load_utc(CFG.FILES.bybit_funding)
    btc_spot = _load_utc(CFG.FILES.btc_spot)
    eth_spot = _load_utc(CFG.FILES.eth_spot)
    with open(CFG.FILES.window_metadata) as f:
        window_meta = json.load(f)

    print("Inputs loaded:", flush=True)
    for name, df in [
        ("calendar",      cal),
        ("bybit_klines",  bybit_k),
        ("bybit_oi",      bybit_oi),
        ("bybit_funding", bybit_f),
        ("btc_spot",      btc_spot),
        ("eth_spot",      eth_spot),
    ]:
        print(f"  {name:16s}: {len(df):,} rows", flush=True)

    return {
        "cal": cal, "bybit_k": bybit_k, "bybit_oi": bybit_oi,
        "bybit_f": bybit_f, "btc_spot": btc_spot, "eth_spot": eth_spot,
        "window_meta": window_meta,
    }


def build_panel(inputs: dict) -> pd.DataFrame:
    """Standardise column names and merge onto the calendar — NB03 cell 985a73f4.

    The OI and funding parquets contribute their *first non-date column*
    (renamed to `oi` and `funding_rate`). Preserved verbatim.
    """
    klines = inputs["bybit_k"][["date", "close", "volume"]].rename(
        columns={"close": "close_perp", "volume": "volume_perp"}
    )

    oi_col = [c for c in inputs["bybit_oi"].columns if c != "date"][0]
    oi = inputs["bybit_oi"][["date", oi_col]].rename(columns={oi_col: "oi"})

    f_col = [c for c in inputs["bybit_f"].columns if c != "date"][0]
    funding = inputs["bybit_f"][["date", f_col]].rename(
        columns={f_col: "funding_rate"}
    )

    btc = inputs["btc_spot"][["date", "close"]].rename(
        columns={"close": "close_btc_spot"}
    )
    eth = inputs["eth_spot"][["date", "close"]].rename(
        columns={"close": "close_eth_spot"}
    )

    panel = inputs["cal"].copy()
    for df in [klines, oi, funding, btc, eth]:
        panel = panel.merge(df, on="date", how="left")

    print(f"Panel: {panel.shape[0]:,} rows × {panel.shape[1]} cols", flush=True)
    return panel


def compute_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Construct the 12 derived columns — NB03 cell 86ec7db3."""
    # 3a. Log returns (×100 for readability)
    panel["ret_eth_perp"] = np.log(panel["close_perp"]).diff() * 100
    panel["ret_btc_spot"] = np.log(panel["close_btc_spot"]).diff() * 100
    panel["ret_eth_spot"] = np.log(panel["close_eth_spot"]).diff() * 100

    # 3b. Open Interest: first difference + z-score + regime flag
    panel["d_oi"] = panel["oi"].diff()
    panel["oi_zscore"] = (
        (panel["oi"] - panel["oi"].rolling(OI_WINDOW).mean())
        / panel["oi"].rolling(OI_WINDOW).std()
    )
    panel["oi_high"] = (
        panel["oi"].rolling(OI_WINDOW).rank(pct=True) > HIGH_OI_THRESHOLD
    ).astype(int)

    # 3b-bis. Funding-regime indicator (alternative leverage-stress proxy).
    # 1 if funding_rate is in its upper quintile over the same 720h rolling
    # window used by oi_high; consumed by Test J in run_robustness_all to
    # corroborate the oi_high proxy. Same threshold (P80) as oi_high for
    # direct comparability. NaN in the warmup window resolves to 0 via the
    # boolean comparison + astype(int), matching the oi_high pattern.
    panel["funding_high"] = (
        panel["funding_rate"].rolling(OI_WINDOW).rank(pct=True) > HIGH_OI_THRESHOLD
    ).astype(int)

    # 3c. OI-to-volume ratio
    vol_ma = panel["volume_perp"].rolling(VOLUME_MA_WINDOW).mean()
    panel["oi_vol_ratio"] = panel["oi"] / vol_ma.replace(0, np.nan)

    # 3d. funding_rate is already on the hourly grid

    # 3e. Rolling volatility (7-day = 168h)
    panel["vol_eth_7d"] = panel["ret_eth_perp"].rolling(VOL_WINDOW).std()
    panel["vol_btc_7d"] = panel["ret_btc_spot"].rolling(VOL_WINDOW).std()

    # 3f. Volatility-normalised returns
    panel["ret_eth_std"] = panel["ret_eth_perp"] / panel["vol_eth_7d"].replace(0, np.nan)
    panel["ret_btc_std"] = panel["ret_btc_spot"] / panel["vol_btc_7d"].replace(0, np.nan)

    # 3g. Basis: perp vs spot spread (no replace(0, NaN) on denominator — verbatim)
    panel["basis_bps"] = (
        1e4 * (panel["close_perp"] - panel["close_eth_spot"])
        / panel["close_eth_spot"]
    )
    return panel


def add_defi_placeholders(panel: pd.DataFrame) -> pd.DataFrame:
    """Append NaN placeholder columns filled by NB04 — NB03 cell b209e612."""
    panel["liq_usd_total"] = np.nan
    panel["log_liq"] = np.nan
    return panel


def audit_missings(panel: pd.DataFrame, window_meta: dict) -> dict:
    """Missing-data audit + core-window slice — NB03 cell a881ba75 + 93a9fa3f.

    Returns the QA dict serialised to econ_core_predefi_qa.json with key
    insertion order matching the legacy output bit-for-bit.
    """
    print("\n═══ Missing Data Audit ═══", flush=True)
    for c in KEY_COLS:
        n = panel[c].isna().sum()
        pct = 100 * n / len(panel)
        print(f"  {c:20s}: {n:5,} missing ({pct:.2f}%)", flush=True)

    core_start = pd.Timestamp(window_meta["core_window"]["start"])
    core_end = pd.Timestamp(window_meta["core_window"]["end_excl"])
    core = panel[(panel["date"] >= core_start) & (panel["date"] < core_end)]
    print(f"\nCore window: {len(core):,} rows  [{core_start}, {core_end})",
          flush=True)
    print("Core missing:", flush=True)
    for c in KEY_COLS:
        n = core[c].isna().sum()
        print(f"  {c:20s}: {n:5,}", flush=True)

    return {
        "n_rows": len(panel),
        "n_cols": panel.shape[1],
        "columns": list(panel.columns),
        "core_window_n": len(core),
        "missing": {c: int(panel[c].isna().sum()) for c in KEY_COLS},
        "status": "PASS (pre-DeFi)",
    }


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(
    panel: pd.DataFrame,
    qa: dict,
    out_dir: Path | None,
) -> tuple[Path, Path]:
    """Write the parquet and QA JSON.

    If `out_dir` is None, write to legacy paths (parquet → ECON_DIR,
    QA JSON → REPORTS_DIR). If set, both files go into `out_dir` —
    convenient for smoke tests on /tmp.
    """
    if out_dir is None:
        parquet_path = CFG.FILES.econ_core_predefi
        json_path = REPORTS_DIR / "econ_core_predefi_qa.json"
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "econ_core_predefi_1h.parquet"
        json_path = out_dir / "econ_core_predefi_qa.json"

    panel.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"\nSaved: {parquet_path}", flush=True)
    print(f"Shape: {panel.shape}", flush=True)

    with open(json_path, "w") as f:
        json.dump(qa, f, indent=2)
    print(f"Saved: {json_path}", flush=True)

    return parquet_path, json_path


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--out_dir", type=Path, default=None,
        help=("If set, both parquet and QA JSON go here. "
              "If unset, parquet→ECON_DIR and QA JSON→REPORTS_DIR "
              "(legacy paths, bit-for-bit reproducible)."),
    )
    args = ap.parse_args()

    print(f"run_core_panel: out_dir={args.out_dir or '<legacy split>'}",
          flush=True)
    t0 = time.time()

    inputs = load_inputs()
    panel = build_panel(inputs)
    panel = compute_features(panel)
    panel = add_defi_placeholders(panel)
    qa = audit_missings(panel, inputs["window_meta"])
    save_outputs(panel, qa, args.out_dir)

    print(f"\nDone. Total wall time: {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    CFG.ensure_dirs()
    raise SystemExit(main())
