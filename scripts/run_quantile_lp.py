#!/usr/bin/env python3
"""
run_quantile_lp.py — CLI for the main quantile local-projection table (NB07).

Factorises 07_quantile_lp.ipynb into a reproducible script. The notebook's
estimation cells (b705d557, 32649a41) are replaced by a parallelisable QuantReg
fit loop over (tau, h); the IO cell (9b74eb82) becomes save_outputs().
07_quantile_lp_report.ipynb is the reading-side companion.

CHANGELOG vs 07_quantile_lp.ipynb
─────────────────────────────────
NEW
- Single CLI for the main and pre-trend QuantReg tables (--skip_pretrend).
- Optional joblib parallelisation across (tau, h) tuples; default is
  --n_jobs 1 for strict bit-for-bit reproducibility (NB07 ran sequentially).
- df_est is built via src.estimation.build_df_est_raw — zero
  duplication of the warmup / shock / cumret_h pipeline.
- Shock-distribution stats and run provenance are dumped to
  quantile_lp_meta.json so 07_quantile_lp_report does not need to reload
  the 41k-row panel just to print scenario impacts.
- Final DataFrame is sorted with kind="mergesort" (stable) by (tau, h)
  before write — guarantees row order is identical regardless of n_jobs.

CONTRACT WITH NB08 (do not break)
- data/econ/quantile_lp_results.csv is consumed by:
    * run_robustness_all.run_test_D1 (kernel-vs-bootstrap SE table)
    * 08_robustness_report.ipynb (OLS-mean vs QLP comparison cell)
  Required columns, in this exact order:
    tau, h, beta_shock, se_shock, pval_shock,
    beta_interaction, se_interaction, pval_interaction, n_obs
  Any schema drift breaks NB08.

Implementation notes
- Shock is RAW (log_liq.shift(1)). Robustness Tests A and B use the
  BTC-orthogonalised shock instead; this script keeps the RAW shock.
  See run_robustness_all.py.
- Quantile grid: {0.01, 0.05, 0.10, 0.50, 0.90, 0.95} is the default;
  CFG.ECON.quantiles lists the 9-value appendix grid.
- shock_x_oi_high = shock * oi_high (no .fillna(0)); run_robustness_all
  uses .fillna(0). Numerically equivalent after warmup.
- bandwidth="hsheather" and max_iter=20000 are set in this script; max_iter
  is reached at tau=0.50 (IterationLimitWarning); convergence is verified
  post-hoc by inspecting coefficient stability.
- bandwidth="hsheather" justification: the Hall-Sheather (1988) bandwidth
  is an asymptotic approximation tuned for central quantiles where the
  conditional density is well-estimated. At tail quantiles (tau in {0.01,
  0.05, 0.95, 0.99}), few observations near the quantile destabilise
  the kernel density estimate, causing the kernel-SE to systematically
  underestimate the true sampling variance. This is documented in
  Chernozhukov, Fernandez-Val, and Kaji (2016) "Extremal Quantile
  Regression: An Overview" and confirmed empirically here: Test D1
  reports a ratio se_bootstrap / se_kernel in [2.2, 3.5] at tau=0.01.
  For inference at tail quantiles, the paper's main quantile-LP table
  (tab:qlp) therefore reports block-bootstrap CIs from Test N (n_boot=1000, 5-level
  SeedSequence) instead of kernel SE. Kernel SE remains valid and is
  retained for tau=0.50 (central).

Usage
-----
    python run_quantile_lp.py                                   # full run
    python run_quantile_lp.py --n_jobs 4                        # parallel
    python run_quantile_lp.py --quantiles 0.01,0.50 \
        --horizons 0,1 --skip_pretrend --out_dir /tmp/lp_smoke  # smoke

Validation (after first run)
----------------------------
    diff -q data/econ/quantile_lp_results.csv data/econ/quantile_lp_results.legacy.csv
    diff -q data/econ/pretrend_results.csv    data/econ/pretrend_results.legacy.csv
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Maximum number of iterations")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from config import CFG, ECON_DIR  # noqa: E402

import statsmodels.api as sm  # noqa: E402
from statsmodels.regression.quantile_regression import QuantReg  # noqa: E402

from src.estimation import build_df_est_raw  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Constants — match NB07 cells b705d557 / 32649a41 verbatim
# ──────────────────────────────────────────────────────────────
QUANTILES_DEFAULT: list[float] = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95]
PRE_HORIZONS_DEFAULT: list[int] = [-2, -1]
CONTROLS: list[str] = ["ret_btc_spot", "vol_eth_7d", "funding_rate", "basis_bps"]
REGRESSORS: list[str] = ["shock", "shock_x_oi_high", "oi_high"] + CONTROLS

QR_FIT_KWARGS: dict = dict(vcov="robust", kernel="epa", bandwidth="hsheather")
MIN_OBS = 500
DEFAULT_MAX_ITER = 20000

OUT_COLS: list[str] = [
    "tau", "h",
    "beta_shock", "se_shock", "pval_shock",
    "beta_interaction", "se_interaction", "pval_interaction",
    "n_obs",
]


# ──────────────────────────────────────────────────────────────
# Worker — picklable, warnings-safe under loky
# ──────────────────────────────────────────────────────────────
def _fit_one(
    tau: float,
    h: int,
    y_col: str,
    df_est: pd.DataFrame,
    regressors: list[str],
    controls: list[str],
    max_iter: int,
) -> dict | None:
    """One QuantReg fit; returns row dict or None if N<MIN_OBS or fit fails.

    Reproduces NB07 cell b705d557 inner body verbatim: the mask uses
    [y_col, "shock"] + controls (interaction & oi_high are NOT in mask),
    X is built from `regressors` then .fillna(0).
    """
    import warnings as _w
    _w.filterwarnings("ignore")

    mask = df_est[[y_col, "shock"] + controls].notna().all(axis=1)
    y = df_est.loc[mask, y_col]
    X = sm.add_constant(df_est.loc[mask, regressors].fillna(0))

    if len(y) < MIN_OBS:
        return None

    try:
        res = QuantReg(y, X).fit(q=tau, max_iter=max_iter, **QR_FIT_KWARGS)
    except Exception as e:
        print(f"  warn tau={tau} h={h}: {e}", flush=True)
        return None

    return {
        "tau":              float(tau),
        "h":                int(h),
        "beta_shock":       res.params.get("shock", np.nan),
        "se_shock":         res.bse.get("shock", np.nan),
        "pval_shock":       res.pvalues.get("shock", np.nan),
        "beta_interaction": res.params.get("shock_x_oi_high", np.nan),
        "se_interaction":   res.bse.get("shock_x_oi_high", np.nan),
        "pval_interaction": res.pvalues.get("shock_x_oi_high", np.nan),
        "n_obs":            int(res.nobs),
    }


# ──────────────────────────────────────────────────────────────
# Drivers — main + pre-trend
# ──────────────────────────────────────────────────────────────
def _dispatch(
    jobs: list[tuple[float, int, str]],
    df_est: pd.DataFrame,
    n_jobs: int,
    max_iter: int,
    label: str,
) -> list[dict]:
    """Run _fit_one over a list of (tau, h, y_col) jobs. Order-agnostic."""
    t0 = time.time()
    if n_jobs == 1:
        out = [_fit_one(tau, h, y_col, df_est, REGRESSORS, CONTROLS, max_iter)
               for tau, h, y_col in jobs]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_fit_one)(tau, h, y_col, df_est, REGRESSORS, CONTROLS, max_iter)
            for tau, h, y_col in jobs
        )
    rows = [r for r in out if r is not None]
    print(f"  [{label}] {len(rows)}/{len(jobs)} fits in {time.time()-t0:.1f}s",
          flush=True)
    return rows


def _to_sorted_df(rows: list[dict]) -> pd.DataFrame:
    """Stable sort by (tau, h); enforce dtypes and column order for NB08."""
    df = pd.DataFrame(rows)
    df["tau"] = df["tau"].astype(float)
    df["h"] = df["h"].astype(int)
    df = df.sort_values(["tau", "h"], kind="mergesort").reset_index(drop=True)
    return df[OUT_COLS]


def compute_main(
    df_est: pd.DataFrame,
    quantiles: list[float],
    horizons: list[int],
    n_jobs: int,
    max_iter: int,
) -> pd.DataFrame:
    """Main quantile-LP table: tau × h × cumret_h{h}."""
    jobs = [(tau, h, f"cumret_h{h}") for tau in quantiles for h in horizons]
    rows = _dispatch(jobs, df_est, n_jobs, max_iter, "main")
    return _to_sorted_df(rows)


def compute_pretrend(
    df_est: pd.DataFrame,
    quantiles: list[float],
    pre_horizons: list[int],
    n_jobs: int,
    max_iter: int,
) -> pd.DataFrame:
    """Pre-trend table: tau × h ∈ pre_horizons × ret_eth_perp.shift(|h|).

    Mirrors NB07 cell 32649a41: the pre-trend dependent variable for h<0 is
    the realized ETH return |h| hours before the (already lagged) shock.
    """
    df_est = df_est.copy()
    for h in pre_horizons:
        df_est[f"cumret_pre_h{abs(h)}"] = df_est["ret_eth_perp"].shift(abs(h))
    jobs = [(tau, h, f"cumret_pre_h{abs(h)}")
            for tau in quantiles for h in pre_horizons]
    rows = _dispatch(jobs, df_est, n_jobs, max_iter, "pretrend")
    return _to_sorted_df(rows)


# ──────────────────────────────────────────────────────────────
# Metadata — shock distribution + run provenance
# ──────────────────────────────────────────────────────────────
def compute_meta(df_est: pd.DataFrame) -> dict:
    """Shock summary stats + reproducibility provenance (consumed by NB report)."""
    shock_all = df_est["shock"].dropna()
    shock_nz = shock_all[shock_all > 0]
    return {
        "shock_mean":         float(shock_nz.mean()),
        "shock_median":       float(shock_nz.median()),
        "shock_std":          float(shock_nz.std()),
        "shock_p95":          float(shock_nz.quantile(0.95)),
        "shock_p99":          float(shock_nz.quantile(0.99)),
        "n_total":            int(len(shock_all)),
        "n_nonzero":          int(len(shock_nz)),
        "run_timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        "python_version":     sys.version.split()[0],
        "platform":           platform.platform(),
    }


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(
    df_main: pd.DataFrame,
    df_pre: pd.DataFrame | None,
    meta: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    main_parquet = out_dir / "quantile_lp_results.parquet"
    main_csv     = out_dir / "quantile_lp_results.csv"
    df_main.to_parquet(main_parquet, index=False)
    df_main.to_csv(main_csv, index=False)
    print(f"  wrote {main_parquet}", flush=True)
    print(f"  wrote {main_csv}", flush=True)

    if df_pre is not None:
        pre_csv = out_dir / "pretrend_results.csv"
        df_pre.to_csv(pre_csv, index=False)
        print(f"  wrote {pre_csv}", flush=True)

    meta_path = out_dir / "quantile_lp_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--quantiles", type=_parse_floats, default=QUANTILES_DEFAULT,
                    help=f"Comma-separated. Default: {QUANTILES_DEFAULT}")
    ap.add_argument("--horizons", type=_parse_ints,
                    default=list(CFG.ECON.lp_horizons),
                    help="Comma-separated. Default: CFG.ECON.lp_horizons.")
    ap.add_argument("--pre_horizons", type=_parse_ints,
                    default=PRE_HORIZONS_DEFAULT,
                    help=f"Default: {PRE_HORIZONS_DEFAULT}")
    ap.add_argument("--skip_pretrend", action="store_true",
                    help="Skip the h<0 pre-trend table.")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="1 = sequential (default, bit-for-bit reproducible). "
                         ">1 = joblib loky.")
    ap.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER)
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    print(f"run_quantile_lp: quantiles={args.quantiles}", flush=True)
    h_preview = (f"{args.horizons[:3]}…{args.horizons[-3:]}"
                 if len(args.horizons) > 6 else str(args.horizons))
    print(f"  horizons={h_preview}  n={len(args.horizons)}", flush=True)
    print(f"  n_jobs={args.n_jobs}  max_iter={args.max_iter}", flush=True)

    t0 = time.time()
    print("Building estimation sample …", flush=True)
    df_est = build_df_est_raw(horizons=args.horizons)
    print(f"  rows={len(df_est):,}  cols={df_est.shape[1]}", flush=True)

    print("Fitting main table …", flush=True)
    df_main = compute_main(df_est, args.quantiles, args.horizons,
                           args.n_jobs, args.max_iter)

    df_pre: pd.DataFrame | None = None
    if not args.skip_pretrend:
        print("Fitting pre-trend table …", flush=True)
        df_pre = compute_pretrend(df_est, args.quantiles, args.pre_horizons,
                                  args.n_jobs, args.max_iter)

    meta = compute_meta(df_est)
    save_outputs(df_main, df_pre, meta, args.out_dir)

    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
