#!/usr/bin/env python3
"""
run_skew_test.py  —  [ROBUSTNESS / FLAGGED — does NOT change the main spec]

Direct conditional-SKEW test (complements the exceedance / placebo
diagnostics in run_exceedance.py and run_placebo_symmetric.py).

QUESTION
--------
Is the downside-specificity of the liquidation shock GENUINE skewness, i.e. a
shift of probability mass toward big DOWN moves NET OF VOLATILITY — or is it a
pure scale (conditional-variance) artefact? The exceedance tests answer this
indirectly via the left>right tail gap; this script answers it directly by
removing scale BEFORE measuring asymmetry.

TWO MEASURES (both regressed on the SAME RHS as the NB07 main table —
RAW shock + 7 controls — with moving-block-bootstrap (block=24) CIs)
-------------------------------------------------------------------
(1) beta_skew — the effect of the shock on  [P(big down) - P(big up)]  with
    SCALE REMOVED. Per-period (NON-OVERLAPPING, h=0) ETH returns are
    standardized by their conditional volatility vol_eth_7d:
        z_t = ret_eth_perp_t / vol_eth_7d_t            (== panel `ret_eth_std`)
    Symmetric tail indicators are then formed at the 5% and 1% cutoffs of the
    STANDARDIZED distribution:
        down_t(tau) = 1[z_t <= q_tau(z)]
        up_t(tau)   = 1[z_t >= q_{1-tau}(z)]
    The dependent variable is the SIGNED tail-skew indicator
        skew_t(tau) = down_t(tau) - up_t(tau)   in {-1, 0, +1}
    and  beta_skew = beta_shock on skew_t. Because OLS is linear, the single
    regression on (down - up) yields exactly  beta_down - beta_up  (the
    asymmetric component), with volatility already netted out via z.
    Reported at tau in {0.05, 0.01} as rows  skew_tail05 / skew_tail01.

(2) z3_winsor — a winsorized conditional-skewness proxy: the standardized
    return cubed, winsorized at the 1%/99% standardized cutoffs BEFORE cubing
    (so a handful of extreme hours cannot dominate the third moment), regressed
    on shock + 7 controls.  beta>0 => shock pushes the conditional third moment
    UP (less left-skew); beta<0 => more left-skew.

LOCKED-SPEC FAITHFULNESS
------------------------
- RHS = run_quantile_lp.REGRESSORS verbatim:
      ["shock","shock_x_oi_high","oi_high","ret_btc_spot",
       "vol_eth_7d","funding_rate","basis_bps"]
  i.e. RAW shock (= log_liq.shift(1)) + the full 7-regressor set including
  vol_eth_7d. This is the same object the project calls "RAW shock + 7 controls"
  (run_robustness_all Test M, line ~1033). beta is extracted on `shock`.
- Estimation sample built by src.estimation.build_df_est_raw (same warmup /
  shock / control pipeline as NB07 — zero duplication).
- Returns are PER-PERIOD (h=0, ret_eth_perp): non-overlapping; one obs per hour.
- Inference: moving-block bootstrap, block = CFG.ECON.block_boot_size (24h),
  OLS refit per replication, percentile [2.5, 97.5] CI; two-sided p-value via
  the project's centered-distribution convention (cf. bootstrap.summarize_pair):
      p = mean(|beta_boot - mean(beta_boot)| >= |beta_point|).
  This is the SAME moving-block construction as bootstrap.one_rep_scalar, with
  OLS substituted for QuantReg (these are mean regressions on a constructed LHS,
  not quantile regressions of returns).

EXPECTED (prior smoke): ~null — 1% borderline, CI ~ [+0.0000, +0.0008].
A near-null here is the INTENDED reading: once scale is removed, the residual
downside-specificity is small => the tail asymmetry in returns is largely a
volatility-amplification channel, with at most a faint genuine-skew component.

OUTPUT
------
- data/econ/skew_test.csv        columns: measure,beta,ci_lo,ci_hi,pval,n_obs
- data/econ/skew_test_meta.json   run provenance + spec record

CLI
---
    .venv/bin/python scripts/aux/run_skew_test.py                    # full
    .venv/bin/python scripts/aux/run_skew_test.py --n_boot 150       # smoke
    .venv/bin/python scripts/aux/run_skew_test.py --n_boot 150 \
        --n_jobs 4 --out_dir /tmp/skew_smoke

SMOKE NOTE: --n_boot 150 is a fast local check; the canonical CI in the paper
uses --n_boot 1000 (CFG.ECON.lp_n_boot) and is re-run on the VM.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # scripts/aux/ -> project root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import CFG, ECON_DIR            # noqa: E402
from statsmodels.regression.linear_model import OLS  # noqa: E402

from src.estimation import build_df_est_raw, prepare_arrays  # noqa: E402
import run_quantile_lp as rqlp              # noqa: E402  (carries REGRESSORS)


# ──────────────────────────────────────────────────────────────
# Spec constants — RHS verbatim from the NB07 main table
# ──────────────────────────────────────────────────────────────
# ["shock","shock_x_oi_high","oi_high","ret_btc_spot",
#  "vol_eth_7d","funding_rate","basis_bps"]  — RAW shock + 7 controls.
REGRESSORS: list[str] = list(rqlp.REGRESSORS)
SHOCK_COL_IDX: int = 1 + REGRESSORS.index("shock")   # +1 for the const column

VOL_COL: str = "vol_eth_7d"          # conditional volatility used to standardize
RET_COL: str = "ret_eth_perp"        # per-period (non-overlapping) ETH return
TAILS: tuple[float, ...] = (0.05, 0.01)   # symmetric cutoffs of the STD dist
WINSOR_Q: float = 0.01               # winsorize z at [q, 1-q] before cubing

OUT_COLS: list[str] = ["measure", "beta", "ci_lo", "ci_hi", "pval", "n_obs"]


# ──────────────────────────────────────────────────────────────
# LHS builders (scale removed BEFORE forming the measures)
# ──────────────────────────────────────────────────────────────
def add_measures(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Attach standardized-return-based skew measures to df_est.

    Returns (df, measure_cols) where measure_cols is the ordered list of
    dependent-variable column names to regress on shock + controls.

    z_t = ret_eth_perp_t / vol_eth_7d_t  (conditional-vol standardization;
    equals the panel's `ret_eth_std`). All quantile cutoffs are computed on
    the NON-NaN standardized series over the estimation sample.
    """
    df = df.copy()
    vol = df[VOL_COL].replace(0, np.nan)
    z = df[RET_COL] / vol
    df["_z_std"] = z
    z_valid = z.dropna()

    measure_cols: list[str] = []

    # (1) Signed tail-skew indicators at symmetric STD cutoffs: down - up.
    for tau in TAILS:
        q_lo = float(z_valid.quantile(tau))
        q_hi = float(z_valid.quantile(1.0 - tau))
        down = (z <= q_lo).astype(float)
        up = (z >= q_hi).astype(float)
        sk = down - up                      # in {-1, 0, +1}; NaN where z is NaN
        sk[z.isna()] = np.nan
        col = f"skew_tail{int(round(tau * 100)):02d}"
        df[col] = sk
        measure_cols.append(col)

    # (2) Winsorized z^3 conditional-skewness proxy.
    wlo = float(z_valid.quantile(WINSOR_Q))
    whi = float(z_valid.quantile(1.0 - WINSOR_Q))
    z_w = z.clip(lower=wlo, upper=whi)
    df["z3_winsor"] = z_w ** 3
    measure_cols.append("z3_winsor")

    return df, measure_cols


# ──────────────────────────────────────────────────────────────
# Moving-block bootstrap (OLS) — same block construction as
# bootstrap.one_rep_scalar, with OLS in place of QuantReg.
# ──────────────────────────────────────────────────────────────
def _ols_beta(y: np.ndarray, X: np.ndarray, col_idx: int) -> float:
    try:
        res = OLS(y, X).fit()
        return float(res.params[col_idx])
    except Exception:
        return np.nan


def _one_rep(
    seed_state: np.random.SeedSequence,
    y: np.ndarray,
    X: np.ndarray,
    block_size: int,
    col_idx: int,
) -> float:
    """One moving-block resample + OLS refit; returns beta at col_idx.

    Identical index construction to src.bootstrap.one_rep_scalar
    (rng.integers(0, n - block_size, n_blocks); blocks tiled then clipped),
    so the resampling scheme is the project's canonical 24h moving block.
    """
    rng = np.random.default_rng(seed_state)
    n = len(y)
    if n < block_size:
        raise ValueError(
            f"Panel size n={n} smaller than block_size={block_size}. "
            f"Cannot perform block bootstrap. Check warmup truncation upstream."
        )
    n_blocks = n // block_size
    block_starts = rng.integers(0, n - block_size, size=n_blocks)
    idx = (block_starts[:, None] + np.arange(block_size)[None, :]).ravel()
    idx = idx[idx < n]
    return _ols_beta(y[idx], X[idx], col_idx)


def bootstrap_beta(
    df: pd.DataFrame,
    y_col: str,
    seed: int,
    test_id: int,
    n_boot: int,
    block_size: int,
    n_jobs: int,
) -> dict:
    """Point OLS beta on `shock` + bootstrap CI/pval for dependent var y_col.

    Uses src.estimation.prepare_arrays for a NaN-free const-prefixed design,
    so the row sample for each measure is exactly the rows where the LHS and
    all 7 regressors are observed.
    """
    y, X = prepare_arrays(df, y_col, REGRESSORS)
    n_obs = int(len(y))
    beta_point = _ols_beta(y, X, SHOCK_COL_IDX)

    # 4-level SeedSequence keying (base_seed, test_id, measure_hash, b) —
    # mirrors the canonical run_robustness_all scheme; measure index keeps
    # the three measures independent under a shared base seed.
    keys = [np.random.SeedSequence([seed, test_id, b]) for b in range(n_boot)]
    args = (y, X, block_size, SHOCK_COL_IDX)
    if n_jobs == 1:
        betas = [_one_rep(s, *args) for s in keys]
    else:
        from joblib import Parallel, delayed
        betas = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one_rep)(s, *args) for s in keys
        )
    arr = np.asarray(betas, dtype=np.float64)
    v = arr[~np.isnan(arr)]
    if len(v) == 0:
        ci_lo = ci_hi = pval = np.nan
    else:
        ci_lo = float(np.percentile(v, 2.5))
        ci_hi = float(np.percentile(v, 97.5))
        # Two-sided p via centered bootstrap distribution (project convention,
        # cf. bootstrap.summarize_pair): p = mean(|centered| >= |beta_point|).
        centered = v - np.mean(v)
        pval = float(np.mean(np.abs(centered) >= abs(beta_point)))

    return {
        "measure": y_col,
        "beta": float(beta_point),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "pval": pval,
        "n_obs": n_obs,
        "_n_success": int(len(v)),
    }


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(rows: list[dict], meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[OUT_COLS]
    csv_path = out_dir / "skew_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}", flush=True)

    meta_path = out_dir / "skew_test_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)

    # Per-convention verification echo (head/tail/shape of the written CSV).
    print(f"\n  skew_test.csv  shape={df.shape}", flush=True)
    print(df.to_string(index=False), flush=True)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--n_boot", type=int, default=CFG.ECON.lp_n_boot,
                    help=f"Bootstrap reps. Default CFG.ECON.lp_n_boot="
                         f"{CFG.ECON.lp_n_boot}. Smoke: 150.")
    ap.add_argument("--block_size", type=int, default=CFG.ECON.block_boot_size,
                    help=f"Moving-block length (h). Default "
                         f"{CFG.ECON.block_boot_size}.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_id", type=int, default=20,
                    help="Seed-namespace id (keeps reps independent of other "
                         "robustness tests sharing the base seed).")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="1 = sequential (default, bit-for-bit). >1 = joblib loky.")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    print(f"run_skew_test: n_boot={args.n_boot}  block_size={args.block_size}  "
          f"seed={args.seed}  n_jobs={args.n_jobs}", flush=True)
    print(f"  RHS (RAW shock + 7 controls) = {REGRESSORS}", flush=True)

    t0 = time.time()
    print("Building estimation sample (build_df_est_raw, h=0 per-period) …",
          flush=True)
    # h=0 only: per-period NON-OVERLAPPING returns; LHS is constructed from
    # ret_eth_perp directly inside add_measures, so no cumret horizon needed.
    df_est = build_df_est_raw(horizons=[0]).reset_index(drop=True)
    df_meas, measure_cols = add_measures(df_est)
    print(f"  rows={len(df_meas):,}  measures={measure_cols}", flush=True)

    # Standardized-distribution diagnostics (for the meta record).
    z = df_meas["_z_std"].dropna()
    rows: list[dict] = []
    for col in measure_cols:
        tr = time.time()
        r = bootstrap_beta(df_meas, col, args.seed, args.test_id,
                           args.n_boot, args.block_size, args.n_jobs)
        rows.append(r)
        print(f"  [{col:12s}] beta={r['beta']:+.5f}  "
              f"CI=[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]  "
              f"p={r['pval']:.4f}  n={r['n_obs']}  "
              f"(boot_ok={r['_n_success']}/{args.n_boot}, "
              f"{time.time()-tr:.1f}s)", flush=True)

    meta = {
        "spec": {
            "shock": "RAW = log_liq.shift(1)",
            "regressors": REGRESSORS,
            "shock_col_idx": SHOCK_COL_IDX,
            "returns": "per-period (non-overlapping) ret_eth_perp, h=0",
            "standardization": f"{RET_COL} / {VOL_COL} (== panel ret_eth_std)",
            "tail_cutoffs_pct": [t * 100 for t in TAILS],
            "winsor_q": WINSOR_Q,
            "measures": {
                "skew_tailNN": "beta_shock on (1[z<=q_tau] - 1[z>=q_{1-tau}]) "
                               "= beta_down - beta_up, scale removed",
                "z3_winsor": "beta_shock on winsorized standardized-return cubed",
            },
        },
        "inference": {
            "method": "moving-block bootstrap (OLS refit), "
                      "same block construction as bootstrap.one_rep_scalar",
            "block_size": int(args.block_size),
            "n_boot": int(args.n_boot),
            "ci": "percentile [2.5, 97.5]",
            "pval": "two-sided centered: mean(|b-mean(b)| >= |b_point|)",
            "seed": int(args.seed),
            "test_id": int(args.test_id),
        },
        "z_std_diagnostics": {
            "mean": float(z.mean()),
            "std": float(z.std()),
            "skew": float(z.skew()),
            "q01": float(z.quantile(0.01)),
            "q05": float(z.quantile(0.05)),
            "q95": float(z.quantile(0.95)),
            "q99": float(z.quantile(0.99)),
            "n": int(len(z)),
        },
        "results": {r["measure"]: {k: r[k] for k in OUT_COLS if k != "measure"}
                    for r in rows},
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    save_outputs(rows, meta, args.out_dir)
    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
