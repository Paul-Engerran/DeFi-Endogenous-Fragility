#!/usr/bin/env python3
"""
run_vol_response.py  —  [AUXILIARY CHANNEL EVIDENCE — does NOT change the main spec]

Standalone VOLATILITY-RESPONSE local projection.

This is the channel the paper CONFIRMS (cf. OECD): a DeFi-liquidation shock
raises *future realized volatility* of ETH returns. It is the volatility-side
companion to the main return-quantile LP (`run_quantile_lp.py`) and backs the
robustness claim A4. It introduces NO new regressor relative to the locked
specification — it only swaps the dependent variable from future cumulative
RETURNS to future realized VOLATILITY, keeping the RAW shock and the same 7
regressors.

WHAT IT ESTIMATES
─────────────────
For each horizon h and each volatility measure, an OLS local projection

    Vol_{t+h}  =  α  +  β · shock_t  +  γ · (shock_t × oi_high_t)
                     +  δ · oi_high_t  +  controls_t  +  u_{t+h}

is fitted, where the dependent variable is one of:

  measure = "rv"      Realized volatility over the FORWARD window (t, t+h]:
                          RV_{t+h} = sqrt( Σ_{j=1..h} ret_eth_perp_{t+j}^2 )
                      i.e. the square-root of realized variance accumulated
                      over the next h hourly perp returns (units: % , since
                      ret_eth_perp is already in percent). This mirrors the
                      cumulative LHS of the return LP (cumret_h{h}) but on the
                      second moment. h=0 is defined as the contemporaneous
                      single-period magnitude |ret_eth_perp_t| (degenerate
                      window), consistent with the main LP where cumret_h0 is
                      the contemporaneous return.

  measure = "absret"  Robustness measure: the absolute future return at the
                      single horizon point, |ret_eth_perp_{t+h}| (h=0 →
                      |ret_eth_perp_t|). This is the |r_{t+h}| series.

INFERENCE (two independent CIs per (h, measure))
────────────────────────────────────────────────
  1. Newey-West HAC SE on the OLS point estimate, with
       maxlags = max(h + 1, CFG.ECON.nw_lags)   (nw_lags = 12)
     — identical convention to Test D2 in run_robustness_all.py.
  2. Moving-block-bootstrap 95% CI, block length CFG.ECON.block_boot_size = 24,
     reusing src.bootstrap (make_seed_sequences / run_parallel_boot / summarize)
     with an OLS one-rep worker. ci_lo / ci_hi in the output are the BOOTSTRAP
     percentile CI (2.5 / 97.5); se_hac / pval are from the HAC fit.

CONVENTIONS MATCHED TO THE PIPELINE
───────────────────────────────────
- Estimation sample: src.estimation.build_df_est_raw (same warmup, same RAW
  shock = log_liq.shift(1), same shock_x_oi_high = shock * oi_high).
- 7 regressors, in this exact order (= REGRESSORS in run_quantile_lp.py):
    shock, shock_x_oi_high, oi_high, ret_btc_spot, vol_eth_7d,
    funding_rate, basis_bps
- NaN handling: rows kept where [y_col, "shock"] + controls are non-NaN,
  then X.fillna(0) — identical to Test D2 / NB07 _fit_one.
- Block bootstrap: src.bootstrap primitives, block_size = 24.

OECD COMPARISON (to report alongside β)
───────────────────────────────────────
OECD's volatility-on-liquidations elasticity: OLS ≈ +0.017–0.024 % vol per 1%
liquidated; IV ≈ 0.112. Our shock is log_liq (≈ ln(1 + USD liquidated)), so the
β scale is NOT directly an "elasticity per 1% liquidated" — it is "% realized
vol per unit log-liquidations". The meta JSON records this caveat; sign
(expected positive) and order-of-magnitude are the comparable quantities.

CLI
───
    # smoke (fast):
    .venv/bin/python scripts/aux/run_vol_response.py \
        --n_boot 150 --horizons 0,1,3,6,12

    # full:
    .venv/bin/python scripts/aux/run_vol_response.py \
        --horizons 0,1,2,3,...,24            # or omit --horizons for 0..24

OUTPUT
──────
  data/econ/vol_response.csv
      columns: h, measure, beta, se_hac, ci_lo, ci_hi, pval, n_obs
  data/econ/vol_response_meta.json
      run provenance + spec + OECD caveat.
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

# scripts/aux/ → ROOT is parents[2]; also expose scripts/ for sibling imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import CFG, ECON_DIR  # noqa: E402

import statsmodels.api as sm  # noqa: E402
from statsmodels.regression.linear_model import OLS  # noqa: E402

from src.estimation import build_df_est_raw  # noqa: E402
from src.bootstrap import (  # noqa: E402
    make_seed_sequences,
    run_parallel_boot,
    summarize,
)


# ──────────────────────────────────────────────────────────────
# Spec constants — match run_quantile_lp.py REGRESSORS verbatim
# ──────────────────────────────────────────────────────────────
CONTROLS: list[str] = ["ret_btc_spot", "vol_eth_7d", "funding_rate", "basis_bps"]
REGRESSORS: list[str] = ["shock", "shock_x_oi_high", "oi_high"] + CONTROLS

MEASURES: tuple[str, ...] = ("rv", "absret")
RET_COL: str = "ret_eth_perp"          # per-period return driving the LHS moment
MIN_OBS: int = 500                     # same floor as the rest of the pipeline

# bootstrap seed namespace: keep a stable test-id so reps are reproducible and
# independent from the run_robustness_all test family (which uses 1..N).
TEST_ID_VOL: int = 9001

OUT_COLS: list[str] = ["h", "measure", "beta", "se_hac", "ci_lo", "ci_hi",
                       "pval", "n_obs"]


# ──────────────────────────────────────────────────────────────
# Dependent-variable construction
# ──────────────────────────────────────────────────────────────
def add_vol_lhs(df_est: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Materialise rv_h{h} and absret_h{h} forward-looking LHS columns.

    rv_h0      = |ret_t|                       (contemporaneous, degenerate window)
    rv_h{h>0}  = sqrt( Σ_{j=1..h} ret_{t+j}^2 ) (realized vol over (t, t+h])
    absret_h{h}= |ret_{t+h}|                    (single-point future magnitude)

    Built so that the same row mask used for the return LP applies: the forward
    sums use .shift(-j) so that row t carries information about t+1..t+h, and
    the trailing rows that run past the panel become NaN (dropped by the mask).
    """
    df_est = df_est.copy()
    r = df_est[RET_COL]
    r2 = r.pow(2)

    for h in horizons:
        if h == 0:
            df_est[f"rv_h{h}"] = r.abs()
            df_est[f"absret_h{h}"] = r.abs()
            continue
        # realized variance over the forward window (t, t+h]
        rvar = sum(r2.shift(-j) for j in range(1, h + 1))
        df_est[f"rv_h{h}"] = np.sqrt(rvar)
        df_est[f"absret_h{h}"] = r.shift(-h).abs()
    return df_est


# ──────────────────────────────────────────────────────────────
# OLS one-rep worker for the moving-block bootstrap (picklable)
# ──────────────────────────────────────────────────────────────
def _ols_one_rep(
    seed_state: np.random.SeedSequence,
    y: np.ndarray,
    X: np.ndarray,
    block_size: int,
    coef_idx: int,
) -> float:
    """One moving-block resample; returns OLS β at `coef_idx` (NaN on failure).

    Mirrors src.bootstrap.one_rep_scalar's resampling exactly (same block-start
    draw and ravel/truncate), but fits OLS instead of QuantReg — appropriate for
    the conditional-mean volatility LP.
    """
    import warnings as _w
    _w.filterwarnings("ignore")
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
    try:
        res = OLS(y[idx], X[idx]).fit()
        return float(res.params[coef_idx])
    except Exception:
        return np.nan


# ──────────────────────────────────────────────────────────────
# Per-(measure, h) estimation: HAC point + block-bootstrap CI
# ──────────────────────────────────────────────────────────────
def _prepare_yx(
    df_est: pd.DataFrame, y_col: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """NaN-free (y, X) with constant prepended; mask = [y, shock]+controls.

    Identical masking/fillna convention to Test D2 and run_quantile_lp._fit_one.
    Returns the int positional index of `shock` in the const-prefixed X too.
    """
    mask = df_est[[y_col, "shock"] + CONTROLS].notna().all(axis=1)
    y = df_est.loc[mask, y_col].to_numpy(dtype=np.float64)
    X = sm.add_constant(df_est.loc[mask, REGRESSORS].fillna(0)).to_numpy(
        dtype=np.float64
    )
    shock_idx = 1 + REGRESSORS.index("shock")  # +1 for the constant column
    return y, X, shock_idx


def estimate_one(
    df_est: pd.DataFrame,
    measure: str,
    h: int,
    n_boot: int,
    n_jobs: int,
    batch_size: int,
    ckpt_root: Path,
    seed: int,
) -> dict | None:
    """HAC point estimate + moving-block-bootstrap CI for one (measure, h)."""
    y_col = f"{measure}_h{h}"
    y, X, shock_idx = _prepare_yx(df_est, y_col)

    if len(y) < MIN_OBS:
        print(f"  skip {measure} h={h}: n={len(y)} < {MIN_OBS}", flush=True)
        return None

    # ── HAC point estimate (Newey-West), Test-D2 lag convention ──
    actual_lags = max(h + 1, CFG.ECON.nw_lags)
    res_hac = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": actual_lags})
    beta = float(res_hac.params[shock_idx])
    se_hac = float(res_hac.bse[shock_idx])
    pval = float(res_hac.pvalues[shock_idx])

    # ── moving-block bootstrap CI (reuse src.bootstrap driver) ──
    seeds = make_seed_sequences(seed, TEST_ID_VOL, h, n=n_boot)
    # encode measure, n_boot AND seed into the checkpoint label so rv/absret
    # don't collide and a re-run with a different replication count or seed
    # can never silently reuse stale chunks.
    label = f"{measure}_n{n_boot}_s{seed}_h{h:02d}"
    betas = run_parallel_boot(
        one_rep_fn=_ols_one_rep,
        seeds=seeds,
        args_tuple=(y, X, CFG.ECON.block_boot_size, shock_idx),
        n_jobs=n_jobs,
        batch_size=batch_size,
        ckpt_path=ckpt_root,
        out_shape_per_rep=(),
        label=label,
    )
    s = summarize(betas)

    return {
        "h": int(h),
        "measure": measure,
        "beta": beta,
        "se_hac": se_hac,
        "ci_lo": s["ci_lo"],   # bootstrap 2.5 pct
        "ci_hi": s["ci_hi"],   # bootstrap 97.5 pct
        "pval": pval,
        "n_obs": int(res_hac.nobs),
    }


# ──────────────────────────────────────────────────────────────
# Metadata
# ──────────────────────────────────────────────────────────────
def compute_meta(
    df_est: pd.DataFrame, horizons: list[int], n_boot: int, seed: int
) -> dict:
    shock_all = df_est["shock"].dropna()
    shock_nz = shock_all[shock_all > 0]
    return {
        "spec": {
            "lhs": "future realized volatility of ret_eth_perp",
            "measures": list(MEASURES),
            "rv_definition": "rv_h{h>0}=sqrt(sum_{j=1..h} ret_{t+j}^2); "
                             "rv_h0=|ret_t|",
            "absret_definition": "absret_h{h}=|ret_{t+h}|; absret_h0=|ret_t|",
            "shock": "RAW: log_liq.shift(1)",
            "regressors": REGRESSORS,
            "controls": CONTROLS,
            "estimator": "OLS local projection",
            "hac_maxlags_rule": "max(h+1, nw_lags)",
            "nw_lags": int(CFG.ECON.nw_lags),
            "block_boot_size": int(CFG.ECON.block_boot_size),
            "ci_source": "moving-block bootstrap percentile (2.5/97.5)",
            "se_source": "Newey-West HAC",
            "min_obs": MIN_OBS,
        },
        "oecd_comparison": {
            "oecd_ols_pct_vol_per_1pct_liquidated": [0.017, 0.024],
            "oecd_iv": 0.112,
            "scale_caveat": (
                "Our shock is log_liq (~ln(1+USD liquidated)), so beta is "
                "'% realized vol per unit log-liquidations', NOT an elasticity "
                "per 1% liquidated. Compare sign (expected +) and order of "
                "magnitude, not the literal coefficient."
            ),
        },
        "run": {
            "horizons": list(horizons),
            "n_boot": int(n_boot),
            "seed": int(seed),
            "test_id_vol": TEST_ID_VOL,
            "n_total_shock": int(len(shock_all)),
            "n_nonzero_shock": int(len(shock_nz)),
            "shock_mean_nz": float(shock_nz.mean()) if len(shock_nz) else None,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(df: pd.DataFrame, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "vol_response.csv"
    df.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}", flush=True)

    meta_path = out_dir / "vol_response_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[4])
    ap.add_argument("--horizons", type=_parse_ints,
                    default=list(CFG.ECON.lp_horizons),
                    help="Comma-separated. Default: CFG.ECON.lp_horizons (0..24).")
    ap.add_argument("--measures", type=lambda s: [x.strip() for x in s.split(",")
                                                  if x.strip()],
                    default=list(MEASURES),
                    help=f"Subset of {list(MEASURES)}. Default: both.")
    ap.add_argument("--n_boot", type=int, default=CFG.ECON.lp_n_boot,
                    help="Block-bootstrap reps. Smoke: 150. Default: "
                         f"{CFG.ECON.lp_n_boot}.")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="1 = sequential (default, reproducible). >1 = joblib loky.")
    ap.add_argument("--batch_size", type=int, default=50,
                    help="Bootstrap checkpoint batch size.")
    ap.add_argument("--seed", type=int, default=12345,
                    help="Base seed for the bootstrap SeedSequence namespace.")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    ap.add_argument("--ckpt_dir", type=Path, default=ECON_DIR / "_vol_ckpt",
                    help="Per-batch bootstrap checkpoint directory.")
    args = ap.parse_args()

    for m in args.measures:
        if m not in MEASURES:
            ap.error(f"unknown measure {m!r}; choose from {list(MEASURES)}")

    h_preview = (f"{args.horizons[:3]}…{args.horizons[-3:]}"
                 if len(args.horizons) > 6 else str(args.horizons))
    print("run_vol_response: VOLATILITY-RESPONSE LP (raw shock, 7 regressors)",
          flush=True)
    print(f"  measures={args.measures}  horizons={h_preview} "
          f"n={len(args.horizons)}", flush=True)
    print(f"  n_boot={args.n_boot}  n_jobs={args.n_jobs}  "
          f"block={CFG.ECON.block_boot_size}  nw_lags={CFG.ECON.nw_lags}",
          flush=True)

    t0 = time.time()
    print("Building estimation sample (build_df_est_raw) …", flush=True)
    df_est = build_df_est_raw(horizons=args.horizons)
    df_est = add_vol_lhs(df_est, args.horizons)
    print(f"  rows={len(df_est):,}  cols={df_est.shape[1]}", flush=True)

    ckpt_root = args.ckpt_dir
    rows: list[dict] = []
    for measure in args.measures:
        print(f"Fitting measure='{measure}' …", flush=True)
        for h in args.horizons:
            r = estimate_one(
                df_est, measure, h,
                n_boot=args.n_boot, n_jobs=args.n_jobs,
                batch_size=args.batch_size, ckpt_root=ckpt_root,
                seed=args.seed,
            )
            if r is None:
                continue
            rows.append(r)
            print(f"  {measure} h={h:>2}  beta={r['beta']:+.5f}  "
                  f"se_hac={r['se_hac']:.5f}  "
                  f"CI=[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]  "
                  f"p={r['pval']:.4f}  n={r['n_obs']}", flush=True)

    df = pd.DataFrame(rows)
    # stable order: measure, then h
    df["h"] = df["h"].astype(int)
    df = df.sort_values(["measure", "h"], kind="mergesort").reset_index(drop=True)
    df = df[OUT_COLS]

    meta = compute_meta(df_est, args.horizons, args.n_boot, args.seed)
    save_outputs(df, meta, args.out_dir)

    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
