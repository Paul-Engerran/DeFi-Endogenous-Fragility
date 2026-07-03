#!/usr/bin/env python3
"""
run_oos_predictive.py — [ROBUSTNESS / FLAGGED — does NOT change the main spec]

PSEUDO-OUT-OF-SAMPLE PREDICTIVE TEST of the tail-risk "predictor" claim.

Question
--------
The paper's positive claim is that DeFi liquidations are a SYMMETRIC
leading indicator of ETH tail risk. The referee's attack: an in-sample
coefficient net of controls is "a conditional correlation re-baptised", not a
predictor — a predictor must beat a benchmark OUT-OF-SAMPLE. This script runs
that test: rolling-origin quantile forecasts of the PER-PERIOD ETH return
(per-period = the clean anti-overlap-artifact object, like run_exceedance),
scored by pinball (quantile) loss, with vs without the liquidation terms.

Design
------
Outcome: y_{t+h} = ret_eth_perp at hour t+h (single-hour return, NOT the
overlapping cumulative), forecast with information available at t. Default
horizons h ∈ {1, 3, 6, 12, 24} (h=0 is contemporaneous, excluded by default —
it would not be "prediction").

Models (nested — the increment is EXACTLY the liquidation information):
  M0 "qr_controls" (benchmark): QuantReg  y_{t+h} ~ const + oi_high + controls
       where controls = [ret_btc_spot, vol_eth_7d, funding_rate, basis_bps].
       This is the locked spec STRIPPED of the liquidation terms — i.e. the
       volatility/market-state-only information set.
  M1 "full":                    M0 regressors + shock + shock_x_oi_high
       (the locked-spec regressor set verbatim).
  Optional extra benchmark "garch11" (--benchmarks qr_controls,garch11):
       zero-mean GARCH(1,1) fitted on training returns only (arch package);
       the h-step quantile forecast is sigma_{t+h|t} (analytic GARCH forecast
       using observed data up to t with TRAINING-ONLY parameters) times the
       empirical tau-quantile of the TRAINING standardised residuals
       (semi-parametric, symmetric-by-model scale forecast). This is the
       "symmetric GARCH" yardstick the referee names: if M1 beats it OOS, the
       predictor claim is solid in the literature's own terms.

Scheme: expanding window, rolling origin. Initial training = first
--train_frac of the usable sample (default 0.60). Models are re-fitted every
--refit_every hours (default 168 = weekly); between refits, parameters are
frozen and applied to each new information set (standard pseudo-OOS).

Scoring: mean pinball loss rho_tau(y - q_hat) over the test set, per
(tau, h, model). Reported per (tau, h, benchmark):
  loss_bench, loss_full, skill = 1 - loss_full/loss_bench  (>0 → liquidations
  ADD out-of-sample tail-forecast value), and a Diebold-Mariano test on the
  loss differential d_t = L_bench,t - L_full,t with Newey-West (HAC) variance,
  lags = max(24, h+1). Positive dm_t & small dm_pval → significant OOS gain.

Symmetry reading: tau defaults to {0.01, 0.05, 0.95, 0.99} — BOTH tails, so
the output shows whether the OOS gain is symmetric (the paper's thesis) or
one-sided.

Spec-lock note: this is a FLAGGED auxiliary diagnostic. It does not modify the
main specification; it evaluates the predictive content of the locked
regressor set out-of-sample. The per-period LHS and both-tails grid mirror
run_exceedance (the A8/A5 objects).

OUTPUT (data/econ/)
-------------------
  oos_predictive.csv       [tau, h, benchmark, n_train_init, n_test, n_refits,
                            loss_bench, loss_full, skill, dm_t, dm_pval]
  oos_predictive_meta.json provenance + scheme parameters

Run
---
    .venv/bin/python scripts/aux/run_oos_predictive.py \
        --taus 0.05,0.95 --horizons 1,12 --refit_every 2000 \
        --train_frac 0.8 --out_dir /tmp/oos_smoke              # smoke
    .venv/bin/python scripts/aux/run_oos_predictive.py \
        --n_jobs -1 --benchmarks qr_controls,garch11           # canonical
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
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Maximum number of iterations")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import CFG, ECON_DIR  # noqa: E402
from statsmodels.regression.quantile_regression import QuantReg  # noqa: E402
from src.estimation import build_df_est_raw  # noqa: E402
import run_quantile_lp as rqlp  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
RET_COL: str = "ret_eth_perp"
TAUS_DEFAULT: list[float] = [0.01, 0.05, 0.95, 0.99]
HORIZONS_DEFAULT: list[int] = [1, 3, 6, 12, 24]
TRAIN_FRAC_DEFAULT: float = 0.60
REFIT_EVERY_DEFAULT: int = 168          # weekly
MAX_ITER_DEFAULT: int = 5000            # point forecasts, not bootstrap
BENCHMARKS: tuple[str, ...] = ("qr_controls", "garch11")

# Liquidation terms = the increment under test (locked-spec naming).
LIQ_TERMS: list[str] = ["shock", "shock_x_oi_high"]
BASE_TERMS: list[str] = ["oi_high"] + list(rqlp.CONTROLS)

OUT_COLS: list[str] = [
    "tau", "h", "benchmark", "n_train_init", "n_test", "n_refits",
    "loss_bench", "loss_full", "skill", "dm_t", "dm_pval",
]


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """Element-wise pinball (quantile) loss rho_tau(y - q)."""
    u = y - q
    return u * (tau - (u < 0).astype(np.float64))


def newey_west_var(d: np.ndarray, lags: int) -> float:
    """HAC long-run variance of the mean of d (Bartlett kernel)."""
    d = d - d.mean()
    n = len(d)
    g0 = float(np.dot(d, d) / n)
    s = g0
    for l in range(1, min(lags, n - 1) + 1):
        g = float(np.dot(d[l:], d[:-l]) / n)
        s += 2.0 * (1.0 - l / (lags + 1.0)) * g
    return s / n


def dm_test(d: np.ndarray, lags: int) -> tuple[float, float]:
    """Diebold-Mariano t-stat + 2-sided normal p on loss differential d."""
    from scipy import stats
    v = newey_west_var(d, lags)
    if v <= 0 or not np.isfinite(v):
        return np.nan, np.nan
    t = float(d.mean() / np.sqrt(v))
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(t))))
    return t, p


# ──────────────────────────────────────────────────────────────
# QuantReg rolling-origin forecasts (M0 nested in M1)
# ──────────────────────────────────────────────────────────────
def qr_oos_quantiles(
    y: np.ndarray,
    X: np.ndarray,
    split: int,
    refit_every: int,
    tau: float,
    max_iter: int,
) -> np.ndarray:
    """Expanding-window QuantReg forecasts for rows [split:], refit every k rows.

    Returns q_hat aligned with y[split:] (NaN where a refit failed).
    """
    n = len(y)
    q_hat = np.full(n - split, np.nan, dtype=np.float64)
    params = None
    for start in range(split, n, refit_every):
        end = min(start + refit_every, n)
        try:
            res = QuantReg(y[:start], X[:start]).fit(q=tau, max_iter=max_iter)
            params = np.asarray(res.params, dtype=np.float64)
        except Exception:
            pass  # keep previous params (frozen) if this refit fails
        if params is not None:
            q_hat[start - split:end - split] = X[start:end] @ params
    return q_hat


# ──────────────────────────────────────────────────────────────
# GARCH(1,1) benchmark forecasts
# ──────────────────────────────────────────────────────────────
def garch_oos_quantiles(
    ret: np.ndarray,
    split: int,
    refit_every: int,
    tau: float,
    h: int,
) -> np.ndarray:
    """h-step-ahead semi-parametric GARCH(1,1) quantile forecasts for [split:].

    Per refit origin: fit zero-mean GARCH(1,1) on ret[:start] (training data
    only), filter sigma_t over [start:end) with the FROZEN parameters via a
    manual GARCH recursion on observed returns (no look-ahead), and map to an
    h-step quantile with the analytic GARCH variance forecast
        sigma2_{t+h|t} = omega * sum_{j<h-1} phi^j + phi^{h-1} * sigma2_{t+1|t},
        phi = alpha + beta,
    times the empirical tau-quantile of the TRAINING standardised residuals.
    """
    from arch import arch_model
    n = len(ret)
    q_hat = np.full(n - split, np.nan, dtype=np.float64)
    frozen = None  # (omega, alpha, beta, z_tau)
    for start in range(split, n, refit_every):
        end = min(start + refit_every, n)
        try:
            am = arch_model(ret[:start], mean="Zero", vol="GARCH", p=1, q=1,
                            dist="normal", rescale=False)
            res = am.fit(disp="off")
            omega = float(res.params["omega"])
            alpha = float(res.params["alpha[1]"])
            beta = float(res.params["beta[1]"])
            sig_train = np.asarray(res.conditional_volatility, dtype=np.float64)
            z_train = ret[:start] / sig_train
            z_tau = float(np.quantile(z_train, tau))
            frozen = (omega, alpha, beta, z_tau)
        except Exception:
            pass  # keep previous frozen params if this refit fails
        if frozen is None:
            continue
        omega, alpha, beta, z_tau = frozen
        phi = alpha + beta
        # Filter sigma2 forward over [start:end) from the unconditional level,
        # warm-started on the last refit window's tail (observed data only).
        warm = max(0, start - 5000)
        sig2 = omega / max(1e-12, 1.0 - phi)   # unconditional init
        sig2_next = sig2
        for t in range(warm, end):
            # sig2_next is sigma2_{t+1|t} AFTER observing ret[t]
            sig2_next = omega + alpha * ret[t] ** 2 + beta * sig2_next
            if t + 1 >= start and t + 1 < end:
                # h-step variance from origin t+1 (info through t … i.e. the
                # forecast for y at (t+1)+h-1 uses sigma2_{t+1+h-1|t})
                # analytic multi-step: j = h-1 extra steps beyond one-step
                j = h - 1
                if phi < 1.0:
                    sig2_h = (omega * (1.0 - phi ** j) / (1.0 - phi)
                              + (phi ** j) * sig2_next)
                else:
                    sig2_h = sig2_next + omega * j
                q_hat[t + 1 - split] = np.sqrt(max(sig2_h, 1e-12)) * z_tau
    return q_hat


# ──────────────────────────────────────────────────────────────
# One (tau, h, benchmark) cell
# ──────────────────────────────────────────────────────────────
def run_cell(
    tau: float,
    h: int,
    benchmark: str,
    y: np.ndarray,
    X_full: np.ndarray,
    X_base: np.ndarray,
    ret_hist: np.ndarray,
    split: int,
    refit_every: int,
    max_iter: int,
) -> dict:
    """Compute OOS losses + DM test for one (tau, h, benchmark) cell."""
    q_full = qr_oos_quantiles(y, X_full, split, refit_every, tau, max_iter)
    if benchmark == "qr_controls":
        q_bench = qr_oos_quantiles(y, X_base, split, refit_every, tau, max_iter)
    else:  # garch11
        q_bench = garch_oos_quantiles(ret_hist, split, refit_every, tau, h)

    y_test = y[split:]
    ok = ~(np.isnan(q_full) | np.isnan(q_bench) | np.isnan(y_test))
    n_test = int(ok.sum())
    if n_test < 100:
        return {"tau": tau, "h": h, "benchmark": benchmark,
                "n_train_init": split, "n_test": n_test,
                "n_refits": int(np.ceil((len(y) - split) / refit_every)),
                "loss_bench": np.nan, "loss_full": np.nan, "skill": np.nan,
                "dm_t": np.nan, "dm_pval": np.nan}

    L_full = pinball(y_test[ok], q_full[ok], tau)
    L_bench = pinball(y_test[ok], q_bench[ok], tau)
    d = L_bench - L_full                       # >0 → full (with liq) better
    lags = max(24, h + 1)
    dm_t, dm_p = dm_test(d, lags)
    lb, lf = float(L_bench.mean()), float(L_full.mean())
    return {
        "tau": float(tau), "h": int(h), "benchmark": benchmark,
        "n_train_init": int(split), "n_test": n_test,
        "n_refits": int(np.ceil((len(y) - split) / refit_every)),
        "loss_bench": lb, "loss_full": lf,
        "skill": float(1.0 - lf / lb) if lb > 0 else np.nan,
        "dm_t": dm_t, "dm_pval": dm_p,
    }


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(df_out: pd.DataFrame, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "oos_predictive.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}", flush=True)
    meta_path = out_dir / "oos_predictive_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)
    print("\n--- oos_predictive.csv ---", flush=True)
    print(f"shape: {df_out.shape}", flush=True)
    print("HEAD:", flush=True)
    print(df_out.head().to_string(index=False), flush=True)
    print("TAIL:", flush=True)
    print(df_out.tail().to_string(index=False), flush=True)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_strs(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--taus", type=_parse_floats, default=TAUS_DEFAULT,
                    help=f"Comma-separated. Default: {TAUS_DEFAULT} (both tails).")
    ap.add_argument("--horizons", type=_parse_ints, default=HORIZONS_DEFAULT,
                    help=f"Comma-separated, h>=1. Default: {HORIZONS_DEFAULT}")
    ap.add_argument("--benchmarks", type=_parse_strs, default=["qr_controls"],
                    help="Subset of {qr_controls,garch11}. Default: qr_controls. "
                         "garch11 needs the arch package.")
    ap.add_argument("--train_frac", type=float, default=TRAIN_FRAC_DEFAULT,
                    help=f"Initial training share. Default {TRAIN_FRAC_DEFAULT}.")
    ap.add_argument("--refit_every", type=int, default=REFIT_EVERY_DEFAULT,
                    help=f"Refit cadence in hours. Default {REFIT_EVERY_DEFAULT} "
                         "(weekly).")
    ap.add_argument("--max_iter", type=int, default=MAX_ITER_DEFAULT)
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="1 = sequential; -1/-N = joblib loky over (tau,h,bench) "
                         "cells.")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    for b in args.benchmarks:
        if b not in BENCHMARKS:
            ap.error(f"unknown benchmark '{b}' (allowed: {BENCHMARKS})")
    if any(h < 1 for h in args.horizons):
        ap.error("horizons must be >= 1 (h=0 is contemporaneous, not prediction)")

    print("run_oos_predictive: pseudo-OOS pinball test",
          flush=True)
    print(f"  taus={args.taus}  horizons={args.horizons}", flush=True)
    print(f"  benchmarks={args.benchmarks}  train_frac={args.train_frac}  "
          f"refit_every={args.refit_every}  n_jobs={args.n_jobs}", flush=True)

    t0 = time.time()
    print("Building estimation sample (build_df_est_raw) ...", flush=True)
    df_est = build_df_est_raw(horizons=[0]).reset_index(drop=True)
    print(f"  rows={len(df_est):,}", flush=True)

    # Per-period future outcome and regressor matrices on a common mask
    # (mask convention mirrors _fit_one: [y, shock] + controls observed;
    # oi_high/interaction enter via fillna(0) like the main kernel).
    cells = []
    for h in args.horizons:
        d = df_est.copy()
        d["y_lead"] = d[RET_COL].shift(-h)
        mask = d[["y_lead", "shock"] + rqlp.CONTROLS].notna().all(axis=1)
        dd = d.loc[mask].reset_index(drop=True)
        y = dd["y_lead"].to_numpy(dtype=np.float64)
        Xb = np.column_stack(
            [np.ones(len(dd))] + [dd[c].fillna(0).to_numpy(np.float64)
                                  for c in BASE_TERMS])
        Xf = np.column_stack(
            [Xb] + [dd[c].fillna(0).to_numpy(np.float64) for c in LIQ_TERMS])
        ret_hist = dd[RET_COL].to_numpy(dtype=np.float64)
        split = int(len(y) * args.train_frac)
        for tau in args.taus:
            for bench in args.benchmarks:
                cells.append((tau, h, bench, y, Xf, Xb, ret_hist, split))

    print(f"Running {len(cells)} OOS cells ...", flush=True)
    if args.n_jobs == 1:
        rows = [run_cell(tau, h, bench, y, Xf, Xb, rh, split,
                         args.refit_every, args.max_iter)
                for tau, h, bench, y, Xf, Xb, rh, split in cells]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(run_cell)(tau, h, bench, y, Xf, Xb, rh, split,
                              args.refit_every, args.max_iter)
            for tau, h, bench, y, Xf, Xb, rh, split in cells
        )

    df_out = pd.DataFrame(rows, columns=OUT_COLS)
    df_out = df_out.sort_values(["benchmark", "tau", "h"],
                                kind="mergesort").reset_index(drop=True)

    # Console summary: symmetric skill reading.
    for _, r in df_out.iterrows():
        flag = "*" if (r["skill"] > 0 and r["dm_pval"] < 0.10) else " "
        print(f"  {flag} tau={r['tau']:>5}  h={int(r['h']):>2}  "
              f"[{r['benchmark']}]  skill={r['skill']:+.4f}  "
              f"DM t={r['dm_t']:+.2f} p={r['dm_pval']:.3f}", flush=True)

    meta = {
        "script": "scripts/aux/run_oos_predictive.py",
        "purpose": ("Pseudo-OOS pinball test of the liquidation tail-risk "
                    "predictor. Per-period outcome, "
                    "expanding window, rolling origin."),
        "outcome": f"{RET_COL}.shift(-h) (PER-PERIOD future return, anti-artifact)",
        "models": {
            "benchmark_qr_controls": "QuantReg ~ const + oi_high + controls "
                                     "(locked spec minus liquidation terms)",
            "full": "benchmark + shock + shock_x_oi_high (locked spec verbatim)",
            "benchmark_garch11": "zero-mean GARCH(1,1), training-frozen params, "
                                 "analytic h-step sigma x empirical z_tau "
                                 "(semi-parametric symmetric scale benchmark)",
        },
        "taus": [float(t) for t in args.taus],
        "horizons": [int(h) for h in args.horizons],
        "benchmarks": list(args.benchmarks),
        "train_frac": float(args.train_frac),
        "refit_every": int(args.refit_every),
        "max_iter": int(args.max_iter),
        "dm_hac_lags": "max(24, h+1) Bartlett",
        "panel": str(CFG.FILES.econ_core_full),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    save_outputs(df_out, meta, args.out_dir)
    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
