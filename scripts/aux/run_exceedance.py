#!/usr/bin/env python3
"""
run_exceedance.py — [ROBUSTNESS — exceedance / VaR-violation test; does NOT change the main spec]

EXCEEDANCE / VaR-violation test. Does the liquidation shock predict *future
tail exceedances* of ETH returns, and is that prediction SYMMETRIC (downside
vs upside)? This is the clean, artefact-free counterpart to the naive
quantile-LP: it replaces the overlapping cumulative-return left tail with a
per-period tail-violation indicator and a block-bootstrap, so the
"horizon-deepening" overlap artefact cannot
manufacture a downside effect.

DESIGN (locked spec)
────────────────────────────────────────────────────────────────────────────
- Shock                : RAW  log_liq.shift(1)   (built by src.estimation.build_df_est_raw)
- Controls (7 = NB07)  : shock, shock_x_oi_high, oi_high, ret_btc_spot,
                         vol_eth_7d, funding_rate, basis_bps
                         => effects are NET OF VOLATILITY via vol_eth_7d.
- Returns              : PER-PERIOD (non-overlapping) for the clean test;
                         a cumulative variant (rolling sum, shift(-h)) is
                         available via --cumulative and is FLAGGED as
                         robustness only (re-introduces the overlap artefact).
- Tail indicators      : for each tail level alpha and horizon h, with
                         q_lo = empirical alpha-quantile and q_hi = empirical
                         (1-alpha)-quantile of the PER-PERIOD ETH return
                         distribution (unconditional thresholds),
                             D_{t+h} = 1{ r_{t+h} < q_lo }   (downside violation)
                             U_{t+h} = 1{ r_{t+h} > q_hi }   (upside violation)
                         Each side therefore carries ~alpha unconditional mass,
                         so under symmetry the shock should load EQUALLY on D
                         and U (Delta = beta_down - beta_up ~ 0).
- Estimators           : LPM  = statsmodels OLS of the 0/1 indicator on the 7
                         controls (interpretable; beta = shock coef), AND
                         logit = same RHS (robustness).
- Inference            : moving-block bootstrap, block size = 24h
                         (CFG.ECON.block_boot_size), percentile CIs, reusing
                         src.bootstrap.{make_seed_sequences, run_parallel_boot,
                         summarize}. Delta = beta_down - beta_up uses the SAME
                         block resample per replication (paired), mirroring
                         src.bootstrap.one_rep_pair / summarize_pair.

EXPECTATION
    beta_down > 0 and beta_up > 0 (liquidations widen BOTH tails — volatility
    channel), and Delta = beta_down - beta_up ~ 0 (SYMMETRIC: no robust
    downside-specific amplification net of volatility).

OUTPUTS (data/econ/)
    exceedance_results.csv  [alpha, h, side, method, beta, se, ci_lo, ci_hi, pval, n_obs]
    exceedance_paired.csv   [alpha, h, delta, ci_lo, ci_hi, pval]
    exceedance_meta.json    {env, n_boot, seed, ...}
    (--cumulative writes *_cumulative.csv/json so the flagged robustness
     variant can never overwrite the canonical per-period artefacts.)

Usage
-----
    # smoke (local, small n_boot, subset horizons)
    .venv/bin/python scripts/aux/run_exceedance.py --n_boot 150 --horizons 0,1,3,6,12

    # canonical (VM): full grid, 1000 reps (spell out every horizon)
    .venv/bin/python scripts/aux/run_exceedance.py --n_boot 1000 \
        --horizons 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24

    # cumulative robustness variant (artefact-aware, flagged)
    .venv/bin/python scripts/aux/run_exceedance.py --cumulative --n_boot 1000
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from config import CFG, ECON_DIR  # noqa: E402

import statsmodels.api as sm  # noqa: E402
from statsmodels.discrete.discrete_model import Logit  # noqa: E402
from statsmodels.regression.linear_model import OLS  # noqa: E402

from src.bootstrap import make_seed_sequences, run_parallel_boot, summarize  # noqa: E402
from src.estimation import build_df_est_raw  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Constants — match NB07 / run_quantile_lp spec verbatim
# ──────────────────────────────────────────────────────────────
ALPHAS_DEFAULT: list[float] = [0.10, 0.05, 0.01]
HORIZONS_SMOKE: list[int] = [0, 1, 3, 6, 12]

CONTROLS: list[str] = ["ret_btc_spot", "vol_eth_7d", "funding_rate", "basis_bps"]
# 7 NB07 regressors; shock is at index 1 after the const prepended in prepare_arrays.
REGRESSORS: list[str] = ["shock", "shock_x_oi_high", "oi_high"] + CONTROLS
SHOCK_COL_IDX: int = 1   # column index of `shock` in X = [const, *REGRESSORS]

BASE_SEED: int = 42
# Deterministic per-(side) test ids for the 4-level SeedSequence namespace
# (mirrors run_robustness_all: make_seed_sequences(base, test_id, h, n=...)).
TEST_ID_DOWN: int = 10_801      # "exceedance / down"
TEST_ID_UP: int = 10_802        # "exceedance / up"
TEST_ID_PAIRED: int = 10_803    # "exceedance / paired delta"

LOGIT_MAXITER: int = 200

RESULTS_COLS: list[str] = [
    "alpha", "h", "side", "method", "beta", "se", "ci_lo", "ci_hi", "pval", "n_obs",
]
PAIRED_COLS: list[str] = ["alpha", "h", "delta", "ci_lo", "ci_hi", "pval"]


# ──────────────────────────────────────────────────────────────
# Estimation sample — per-period (clean) or cumulative (robustness)
# ──────────────────────────────────────────────────────────────
def build_sample(horizons: list[int], cumulative: bool) -> tuple[pd.DataFrame, dict[float, tuple[float, float]]]:
    """Build df_est with the 7 regressors and the future-return / tail columns.

    Reuses src.estimation.build_df_est_raw for the IDENTICAL warmup, RAW shock,
    and shock_x_oi_high construction as the main table (NB07). On top of it:

    - per-period (default): r_{t+h} = ret_eth_perp.shift(-h)  (non-overlapping).
    - cumulative (--cumulative, FLAGGED robustness): r_{t+h} = rolling(h+1).sum()
      .shift(-h)  (overlapping cumulative — re-introduces the overlap artefact),
      i.e. the cumret_h{h} that build_df_est_raw already materialises.

    Unconditional tail thresholds are the empirical quantiles of the PER-PERIOD
    return (ret_eth_perp) over the estimation window, computed ONCE and shared
    across horizons so the indicator definition does not drift with h. For the
    cumulative variant the same per-period thresholds are reused (the violation
    is still "did the per-period-calibrated tail get breached"); this is the
    conservative choice and keeps down/up strictly comparable.

    Returns
    -------
    df_est : DataFrame after warmup, with `fut_r_h{h}` columns added.
    thresholds : {alpha: (q_lo, q_hi)} empirical lower/upper unconditional quantiles.
    """
    # build_df_est_raw already adds shock, shock_x_oi_high and cumret_h{h}.
    df_est = build_df_est_raw(horizons=horizons).reset_index(drop=True)

    for h in horizons:
        if cumulative:
            # overlapping cumulative return already present as cumret_h{h}
            df_est[f"fut_r_h{h}"] = df_est[f"cumret_h{h}"]
        else:
            # per-period, non-overlapping: the return realised h hours ahead
            df_est[f"fut_r_h{h}"] = (
                df_est["ret_eth_perp"] if h == 0
                else df_est["ret_eth_perp"].shift(-h)
            )
    return df_est


def tail_thresholds(df_est: pd.DataFrame, alphas: list[float]) -> dict[float, tuple[float, float]]:
    """Unconditional empirical lower/upper quantiles of PER-PERIOD ETH returns.

    For tail level alpha: q_lo = quantile(alpha), q_hi = quantile(1 - alpha),
    on the estimation-window per-period return (ret_eth_perp). Each side then
    carries ~alpha unconditional mass, so under symmetry the shock loads
    equally on the down- and up-violation indicators.
    """
    r = df_est["ret_eth_perp"].dropna()
    return {a: (float(r.quantile(a)), float(r.quantile(1.0 - a))) for a in alphas}


def add_indicators(df_est: pd.DataFrame, h: int, q_lo: float, q_hi: float) -> pd.DataFrame:
    """Add D_{t+h}=1{fut_r<q_lo} and U_{t+h}=1{fut_r>q_hi} for horizon h."""
    fut = df_est[f"fut_r_h{h}"]
    df_est[f"D_h{h}"] = (fut < q_lo).astype(float)
    df_est[f"U_h{h}"] = (fut > q_hi).astype(float)
    # NaN future return => indicator must be NaN so it is dropped, not counted 0.
    df_est.loc[fut.isna(), [f"D_h{h}", f"U_h{h}"]] = np.nan
    return df_est


# ──────────────────────────────────────────────────────────────
# Point estimators — LPM (OLS) and logit
# ──────────────────────────────────────────────────────────────
def _prepare_indicator_arrays(
    df_est: pd.DataFrame, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-free (y, X) with const + 7 regressors, mirroring estimation.prepare_arrays.

    The mask requires the indicator and all 7 regressors to be non-NaN. X is
    built from REGRESSORS then .fillna(0) is NOT needed (mask already drops
    NaNs), but we keep parity with run_quantile_lp by relying on the mask only.
    """
    cols = [y_col] + REGRESSORS
    clean = df_est.loc[df_est[cols].notna().all(axis=1), cols].reset_index(drop=True)
    y = clean[y_col].to_numpy(dtype=np.float64)
    X = np.column_stack([
        np.ones(len(clean), dtype=np.float64),
        clean[REGRESSORS].to_numpy(dtype=np.float64),
    ])
    return y, X


def fit_point_lpm(y: np.ndarray, X: np.ndarray) -> dict:
    """LPM = OLS of the 0/1 indicator on [const, 7 regressors]. beta = shock coef.

    SE reported is heteroskedasticity-robust (HC1) — appropriate for a linear
    probability model whose errors are mechanically heteroskedastic.
    """
    res = OLS(y, X).fit(cov_type="HC1")
    return {
        "beta":  float(res.params[SHOCK_COL_IDX]),
        "se":    float(res.bse[SHOCK_COL_IDX]),
        "pval":  float(res.pvalues[SHOCK_COL_IDX]),
        "n_obs": int(res.nobs),
    }


def fit_point_logit(y: np.ndarray, X: np.ndarray) -> dict:
    """Logit of the indicator on [const, 7 regressors]. beta = shock log-odds coef.

    Robustness companion to the LPM. Returns NaNs on non-convergence /
    separation rather than raising (rare-event tails can be unstable).
    """
    try:
        res = Logit(y, X).fit(disp=0, maxiter=LOGIT_MAXITER)
        return {
            "beta":  float(res.params[SHOCK_COL_IDX]),
            "se":    float(res.bse[SHOCK_COL_IDX]),
            "pval":  float(res.pvalues[SHOCK_COL_IDX]),
            "n_obs": int(res.nobs),
        }
    except Exception as e:  # noqa: BLE001
        print(f"  warn logit failed: {e}", flush=True)
        return {"beta": np.nan, "se": np.nan, "pval": np.nan, "n_obs": int(len(y))}


# ──────────────────────────────────────────────────────────────
# Block-bootstrap replications (LPM only — fast, the reported CI)
# ──────────────────────────────────────────────────────────────
def _block_idx(rng: np.random.Generator, n: int, block_size: int) -> np.ndarray:
    """Moving-block resample row index — identical scheme to src.bootstrap."""
    if n < block_size:
        raise ValueError(
            f"Panel size n={n} smaller than block_size={block_size}. "
            f"Cannot perform block bootstrap. Check warmup truncation upstream."
        )
    n_blocks = n // block_size
    block_starts = rng.integers(0, n - block_size, size=n_blocks)
    idx = (block_starts[:, None] + np.arange(block_size)[None, :]).ravel()
    return idx[idx < n]


def _one_rep_lpm(
    seed_state: np.random.SeedSequence,
    y: np.ndarray,
    X: np.ndarray,
    block_size: int,
) -> float:
    """One moving-block LPM replication; returns the shock coef (or NaN).

    Mirrors src.bootstrap.one_rep_scalar but fits OLS (LPM) instead of QuantReg.
    Plain OLS coef (no robust cov needed: we want the point coef, the CI comes
    from the bootstrap distribution itself).
    """
    import warnings as _w
    _w.filterwarnings("ignore")
    rng = np.random.default_rng(seed_state)
    idx = _block_idx(rng, len(y), block_size)
    try:
        res = OLS(y[idx], X[idx]).fit()
        return float(res.params[SHOCK_COL_IDX])
    except Exception:  # noqa: BLE001
        return np.nan


def _one_rep_lpm_pair(
    seed_state: np.random.SeedSequence,
    yD: np.ndarray,
    yU: np.ndarray,
    X: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """One replication, SAME block resample for D and U => paired Delta.

    Mirrors src.bootstrap.one_rep_pair. Returns [beta_down, beta_up] on the
    identical y[idx], X[idx]; Delta = beta_down - beta_up is therefore a paired
    estimate (the only correct way to CI a difference). yD, yU and X share the
    SAME mask/rows (see _prepare_pair_arrays).
    """
    import warnings as _w
    _w.filterwarnings("ignore")
    rng = np.random.default_rng(seed_state)
    idx = _block_idx(rng, len(yD), block_size)
    out = np.full(2, np.nan, dtype=np.float64)
    for i, yv in enumerate((yD, yU)):
        try:
            out[i] = float(OLS(yv[idx], X[idx]).fit().params[SHOCK_COL_IDX])
        except Exception:  # noqa: BLE001
            pass
    return out


def _prepare_pair_arrays(
    df_est: pd.DataFrame, h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(yD, yU, X) on a COMMON mask so the paired Delta resamples the same rows.

    D and U are NaN exactly where the future return is NaN (same rows), so the
    common mask = rows where D, U and all 7 regressors are non-NaN.
    """
    cols = [f"D_h{h}", f"U_h{h}"] + REGRESSORS
    clean = df_est.loc[df_est[cols].notna().all(axis=1), cols].reset_index(drop=True)
    yD = clean[f"D_h{h}"].to_numpy(dtype=np.float64)
    yU = clean[f"U_h{h}"].to_numpy(dtype=np.float64)
    X = np.column_stack([
        np.ones(len(clean), dtype=np.float64),
        clean[REGRESSORS].to_numpy(dtype=np.float64),
    ])
    return yD, yU, X


# ──────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────
def run(
    df_est: pd.DataFrame,
    thresholds: dict[float, tuple[float, float]],
    alphas: list[float],
    horizons: list[int],
    n_boot: int,
    block_size: int,
    n_jobs: int,
    ckpt_dir: Path,
    mode_tag: str = "pp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate LPM + logit point coefs and block-bootstrap CIs for every (alpha, h).

    `mode_tag` ('pp' per-period / 'cum' cumulative) is encoded — together with
    n_boot and block_size — into the checkpoint labels, so a re-run with a
    different mode, replication count or block length can NEVER silently reuse
    a stale checkpoint chunk (the labels differ -> cache miss by construction).

    Returns (df_results, df_paired).
    """
    results_rows: list[dict] = []
    paired_rows: list[dict] = []
    tag = f"{mode_tag}_n{n_boot}_b{block_size}"

    for alpha in alphas:
        q_lo, q_hi = thresholds[alpha]
        a_int = int(round(alpha * 1000))  # 100/50/10 — distinct seed namespace key
        print(f"\n[alpha={alpha:.2f}]  q_lo={q_lo:+.4f}  q_hi={q_hi:+.4f}", flush=True)

        for h in horizons:
            df_est = add_indicators(df_est, h, q_lo, q_hi)

            # ---- point estimates: LPM + logit, for both sides ----
            for side, ind_col in (("down", f"D_h{h}"), ("up", f"U_h{h}")):
                y, X = _prepare_indicator_arrays(df_est, ind_col)

                lpm = fit_point_lpm(y, X)
                # bootstrap CI for the LPM coef (the reported interval)
                seeds = make_seed_sequences(
                    BASE_SEED,
                    TEST_ID_DOWN if side == "down" else TEST_ID_UP,
                    a_int, h, n=n_boot,
                )
                boot = run_parallel_boot(
                    one_rep_fn=_one_rep_lpm,
                    seeds=seeds,
                    args_tuple=(y, X, block_size),
                    n_jobs=n_jobs,
                    batch_size=max(1, n_boot // 4),
                    ckpt_path=ckpt_dir,
                    out_shape_per_rep=(),
                    label=f"lpm_{tag}_a{a_int}_{side}_h{h}",
                )
                bs = summarize(boot)
                results_rows.append({
                    "alpha": alpha, "h": h, "side": side, "method": "lpm",
                    "beta": lpm["beta"], "se": lpm["se"],
                    "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                    "pval": lpm["pval"], "n_obs": lpm["n_obs"],
                })

                # logit (robustness): point only; CI from analytic SE (Wald).
                logit = fit_point_logit(y, X)
                lo = (logit["beta"] - 1.96 * logit["se"]
                      if np.isfinite(logit["se"]) else np.nan)
                hi = (logit["beta"] + 1.96 * logit["se"]
                      if np.isfinite(logit["se"]) else np.nan)
                results_rows.append({
                    "alpha": alpha, "h": h, "side": side, "method": "logit",
                    "beta": logit["beta"], "se": logit["se"],
                    "ci_lo": lo, "ci_hi": hi,
                    "pval": logit["pval"], "n_obs": logit["n_obs"],
                })

            # ---- paired Delta = beta_down - beta_up (LPM, same resample) ----
            yD, yU, Xp = _prepare_pair_arrays(df_est, h)
            d_point = (fit_point_lpm(yD, Xp)["beta"] - fit_point_lpm(yU, Xp)["beta"])
            seeds_p = make_seed_sequences(BASE_SEED, TEST_ID_PAIRED, a_int, h, n=n_boot)
            boot_pair = run_parallel_boot(
                one_rep_fn=_one_rep_lpm_pair,
                seeds=seeds_p,
                args_tuple=(yD, yU, Xp, block_size),
                n_jobs=n_jobs,
                batch_size=max(1, n_boot // 4),
                ckpt_path=ckpt_dir,
                out_shape_per_rep=(2,),
                label=f"pair_{tag}_a{a_int}_h{h}",
            )
            mask = ~np.isnan(boot_pair).any(axis=1)
            deltas = boot_pair[mask, 0] - boot_pair[mask, 1]
            if len(deltas) == 0:
                ci_lo = ci_hi = pval = np.nan
            else:
                ci_lo = float(np.percentile(deltas, 2.5))
                ci_hi = float(np.percentile(deltas, 97.5))
                # two-sided bootstrap p-value, centered (cf. summarize_pair)
                centered = deltas - np.mean(deltas)
                pval = float(np.mean(np.abs(centered) >= np.abs(d_point)))
            paired_rows.append({
                "alpha": alpha, "h": h, "delta": float(d_point),
                "ci_lo": ci_lo, "ci_hi": ci_hi, "pval": pval,
            })
            print(f"  h={h:>2}  Delta=beta_down-beta_up = {d_point:+.5f}  "
                  f"CI=[{ci_lo:+.5f},{ci_hi:+.5f}]  p={pval:.3f}", flush=True)

    df_results = (pd.DataFrame(results_rows)
                  .sort_values(["alpha", "h", "side", "method"], kind="mergesort")
                  .reset_index(drop=True)[RESULTS_COLS])
    df_paired = (pd.DataFrame(paired_rows)
                 .sort_values(["alpha", "h"], kind="mergesort")
                 .reset_index(drop=True)[PAIRED_COLS])
    return df_results, df_paired


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(
    df_results: pd.DataFrame,
    df_paired: pd.DataFrame,
    meta: dict,
    out_dir: Path,
    cumulative: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # The FLAGGED cumulative robustness variant gets suffixed filenames so it
    # can never overwrite the canonical per-period artefacts.
    sfx = "_cumulative" if cumulative else ""
    res_csv = out_dir / f"exceedance_results{sfx}.csv"
    pair_csv = out_dir / f"exceedance_paired{sfx}.csv"
    meta_path = out_dir / f"exceedance_meta{sfx}.json"

    df_results.to_csv(res_csv, index=False)
    df_paired.to_csv(pair_csv, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  wrote {res_csv}", flush=True)
    print(f"  wrote {pair_csv}", flush=True)
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
    ap.add_argument("--alphas", type=_parse_floats, default=ALPHAS_DEFAULT,
                    help=f"Comma-separated tail levels. Default: {ALPHAS_DEFAULT}")
    ap.add_argument("--horizons", type=_parse_ints, default=HORIZONS_SMOKE,
                    help=f"Comma-separated. Default (smoke): {HORIZONS_SMOKE}. "
                         f"Full: 0,1,...,24.")
    ap.add_argument("--n_boot", type=int, default=150,
                    help="Block-bootstrap replications. 150 = smoke, 1000 = canonical.")
    ap.add_argument("--block_size", type=int, default=CFG.ECON.block_boot_size,
                    help=f"Moving-block length (hours). Default: "
                         f"{CFG.ECON.block_boot_size}.")
    ap.add_argument("--cumulative", action="store_true",
                    help="ROBUSTNESS ONLY: use overlapping cumulative returns "
                         "(re-introduces the overlap artefact). Default = "
                         "per-period (clean).")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="1 = sequential (default, bit-for-bit). >1 = joblib loky.")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    mode = "CUMULATIVE (robustness, artefact-aware)" if args.cumulative else "PER-PERIOD (clean)"
    print("run_exceedance — VaR-violation / exceedance test", flush=True)
    print(f"  mode={mode}", flush=True)
    print(f"  alphas={args.alphas}  horizons={args.horizons}", flush=True)
    print(f"  n_boot={args.n_boot}  block_size={args.block_size}  "
          f"n_jobs={args.n_jobs}  seed={BASE_SEED}", flush=True)

    t0 = time.time()
    print("Building estimation sample …", flush=True)
    df_est = build_sample(args.horizons, args.cumulative)
    thresholds = tail_thresholds(df_est, args.alphas)
    print(f"  rows={len(df_est):,}  cols={df_est.shape[1]}", flush=True)

    ckpt_dir = args.out_dir / "_exceedance_ckpt"
    df_results, df_paired = run(
        df_est, thresholds, args.alphas, args.horizons,
        n_boot=args.n_boot, block_size=args.block_size,
        n_jobs=args.n_jobs, ckpt_dir=ckpt_dir,
        mode_tag="cum" if args.cumulative else "pp",
    )

    meta = {
        "test": "exceedance / VaR-violation (A4/A5/A8)",
        "mode": "cumulative" if args.cumulative else "per_period",
        "shock": "raw log_liq.shift(1)",
        "regressors": REGRESSORS,
        "controls_net_of_volatility_via": "vol_eth_7d",
        "alphas": args.alphas,
        "horizons": args.horizons,
        "n_boot": int(args.n_boot),
        "block_size": int(args.block_size),
        "seed": int(BASE_SEED),
        "seed_namespace": {
            "down": TEST_ID_DOWN, "up": TEST_ID_UP, "paired": TEST_ID_PAIRED,
            "scheme": "make_seed_sequences(base_seed, test_id, alpha_int, h, n)",
        },
        "thresholds_unconditional": {
            str(a): {"q_lo": thresholds[a][0], "q_hi": thresholds[a][1]}
            for a in args.alphas
        },
        "lpm_cov": "HC1 (point SE); CI = block-bootstrap percentile",
        "logit_cov": "Wald analytic SE (point + 1.96*se CI)",
        "n_rows_estimation": int(len(df_est)),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "panel": str(CFG.FILES.econ_core_full),
    }
    save_outputs(df_results, df_paired, meta, args.out_dir,
                 cumulative=args.cumulative)

    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
