#!/usr/bin/env python3
"""
run_pure_null.py  —  [DIAGNOSTIC / FLAGGED A3 — does NOT change the main spec]

PURE-NULL / reshuffled artifact-by-horizon: the smoking gun for A3.

Claim under test (A3)
---------------------
The main specification runs a quantile local projection (QLP) of the tail
quantile tau=0.01 on OVERLAPPING CUMULATIVE returns  cumret_h{h}  (the rolling
(h+1)-sum of one-hour ETH returns). Overlapping cumulative LHS + a fat-tailed,
volatility-clustered return series mechanically manufactures a *downside-
deepening* tail coefficient that GROWS with the horizon h — EVEN WHEN THE SHOCK
CARRIES NO INFORMATION ABOUT RETURNS. This script demonstrates that artifact by
estimating the EXACT baseline beta(tau=0.01, h) under a principled "zero-world"
null and reporting the spurious |beta| per horizon against the true |beta|.

The reshuffle scheme (principled null — preserves sigma_t, kills shock-info)
---------------------------------------------------------------------------
1. KEEP REAL, UNTOUCHED: the LHS cumret_h{h} (hence the *entire* return series
   and its conditional heteroskedasticity / volatility clustering sigma_t), the
   regime dummy oi_high, and every control (ret_btc_spot, vol_eth_7d,
   funding_rate, basis_bps). Nothing about the dependent variable's distribution,
   serial dependence, or fat tails is altered.
2. KILL SHOCK<->RETURN INFO: replace the shock column by a CIRCULAR SHIFT of the
   real shock series by a random offset k drawn per seed,
   shock_null = np.roll(shock_real, k). A circular shift exactly preserves the
   shock's marginal distribution AND its own autocorrelation structure (the
   liquidation-clustering of the regressor), but severs its temporal alignment
   with returns, so the true contemporaneous/lead-lag link shock_t -> return_t+h
   is destroyed. The interaction is rebuilt consistently from the shifted shock,
   shock_x_oi_high = shock_null * oi_high, so the leverage-cycle term is nulled
   too. Any non-zero beta on shock_null is therefore PURELY MECHANICAL: it is the
   artifact of regressing tail quantiles of an overlapping cumulative,
   heteroskedastic LHS on a regressor that — by construction — knows nothing about
   the returns.

Why circular shift (not iid permutation): an iid shuffle of shock would also kill
shock-info, but it would destroy the regressor's OWN serial dependence, making the
null easier than the real design. The circular shift is the conservative null — it
holds fixed everything except the one thing we want to break (shock<->return
phase), so the residual beta is attributable to the cumulative-LHS geometry alone.
To draw an ENSEMBLE we vary the offset k per seed (offsets sampled uniformly in
[block, n-block] to avoid near-identity shifts), i.e. a block/phase permutation of
the shock series across seeds.

Two nulls (--null_mode) — resolving the innovation-shuffle vs circular-shift discrepancy
----------------------------------------------------------------------------
The circular-shift artifact ratio at h=12 (~18 %) and the shuffled-innovations
test (~72 % at h=12) are DIFFERENT nulls, not a contradiction; this script can
now produce both so the canonical run reconciles them:

  --null_mode circular_shift  (DEFAULT — current behaviour, the CONSERVATIVE null)
      Keep the REAL LHS (cumret_h{h}) untouched — its volatility clustering AND its
      own serial structure are preserved — and only sever the shock<->return phase
      by circularly shifting the shock (see above). Only the shock<->return *link*
      is broken; everything intrinsic to the return series is real. Smaller residual
      artifact (~18 % @ h=12): the long-horizon beta is therefore MOSTLY GENUINE.

  --null_mode innov_shuffle   (the coarser null)
      Construction: make the shock carry NO information about
      returns by RESAMPLING THE LHS instead of moving the shock. The per-period
      return is split into a conditional mean m_t = OLS(ret ~ const+shock+controls)
      and an innovation e_t = ret - m_t; per seed we DROP the shock's mean
      contribution (use the constant-only mean m0 = mean(ret)) and SHUFFLE the
      innovations e_t, then rebuild the per-period return ret_null = m0 + shuffle(e)
      and re-materialise every cumret_h{h} from it via the SAME
      rolling(h+1).sum().shift(-h) convention as build_df_est_raw. The shock and all
      controls are the real columns; only the dependent variable is reshuffled so it
      is informationally independent of the shock. This destroys the return series'
      OWN serial dependence too (not just the shock phase), so it is the COARSER /
      easier null — it isolates the overlapping-cumulative geometry acting on iid-
      reshuffled (but real-magnitude, heteroskedasticity-stripped) innovations,
      reproducing the larger ratio (~72 % @ h=12).

Reading the pair: circular_shift is the defensible answer to "is the shock's beta
mechanical given the real return process" (→ mostly genuine); innov_shuffle is the
coarser upper bound on the pure overlapping-cumulative geometry. The canonical run
emits BOTH (two CSVs, suffixed by mode) so the 18 %-vs-72 % gap is explained as a
null-construction difference, not an instability.

Estimator faithfulness
----------------------
This script does NOT re-implement the estimator. It reuses the project's EXACT
baseline kernel: src.estimation.build_df_est_raw to build the panel + cumret_h{h},
and run_quantile_lp._fit_one with the same REGRESSORS / CONTROLS / QR_FIT_KWARGS.
Only the `shock` column (and its derived interaction) is replaced by the null
draw; cumret_h{h}, oi_high and the controls are the real columns. The reported
beta is _fit_one(...)["beta_shock"] at tau=0.01 — bit-identical machinery to the
main table, run on a shock that carries no information.

EXPECTED (the smoking gun): the ratio spurious|beta| / true|beta| is small at
h=0 (~few %) and rises steeply with h (artifact grows with the overlap), e.g.
~2% at h=0 climbing toward ~70% by h=12.

Local-diagnostic note: MAX_ITER is lowered to 2000 here for speed; the canonical
table uses 20000 (coefficient diff < 1e-3, immaterial for this ratio). Canonical
numbers (if cited) to be re-run on the VM.

Outputs (data/econ/)
--------------------
The output filenames are suffixed by --null_mode so the two nulls can coexist:
  circular_shift -> pure_null_circular_shift_by_horizon.csv  (+ _meta.json)
  innov_shuffle  -> pure_null_innov_shuffle_by_horizon.csv   (+ _meta.json)
(The legacy unsuffixed names pure_null_by_horizon.csv / pure_null_meta.json are
ALSO written for the DEFAULT circular_shift mode, for backward compatibility with
any reader expecting them.)

- pure_null_<mode>_by_horizon.csv : [h, true_abs_beta, artifact_beta_mean,
  artifact_beta_sd, ratio, n_seeds]
  where artifact_beta_mean = mean over seeds of |beta_null(0.01, h)|,
        artifact_beta_sd   = sd   over seeds of |beta_null(0.01, h)|,
        ratio              = artifact_beta_mean / true_abs_beta.
- pure_null_<mode>_meta.json : run provenance + null-scheme description + the
  per-seed offsets (circular_shift) or seed list (innov_shuffle).

Run
---
    .venv/bin/python scripts/aux/run_pure_null.py                       # smoke (circular_shift)
    .venv/bin/python scripts/aux/run_pure_null.py --n_seeds 50 \
        --horizons 0,1,2,3,6,12
    # canonical (VM): BOTH nulls, full horizons
    .venv/bin/python scripts/aux/run_pure_null.py --null_mode circular_shift \
        --n_seeds 500 --horizons 0,1,2,3,4,6,8,12,18,24 --max_iter 20000
    .venv/bin/python scripts/aux/run_pure_null.py --null_mode innov_shuffle \
        --n_seeds 500 --horizons 0,1,2,3,4,6,8,12,18,24 --max_iter 20000
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import CFG, ECON_DIR                       # noqa: E402
import statsmodels.api as sm                            # noqa: E402 (innov_shuffle mean fit)
from src.estimation import build_df_est_raw            # noqa: E402
import run_quantile_lp as rqlp                          # noqa: E402 (carries _fit_one + constants)


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
TAU_NULL: float = 0.01                 # the tail quantile where the artifact lives
HORIZONS_DEFAULT: list[int] = [0, 1, 2, 3, 6, 12]
N_SEEDS_DEFAULT: int = 50              # smoke default
MAX_ITER_DEFAULT: int = 2000          # local diagnostic; canonical (VM) = 20000
BASE_SEED: int = 20260607             # deterministic offset stream (project date)

NULL_MODES: tuple[str, ...] = ("circular_shift", "innov_shuffle")
NULL_MODE_DEFAULT: str = "circular_shift"   # current behaviour (conservative null)
RET_COL: str = "ret_eth_perp"          # per-period return underlying cumret_h{h}

OUT_COLS: list[str] = [
    "h", "true_abs_beta", "artifact_beta_mean", "artifact_beta_sd",
    "ratio", "n_seeds",
    # Signed-bias decomposition + permutation inference (mean |beta| vs signed
    # bias). Appended columns — readers of the original
    # 6 columns are unaffected.
    "true_beta",                  # SIGNED true beta(0.01, h)
    "artifact_beta_signed_mean",  # mean over seeds of SIGNED beta_null
    "artifact_beta_signed_sd",    # sd of SIGNED beta_null
    "null_q025", "null_q975",     # 2.5/97.5 pct of the SIGNED null distribution
    "perm_pval_abs",              # mean(|beta_null| >= |beta_true|)  (permutation p)
]


# ──────────────────────────────────────────────────────────────
# Estimator wrappers (reuse the EXACT baseline kernel)
# ──────────────────────────────────────────────────────────────
def fit_beta(df_est: pd.DataFrame, tau: float, h: int, max_iter: int) -> float:
    """beta_shock from the project's baseline _fit_one (NaN on failure / N<MIN)."""
    r = rqlp._fit_one(tau, h, f"cumret_h{h}", df_est, rqlp.REGRESSORS,
                      rqlp.CONTROLS, max_iter)
    return np.nan if r is None else float(r["beta_shock"])


def make_null_panel(df_est: pd.DataFrame, offset: int) -> pd.DataFrame:
    """Return a copy of df_est with shock circularly shifted by `offset`.

    Implements the principled null (see module docstring):
      - shock          -> np.roll(real_shock, offset)   (preserves marginal + ACF,
                                                          severs shock<->return phase)
      - shock_x_oi_high -> shifted_shock * oi_high       (interaction rebuilt)
    Everything else — cumret_h{h}, oi_high, controls — is the REAL column, so the
    return series' heteroskedasticity / volatility clustering sigma_t is preserved.
    """
    d = df_est.copy()
    shock_real = d["shock"].to_numpy(dtype=np.float64)
    shock_null = np.roll(shock_real, offset)
    d["shock"] = shock_null
    # Rebuild the interaction from the shifted shock (oi_high stays real),
    # mirroring build_df_est_raw: shock_x_oi_high = shock * oi_high (no fillna).
    d["shock_x_oi_high"] = d["shock"] * d["oi_high"]
    return d


# ──────────────────────────────────────────────────────────────
# innov_shuffle null — resample the LHS
# ──────────────────────────────────────────────────────────────
def fit_ret_innovations(df_est: pd.DataFrame) -> tuple[np.ndarray, float, np.ndarray]:
    """Split the per-period return into a conditional mean and its innovations.

    Mirrors run_placebo_symmetric._fit_mean: OLS(ret ~ const + shock + controls)
    over the rows where ret + shock + controls are observed. Returns
        (mask_over_all_rows, m0, innovations_on_mask)
    where
      - m0          = the CONSTANT-only mean of the masked return (the mean with
                      the shock's contribution removed), a scalar;
      - innovations = ret - m_t (the OLS residuals on the masked rows).
    The null return is rebuilt as m0 + shuffle(innovations): the shock therefore
    carries NO information about the LHS (its mean effect is dropped and the
    residuals are reshuffled). This is the innovation-shuffle construction.
    """
    feats = ["shock"] + list(rqlp.CONTROLS)
    mask = df_est[[RET_COL] + feats].notna().all(axis=1)
    Xc = sm.add_constant(df_est.loc[mask, feats].fillna(0.0))
    rr = df_est.loc[mask, RET_COL].to_numpy(dtype=np.float64)
    mean_fit = sm.OLS(rr, Xc).fit()
    m_t = np.asarray(mean_fit.predict(Xc), dtype=np.float64)
    innov = rr - m_t
    return mask.to_numpy(), float(rr.mean()), innov


def make_innov_null_panel(
    df_est: pd.DataFrame,
    mask: np.ndarray,
    m0: float,
    innov: np.ndarray,
    horizons: list[int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a copy of df_est with the LHS resampled (shock carries no info).

    Implements the coarser innovation-shuffle null:
      - ret_eth_perp on the masked rows := m0 + rng.permutation(innov)
        (constant mean + reshuffled innovations; the shock's mean link is dropped
         and the residual serial structure is destroyed -> the LHS is
         informationally independent of the shock);
      - every cumret_h{h} is RE-MATERIALISED from the reshuffled per-period return
        via the SAME rolling(h+1).sum().shift(-h) convention as build_df_est_raw;
      - shock, shock_x_oi_high, oi_high and all controls stay REAL/untouched.
    Rows outside the mean-fit mask keep their real return (they are dropped by the
    estimator's own [y, shock]+controls mask anyway).
    """
    d = df_est.copy()
    ret = d[RET_COL].to_numpy(dtype=np.float64).copy()
    ret[mask] = m0 + rng.permutation(innov)
    d[RET_COL] = ret
    # Re-materialise cumret_h{h} from the reshuffled per-period return (verbatim
    # build_df_est_raw convention): h==0 -> contemporaneous; h>0 -> rolling sum.
    for h in horizons:
        col = f"cumret_h{h}"
        if h == 0:
            d[col] = d[RET_COL]
        else:
            d[col] = d[RET_COL].rolling(h + 1).sum().shift(-h)
    return d


def draw_offsets(n_seeds: int, n_rows: int, block: int) -> list[int]:
    """Per-seed circular-shift offsets, sampled uniformly in [block, n_rows-block].

    Deterministic given BASE_SEED. Bounds avoid near-identity shifts (k≈0 or
    k≈n) that would leave most rows aligned with their real shock.
    """
    rng = np.random.default_rng(BASE_SEED)
    lo, hi = block, max(block + 1, n_rows - block)
    return [int(rng.integers(lo, hi)) for _ in range(n_seeds)]


# ──────────────────────────────────────────────────────────────
# Batch workers (picklable, used by both sequential and joblib paths).
# Results are deterministic per (seed, h) — offsets / SeedSequences are
# pre-drawn before dispatch — so the partitioning into batches and the worker
# count CANNOT change the numbers, only the wall time.
# ──────────────────────────────────────────────────────────────
def _circular_betas_batch(
    df_est: pd.DataFrame,
    offsets_batch: list[int],
    horizons: list[int],
    max_iter: int,
) -> np.ndarray:
    """SIGNED beta_null(0.01, h) for a batch of circular-shift offsets."""
    import warnings as _w
    _w.filterwarnings("ignore")
    out = np.full((len(offsets_batch), len(horizons)), np.nan, dtype=np.float64)
    for i, k in enumerate(offsets_batch):
        d_null = make_null_panel(df_est, k)
        for j, h in enumerate(horizons):
            out[i, j] = fit_beta(d_null, TAU_NULL, h, max_iter)
    return out


def _innov_betas_batch(
    df_est: pd.DataFrame,
    child_seeds_batch: list[np.random.SeedSequence],
    mask: np.ndarray,
    m0: float,
    innov: np.ndarray,
    horizons: list[int],
    max_iter: int,
) -> np.ndarray:
    """SIGNED beta_null(0.01, h) for a batch of innovation-shuffle seeds."""
    import warnings as _w
    _w.filterwarnings("ignore")
    out = np.full((len(child_seeds_batch), len(horizons)), np.nan, dtype=np.float64)
    for i, cs in enumerate(child_seeds_batch):
        rng_s = np.random.default_rng(cs)
        d_null = make_innov_null_panel(df_est, mask, m0, innov, horizons, rng_s)
        for j, h in enumerate(horizons):
            out[i, j] = fit_beta(d_null, TAU_NULL, h, max_iter)
    return out


def _chunk(seq: list, n_chunks: int) -> list[list]:
    """Split seq into ~equal contiguous chunks (preserving order)."""
    n_chunks = max(1, min(n_chunks, len(seq)))
    size = (len(seq) + n_chunks - 1) // n_chunks
    return [seq[i:i + size] for i in range(0, len(seq), size)]


# ──────────────────────────────────────────────────────────────
# Null-preservation diagnostics (anti null-shopping guard).
# The choice between the two nulls must be justified on STATISTICAL grounds
# BEFORE looking at the artifact number. These diagnostics
# quantify exactly WHAT each null preserves / destroys, and are written to the
# meta so the justification is an artefact, not an assertion.
# ──────────────────────────────────────────────────────────────
def _acf(x: np.ndarray, lags: list[int]) -> dict:
    x = x[~np.isnan(x)]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    out = {}
    for l in lags:
        out[f"lag{l}"] = (float(np.dot(x[l:], x[:-l]) / denom)
                          if denom > 0 and l < len(x) else np.nan)
    return out


def null_preservation_diagnostics(
    df_est: pd.DataFrame,
    null_mode: str,
    offsets: list[int],
    mask: np.ndarray | None,
    m0: float | None,
    innov: np.ndarray | None,
) -> dict:
    """Quantify what the chosen null PRESERVES vs DESTROYS (one representative draw).

    circular_shift: the shock's marginal is preserved EXACTLY (same values,
    reordered) and its autocorrelation is preserved up to circular wrap; the
    LHS (returns + vol clustering) is untouched by construction. Reported:
    max |ACF_real - ACF_shifted| over lags {1,6,24} for the first offset.

    innov_shuffle: the LHS marginal magnitudes are preserved (innovations are
    permuted, not redrawn) but the return's serial dependence (vol clustering)
    is destroyed. Reported: ACF of |ret| (the vol-clustering signature) for
    real vs one reshuffled draw, plus moment preservation of the return.
    """
    lags = [1, 6, 24]
    diags: dict = {"mode": null_mode}
    shock = df_est["shock"].to_numpy(dtype=np.float64)
    ret = df_est[RET_COL].to_numpy(dtype=np.float64)

    if null_mode == "circular_shift":
        shifted = np.roll(shock, offsets[0])
        acf_real = _acf(shock, lags)
        acf_shift = _acf(shifted, lags)
        diags.update({
            "shock_marginal": "EXACT (values reordered, distribution identical)",
            "shock_acf_real": acf_real,
            "shock_acf_shifted_first_offset": acf_shift,
            "shock_acf_max_abs_diff": float(max(
                abs(acf_real[k] - acf_shift[k]) for k in acf_real)),
            "lhs": "UNTOUCHED (real returns; sigma_t / vol clustering preserved)",
            "destroys": "ONLY the shock<->return temporal alignment",
        })
    else:  # innov_shuffle
        rng = np.random.default_rng(np.random.SeedSequence(BASE_SEED).spawn(1)[0])
        ret_null = ret.copy()
        ret_null[mask] = m0 + rng.permutation(innov)
        absr, absn = np.abs(ret[mask]), np.abs(ret_null[mask])
        diags.update({
            "shock": "UNTOUCHED (real column)",
            "lhs_marginal": "magnitudes preserved (innovations permuted, not redrawn)",
            "lhs_moments_real": {"std": float(np.nanstd(ret[mask])),
                                 "skew": float(pd.Series(ret[mask]).skew()),
                                 "kurt": float(pd.Series(ret[mask]).kurt())},
            "lhs_moments_null_first_draw": {"std": float(np.nanstd(ret_null[mask])),
                                            "skew": float(pd.Series(ret_null[mask]).skew()),
                                            "kurt": float(pd.Series(ret_null[mask]).kurt())},
            "absret_acf_real": _acf(absr, lags),
            "absret_acf_null_first_draw": _acf(absn, lags),
            "destroys": ("the return's OWN serial dependence (vol clustering) "
                         "AND the shock<->return link — the coarser null"),
        })
    return diags


# ──────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────
def run(
    horizons: list[int],
    n_seeds: int,
    max_iter: int,
    null_mode: str = NULL_MODE_DEFAULT,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict]:
    print("Building estimation sample (build_df_est_raw) ...", flush=True)
    df_est = build_df_est_raw(horizons=horizons).reset_index(drop=True)
    n_rows = len(df_est)
    print(f"  rows={n_rows:,}  cols={df_est.shape[1]}  null_mode={null_mode}  "
          f"n_jobs={n_jobs}", flush=True)

    block = int(CFG.ECON.block_boot_size)   # 24h — shift-bound scale (project const)
    # circular_shift: per-seed offsets. innov_shuffle: per-seed RNG stream.
    offsets = draw_offsets(n_seeds, n_rows, block)   # also recorded for innov (unused there)
    mean_info: dict = {}

    # ---- True (real-shock) baseline beta(tau=0.01, h) — SIGNED ----
    print(f"True baseline beta(tau={TAU_NULL}, h) ...", flush=True)
    true_beta = {}
    for h in horizons:
        true_beta[h] = fit_beta(df_est, TAU_NULL, h, max_iter)
    print("  true beta: "
          + " ".join(f"h{h}={true_beta[h]:+.3f}" for h in horizons), flush=True)

    # ---- Null ensemble: SIGNED spurious beta per (seed, h), batched ----
    # Offsets / SeedSequences are pre-drawn above, so each (seed, h) cell is
    # deterministic and the batching / worker count cannot change the numbers.
    t0 = time.time()
    if null_mode == "circular_shift":
        print(f"Null ensemble: {n_seeds} circular-shift reshuffles "
              f"(tau={TAU_NULL}) ...", flush=True)
        work_items = offsets
        worker_args = lambda batch: (df_est, batch, horizons, max_iter)  # noqa: E731
        worker = _circular_betas_batch
    elif null_mode == "innov_shuffle":
        print(f"Null ensemble: {n_seeds} innovation-shuffle resamples "
              f"(coarse null, tau={TAU_NULL}) ...", flush=True)
        mask, m0, innov = fit_ret_innovations(df_est)
        mean_info = {"ret_const_mean_m0": float(m0), "n_mask": int(mask.sum())}
        print(f"  LHS mean fit: m0={m0:+.5f}  innovations n={int(mask.sum()):,}",
              flush=True)
        # Deterministic independent per-seed RNG stream (parity with circular_shift's
        # BASE_SEED determinism); the spawn keys vary the innovation permutation.
        work_items = np.random.SeedSequence(BASE_SEED).spawn(n_seeds)
        worker_args = lambda batch: (df_est, batch, mask, m0, innov, horizons,  # noqa: E731
                                     max_iter)
        worker = _innov_betas_batch
    else:  # pragma: no cover — guarded by argparse choices
        raise ValueError(f"unknown null_mode {null_mode!r}; choose from {NULL_MODES}")

    # ~4 batches per worker keeps the df_est pickling overhead small while
    # load-balancing; n_jobs=1 runs the same batches sequentially (same numbers).
    eff_jobs = n_jobs if n_jobs > 0 else max(1, (__import__("os").cpu_count() or 2) - 1)
    batches = _chunk(list(work_items), max(1, 4 * eff_jobs))
    if n_jobs == 1:
        chunks = []
        done = 0
        for b in batches:
            chunks.append(worker(*worker_args(b)))
            done += len(b)
            print(f"    seed {done}/{n_seeds}  ({time.time() - t0:.0f}s)", flush=True)
    else:
        from joblib import Parallel, delayed
        chunks = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(worker)(*worker_args(b)) for b in batches
        )
        print(f"    {n_seeds} seeds done  ({time.time() - t0:.0f}s)", flush=True)
    betas = np.vstack(chunks)          # (n_seeds, n_horizons), SIGNED

    # ---- Null-preservation diagnostics (anti null-shopping guard) ----
    diags = null_preservation_diagnostics(
        df_est, null_mode, offsets,
        mask if null_mode == "innov_shuffle" else None,
        m0 if null_mode == "innov_shuffle" else None,
        innov if null_mode == "innov_shuffle" else None,
    )

    # ---- Summary table ----
    rows = []
    for j, h in enumerate(horizons):
        col = betas[:, j]
        col = col[~np.isnan(col)]
        a = np.abs(col)
        mean_abs = float(np.mean(a)) if a.size else np.nan
        sd_abs = float(np.std(a, ddof=1)) if a.size > 1 else np.nan
        tb_signed = true_beta[h]
        tb = abs(tb_signed)
        ratio = float(mean_abs / tb) if (tb and not np.isnan(tb)) else np.nan
        rows.append({
            "h": int(h),
            "true_abs_beta": float(tb),
            "artifact_beta_mean": mean_abs,
            "artifact_beta_sd": sd_abs,
            "ratio": ratio,
            "n_seeds": int(a.size),
            "true_beta": float(tb_signed),
            "artifact_beta_signed_mean": float(np.mean(col)) if col.size else np.nan,
            "artifact_beta_signed_sd": (float(np.std(col, ddof=1))
                                        if col.size > 1 else np.nan),
            "null_q025": (float(np.percentile(col, 2.5)) if col.size else np.nan),
            "null_q975": (float(np.percentile(col, 97.5)) if col.size else np.nan),
            "perm_pval_abs": (float(np.mean(a >= tb))
                              if (a.size and not np.isnan(tb)) else np.nan),
        })
    df_out = pd.DataFrame(rows).sort_values("h", kind="mergesort").reset_index(drop=True)
    df_out = df_out[OUT_COLS]

    if null_mode == "circular_shift":
        null_scheme = "circular_shift_of_shock"
        null_description = (
            "Keep real LHS cumret_h{h} (and thus sigma_t / volatility clustering "
            "AND the return's own serial structure), oi_high, and all controls; "
            "replace shock by np.roll(shock_real, k) (per-seed offset k), preserving "
            "the shock's marginal distribution and autocorrelation while severing "
            "its temporal alignment with returns; rebuild shock_x_oi_high = "
            "shifted_shock * oi_high. CONSERVATIVE null: any non-zero beta is "
            "purely mechanical (overlapping-cumulative-LHS artifact)."
        )
    else:  # innov_shuffle
        null_scheme = "innovation_shuffle_of_returns"
        null_description = (
            "Innovation-shuffle null: make the shock carry NO info about "
            "returns by RESAMPLING THE LHS. Split ret_eth_perp into a conditional "
            "mean m_t = OLS(ret ~ const+shock+controls) and innovations e = ret-m_t; "
            "per seed rebuild ret_null = m0 + permutation(e) (m0 = constant-only mean, "
            "i.e. the shock's mean effect is DROPPED) and re-materialise every "
            "cumret_h{h} via rolling(h+1).sum().shift(-h). shock, shock_x_oi_high, "
            "oi_high and controls stay REAL. COARSER null: also destroys the return's "
            "own serial dependence, isolating the overlapping-cumulative geometry on "
            "reshuffled real-magnitude innovations (artifact ratio ~72% @ h=12)."
        )

    meta = {
        "tau":               TAU_NULL,
        "lhs":               "cumret_h{h}  (overlapping cumulative ETH returns)",
        "null_mode":         null_mode,
        "null_scheme":       null_scheme,
        "null_description":  null_description,
        # Anti null-shopping guard: what this null PRESERVES vs
        # DESTROYS, quantified — the statistical justification of the null
        # choice is an artefact, recorded BEFORE anyone reads the ratio.
        "null_preservation_diagnostics": diags,
        "n_jobs":            int(n_jobs),
        "estimator":         "run_quantile_lp._fit_one (baseline REGRESSORS/CONTROLS/QR_FIT_KWARGS)",
        "regressors":        list(rqlp.REGRESSORS),
        "controls":          list(rqlp.CONTROLS),
        "qr_fit_kwargs":     dict(rqlp.QR_FIT_KWARGS),
        "horizons":          [int(h) for h in horizons],
        "n_seeds":           int(n_seeds),
        "max_iter":          int(max_iter),
        "base_seed":         int(BASE_SEED),
        "shift_block_bound": int(block),
        # offsets are the circular-shift draws; for innov_shuffle the per-seed
        # randomness is the SeedSequence(BASE_SEED).spawn(n_seeds) innovation
        # permutation (offsets are recorded but unused by that mode).
        "offsets":           ([int(k) for k in offsets]
                              if null_mode == "circular_shift" else None),
        "innov_mean_fit":    (mean_info or None),
        "n_rows":            int(n_rows),
        "panel":             str(CFG.FILES.econ_core_full),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version":    sys.version.split()[0],
        "platform":          platform.platform(),
    }
    return df_out, meta


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(df_out: pd.DataFrame, meta: dict, out_dir: Path,
                 null_mode: str = NULL_MODE_DEFAULT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Mode-suffixed names so the two nulls coexist on disk.
    csv_path = out_dir / f"pure_null_{null_mode}_by_horizon.csv"
    meta_path = out_dir / f"pure_null_{null_mode}_meta.json"
    df_out.to_csv(csv_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {csv_path}", flush=True)
    print(f"  wrote {meta_path}", flush=True)
    # Backward-compat: the DEFAULT circular_shift mode also writes the legacy
    # unsuffixed names any existing reader may expect.
    if null_mode == NULL_MODE_DEFAULT:
        legacy_csv = out_dir / "pure_null_by_horizon.csv"
        legacy_meta = out_dir / "pure_null_meta.json"
        df_out.to_csv(legacy_csv, index=False)
        with open(legacy_meta, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  wrote {legacy_csv}  (legacy alias)", flush=True)
        print(f"  wrote {legacy_meta}  (legacy alias)", flush=True)
    # Convention: after modifying a CSV, print head/tail/shape.
    print(f"\n--- pure_null_{null_mode}_by_horizon.csv ---", flush=True)
    print(f"shape: {df_out.shape}", flush=True)
    print("HEAD:", flush=True)
    print(df_out.head().to_string(index=False), flush=True)
    print("TAIL:", flush=True)
    print(df_out.tail().to_string(index=False), flush=True)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--n_seeds", type=int, default=N_SEEDS_DEFAULT,
                    help=f"Number of circular-shift reshuffles. Default: {N_SEEDS_DEFAULT} (smoke).")
    ap.add_argument("--horizons", type=_parse_ints, default=HORIZONS_DEFAULT,
                    help=f"Comma-separated. Default: {HORIZONS_DEFAULT}")
    ap.add_argument("--max_iter", type=int, default=MAX_ITER_DEFAULT,
                    help=f"QuantReg max_iter. Default {MAX_ITER_DEFAULT} (local); canonical=20000.")
    ap.add_argument("--null_mode", choices=NULL_MODES, default=NULL_MODE_DEFAULT,
                    help="Null construction. 'circular_shift' (DEFAULT, conservative: "
                         "shift the shock, keep the real LHS) or 'innov_shuffle' "
                         "(coarse null: resample the return innovations so the "
                         "shock carries no info). Run BOTH for the canonical batch to "
                         "quantify how much of the naive long-horizon coefficient a pure-noise null reproduces.")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="1 = sequential (default). -1/-N = joblib loky across "
                         "pre-drawn seed batches — bit-identical results, only "
                         "wall time changes (offsets/SeedSequences are drawn "
                         "BEFORE dispatch).")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    print(f"run_pure_null: tau={TAU_NULL}  n_seeds={args.n_seeds}  "
          f"null_mode={args.null_mode}", flush=True)
    print(f"  horizons={args.horizons}  max_iter={args.max_iter}  "
          f"n_jobs={args.n_jobs}", flush=True)

    t0 = time.time()
    df_out, meta = run(args.horizons, args.n_seeds, args.max_iter, args.null_mode,
                       args.n_jobs)
    save_outputs(df_out, meta, args.out_dir, args.null_mode)

    print(f"\nDone. Total wall time: {(time.time() - t0) / 60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
