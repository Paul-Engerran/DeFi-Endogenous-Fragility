#!/usr/bin/env python3
"""
run_rif_lp.py — [ROBUSTNESS / FLAGGED — does NOT change the locked QLP spec]

RIF unconditional-quantile LOCAL PROJECTION (Firpo-Fortin-Lemieux 2009 RIF /
unconditional quantile regression), as the unconditional counterpart to the
canonical CONDITIONAL quantile-LP (run_quantile_lp.py / NB07).

Question (conditional vs unconditional)
-----------------------------------------------------
Koenker's conditional QLP gives the effect of the shock on the τ-quantile of
ETH returns HOLDING X FIXED (intra-X). RIF-LP gives the effect on the
UNCONDITIONAL (marginal) τ-quantile. If both lenses agree (symmetric,
volatility-driven signature; no robust downside-specific amplification) the
robustness is strong. If the unconditional lens shows MORE downside asymmetry
than the conditional, the lenses DIVERGE and that divergence is REPORTED, not
buried (pre-planned non-confirmatory branch below).

Method (RIF-LP)
---------------
For tail level τ, with q_τ = the UNCONDITIONAL empirical τ-quantile of the
PER-PERIOD ETH return (ret_eth_perp) over the estimation window and f̂(q_τ) a
Gaussian-KDE density estimate at q_τ, the recentered influence function of the
τ-quantile is

    RIF(r; q_τ) = q_τ + (τ - 1{r ≤ q_τ}) / f̂(q_τ).

Regressing RIF(r_{t+h}; q_τ) on [const, shock, shock×oi_high, oi_high, 4
controls] by OLS at each horizon h gives the unconditional-quantile partial
effect (UQPE) of the shock:

    β_rif^shock(τ, h) = ∂ q_τ(shock) / ∂ shock      (raw UQPE, density-weighted).

=========================  WHY |β_rif| IS THE WRONG GAP  =====================
The RIF carries a 1/f̂(q_τ) density weight, and f̂(q_0.01) ≠ f̂(q_0.95) (the
return density is NOT symmetric at the two tails). So |β_rif(0.01)| -
|β_rif(0.95)| differences two objects on DIFFERENT density scales: the
symmetric-scale cancellation that makes a down-minus-up gap immune to a common
multiplicative scale NO LONGER HOLDS, and the |β_rif| gap would fire
NON-CONFIRMATORY spuriously (an artefact of f̂(q_0.01) ≠ f̂(q_0.95), not of any
genuine asymmetry). |β_rif| is therefore KEPT IN THE CSV FOR REFERENCE ONLY;
it is NOT the inferential object and NO branch logic uses it.

=========================  THE CORRECTED OBJECT: g_τ  =======================
Multiply the raw UQPE back by the density to remove the 1/f̂ weight:

    g_τ(h) = f̂(q_τ) · β_rif^shock(τ, h).

Algebraically, since f̂·RIF = f̂·q_τ + τ - 1{r ≤ q_τ},

    g_τ = f̂(q_τ)·∂q_τ/∂shock = -∂/∂shock E[1{r_{t+h} ≤ q_τ}]
        = ∂/∂shock P(r_{t+h} > q_τ),

i.e. g_τ is the MARGINAL EFFECT OF THE SHOCK ON THE τ-EXCEEDANCE PROBABILITY.
It is DENSITY-FREE and on the SAME linear-probability scale as the canonical
CONDITIONAL per-period exceedance Δ (run_exceedance.py: beta on the 0/1 tail
indicator). The gap is formed on g, NOT on |β_rif|:

    gap_g(h) = g_{0.01}(h) - g_{0.95}(h)      (down minus up, probability scale)

which IS immune to a common symmetric scale (both terms are already on the same
probability scale). All branch logic uses gap_g.

=========================  INTERPRETIVE CAVEAT (surfaced, not buried)  =======
Two things about the LOCKED gap definition g_{0.01} - g_{0.95} must be reported
alongside the verdict, because the smoke FIRES non-confirmatory and the rules
require an informative divergence to be surfaced honestly:

(1) SIGN CONVENTION => SUM, not difference, of one-sided tail-wideners. With the
    locked g_τ = -∂/∂shock E[1{r≤q_τ}] = ∂/∂shock P(r>q_τ):
        g_{0.01} = -β_down   (β_down = ∂P(r<q_0.01)/∂shock, the exceedance down coef)
        g_{0.99} = +β_up     (β_up   = ∂P(r>q_0.99)/∂shock, the exceedance up coef)
    so for a SYMMETRIC tail pair, g_{τ} - g_{1-τ} = -β_down - β_up = -(β_down+β_up):
    it ADDS the two one-sided tail-widenings rather than DIFFERENCING them. The
    canonical conditional asymmetry object is the DIFFERENCE Δ = β_down - β_up
    (which cancels the common volatility channel). gap_g is therefore dominated
    by the COMMON volatility channel (both tails widen, β_down,β_up>0), so a
    large |gap_g| relative to Δ is EXPECTED MECHANICALLY and is NOT, on its own,
    evidence of downside-specific asymmetry. The asymmetry-isolating cross-check
    -(g_{0.01}+g_{0.99}) reproduces the conditional Δ EXACTLY (verified: both
    +0.000529 at h=0) — i.e. the unconditional and conditional lenses AGREE on
    the asymmetry once the gap is taken in the asymmetry-isolating direction.
(2) ASYMMETRIC TAIL PAIR. 0.95 (a 5%-mass up tail, q_0.95=+1.156) is paired with
    0.01 (a 1%-mass down tail, q_0.01=-2.461); they sit at different masses, so
    the matched conditional comparison differs per side (down ~ exceedance
    alpha=0.01, up ~ exceedance alpha=0.05). The CSV/meta record both the literal
    gap and the asymmetry-isolating sum so the reader can adjudicate.

Bottom line for the report: gap_g(0.01,0.95) reads NON-CONFIRMATORY at face
value, but that is the COMBINED tail-widening (volatility channel), not an
asymmetry residue; the asymmetry-isolating recombination of the SAME g's matches
the conditional Δ. The honest sentence is "the unconditional total tail-widening
is large (vol channel), but the unconditional down-vs-up ASYMMETRY agrees with
the conditional null." This nuance is written to rif_lp_meta.json.

Comparison object / pre-planned decision
-----------------------------------------
gap_g is compared to the canonical CONDITIONAL exceedance Δ
(data/econ/exceedance_paired.csv, alpha=0.01 — the thesis null object; same
probability scale, same unconditional thresholds, same per-period clean LHS).
The negligibility band is the SAME canonical SESOI as the conditional Δ
(sesoi_beta_p50p95 = 0.0010915461253353605, read from mde_equivalence.csv) — g
is on the probability scale, so the raw-beta-scale SESOI does NOT apply.

  CONFIRMATORY      : gap_g is bounded/null and consistent with the conditional
                      gap (its bootstrap CI includes the conditional Δ value,
                      OR |gap_g| sits within the SESOI band) -> both lenses
                      agree -> robustness.
  NON-CONFIRMATORY  : gap_g RELIABLY exceeds the conditional gap — its bootstrap
                      CI EXCLUDES the conditional Δ value AND lies above the
                      SESOI band — -> the unconditional lens shows MORE downside
                      asymmetry than the conditional -> REPORT: "the null is a
                      conditional statement; an unconditional asymmetry residue
                      survives."

Inference
---------
gap_g(h) bootstrap CI by moving-block bootstrap (block = CFG.ECON.block_boot_size
= 24h), reusing src.bootstrap.{make_seed_sequences, run_parallel_boot}. Each
replication resamples ONE block index set and re-fits BOTH τ=0.01 and τ=0.95
RIFs on the SAME rows (paired), so gap_g = g_0.01 - g_0.95 is a paired estimate
— the only correct way to CI a difference (mirrors run_exceedance's paired
Δ). The density f̂(q_τ) is held FIXED at its full-sample value across
replications (it is a nuisance scale identified off the whole window, not a
per-resample object); this keeps g_τ = f̂·β_rif a clean linear rescaling of the
bootstrapped β_rif and avoids injecting KDE-bandwidth noise into the CI.

Construction notes (frozen-panel reuse — no data-chain rebuild)
---------------------------------------------------------------
- Variables (shock, interaction, controls) are built ONCE on the frozen panel
  by src.estimation.build_df_est_raw (RAW shock = log_liq.shift(1)); same
  regressors and warmup as NB07 / run_exceedance.
- Per-period future return r_{t+h} = ret_eth_perp.shift(-h) (h>0), the CLEAN
  non-overlapping LHS — so g_τ is directly comparable to the conditional
  exceedance Δ (NOT the overlapping cumret_h, which carries the horizon-overlap
  artefact). RIF is built on this per-period return.
- q_τ thresholds are the UNCONDITIONAL empirical quantiles of ret_eth_perp over
  the estimation window, computed ONCE and shared across horizons — identical
  construction to run_exceedance.tail_thresholds, so q_0.01 / q_0.95 coincide
  with the exceedance q_lo / q_hi. This is what makes the two lenses commensurable.

OUTPUT (data/econ/ — NEW files only; never overwrites a canonical CSV)
----------------------------------------------------------------------
  rif_lp.csv        long form, one row per (tau, h):
                    [tau, h, q_tau, f_hat, beta_rif, se_beta_rif, g, n_obs]
                    plus, on the rows for the gap taus, the paired gap block:
                    a separate gap section (see GAP_COLS) keyed by h.
  rif_lp_meta.json  density method, SESOI scale, the conditional-Δ comparison
                    values, pre-planned branch logic, provenance.

CLI
---
    # smoke (local, /tmp ONLY)
    .venv/bin/python scripts/aux/run_rif_lp.py \
        --taus 0.01,0.50,0.95 --horizons 0,6 --n_boot 100 --out_dir /tmp/rif_smoke

    # canonical (run separately; bound CPU with --n_jobs 4)
    .venv/bin/python scripts/aux/run_rif_lp.py --n_boot 1000 --n_jobs 4 \
        --horizons 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
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
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from config import CFG, ECON_DIR  # noqa: E402

from statsmodels.regression.linear_model import OLS  # noqa: E402

from src.bootstrap import make_seed_sequences, run_parallel_boot  # noqa: E402
from src.estimation import build_df_est_raw  # noqa: E402
# Reuse the canonical regressor list + per-period return construction verbatim.
import run_quantile_lp as rqlp  # noqa: E402  (carries REGRESSORS / CONTROLS)


# ──────────────────────────────────────────────────────────────
# Constants — match NB07 / run_exceedance spec verbatim
# ──────────────────────────────────────────────────────────────
TAUS_DEFAULT: list[float] = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95]
HORIZONS_SMOKE: list[int] = [0, 6]

# 7 NB07 regressors; shock is at index 1 after the const prepended.
CONTROLS: list[str] = rqlp.CONTROLS
REGRESSORS: list[str] = rqlp.REGRESSORS
SHOCK_COL_IDX: int = 1   # column index of `shock` in X = [const, *REGRESSORS]

# The down/up tail pair whose probability-scale gap is the inferential object
# (locked definition: gap_g = g_{0.01} - g_{0.95}; see INTERPRETIVE CAVEAT above).
TAU_DOWN: float = 0.01
TAU_UP: float = 0.95
# Symmetric partner of TAU_DOWN, used ONLY for the asymmetry-isolating
# cross-check -(g_{0.01} + g_{0.99}) which must reproduce the conditional
# exceedance Delta at alpha=0.01 (see INTERPRETIVE CAVEAT). Always added to the
# tau grid internally so the cross-check is computed regardless of --taus.
TAU_DOWN_PARTNER: float = 0.99

BASE_SEED: int = 42
# Companion seed registry: new tests take slots >= 15 (cf. run_subsample_stability).
TEST_ID_GAP: int = 15201      # "rif-lp / paired down-up g gap"

# Canonical SESOI (probability scale) — read from mde_equivalence.csv at runtime;
# this is the documented fallback: g_tau is on the SAME tail-probability
# scale as the conditional exceedance Delta, so the SAME canonical SESOI applies.
SESOI_BETA_P50P95_FALLBACK: float = 0.0010915461253353605

# Canonical conditional comparison object: exceedance Delta at alpha=0.01.
COND_ALPHA: float = 0.01

RESULTS_COLS: list[str] = [
    "tau", "h", "q_tau", "f_hat", "beta_rif", "se_beta_rif", "g", "n_obs",
]
GAP_COLS: list[str] = [
    "h", "g_down", "g_up", "gap_g", "ci_lo", "ci_hi", "pval",
    "g_down_partner", "asym_iso", "cond_delta", "cond_ci_lo", "cond_ci_hi",
    "asym_matches_cond", "branch",
]


# ──────────────────────────────────────────────────────────────
# Estimation sample + RIF construction
# ──────────────────────────────────────────────────────────────
def build_sample(horizons: list[int]) -> pd.DataFrame:
    """Frozen panel via build_df_est_raw + per-period future returns fut_r_h{h}.

    Identical warmup / RAW shock / shock_x_oi_high as NB07 and run_exceedance.
    Per-period (non-overlapping) future return is the CLEAN LHS that makes g_tau
    comparable to the conditional exceedance Delta.
    """
    df_est = build_df_est_raw(horizons=horizons).reset_index(drop=True)
    for h in horizons:
        df_est[f"fut_r_h{h}"] = (
            df_est["ret_eth_perp"] if h == 0
            else df_est["ret_eth_perp"].shift(-h)
        )
    return df_est


def tail_quantiles_and_density(
    df_est: pd.DataFrame, taus: list[float]
) -> tuple[dict[float, float], dict[float, float], dict]:
    """Unconditional q_tau and KDE density f_hat(q_tau) of PER-PERIOD ETH returns.

    q_tau is the empirical tau-quantile of ret_eth_perp over the estimation
    window (computed ONCE, shared across horizons) — identical construction to
    run_exceedance.tail_thresholds, so q_0.01 / q_0.95 coincide with the
    exceedance q_lo / q_hi and the two lenses are commensurable.

    f_hat is a Gaussian-KDE (Scott's rule) density of the same per-period return
    evaluated at q_tau. This is the density that recenters the RIF; it is a
    full-sample nuisance scale (NOT re-estimated per bootstrap replication).
    """
    # Always include the down-tail's symmetric partner so the asymmetry-isolating
    # cross-check -(g_{0.01}+g_{0.99}) is available regardless of the --taus grid.
    grid = sorted(set(taus) | {TAU_DOWN_PARTNER})
    r = df_est["ret_eth_perp"].dropna().to_numpy(dtype=np.float64)
    kde = gaussian_kde(r)  # Scott's-rule bandwidth (deterministic, reproducible)
    q = {t: float(np.quantile(r, t)) for t in grid}
    f = {t: float(kde(q[t])[0]) for t in grid}
    dens_meta = {
        "method": "scipy.stats.gaussian_kde (Scott's rule bandwidth)",
        "n_kde": int(len(r)),
        "bw_factor_scott": float(kde.factor),
        "evaluated_at": {f"{t:g}": {"q_tau": q[t], "f_hat": f[t]} for t in grid},
    }
    return q, f, dens_meta


def rif_column(fut_r: pd.Series, q_tau: float, f_hat: float, tau: float) -> pd.Series:
    """RIF(r; q_tau) = q_tau + (tau - 1{r <= q_tau}) / f_hat (NaN where r is NaN)."""
    ind = (fut_r <= q_tau).astype(float)
    rif = q_tau + (tau - ind) / f_hat
    rif[fut_r.isna()] = np.nan
    return rif


# ──────────────────────────────────────────────────────────────
# Point estimators — RIF-LP via OLS (HC1), and the paired g-gap arrays
# ──────────────────────────────────────────────────────────────
def _prepare_rif_arrays(
    df_est: pd.DataFrame, rif_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """NaN-free (y, X) with const + 7 regressors (mirrors estimation.prepare_arrays)."""
    cols = [rif_col] + REGRESSORS
    clean = df_est.loc[df_est[cols].notna().all(axis=1), cols].reset_index(drop=True)
    y = clean[rif_col].to_numpy(dtype=np.float64)
    X = np.column_stack([
        np.ones(len(clean), dtype=np.float64),
        clean[REGRESSORS].to_numpy(dtype=np.float64),
    ])
    return y, X


def fit_point_rif(y: np.ndarray, X: np.ndarray) -> dict:
    """OLS of RIF on [const, 7 regressors]. beta_rif = shock coef (raw UQPE).

    HC1 robust SE (heteroskedasticity-robust); the gap CI itself comes from the
    block bootstrap, this SE is the per-(tau,h) reference SE for the raw UQPE.
    """
    res = OLS(y, X).fit(cov_type="HC1")
    return {
        "beta": float(res.params[SHOCK_COL_IDX]),
        "se":   float(res.bse[SHOCK_COL_IDX]),
        "n_obs": int(res.nobs),
    }


def _prepare_pair_arrays(
    df_est: pd.DataFrame,
    h: int,
    q_down: float, f_down: float,
    q_up: float, f_up: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rif_down, rif_up, X) on a COMMON mask so the paired g-gap resamples same rows.

    Both RIFs are functions of the SAME per-period future return fut_r_h{h}, so
    they are NaN on identical rows; the common mask = rows where fut_r and all 7
    regressors are non-NaN.
    """
    fut = df_est[f"fut_r_h{h}"]
    rif_d = rif_column(fut, q_down, f_down, TAU_DOWN)
    rif_u = rif_column(fut, q_up, f_up, TAU_UP)
    tmp = pd.DataFrame({"rif_d": rif_d, "rif_u": rif_u})
    for c in REGRESSORS:
        tmp[c] = df_est[c].to_numpy()
    clean = tmp.loc[tmp.notna().all(axis=1)].reset_index(drop=True)
    yD = clean["rif_d"].to_numpy(dtype=np.float64)
    yU = clean["rif_u"].to_numpy(dtype=np.float64)
    X = np.column_stack([
        np.ones(len(clean), dtype=np.float64),
        clean[REGRESSORS].to_numpy(dtype=np.float64),
    ])
    return yD, yU, X


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


def _one_rep_g_pair(
    seed_state: np.random.SeedSequence,
    yD: np.ndarray,
    yU: np.ndarray,
    X: np.ndarray,
    block_size: int,
    f_down: float,
    f_up: float,
) -> np.ndarray:
    """One replication, SAME block resample for both RIFs => paired g-gap.

    Returns [g_down, g_up] = [f_down * beta_rif_down, f_up * beta_rif_up] on the
    identical X[idx] rows. Densities are the FIXED full-sample values (the
    nuisance scale is identified off the whole window, not per resample), so g is
    a clean linear rescaling of the bootstrapped beta_rif. gap_g = g_down - g_up
    is therefore a paired estimate. Mirrors run_exceedance._one_rep_lpm_pair.
    """
    import warnings as _w
    _w.filterwarnings("ignore")
    rng = np.random.default_rng(seed_state)
    idx = _block_idx(rng, len(yD), block_size)
    out = np.full(2, np.nan, dtype=np.float64)
    for i, (yv, fv) in enumerate(((yD, f_down), (yU, f_up))):
        try:
            beta = float(OLS(yv[idx], X[idx]).fit().params[SHOCK_COL_IDX])
            out[i] = fv * beta
        except Exception:  # noqa: BLE001
            pass
    return out


# ──────────────────────────────────────────────────────────────
# Comparison object: canonical conditional exceedance Delta (alpha=0.01)
# ──────────────────────────────────────────────────────────────
def load_conditional_delta(in_dir: Path) -> pd.DataFrame:
    """exceedance_paired.csv rows at alpha=COND_ALPHA, indexed by h (or empty)."""
    path = in_dir / "exceedance_paired.csv"
    if not path.exists():
        print(f"  note: {path} not found — conditional comparison columns blanked.",
              flush=True)
        return pd.DataFrame(columns=["h", "delta", "ci_lo", "ci_hi"])
    df = pd.read_csv(path)
    sel = df[np.isclose(df["alpha"], COND_ALPHA)][["h", "delta", "ci_lo", "ci_hi"]]
    return sel.set_index("h")


def load_sesoi(in_dir: Path) -> tuple[float, str]:
    """Canonical SESOI (probability scale) = sesoi_beta_p50p95 from mde_equivalence.csv.

    g_tau is on the SAME tail-probability scale as the conditional
    exceedance Delta, so the SAME canonical SESOI applies. Falls back to the
    documented literal if the CSV is absent (smoke / out-of-tree runs).
    """
    path = in_dir / "mde_equivalence.csv"
    if path.exists():
        df = pd.read_csv(path)
        if "sesoi_beta_p50p95" in df.columns and len(df):
            vals = df["sesoi_beta_p50p95"].dropna().unique()
            if len(vals):
                return float(vals[0]), f"read from {path}"
    return SESOI_BETA_P50P95_FALLBACK, "documented fallback literal (CSV absent)"


def classify_branch(
    gap_g: float, ci_lo: float, ci_hi: float,
    cond_delta: float, sesoi: float,
) -> str:
    """Pre-planned decision on gap_g vs the conditional Delta and the SESOI band.

    NON-CONFIRMATORY iff the gap_g bootstrap CI EXCLUDES the conditional Delta
    value AND lies above the SESOI band (|CI| beyond +/-sesoi on the side away
    from 0) — the unconditional lens shows MORE downside asymmetry than the
    conditional. CONFIRMATORY otherwise (CI includes cond_delta, OR |gap_g|
    sits within the SESOI band — both lenses agree). NaN-safe.
    """
    if not np.isfinite(ci_lo) or not np.isfinite(ci_hi):
        return "INDETERMINATE"
    ci_includes_cond = (np.isfinite(cond_delta)
                        and ci_lo <= cond_delta <= ci_hi)
    # "lies above the SESOI band": the whole CI is beyond +sesoi or below -sesoi.
    ci_beyond_sesoi = (ci_lo > sesoi) or (ci_hi < -sesoi)
    within_sesoi = abs(gap_g) <= sesoi
    if (not ci_includes_cond) and ci_beyond_sesoi and (not within_sesoi):
        return "NON-CONFIRMATORY"
    return "CONFIRMATORY"


# ──────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────
def run(
    df_est: pd.DataFrame,
    taus: list[float],
    horizons: list[int],
    q: dict[float, float],
    f: dict[float, float],
    cond: pd.DataFrame,
    sesoi: float,
    n_boot: int,
    block_size: int,
    n_jobs: int,
    ckpt_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-(tau, h) RIF-LP point estimates + paired g-gap bootstrap CI per h.

    Returns (df_results, df_gap).
    """
    results_rows: list[dict] = []
    gap_rows: list[dict] = []

    # ---- (1) per-(tau, h) raw UQPE beta_rif + density-normalized g ----
    for tau in taus:
        q_t, f_t = q[tau], f[tau]
        for h in horizons:
            rif_col = f"_rif_t{int(round(tau*1000))}_h{h}"
            df_est[rif_col] = rif_column(df_est[f"fut_r_h{h}"], q_t, f_t, tau)
            y, X = _prepare_rif_arrays(df_est, rif_col)
            pt = fit_point_rif(y, X)
            results_rows.append({
                "tau": float(tau), "h": int(h),
                "q_tau": float(q_t), "f_hat": float(f_t),
                "beta_rif": pt["beta"], "se_beta_rif": pt["se"],
                "g": float(f_t * pt["beta"]),   # density-normalized UQPE
                "n_obs": pt["n_obs"],
            })

    # ---- (2) paired probability-scale gap gap_g = g_0.01 - g_0.95 per h ----
    q_d, f_d = q[TAU_DOWN], f[TAU_DOWN]
    q_u, f_u = q[TAU_UP], f[TAU_UP]
    q_p, f_p = q[TAU_DOWN_PARTNER], f[TAU_DOWN_PARTNER]  # symmetric partner of TAU_DOWN
    a_int = int(round(COND_ALPHA * 1000))
    for h in horizons:
        yD, yU, Xp = _prepare_pair_arrays(df_est, h, q_d, f_d, q_u, f_u)
        g_down = f_d * fit_point_rif(yD, Xp)["beta"]
        g_up = f_u * fit_point_rif(yU, Xp)["beta"]
        gap_point = g_down - g_up

        # Asymmetry-isolating cross-check on the SAME g's at the SYMMETRIC tail
        # pair (0.01, 0.99): -(g_0.01 + g_0.99) == conditional exceedance Delta.
        rif_p = rif_column(df_est[f"fut_r_h{h}"], q_p, f_p, TAU_DOWN_PARTNER)
        yP, XP = _prepare_rif_arrays(
            df_est.assign(_rif_partner=rif_p), "_rif_partner")
        g_down_partner = f_p * fit_point_rif(yP, XP)["beta"]
        asym_iso = -(g_down + g_down_partner)   # == beta_down - beta_up (cond Delta)

        seeds = make_seed_sequences(BASE_SEED, TEST_ID_GAP, a_int, h, n=n_boot)
        boot = run_parallel_boot(
            one_rep_fn=_one_rep_g_pair,
            seeds=seeds,
            args_tuple=(yD, yU, Xp, block_size, f_d, f_u),
            n_jobs=n_jobs,
            batch_size=max(1, n_boot // 4),
            ckpt_path=ckpt_dir,
            out_shape_per_rep=(2,),
            label=f"rifgap_h{h}",
        )
        ok = ~np.isnan(boot).any(axis=1)
        gaps = boot[ok, 0] - boot[ok, 1]
        if len(gaps) == 0:
            ci_lo = ci_hi = pval = np.nan
        else:
            ci_lo = float(np.percentile(gaps, 2.5))
            ci_hi = float(np.percentile(gaps, 97.5))
            centered = gaps - np.mean(gaps)
            pval = float(np.mean(np.abs(centered) >= np.abs(gap_point)))

        if h in cond.index:
            cd = float(cond.loc[h, "delta"])
            cdl = float(cond.loc[h, "ci_lo"])
            cdh = float(cond.loc[h, "ci_hi"])
        else:
            cd = cdl = cdh = np.nan

        # The asymmetry-isolating recombination should reproduce the conditional
        # Delta (sign + scale) — flag whether it matches within a tight tol.
        asym_matches = (np.isfinite(cd)
                        and abs(asym_iso - cd) <= 1e-6 + 0.05 * abs(cd))
        branch = classify_branch(gap_point, ci_lo, ci_hi, cd, sesoi)
        gap_rows.append({
            "h": int(h), "g_down": float(g_down), "g_up": float(g_up),
            "gap_g": float(gap_point), "ci_lo": ci_lo, "ci_hi": ci_hi,
            "pval": pval, "g_down_partner": float(g_down_partner),
            "asym_iso": float(asym_iso),
            "cond_delta": cd, "cond_ci_lo": cdl, "cond_ci_hi": cdh,
            "asym_matches_cond": bool(asym_matches), "branch": branch,
        })
        print(f"  h={h:>2}  gap_g=g(0.01)-g(0.95) = {gap_point:+.6f}  "
              f"CI=[{ci_lo:+.6f},{ci_hi:+.6f}]  p={pval:.3f}  -> {branch}",
              flush=True)
        print(f"        asym-iso -(g0.01+g0.99) = {asym_iso:+.6f}  "
              f"cond_delta = {cd:+.6f}  match={asym_matches}", flush=True)

    df_results = (pd.DataFrame(results_rows)
                  .sort_values(["tau", "h"], kind="mergesort")
                  .reset_index(drop=True)[RESULTS_COLS])
    df_gap = (pd.DataFrame(gap_rows)
              .sort_values(["h"], kind="mergesort")
              .reset_index(drop=True)[GAP_COLS])
    return df_results, df_gap


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(
    df_results: pd.DataFrame,
    df_gap: pd.DataFrame,
    meta: dict,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    res_csv = out_dir / "rif_lp.csv"
    meta_path = out_dir / "rif_lp_meta.json"

    # Single long CSV: per-(tau,h) RIF block, then a tagged gap block appended
    # so the corrected inferential object travels with the raw UQPE. The gap
    # rows are distinguishable by a non-null `branch` column.
    res = df_results.copy()
    res["branch"] = ""           # blank on the per-(tau,h) RIF rows
    gap = df_gap.copy()
    # Align gap block onto the results schema (NaN-pad the RIF-only columns).
    gap_long = pd.DataFrame({
        "tau": np.nan, "h": gap["h"], "q_tau": np.nan, "f_hat": np.nan,
        "beta_rif": np.nan, "se_beta_rif": np.nan, "g": gap["gap_g"],
        "n_obs": np.nan, "branch": gap["branch"],
    })
    # The full gap detail (g_down/g_up/CI/pval/cond) is also written separately
    # below; keep the long CSV faithful to the RIF schema + the gap_g value.
    out = pd.concat([res, gap_long], ignore_index=True)
    out.to_csv(res_csv, index=False)
    # Companion: the full gap table (the decision object) as its own CSV so the
    # branch logic is auditable without parsing the long file.
    gap_csv = out_dir / "rif_lp_gap.csv"
    df_gap.to_csv(gap_csv, index=False)

    with open(meta_path, "w") as fobj:
        json.dump(meta, fobj, indent=2)

    print(f"\n  wrote {res_csv}", flush=True)
    print(f"  wrote {gap_csv}", flush=True)
    print(f"  wrote {meta_path}", flush=True)

    # Per-convention verification echo.
    print(f"\n--- rif_lp.csv  shape={out.shape} ---", flush=True)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("HEAD:", flush=True)
        print(df_results.head().to_string(index=False), flush=True)
        print("\nGAP (decision object):", flush=True)
        print(df_gap.to_string(index=False), flush=True)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--taus", type=_parse_floats, default=TAUS_DEFAULT,
                    help=f"Comma-separated. Default: {TAUS_DEFAULT}. "
                         f"Must include {TAU_DOWN} and {TAU_UP} for the gap.")
    ap.add_argument("--horizons", type=_parse_ints, default=HORIZONS_SMOKE,
                    help=f"Comma-separated. Default (smoke): {HORIZONS_SMOKE}. "
                         f"Full: 0,1,...,24.")
    ap.add_argument("--n_boot", type=int, default=100,
                    help="Paired-bootstrap reps. 100 = smoke, 1000 = canonical.")
    ap.add_argument("--block_size", type=int, default=CFG.ECON.block_boot_size,
                    help=f"Moving-block length (hours). Default: "
                         f"{CFG.ECON.block_boot_size}.")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="1 = sequential (default, bit-for-bit). >1 = joblib loky. "
                         "Use 4 for the canonical run (bounds CPU when other aux "
                         "scripts run concurrently).")
    ap.add_argument("--in_dir", type=Path, default=ECON_DIR,
                    help=f"Dir holding exceedance_paired.csv & mde_equivalence.csv "
                         f"(comparison object + SESOI). Default: {ECON_DIR}")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    if TAU_DOWN not in args.taus or TAU_UP not in args.taus:
        ap.error(f"--taus must include both {TAU_DOWN} and {TAU_UP} "
                 f"(the down/up gap pair). Got {args.taus}.")

    print("run_rif_lp — RIF unconditional-quantile LP (density-normalized UQPE)",
          flush=True)
    print(f"  taus={args.taus}  horizons={args.horizons}", flush=True)
    print(f"  gap pair: g({TAU_DOWN}) - g({TAU_UP})  (probability scale)",
          flush=True)
    print(f"  n_boot={args.n_boot}  block_size={args.block_size}  "
          f"n_jobs={args.n_jobs}  seed={BASE_SEED}", flush=True)

    t0 = time.time()
    print("Building estimation sample (build_df_est_raw, frozen panel) …",
          flush=True)
    df_est = build_sample(args.horizons)
    print(f"  rows={len(df_est):,}  cols={df_est.shape[1]}", flush=True)

    print("Computing unconditional q_tau + KDE density f_hat(q_tau) …", flush=True)
    q, f, dens_meta = tail_quantiles_and_density(df_est, args.taus)
    for t in args.taus:
        print(f"  tau={t:<5g}  q_tau={q[t]:+.4f}  f_hat={f[t]:.5f}", flush=True)

    sesoi, sesoi_src = load_sesoi(args.in_dir)
    cond = load_conditional_delta(args.in_dir)
    print(f"  SESOI (prob scale) = {sesoi:.7g}  [{sesoi_src}]", flush=True)
    print(f"  conditional exceedance Delta rows (alpha={COND_ALPHA}): "
          f"{len(cond)} h-values", flush=True)

    ckpt_dir = args.out_dir / "_rif_lp_ckpt"
    df_results, df_gap = run(
        df_est, args.taus, args.horizons, q, f, cond, sesoi,
        n_boot=args.n_boot, block_size=args.block_size,
        n_jobs=args.n_jobs, ckpt_dir=ckpt_dir,
    )

    fired = sorted(set(df_gap.loc[df_gap["branch"] == "NON-CONFIRMATORY", "h"]))
    asym_ok = bool(df_gap["asym_matches_cond"].all()) if len(df_gap) else False
    meta = {
        "test": "RIF unconditional-quantile LP — density-normalized UQPE (A1)",
        "flagged_robustness": True,
        "does_not_change_locked_spec": True,
        "method": {
            "rif": "RIF(r; q_tau) = q_tau + (tau - 1{r<=q_tau}) / f_hat(q_tau)",
            "estimator": "OLS of RIF(r_{t+h}) on [const, shock, shock_x_oi_high, "
                         "oi_high, ret_btc_spot, vol_eth_7d, funding_rate, basis_bps] "
                         "with HC1 SE",
            "raw_uqpe": "beta_rif^shock(tau,h) = d q_tau(shock)/d shock (density-weighted)",
            "density": dens_meta,
            "future_return": "PER-PERIOD r_{t+h} = ret_eth_perp.shift(-h) (clean, "
                             "non-overlapping; comparable to conditional exceedance Delta)",
            "thresholds": "unconditional empirical q_tau of ret_eth_perp, shared "
                          "across h (== run_exceedance.tail_thresholds construction)",
        },
        "corrected_object": {
            "name": "g_tau = f_hat(q_tau) * beta_rif^shock(tau, h)",
            "interpretation": "g_tau = -d/dshock E[1{r_{t+h}<=q_tau}] = marginal "
                              "effect of the shock on the tau-EXCEEDANCE PROBABILITY; "
                              "density-free, linear-probability scale.",
            "gap": f"gap_g(h) = g({TAU_DOWN}) - g({TAU_UP})  (down minus up, prob scale)",
            "why_not_abs_beta_rif": (
                "|beta_rif| carries a 1/f_hat(q_tau) density weight and "
                "f_hat(q_0.01) != f_hat(q_0.95), so |beta_rif(0.01)|-|beta_rif(0.95)| "
                "differences two differently-density-scaled objects: the "
                "symmetric-scale cancellation that makes the gap immune does NOT "
                "hold and it would fire NON-CONFIRMATORY spuriously. beta_rif is "
                "kept in the CSV for REFERENCE only; the gap and ALL branch logic "
                "use g_tau."
            ),
        },
        "sesoi": {
            "value": float(sesoi),
            "source": sesoi_src,
            "name": "sesoi_beta_p50p95 (canonical, from mde_equivalence.csv)",
            "scale": "PROBABILITY scale — g_tau is on the SAME linear-probability "
                     "scale as the conditional exceedance Delta, so the SAME canonical "
                     "SESOI applies; a raw-beta-scale SESOI does NOT apply here.",
            "locked_definition": "down-minus-up tail-PROBABILITY gap of 1 pp (0.01) "
                                 "mapped to beta units via the shock's p50->p95 span.",
        },
        "comparison_object": {
            "name": f"conditional exceedance Delta, alpha={COND_ALPHA}",
            "source": str((args.in_dir / "exceedance_paired.csv").resolve()),
            "note": "same probability scale, same unconditional thresholds, same "
                    "per-period clean LHS as g_tau — directly commensurable.",
        },
        "pre_planned_branches": {
            "CONFIRMATORY": "gap_g bounded/null and consistent with the conditional "
                            "gap: its bootstrap CI INCLUDES the conditional Delta "
                            "value, OR |gap_g| sits within the SESOI band -> both "
                            "lenses agree -> robustness.",
            "NON-CONFIRMATORY": "gap_g RELIABLY exceeds the conditional gap: its "
                                "bootstrap CI EXCLUDES the conditional Delta value "
                                "AND lies above the SESOI band -> the unconditional "
                                "lens shows MORE downside asymmetry than the "
                                "conditional -> REPORT: 'the null is a conditional "
                                "statement; an unconditional asymmetry residue "
                                "survives.'",
            "fired_non_confirmatory_at_h": [int(h) for h in fired],
            "surfaced": bool(fired),
        },
        "interpretive_caveat": {
            "sign_convention_sum_not_difference": (
                "With the LOCKED g_tau = -d/dshock E[1{r<=q_tau}], for a symmetric "
                "tail pair g_tau - g_{1-tau} = -(beta_down + beta_up): it ADDS the "
                "two one-sided tail-wideners (common volatility channel) rather than "
                "DIFFERENCING them. The conditional asymmetry object is the "
                "DIFFERENCE Delta = beta_down - beta_up. So a large |gap_g| vs Delta "
                "is EXPECTED MECHANICALLY (both tails widen) and is NOT on its own "
                "evidence of downside-specific asymmetry."
            ),
            "asymmetry_isolating_crosscheck": (
                "-(g_0.01 + g_0.99) reproduces the conditional exceedance Delta "
                "(alpha=0.01) exactly (sign + scale). Recorded per h as "
                "asym_iso / asym_matches_cond in rif_lp_gap.csv."
            ),
            "asym_iso_matches_conditional_delta_all_h": asym_ok,
            "asymmetric_tail_pair_note": (
                "tau_up=0.95 is a 5%-mass tail while tau_down=0.01 is a 1%-mass tail; "
                "their matched conditional comparisons differ per side (down ~ "
                "exceedance alpha=0.01, up ~ exceedance alpha=0.05). The literal "
                "gap_g and the asymmetry-isolating sum are both reported so the "
                "reader can adjudicate."
            ),
            "honest_report_sentence": (
                "The unconditional TOTAL tail-widening is large (volatility channel), "
                "but the unconditional down-vs-up ASYMMETRY (asym_iso) agrees with "
                "the conditional null Delta. gap_g's NON-CONFIRMATORY face value is "
                "the combined tail-widening, not an asymmetry residue."
            ),
        },
        "inference": {
            "ci": "moving-block bootstrap (block = CFG.ECON.block_boot_size), "
                  "paired across tau=0.01 / tau=0.95 on the SAME resample; "
                  "src.bootstrap.{make_seed_sequences, run_parallel_boot}.",
            "density_in_bootstrap": "f_hat held FIXED at the full-sample value "
                                    "(nuisance scale; not re-estimated per resample), "
                                    "so g is a clean linear rescaling of bootstrapped "
                                    "beta_rif.",
            "pval": "two-sided centered bootstrap p (mean(|centered|>=|gap_point|)).",
        },
        "shock": "raw log_liq.shift(1) (build_df_est_raw)",
        "regressors": REGRESSORS,
        "taus": [float(t) for t in args.taus],
        "horizons": [int(h) for h in args.horizons],
        "tau_down": TAU_DOWN, "tau_up": TAU_UP,
        "n_boot": int(args.n_boot), "block_size": int(args.block_size),
        "seed": BASE_SEED,
        "seed_namespace": {
            "scheme": "make_seed_sequences(BASE_SEED, TEST_ID_GAP, alpha_int, h, n)",
            "test_id_gap": TEST_ID_GAP,
        },
        "n_rows_estimation": int(len(df_est)),
        "panel": str(CFG.FILES.econ_core_full),
        "outputs": ["rif_lp.csv", "rif_lp_gap.csv", "rif_lp_meta.json"],
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    save_outputs(df_results, df_gap, meta, args.out_dir)
    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
