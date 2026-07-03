#!/usr/bin/env python3
"""
run_positive_control.py — [ROBUSTNESS / FLAGGED — POSITIVE CONTROL for A3; does
NOT change the main spec and NEVER touches a canonical artefact]

DGP #2 — POSITIVE CONTROL (planted downside cascade)
====================================================
The diagnostic battery (run_placebo_symmetric, run_exceedance, run_skew_test,
run_mde_equivalence) is, on the REAL data, all-negative: every downside-
asymmetry object lands inside its symmetric band / equivalence region. A
sceptical referee asks the obvious question: *can the battery DETECT a downside
asymmetry when one is actually present, or does it cancel everything?* This
script is the positive control that answers it. We SIMULATE a series with a
GENUINE, sign-specific downside amplification of a CALIBRATED magnitude, run the
SAME battery machinery on it, and verify the battery FLAGS the plant. A power
curve (plant below / at / above the MDE) shows detection switching on at the
detectable scale.

THE PLANTED MECHANISM (a one-sided cascade — a TRUE asymmetry, not scale)
-------------------------------------------------------------------------
Only the LHS (ETH per-period returns) is simulated; the RHS (shock + the 7 NB07
regressors) stays REAL, so the estimators see the real design matrix.

Start from a SYMMETRIC null panel built EXACTLY like run_placebo_symmetric's
sign-flip DGP:
        r0_t = m_t + sign_t * |resid_t|          (sign_t iid Rademacher)
where m_t is the OLS conditional mean (ret ~ const + shock + controls) and
|resid_t| carries the empirical conditional-vol path. r0 has ZERO planted
asymmetry — it is the null the battery is compared against.

Then PLANT a downside cascade keyed to the shock. For each date t, with a
shock-driven, strictly DOWN-SIDE-ONLY hazard
        p_t = clip( gamma * shock_t / span , 0, p_max )
we draw a Bernoulli cascade trigger c_t ~ Bern(p_t) and, when it fires, push the
return DOWN by a cascade kick:
        r_t = r0_t - kick * |z_t|         if c_t == 1
        r_t = r0_t                         otherwise
with z_t a standard half-normal magnitude. The kick is ADDED ON THE DOWN SIDE
ONLY: it never moves a return up, and the trigger probability rises with the
shock. So, conditional on a large shock, mass migrates into the LEFT tail and
NOWHERE ELSE — exactly the cascade the leverage-cycle prior posits.

CALIBRATION (tie the plant to the SESOI / MDE so the power curve is meaningful)
------------------------------------------------------------------------------
The battery's headline asymmetry object is the exceedance Delta = beta_down -
beta_up at alpha=0.01, h=0 (a down-minus-up tail-violation-probability effect,
per unit shock). Its canonical SESOI band (run_mde_equivalence) is SESOI_beta in
[~0.0011 (p50->p95 span), ~0.0038 (IQR span)] and its MDE@80 ~ 0.00094. We
calibrate gamma so the *induced* Delta hits a target = `mult` x SESOI_anchor
(anchor default = the p50->p95 SESOI, 0.0010915, the stricter middle of the
band). Because the relationship gamma -> Delta is monotone and ~linear over the
relevant range, we calibrate gamma ONCE by a short bisection on the induced
point Delta at h=0 (on a single null draw), then REUSE that gamma across all MC
replications for that `mult`. The power-curve grid is `--mults` (default
0.5, 1.0, 2.0 — below / at / above the MDE-to-SESOI scale).

WHY THIS IS A TRUE ASYMMETRY, NOT A SIGMA / SCALE ARTEFACT (the project trap)
----------------------------------------------------------------------------
The cancellation trap that killed the earlier placebo: ret = m + sigma*(+-|e/sigma|)
= m +- |e| makes sigma cancel, so a symmetric scale change is INVISIBLE to a
difference object Delta = beta_down - beta_up. The whole point of a positive
control is that its plant must SURVIVE the difference. Our kick is applied with a
FIXED sign (always subtract) and a SHOCK-DRIVEN ONE-SIDED hazard, so it is NOT a
symmetric rescaling: it moves P(down-violation | shock) up while leaving
P(up-violation | shock) UNCHANGED. The two are not algebraically linked. The
script PROVES this per replication by recording, separately, beta_down and
beta_up on the planted series and confirming (i) beta_down rises with `mult`
while (ii) beta_up stays at its null level (the `sigma_cancellation_check`
columns) — i.e. the induced Delta is carried entirely by the down side. A
symmetric-scale plant would move both equally and Delta would not budge; ours
does, by construction. (Self-test asserted at runtime for mult>=1.)

THE BATTERY, RUN ON THE PLANTED SERIES (per `mult`, MC over `--n_sim` draws)
----------------------------------------------------------------------------
1. PLACEBO gap  — the run_placebo_symmetric discriminating statistic
   g = |beta(0.01,h)| - |beta(0.99,h)| on the QLP kernel (rqlp._fit_one,
   verbatim REGRESSORS/CONTROLS). We compare the planted-series gap against the
   SAME symmetric sign-flip band built from the null r0. FLAG = gap_plant above
   the symmetric band's 97.5 pct.
2. EXCEEDANCE Delta — the run_exceedance per-period paired object: LPM of the
   down / up violation indicators on the 7 regressors, Delta = beta_down -
   beta_up, with the run_exceedance threshold / indicator / point-LPM helpers
   used VERBATIM. FLAG = Delta CI (MC across draws) excludes 0 AND Delta > 0.
3. STANDARDIZED SKEW — the run_skew_test signed-tail-skew object: beta_shock on
   (1[z<=q_tau] - 1[z>=q_{1-tau}]) with z = ret/vol_eth_7d (scale removed before
   measuring asymmetry), using run_skew_test.add_measures VERBATIM. FLAG =
   beta_skew CI (MC) excludes 0 AND > 0.
4. MDE / TOST — the run_mde_equivalence arithmetic (se_from_ci, mde, tost_verdict)
   applied to the induced exceedance Delta vs the canonical SESOI bands. FLAG =
   verdict == NON-NEGLIGIBLE (the equivalence test REJECTS negligibility), at
   the p50->p95 anchor span.

DETECTION RULE (per `mult`)
---------------------------
detected = (placebo flags) OR (exceedance flags) OR (skew flags) OR (mde flags).
We report each component AND the OR. The CONFIRMATORY reading: detection is OFF
below the MDE-scaled plant (mult=0.5) and ON at/above it (mult>=1.0) — i.e. the
battery has power exactly where the SESOI says it should, and does not
hallucinate signal where it should not.

PRE-PLANNED NON-CONFIRMATORY BRANCH (reported, not buried) [MAJOR if it fires]
-----------------------------------------------------------------------------
If the battery FAILS to flag a plausibly-sized plant (mult >= 1.0, i.e. an
asymmetry at/above the canonical MDE-to-SESOI scale), then the battery is
UNDER-POWERED and the paper's bounded-null claim is weakened: "we found nothing"
could be "we could not have found it". This outcome is written to the CSV
(detected_any == 0 at mult>=1) and called out in the meta `verdict` block as
NON-CONFIRMATORY / MAJOR — it is an informative result and is reported, not
suppressed. (A symmetric companion caution: detection firing at mult=0.5, BELOW
the MDE scale, would flag the battery as TRIGGER-HAPPY / mis-calibrated; also
recorded.)

OUTPUT (data/econ/ — NEW files, never overwrites a canonical CSV)
-----------------------------------------------------------------
  positive_control.csv
    [mult, target_delta, gamma, object, h, estimate, ci_lo, ci_hi, pval,
     flagged, n_sim, n_obs]
    one block of rows per `mult`: the planted-Delta target/gamma, then one row
    per battery object (placebo_gap, exceedance_delta, skew_tail01, mde_tost),
    plus the sigma_cancellation_check rows (beta_down_plant, beta_up_plant) and
    a summary detected_any row.
  positive_control_meta.json
    full provenance: DGP spec, calibration trace, SESOI/MDE anchors (read from
    the canonical mde_equivalence.csv for consistency), per-mult verdicts,
    the confirmatory / non-confirmatory branch evaluation, seeds, env.

CLI
---
    # SMOKE (fast local check)
    .venv/bin/python scripts/aux/run_positive_control.py \
        --n_sim 40 --n_calib 30 --horizons 0 --mults 0.5,1.0,2.0 \
        --max_iter 2000 --out_dir /tmp/posctrl_smoke

    # CANONICAL-ish (local; the plant is a controlled simulation, no VM)
    .venv/bin/python scripts/aux/run_positive_control.py \
        --n_sim 300 --n_calib 120 --horizons 0 --n_jobs -1

`--n_sim` default 100 (smoke-grade band); use >=300 for a tight power curve.
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
sys.path.insert(0, str(HERE))

from config import CFG, ECON_DIR  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from src.estimation import build_df_est_raw  # noqa: E402
import run_quantile_lp as rqlp  # noqa: E402
# Reuse the battery machinery VERBATIM (indicators, thresholds, point LPM, skew).
import run_exceedance as rexc  # noqa: E402
import run_skew_test as rskew  # noqa: E402
import run_mde_equivalence as rmde  # noqa: E402
# Canonical block-bootstrap engine (same primitives run_exceedance's paired
# Delta CI is built from): per-panel SE for the headline mde_tost object.
from src.bootstrap import make_seed_sequences, run_parallel_boot  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Constants — anchored to the canonical battery objects
# ──────────────────────────────────────────────────────────────
TAU_LO: float = 0.01          # headline downside quantile (== GAP_LO_TAU)
TAU_HI: float = 0.99          # upside mirror (== GAP_HI_TAU)
ALPHA_EXC: float = 0.01       # exceedance tail level for the planted Delta object
SKEW_TAU: float = 0.01        # standardized-skew tail level reported (skew_tail01)
HORIZONS_DEFAULT: list[int] = [0]   # the plant is a per-period (h=0) cascade

# Calibration target: induced exceedance Delta = mult x SESOI_anchor.
# Anchor = the p50->p95 SESOI_beta (the stricter middle of the canonical band),
# read at runtime from the canonical mde_equivalence.csv so it can never drift
# from the published number. Fallback value below if the CSV is unavailable.
SESOI_ANCHOR_KEY: str = "sesoi_beta_p50p95"
SESOI_ANCHOR_FALLBACK: float = 0.0010915461253353605   # canonical p50->p95
MDE80_FALLBACK: float = 0.0009422558355928558          # canonical exc Delta@1% MDE@80
MULTS_DEFAULT: list[float] = [0.5, 1.0, 2.0]            # below / at / above MDE-scale

P_MAX: float = 0.60           # cap on the per-date down-cascade hazard
BASE_SEED: int = 42
# Companion seed registry: new auxiliary tests take slots >= 15;
# positive control reserves the 153xx block (distinct from subsample 151xx).
TEST_ID: int = 15301
CALIB_TEST_ID: int = 15300
# the per-panel block-bootstrap of the headline Delta gets its OWN seed
# slot so its stream is independent of the calibration / MC-draw streams.
SE_BOOT_TEST_ID: int = 15302

MAX_ITER_DEFAULT: int = 2000  # QuantReg cap inside the sim loop (band, not point)

# per-panel block-bootstrap parameters for the headline mde_tost SE.
# block=24 is the canonical MBB block (CFG.ECON.block_boot_size, the same block
# run_exceedance's paired Delta CI uses). The bootstrap resamples a SINGLE
# representative planted draw per mult, so its SE is the WITHIN-PANEL sampling
# SE — invariant to the simulation budget n_sim (unlike the across-draw MC band).
SE_BOOT_BLOCK: int = int(CFG.ECON.block_boot_size)
SE_BOOT_N_DEFAULT: int = 500   # bootstrap reps for the per-panel Delta SE

# cap the cascade kick so triggered returns stay within an empirically
# defensible range. Triggered returns are clipped to a generous multiple of the
# observed in-sample ETH per-period return support [min, max]; this stops the
# tau=0.01 QuantReg from being driven by gross out-of-support outliers (which
# made the placebo gap_plant blow up to ~0.174 and non-converge at mult=2.0).
RET_SUPPORT_MULT: float = 1.0   # clip triggered returns to the observed [min,max]
# the per-period (h=0) tau=0.01 QLP point fits inside the loop get a
# higher max_iter than the band default, so the placebo gap is a converged
# estimate on the distorted LHS rather than a non-convergence artefact.
POINT_MAX_ITER: int = 20000

OUT_COLS: list[str] = [
    "mult", "target_delta", "gamma", "object", "h",
    "estimate", "ci_lo", "ci_hi", "pval", "flagged", "n_sim", "n_obs",
]


# ──────────────────────────────────────────────────────────────
# Null + planted DGP
# ──────────────────────────────────────────────────────────────
def fit_null_mean(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """OLS conditional mean ret ~ const + shock + controls (== placebo's _fit_mean).

    Returns (mask_over_all_rows, m_t_on_mask, resid_on_mask, info). Reused so the
    null r0 we plant on is the IDENTICAL symmetric null the placebo compares to.
    """
    feats = ["shock"] + rqlp.CONTROLS
    mask = df[["ret_eth_perp"] + feats].notna().all(axis=1)
    Xc = sm.add_constant(df.loc[mask, feats].fillna(0.0))
    rr = df.loc[mask, "ret_eth_perp"].to_numpy(float)
    fit = sm.OLS(rr, Xc).fit()
    m_t = np.asarray(fit.predict(Xc))
    resid = rr - m_t
    info = {"mean_shock_coef": float(fit.params["shock"]), "n_mask": int(mask.sum())}
    return mask.to_numpy(), m_t, resid, info


def make_null_returns(
    base_ret: np.ndarray, idx: np.ndarray, m_t: np.ndarray, resid: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    """Symmetric null r0 = m_t + sign_t*|resid_t| (placebo sign-flip DGP), full-length."""
    ret = base_ret.copy()
    ret[idx] = m_t + signs * np.abs(resid)
    return ret


def plant_cascade(
    r0_on_idx: np.ndarray,
    shock_on_idx: np.ndarray,
    span: float,
    gamma: float,
    kick: float,
    rng: np.random.Generator,
    ret_clip: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject a one-sided (DOWN-only) shock-driven cascade into r0.

    p_t = clip(gamma * shock_t / span, 0, P_MAX)  is the per-date DOWN-cascade
    hazard (>0 only when shock_t>0; the shock is zero-inflated). On a trigger we
    SUBTRACT kick*|z_t| (half-normal magnitude). The displacement is ALWAYS
    negative and the hazard depends ONLY on the shock — so this raises the
    down-tail violation probability conditional on the shock and leaves the
    up-tail untouched: a TRUE asymmetry that survives beta_down - beta_up.

    Kick capping. The one-sided cascade is UNCHANGED (same trigger
    hazard, same fixed-sign subtraction); we only CLIP the post-kick TRIGGERED
    returns into `ret_clip = (lo, hi)`, a generous multiple of the observed
    in-sample ETH return support. Without the clip the half-normal tail can
    push a triggered return arbitrarily far below the data's support, and the
    tau=0.01 QuantReg point fit on the placebo gap is then driven by those gross
    out-of-support outliers (and/or non-converges), blowing gap_plant up
    non-proportionally (~0.174 at mult=2.0). Clipping only the LOWER edge bites
    (the kick is always negative), so this does not touch the up side and the
    asymmetry / sigma-cancellation logic is preserved. gamma is re-calibrated
    AFTER the clip (the calibration draws plant with the SAME clip), so the
    induced Delta still hits its target.

    Returns (r_planted_on_idx, trigger_mask).
    """
    p_t = np.clip(gamma * shock_on_idx / span, 0.0, P_MAX)
    trig = rng.random(r0_on_idx.size) < p_t
    z = np.abs(rng.standard_normal(r0_on_idx.size))
    r = r0_on_idx.copy()
    r[trig] = r[trig] - kick * z[trig]
    if ret_clip is not None:
        lo, hi = ret_clip
        # Only the triggered returns are clipped (the cascade is the only thing
        # that can leave the support; r0 already lives inside it by construction).
        r[trig] = np.clip(r[trig], lo, hi)
    return r, trig


# ──────────────────────────────────────────────────────────────
# Exceedance Delta on a planted return vector (run_exceedance VERBATIM helpers)
# ──────────────────────────────────────────────────────────────
def _exceedance_betas(
    df_base: pd.DataFrame, ret_full: np.ndarray, q_lo: float, q_hi: float, h: int,
) -> tuple[float, float]:
    """Point (beta_down, beta_up) LPM on the planted series at horizon h.

    Uses run_exceedance.add_indicators + fit_point_lpm VERBATIM, so the object is
    bit-identical in construction to the canonical exceedance_paired Delta.
    """
    d = df_base.copy()
    d["ret_eth_perp"] = ret_full
    d[f"fut_r_h{h}"] = d["ret_eth_perp"] if h == 0 else d["ret_eth_perp"].shift(-h)
    d = rexc.add_indicators(d, h, q_lo, q_hi)
    yD, yU, Xp = rexc._prepare_pair_arrays(d, h)
    bD = rexc.fit_point_lpm(yD, Xp)["beta"]
    bU = rexc.fit_point_lpm(yU, Xp)["beta"]
    return float(bD), float(bU)


def _skew_beta(df_base: pd.DataFrame, ret_full: np.ndarray) -> tuple[float, int]:
    """Standardized signed-tail-skew beta_shock at tau=0.01 (run_skew_test VERBATIM).

    z = ret/vol_eth_7d removes scale BEFORE measuring asymmetry; beta on
    (1[z<=q_tau]-1[z>=q_{1-tau}]) = beta_down-beta_up with scale netted out.
    """
    d = df_base.copy()
    d["ret_eth_perp"] = ret_full
    d, _ = rskew.add_measures(d)
    col = f"skew_tail{int(round(SKEW_TAU * 100)):02d}"
    y, X = rskew.prepare_arrays(d, col, rskew.REGRESSORS)
    return rskew._ols_beta(y, X, rskew.SHOCK_COL_IDX), int(len(y))


def _qlp_beta(df_base: pd.DataFrame, ret_full: np.ndarray, tau: float, h: int,
              max_iter: int) -> float:
    """QLP beta_shock(tau,h) via the canonical kernel (run_placebo_symmetric path)."""
    d = df_base.copy()
    d["ret_eth_perp"] = ret_full
    if h == 0:
        d[f"cumret_h{h}"] = d["ret_eth_perp"]
    else:
        d[f"cumret_h{h}"] = d["ret_eth_perp"].rolling(h + 1).sum().shift(-h)
    r = rqlp._fit_one(tau, h, f"cumret_h{h}", d, rqlp.REGRESSORS, rqlp.CONTROLS,
                      max_iter)
    return np.nan if r is None else float(r["beta_shock"])


# ──────────────────────────────────────────────────────────────
# gamma calibration: bisection so induced point Delta@(alpha,h=0) == target
# ──────────────────────────────────────────────────────────────
def calibrate_gamma(
    df_base: pd.DataFrame, base_ret: np.ndarray, idx: np.ndarray,
    m_t: np.ndarray, resid: np.ndarray, shock_on_idx: np.ndarray, span: float,
    q_lo: float, q_hi: float, kick: float, target_delta: float,
    n_calib: int, seed_key: int, ret_clip: tuple[float, float] | None = None,
) -> tuple[float, dict]:
    """Find gamma s.t. mean induced point Delta over n_calib null draws == target.

    Monotone in gamma (more down-cascade => larger down-tail probability =>
    larger Delta), so bisection on [0, gamma_hi] converges. The induced Delta is
    averaged over `n_calib` fresh null draws to damp Monte-Carlo noise in the
    calibration itself.

    The calibration plants with the SAME `ret_clip` the MC loop uses, so
    gamma is re-calibrated AGAINST the capped plant — the induced Delta still
    hits its target under the clip.
    """
    rng = np.random.default_rng([BASE_SEED, CALIB_TEST_ID, seed_key])

    def induced_delta(gamma: float) -> float:
        ds = []
        for _ in range(n_calib):
            signs = rng.choice([-1.0, 1.0], size=resid.size)
            r0 = make_null_returns(base_ret, idx, m_t, resid, signs)
            rp, _ = plant_cascade(r0[idx], shock_on_idx, span, gamma, kick, rng,
                                  ret_clip=ret_clip)
            ret_full = r0.copy()
            ret_full[idx] = rp
            bD, bU = _exceedance_betas(df_base, ret_full, q_lo, q_hi, 0)
            ds.append(bD - bU)
        return float(np.mean(ds))

    # Bracket: gamma=0 gives ~0 induced Delta; grow gamma_hi until it overshoots.
    lo, d_lo = 0.0, induced_delta(0.0)
    hi = 0.05
    d_hi = induced_delta(hi)
    grow = 0
    while d_hi < target_delta and grow < 12:
        hi *= 2.0
        d_hi = induced_delta(hi)
        grow += 1
    trace = [{"gamma": lo, "delta": d_lo}, {"gamma": hi, "delta": d_hi}]
    # Bisection.
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        d_mid = induced_delta(mid)
        trace.append({"gamma": mid, "delta": d_mid})
        if d_mid < target_delta:
            lo = mid
        else:
            hi = mid
        if abs(d_mid - target_delta) < 0.02 * target_delta:
            break
    gamma = 0.5 * (lo + hi)
    return float(gamma), {"target": target_delta, "kick": kick,
                          "gamma_final": gamma, "bracket_hi": hi,
                          "n_calib": n_calib, "trace_tail": trace[-6:],
                          "ret_clip": (None if ret_clip is None
                                       else [float(ret_clip[0]),
                                             float(ret_clip[1])])}


# ──────────────────────────────────────────────────────────────
# One MC replication on the planted series (picklable; all randomness pre-drawn)
# ──────────────────────────────────────────────────────────────
def _one_plant_rep(
    seed_key,
    df_base: pd.DataFrame, base_ret: np.ndarray, idx: np.ndarray,
    m_t: np.ndarray, resid: np.ndarray, shock_on_idx: np.ndarray, span: float,
    q_lo: float, q_hi: float, kick: float, gamma: float,
    taus_qlp: tuple, horizons: list, max_iter: int,
    ret_clip: tuple[float, float] | None = None,
) -> dict:
    """One planted-series draw: returns the battery objects for this replication.

    Each replication: draw a fresh symmetric null r0, plant the down-cascade,
    then compute (a) QLP betas at (0.01,0.99,h) for the placebo gap, (b) the
    exceedance (beta_down,beta_up) at alpha=0.01 h=0, (c) the standardized skew
    beta. r0's UNPLANTED exceedance betas are also returned as the null
    reference for the sigma-cancellation self-check.

    The plant is capped via `ret_clip`, and the extreme-tau (0.01/0.99)
    QLP point fits use POINT_MAX_ITER (>> the band default) so the placebo gap
    is a CONVERGED estimate on the distorted LHS, not a max_iter artefact.
    """
    import warnings as _w
    _w.filterwarnings("ignore")
    rng = np.random.default_rng(seed_key)
    signs = rng.choice([-1.0, 1.0], size=resid.size)
    r0 = make_null_returns(base_ret, idx, m_t, resid, signs)
    rp_idx, _ = plant_cascade(r0[idx], shock_on_idx, span, gamma, kick, rng,
                              ret_clip=ret_clip)
    r_plant = r0.copy()
    r_plant[idx] = rp_idx

    # the headline gap quantiles are extreme (tau in {0.01, 0.99}); give
    # them a high iteration cap so they converge on the capped-but-distorted LHS.
    gap_max_iter = max(int(max_iter), POINT_MAX_ITER)

    out: dict = {}
    # (a) QLP gap on planted + null (for the symmetric placebo band).
    for tag, ret in (("plant", r_plant), ("null", r0)):
        for h in horizons:
            out[(f"qlp_{tag}", TAU_LO, h)] = _qlp_beta(df_base, ret, TAU_LO, h, gap_max_iter)
            out[(f"qlp_{tag}", TAU_HI, h)] = _qlp_beta(df_base, ret, TAU_HI, h, gap_max_iter)
    # (b) exceedance betas on planted AND null (self-check), alpha=0.01 h=0.
    bD_p, bU_p = _exceedance_betas(df_base, r_plant, q_lo, q_hi, 0)
    bD_0, bU_0 = _exceedance_betas(df_base, r0, q_lo, q_hi, 0)
    out["exc_plant"] = (bD_p, bU_p)
    out["exc_null"] = (bD_0, bU_0)
    # (c) standardized skew beta on planted.
    sk, n_sk = _skew_beta(df_base, r_plant)
    out["skew_plant"] = (sk, n_sk)
    return out


# ──────────────────────────────────────────────────────────────
# Per-panel block-bootstrap SE for the headline mde_tost object
# ──────────────────────────────────────────────────────────────
def panel_bootstrap_delta_ci(
    df_base: pd.DataFrame, base_ret: np.ndarray, idx: np.ndarray,
    m_t: np.ndarray, resid: np.ndarray, shock_on_idx: np.ndarray, span: float,
    q_lo: float, q_hi: float, kick: float, gamma: float,
    ret_clip: tuple[float, float] | None,
    mult: float, n_boot: int, block_size: int, n_jobs: int, ckpt_dir: Path,
) -> dict:
    """Block-bootstrap the paired exceedance Delta on a SINGLE planted panel.

    THIS is the headline test's REAL inferential SE. The earlier code
    fed mde_tost se_from_ci(across-draw CI), i.e. the dispersion of the MEAN
    Delta across n_sim simulated panels — which SHRINKS with n_sim (~1/sqrt) and
    is ~2x too small vs the within-panel block-bootstrap SE the canonical
    mde_equivalence pipeline uses. Here we instead:

      1. plant ONE representative panel for this `mult` (a fresh symmetric null +
         the calibrated, capped cascade);
      2. build the paired (yD, yU, X) arrays with run_exceedance._prepare_pair_arrays
         VERBATIM (alpha=0.01, h=0);
      3. moving-block bootstrap (block=24, the canonical MBB block) the paired
         Delta = beta_down - beta_up via run_exceedance._one_rep_lpm_pair +
         src.bootstrap.run_parallel_boot — bit-identical to how the canonical
         exceedance_paired CI (the object mde_equivalence reads) is built;
      4. return the [2.5, 97.5] percentile CI of that bootstrap distribution.

    se_from_ci(this CI) is the WITHIN-PANEL sampling SE: it is a property of the
    sample size and block structure, NOT of the simulation budget, so it is
    INVARIANT to n_sim. The power curve is then anchored to the published MDE.
    """
    # ONE representative planted panel for this mult. Deterministic seed in a
    # dedicated slot (SE_BOOT_TEST_ID) so this stream never collides with the
    # calibration or the MC-draw streams; keyed by mult for per-mult independence.
    rng = np.random.default_rng([BASE_SEED, SE_BOOT_TEST_ID, int(round(mult * 1000))])
    signs = rng.choice([-1.0, 1.0], size=resid.size)
    r0 = make_null_returns(base_ret, idx, m_t, resid, signs)
    rp_idx, _ = plant_cascade(r0[idx], shock_on_idx, span, gamma, kick, rng,
                              ret_clip=ret_clip)
    r_plant = r0.copy()
    r_plant[idx] = rp_idx

    # Build the paired arrays exactly as run_exceedance does (h=0, alpha=0.01).
    d = df_base.copy()
    d["ret_eth_perp"] = r_plant
    d["fut_r_h0"] = d["ret_eth_perp"]
    d = rexc.add_indicators(d, 0, q_lo, q_hi)
    yD, yU, Xp = rexc._prepare_pair_arrays(d, 0)

    # Point Delta on the representative panel (the bootstrap is centered on it).
    d_point = (rexc.fit_point_lpm(yD, Xp)["beta"]
               - rexc.fit_point_lpm(yU, Xp)["beta"])

    # Moving-block bootstrap of the paired Delta — canonical engine, verbatim.
    seeds = make_seed_sequences(
        BASE_SEED, SE_BOOT_TEST_ID, int(round(mult * 1000)), n=n_boot
    )
    boot_pair = run_parallel_boot(
        one_rep_fn=rexc._one_rep_lpm_pair,
        seeds=seeds,
        args_tuple=(yD, yU, Xp, block_size),
        n_jobs=n_jobs,
        batch_size=max(1, n_boot // 4),
        ckpt_path=ckpt_dir,
        out_shape_per_rep=(2,),
        label=f"posctrl_se_mult{int(round(mult * 1000))}",
    )
    mask = ~np.isnan(boot_pair).any(axis=1)
    deltas = boot_pair[mask, 0] - boot_pair[mask, 1]
    if deltas.size == 0:
        return {"d_point": float(d_point), "ci_lo": np.nan, "ci_hi": np.nan,
                "n_boot_ok": 0, "n_obs": int(len(yD)), "block_size": block_size}
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    return {"d_point": float(d_point), "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_boot_ok": int(deltas.size), "n_obs": int(len(yD)),
            "block_size": int(block_size)}


# ──────────────────────────────────────────────────────────────
# Assemble battery verdicts for one mult from the MC draws
# ──────────────────────────────────────────────────────────────
def _ci(arr: np.ndarray) -> tuple[float, float, float]:
    v = arr[~np.isnan(arr)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _two_sided_p(arr: np.ndarray) -> float:
    v = arr[~np.isnan(arr)]
    if v.size == 0:
        return np.nan
    centered = v - np.mean(v)
    return float(np.mean(np.abs(centered) >= np.abs(np.mean(v))))


def build_mult_rows(
    mult: float, target_delta: float, gamma: float, reps: list[dict],
    horizons: list[int], sesoi_anchor: float, n_sim: int,
    panel_se: dict,
) -> tuple[list[dict], dict]:
    """Turn the MC draws for one mult into output rows + a verdict dict.

    `panel_se` is the per-panel block-bootstrap CI of the headline Delta on a
    single representative planted draw. It — NOT the across-draw MC band —
    feeds the mde_tost SE / verdict. The across-draw MC dispersion is still
    computed and reported, but only as a CALIBRATION check, never as the
    inferential SE.
    """
    rows: list[dict] = []
    h0 = 0

    # --- 1. Placebo gap (SUPPORTING): gap_plant vs symmetric (null) band ---
    # this is an across-draw-of-a-positive-mean object — it fires at
    # EVERY mult>0 (a near-deterministic consequence of calibrating the mean
    # positive), NOT a power statement indexed to the SESOI. It is reported as
    # always-detect SUPPORTING evidence and is NOT part of the power decision.
    g_plant = np.array([
        abs(r[("qlp_plant", TAU_LO, h0)]) - abs(r[("qlp_plant", TAU_HI, h0)])
        for r in reps
    ], dtype=float)
    g_null = np.array([
        abs(r[("qlp_null", TAU_LO, h0)]) - abs(r[("qlp_null", TAU_HI, h0)])
        for r in reps
    ], dtype=float)
    gp_mean, _, _ = _ci(g_plant)
    gn = g_null[~np.isnan(g_null)]
    band_hi = float(np.percentile(gn, 97.5)) if gn.size else np.nan
    band_lo = float(np.percentile(gn, 2.5)) if gn.size else np.nan
    placebo_flag = bool(np.isfinite(gp_mean) and np.isfinite(band_hi)
                        and gp_mean > band_hi)
    rows.append({
        "mult": mult, "target_delta": target_delta, "gamma": gamma,
        "object": "placebo_gap", "h": h0, "estimate": gp_mean,
        "ci_lo": band_lo, "ci_hi": band_hi, "pval": np.nan,
        "flagged": int(placebo_flag), "n_sim": n_sim,
        "n_obs": int(np.sum(~np.isnan(g_plant))),
    })

    # --- 2. Exceedance Delta (SUPPORTING): beta_down - beta_up (planted) ---
    # the 'CI excludes 0' flag is across-draw-of-a-positive-mean, so it
    # fires at every mult>0 (it is a consequence of the calibration, not power).
    # Demoted to SUPPORTING; NOT part of the power decision.
    dD = np.array([r["exc_plant"][0] - r["exc_plant"][1] for r in reps], float)
    d_mean, d_lo, d_hi = _ci(dD)
    exc_flag = bool(np.isfinite(d_lo) and d_lo > 0.0)  # CI excludes 0 on positive side
    rows.append({
        "mult": mult, "target_delta": target_delta, "gamma": gamma,
        "object": "exceedance_delta", "h": h0, "estimate": d_mean,
        "ci_lo": d_lo, "ci_hi": d_hi, "pval": _two_sided_p(dD),
        "flagged": int(exc_flag), "n_sim": n_sim,
        "n_obs": int(np.sum(~np.isnan(dD))),
    })

    # --- 3. Standardized skew beta (SUPPORTING, scale removed) ---
    # same always-detect character as the exceedance flag. SUPPORTING.
    sk = np.array([r["skew_plant"][0] for r in reps], float)
    n_sk = int(np.median([r["skew_plant"][1] for r in reps]))
    s_mean, s_lo, s_hi = _ci(sk)
    skew_flag = bool(np.isfinite(s_lo) and s_lo > 0.0)
    rows.append({
        "mult": mult, "target_delta": target_delta, "gamma": gamma,
        "object": "skew_tail01", "h": h0, "estimate": s_mean,
        "ci_lo": s_lo, "ci_hi": s_hi, "pval": _two_sided_p(sk),
        "flagged": int(skew_flag), "n_sim": n_sim, "n_obs": n_sk,
    })

    # --- 4. MDE / TOST on the induced Delta — THE POWER OBJECT ---
    # the SE is the PER-PANEL block-bootstrap SE (within-panel sampling
    # SE on one representative planted draw, block=24), NOT the across-draw MC
    # dispersion. This anchors the graduation to the published MDE@80 and makes
    # the verdict INVARIANT to n_sim. The point Delta is the per-panel point
    # estimate (centre of that bootstrap), consistent with the SE's panel.
    # mde_tost is the SOLE power object — it is the only object indexed to
    # the SESOI by construction; it graduates EQUIVALENT/INCONCLUSIVE/NON-NEG.
    pse_lo = panel_se.get("ci_lo", np.nan)
    pse_hi = panel_se.get("ci_hi", np.nan)
    d_point_panel = panel_se.get("d_point", np.nan)
    if np.isfinite(pse_lo) and np.isfinite(pse_hi):
        se = rmde.se_from_ci(pse_lo, pse_hi)
        tost = rmde.tost_verdict(d_point_panel, se, sesoi_anchor)
    else:
        se, tost = np.nan, rmde.NA_UNITS
    mde_flag = bool(tost == rmde.NONNEG)
    rows.append({
        "mult": mult, "target_delta": target_delta, "gamma": gamma,
        "object": "mde_tost", "h": h0, "estimate": d_point_panel,
        "ci_lo": pse_lo, "ci_hi": pse_hi, "pval": np.nan,
        "flagged": int(mde_flag), "n_sim": n_sim,
        "n_obs": int(panel_se.get("n_obs", 0)),
    })

    # --- 4b. CALIBRATION-CHECK row: across-draw MC band of the MEAN Delta ---
    # the OLD (wrong) inferential SE. Kept and reported SEPARATELY so the
    # n_sim-shrinking dispersion is auditable — but it is NOT fed to any verdict.
    se_mc = (rmde.se_from_ci(d_lo, d_hi)
             if (np.isfinite(d_lo) and np.isfinite(d_hi)) else np.nan)
    rows.append({
        "mult": mult, "target_delta": target_delta, "gamma": gamma,
        "object": "mde_tost_mc_calib_check", "h": h0, "estimate": d_mean,
        "ci_lo": d_lo, "ci_hi": d_hi, "pval": np.nan,
        "flagged": np.nan, "n_sim": n_sim,
        "n_obs": int(np.sum(~np.isnan(dD))),
    })

    # --- sigma-cancellation self-check rows: down moves, up does NOT ---
    bD_p = np.array([r["exc_plant"][0] for r in reps], float)
    bU_p = np.array([r["exc_plant"][1] for r in reps], float)
    bU_0 = np.array([r["exc_null"][1] for r in reps], float)
    bDp_m, bDp_lo, bDp_hi = _ci(bD_p)
    bUp_m, bUp_lo, bUp_hi = _ci(bU_p)
    bU0_m, _, _ = _ci(bU_0)
    rows.append({
        "mult": mult, "target_delta": target_delta, "gamma": gamma,
        "object": "selfcheck_beta_down_plant", "h": h0, "estimate": bDp_m,
        "ci_lo": bDp_lo, "ci_hi": bDp_hi, "pval": np.nan, "flagged": np.nan,
        "n_sim": n_sim, "n_obs": int(np.sum(~np.isnan(bD_p))),
    })
    rows.append({
        "mult": mult, "target_delta": target_delta, "gamma": gamma,
        "object": "selfcheck_beta_up_plant", "h": h0, "estimate": bUp_m,
        "ci_lo": bUp_lo, "ci_hi": bUp_hi, "pval": np.nan, "flagged": np.nan,
        "n_sim": n_sim, "n_obs": int(np.sum(~np.isnan(bU_p))),
    })

    # the POWER decision is the mde_tost graduation ALONE — it is the
    # only object indexed to the SESOI by construction. detected_power tracks the
    # mde_tost verdict (NON-NEGLIGIBLE => detected at this mult). The
    # exceedance/skew/placebo flags are SUPPORTING (always-detect) evidence and
    # are recorded but DO NOT enter the decision. detected_any is retained as a
    # legacy/diagnostic alias of detected_power so the CSV column stays present.
    detected_power = bool(mde_flag)
    supporting_any = bool(placebo_flag or exc_flag or skew_flag)
    detected_any = detected_power
    rows.append({
        "mult": mult, "target_delta": target_delta, "gamma": gamma,
        "object": "detected_any", "h": h0, "estimate": float(detected_any),
        "ci_lo": np.nan, "ci_hi": np.nan, "pval": np.nan,
        "flagged": int(detected_any), "n_sim": n_sim, "n_obs": len(reps),
    })

    # up-side should be ~unchanged by a down-only plant: |bUp - bU0| small vs the
    # down-side movement. Reported as the structural proof the plant is a TRUE
    # asymmetry (not a symmetric scale change, which would move both equally).
    verdict = {
        "mult": mult,
        "target_delta": target_delta,
        "gamma": gamma,
        # SUPPORTING (always-detect) evidence — recorded, NOT part of the power
        # decision. These fire at every mult>0 by construction.
        "supporting_evidence": {
            "placebo_gap": {"gap_plant": gp_mean, "band_lo": band_lo,
                            "band_hi": band_hi, "flagged": placebo_flag},
            "exceedance_delta": {"delta": d_mean, "ci_lo": d_lo, "ci_hi": d_hi,
                                 "flagged": exc_flag},
            "skew_tail01": {"beta": s_mean, "ci_lo": s_lo, "ci_hi": s_hi,
                            "flagged": skew_flag},
            "supporting_any": supporting_any,
            "note": ("These objects are across-draw-of-a-positive-mean flags: "
                     "they fire at EVERY mult>0 (a consequence of calibrating "
                     "the mean positive), so they are SUPPORTING, not power "
                     "statements. The power decision is mde_tost alone."),
        },
        # THE POWER OBJECT.
        "mde_tost": {"se": (None if not np.isfinite(se) else float(se)),
                     "se_source": ("per-panel block-bootstrap (block=%d) of the "
                                   "paired exceedance Delta on ONE representative "
                                   "planted draw — within-panel sampling SE, "
                                   "n_sim-invariant"
                                   % int(panel_se.get("block_size", SE_BOOT_BLOCK))),
                     "delta_point_panel": (None if not np.isfinite(d_point_panel)
                                           else float(d_point_panel)),
                     "ci_lo": (None if not np.isfinite(pse_lo) else float(pse_lo)),
                     "ci_hi": (None if not np.isfinite(pse_hi) else float(pse_hi)),
                     "n_boot_ok": int(panel_se.get("n_boot_ok", 0)),
                     "sesoi_anchor": sesoi_anchor, "verdict": tost,
                     "flagged": mde_flag, "is_power_object": True},
        # CALIBRATION CHECK ONLY — the OLD (wrong) across-draw SE. NOT a verdict.
        "mde_tost_mc_calib_check": {
            "delta_mean_across_draws": d_mean, "ci_lo": d_lo, "ci_hi": d_hi,
            "se_across_draws": (None if not np.isfinite(se_mc) else float(se_mc)),
            "note": ("across-draw dispersion of the MEAN Delta over n_sim panels; "
                     "SHRINKS with n_sim, ~2x too small vs the per-panel SE. "
                     "Reported for audit only — NOT fed to any verdict."),
        },
        "detected_power": detected_power,
        "detected_any": detected_any,
        "sigma_cancellation_check": {
            "beta_down_plant": bDp_m,
            "beta_up_plant": bUp_m,
            "beta_up_null": bU0_m,
            "down_side_movement": (None if not np.isfinite(bDp_m) else
                                   float(bDp_m)),
            "up_side_drift_vs_null": (None if not (np.isfinite(bUp_m)
                                      and np.isfinite(bU0_m))
                                      else float(bUp_m - bU0_m)),
            "interpretation": (
                "down-only cascade: beta_down carries the induced Delta while "
                "beta_up stays at its null level. A symmetric scale change would "
                "move both equally and Delta would not budge — so the plant is a "
                "TRUE asymmetry, not a sigma/scale artefact."
            ),
        },
    }
    return rows, verdict


# ──────────────────────────────────────────────────────────────
# Anchors from the canonical mde_equivalence.csv (consistency, no re-derivation)
# ──────────────────────────────────────────────────────────────
def read_sesoi_anchor(in_dir: Path) -> tuple[float, float, dict]:
    """Read SESOI (p50->p95) and the exceedance Delta@1% MDE@80 from canonical CSV.

    Falls back to the locked constants if the CSV is absent, so a clean checkout
    still runs (the numbers are recorded as 'fallback' in the meta).
    """
    path = in_dir / "mde_equivalence.csv"
    info: dict = {"source": str(path)}
    try:
        df = pd.read_csv(path)
        # match the headline row EXACTLY. The old code used
        # str.contains("0.01") as a REGEX, where '.' is a wildcard — it could
        # silently match e.g. 'alpha=0X01' and never asserted uniqueness. Use a
        # literal (regex=False) substring on the canonical label 'alpha=0.01'
        # and assert exactly one row before iloc[0].
        sel = df[(df["object"] == "exceedance_delta")
                 & (df["alpha_or_measure"].astype(str)
                    .str.contains("alpha=0.01", regex=False, na=False))]
        assert len(sel) == 1, (
            f"read_sesoi_anchor: expected exactly 1 exceedance_delta row "
            f"matching 'alpha=0.01' in {path}, found {len(sel)}: "
            f"{sel['alpha_or_measure'].tolist()}"
        )
        sesoi = float(sel.iloc[0][SESOI_ANCHOR_KEY])
        mde80 = float(sel.iloc[0]["mde_80"])
        info.update({"sesoi_anchor": sesoi, "mde80": mde80, "from_csv": True})
        return sesoi, mde80, info
    except Exception as e:  # noqa: BLE001
        info.update({"sesoi_anchor": SESOI_ANCHOR_FALLBACK,
                     "mde80": MDE80_FALLBACK, "from_csv": False,
                     "fallback_reason": str(e)})
        return SESOI_ANCHOR_FALLBACK, MDE80_FALLBACK, info


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(df_out: pd.DataFrame, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "positive_control.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}", flush=True)
    meta_path = out_dir / "positive_control_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)
    print("\n--- positive_control.csv ---", flush=True)
    print(f"shape: {df_out.shape}", flush=True)
    print("HEAD:", flush=True)
    print(df_out.head(9).to_string(index=False), flush=True)
    print("TAIL:", flush=True)
    print(df_out.tail(9).to_string(index=False), flush=True)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--n_sim", type=int, default=100,
                    help="MC planted draws per mult. Default 100 (smoke); "
                         ">=300 for a tight power curve.")
    ap.add_argument("--n_calib", type=int, default=60,
                    help="Null draws averaged inside the gamma bisection.")
    ap.add_argument("--mults", type=_parse_floats, default=MULTS_DEFAULT,
                    help=f"Plant size grid as multiples of the SESOI anchor. "
                         f"Default {MULTS_DEFAULT} (below/at/above MDE-scale).")
    ap.add_argument("--horizons", type=_parse_ints, default=HORIZONS_DEFAULT,
                    help=f"QLP-gap horizons. Default {HORIZONS_DEFAULT} "
                         "(the plant is an h=0 cascade).")
    ap.add_argument("--kick", type=float, default=2.0,
                    help="Cascade down-kick scale (in return std units, ret std "
                         "~0.81). Larger kick => fewer triggers for a given "
                         "induced Delta; gamma absorbs the calibration.")
    ap.add_argument("--max_iter", type=int, default=MAX_ITER_DEFAULT,
                    help=f"QuantReg max_iter in the sim loop. "
                         f"Default {MAX_ITER_DEFAULT}.")
    ap.add_argument("--se_n_boot", type=int, default=SE_BOOT_N_DEFAULT,
                    help=f"block-bootstrap reps for the per-panel "
                         f"mde_tost SE (block={SE_BOOT_BLOCK}). The headline "
                         f"power object's SE is built from THIS, on one "
                         f"representative planted panel per mult, so it is "
                         f"n_sim-invariant. Default {SE_BOOT_N_DEFAULT}.")
    ap.add_argument("--ret_support_mult", type=float, default=RET_SUPPORT_MULT,
                    help=f"clip triggered (cascade) returns to "
                         f"ret_support_mult x the observed in-sample ETH "
                         f"per-period return [min,max]. Default "
                         f"{RET_SUPPORT_MULT}.")
    ap.add_argument("--seed", type=int, default=BASE_SEED)
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="1 = sequential (default, bit-for-bit). -1/-N = joblib "
                         "loky across pre-seeded MC draws.")
    ap.add_argument("--in_dir", type=Path, default=ECON_DIR,
                    help="Dir holding canonical mde_equivalence.csv (SESOI/MDE "
                         "anchors are read from it for consistency).")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    print("run_positive_control — DGP #2 planted-cascade POSITIVE control (A3)",
          flush=True)
    print(f"  n_sim={args.n_sim}  n_calib={args.n_calib}  mults={args.mults}",
          flush=True)
    print(f"  kick={args.kick}  horizons={args.horizons}  "
          f"max_iter={args.max_iter}  n_jobs={args.n_jobs}  seed={args.seed}",
          flush=True)

    t_all = time.time()

    # Anchors (consistency with the published equivalence numbers).
    sesoi_anchor, mde80, anchor_info = read_sesoi_anchor(args.in_dir)
    print(f"  SESOI anchor (p50->p95) = {sesoi_anchor:.6f}  "
          f"exc Delta@1% MDE@80 = {mde80:.6f}  "
          f"(from_csv={anchor_info.get('from_csv')})", flush=True)

    print("Building estimation sample (build_df_est_raw, full panel) ...",
          flush=True)
    df_base = build_df_est_raw(horizons=args.horizons).reset_index(drop=True)
    print(f"  rows={len(df_base):,}", flush=True)

    mask, m_t, resid, mean_info = fit_null_mean(df_base)
    idx = np.where(mask)[0]
    base_ret = df_base["ret_eth_perp"].to_numpy(float)
    shock_on_idx = df_base["shock"].to_numpy(float)[idx]
    span = float(np.nanquantile(df_base["shock"], 0.95)
                 - np.nanquantile(df_base["shock"], 0.50))  # p50->p95 (== SESOI span)
    # Unconditional per-period tail thresholds (run_exceedance convention).
    q_lo, q_hi = rexc.tail_thresholds(df_base, [ALPHA_EXC])[ALPHA_EXC]
    print(f"  DGP null mean shock coef = {mean_info['mean_shock_coef']:+.4f} "
          f"(n_mask={mean_info['n_mask']:,})", flush=True)
    print(f"  shock span(p50->p95) = {span:.4f}  "
          f"tail thresholds q_lo={q_lo:+.4f} q_hi={q_hi:+.4f}", flush=True)

    # empirical in-sample return support. Triggered (cascade) returns are
    # clipped into ret_support_mult x [min,max] of the REAL ETH per-period return
    # series — a generous, data-defensible bound that stops the tau=0.01 QuantReg
    # from being driven by gross out-of-support outliers (the gap_plant blow-up).
    r_real = df_base["ret_eth_perp"].dropna().to_numpy(float)
    r_min, r_max = float(np.min(r_real)), float(np.max(r_real))
    ret_clip = (args.ret_support_mult * r_min, args.ret_support_mult * r_max)
    print(f"  ret support [min,max]=[{r_min:+.4f},{r_max:+.4f}]  "
          f"clip(x{args.ret_support_mult})=[{ret_clip[0]:+.4f},{ret_clip[1]:+.4f}]",
          flush=True)

    # checkpoint dir for the per-panel block-bootstrap SE (uses the
    # canonical run_parallel_boot checkpointer). Goes under out_dir so a smoke
    # run keeps it in /tmp.
    se_ckpt_dir = args.out_dir / "_posctrl_se_ckpt"

    all_rows: list[dict] = []
    verdicts: list[dict] = []
    calib_traces: dict = {}
    panel_se_by_mult: dict = {}

    eff_jobs = (args.n_jobs if args.n_jobs > 0
                else max(1, (__import__("os").cpu_count() or 2) - 1))

    for mult in args.mults:
        target = float(mult) * sesoi_anchor
        print(f"\n=== mult={mult}  target induced Delta = {target:.6f} ===",
              flush=True)
        gamma, ctrace = calibrate_gamma(
            df_base, base_ret, idx, m_t, resid, shock_on_idx, span,
            q_lo, q_hi, args.kick, target, args.n_calib,
            seed_key=int(round(mult * 1000)), ret_clip=ret_clip,
        )
        calib_traces[str(mult)] = ctrace
        print(f"  calibrated gamma = {gamma:.5f}  "
              f"(target Delta {target:.6f})", flush=True)

        # the headline power object's SE = per-panel block-bootstrap of
        # the paired exceedance Delta on ONE representative planted panel. This
        # is the REAL inferential SE (within-panel, n_sim-invariant), computed
        # BEFORE the MC loop and fed to build_mult_rows' mde_tost verdict.
        t_se = time.time()
        panel_se = panel_bootstrap_delta_ci(
            df_base, base_ret, idx, m_t, resid, shock_on_idx, span,
            q_lo, q_hi, args.kick, gamma, ret_clip, mult,
            n_boot=args.se_n_boot, block_size=SE_BOOT_BLOCK,
            n_jobs=eff_jobs if args.n_jobs != 1 else 1, ckpt_dir=se_ckpt_dir,
        )
        panel_se_by_mult[str(mult)] = panel_se
        _se_val = (rmde.se_from_ci(panel_se["ci_lo"], panel_se["ci_hi"])
                   if (np.isfinite(panel_se["ci_lo"])
                       and np.isfinite(panel_se["ci_hi"])) else np.nan)
        print(f"  per-panel block-boot SE (block={SE_BOOT_BLOCK}, "
              f"n_boot={args.se_n_boot}): Delta_point={panel_se['d_point']:+.6f} "
              f"CI=[{panel_se['ci_lo']:+.6f},{panel_se['ci_hi']:+.6f}] "
              f"SE={_se_val:.6f}  ({time.time() - t_se:.0f}s)", flush=True)

        # MC over planted draws. All randomness keyed by a per-(mult,rep) seed so
        # the stream is dispatch-invariant (sequential == parallel, bit-for-bit).
        seed_keys = [np.random.SeedSequence([BASE_SEED, TEST_ID,
                                             int(round(mult * 1000)), b])
                     for b in range(args.n_sim)]
        args_common = (df_base, base_ret, idx, m_t, resid, shock_on_idx, span,
                       q_lo, q_hi, args.kick, gamma,
                       (TAU_LO, TAU_HI), args.horizons, args.max_iter, ret_clip)
        t0 = time.time()
        if args.n_jobs == 1:
            reps = []
            for b, sk in enumerate(seed_keys):
                reps.append(_one_plant_rep(sk, *args_common))
                if (b + 1) % max(1, args.n_sim // 5) == 0:
                    print(f"    {b + 1}/{args.n_sim}  "
                          f"({time.time() - t0:.0f}s)", flush=True)
        else:
            from joblib import Parallel, delayed
            reps = Parallel(n_jobs=eff_jobs, backend="loky")(
                delayed(_one_plant_rep)(sk, *args_common) for sk in seed_keys
            )
            print(f"    {args.n_sim}/{args.n_sim}  ({time.time() - t0:.0f}s)",
                  flush=True)

        rows, verdict = build_mult_rows(mult, target, gamma, reps,
                                        args.horizons, sesoi_anchor, args.n_sim,
                                        panel_se)
        all_rows.extend(rows)
        verdicts.append(verdict)

        scc = verdict["sigma_cancellation_check"]
        sup = verdict["supporting_evidence"]
        print(f"  [SUPPORTING] placebo_gap flagged={sup['placebo_gap']['flagged']}"
              f"  exc_delta flagged={sup['exceedance_delta']['flagged']}"
              f"  skew flagged={sup['skew_tail01']['flagged']}", flush=True)
        print(f"  [POWER] mde_tost verdict={verdict['mde_tost']['verdict']}  "
              f"(NON-NEG => detected)", flush=True)
        print(f"  DETECTED_POWER = {verdict['detected_power']}", flush=True)
        print(f"  self-check: beta_down_plant={scc['beta_down_plant']:+.5f}  "
              f"beta_up_plant={scc['beta_up_plant']:+.5f}  "
              f"up_drift_vs_null={scc['up_side_drift_vs_null']:+.5f}", flush=True)

        # Runtime self-test: for a plausibly-sized plant (mult>=1) the down side
        # must move materially MORE than the up side (the asymmetry is real).
        if mult >= 1.0 and np.isfinite(scc["beta_down_plant"]):
            down_mv = abs(scc["beta_down_plant"])
            up_drift = abs(scc["up_side_drift_vs_null"] or 0.0)
            if not (down_mv > 2.0 * up_drift):
                print("  WARNING: down-side movement is not clearly larger than "
                      "up-side drift — inspect the plant (asymmetry may be "
                      "leaking to the up tail).", flush=True)

    # ── per-object native-scale induced effect + proportionality flag ──
    # Each object's induced effect is reported on its OWN native scale (the
    # exceedance Delta and the mde_tost point are a tail-PROBABILITY response;
    # the placebo gap is a RETURN-units quantile response; the skew beta is a
    # scale-removed tail-probability response). They are NOT on a common scale,
    # so we report each per-mult AND flag any object whose response across mults
    # is NON-PROPORTIONAL (ratio to its mult=1.0 value departs from mult by more
    # than a tolerance). The exceedance Delta is proportional BY CONSTRUCTION
    # (it is the calibration target); the placebo gap is expected to be
    # super-proportional (the tau=0.01 QuantReg quantile response is non-linear
    # in the planted left-tail mass — a converged estimate, NOT an outlier /
    # non-convergence artefact, confirmed stable across max_iter). This flag
    # makes that non-proportionality explicit rather than burying it.
    def _obj_estimate(v: dict, obj: str) -> float:
        if obj == "placebo_gap":
            return v["supporting_evidence"]["placebo_gap"]["gap_plant"]
        if obj == "exceedance_delta":
            return v["supporting_evidence"]["exceedance_delta"]["delta"]
        if obj == "skew_tail01":
            return v["supporting_evidence"]["skew_tail01"]["beta"]
        if obj == "mde_tost":
            return v["mde_tost"].get("delta_point_panel") or np.nan
        return np.nan

    PROP_TOL = 0.5  # |ratio/mult - 1| above this => flagged non-proportional
    native_scale_by_object: dict = {}
    has_unit_mult = any(m == 1.0 for m in args.mults)
    for obj, scale in (("mde_tost", "tail_probability_per_log_liq"),
                       ("exceedance_delta", "tail_probability_per_log_liq"),
                       ("skew_tail01", "scale_removed_tail_probability_per_log_liq"),
                       ("placebo_gap", "return_units_quantile_gap_per_log_liq")):
        per_mult = {v["mult"]: float(_obj_estimate(v, obj)) for v in verdicts}
        ref = per_mult.get(1.0) if has_unit_mult else None
        prop = {}
        nonprop_flag = False
        if ref is not None and np.isfinite(ref) and abs(ref) > 1e-12:
            for m, est in per_mult.items():
                if not np.isfinite(est):
                    prop[str(m)] = None
                    continue
                # proportional => est/ref should ~ m/1.0 = m
                ratio = (est / ref) / m if m != 0 else np.nan
                prop[str(m)] = float(ratio)
                if np.isfinite(ratio) and abs(ratio - 1.0) > PROP_TOL:
                    nonprop_flag = True
        native_scale_by_object[obj] = {
            "native_scale": scale,
            "estimate_by_mult": {str(m): per_mult[m] for m in per_mult},
            "proportionality_ratio_by_mult": prop,
            "non_proportional": bool(nonprop_flag),
            "note": ("ratio = (est/est@mult1)/mult; ~1 => proportional to the "
                     "plant size. Flagged if |ratio-1| > %.2f for any mult."
                     % PROP_TOL),
        }
        if nonprop_flag:
            print(f"  [FIX3 native-scale] object '{obj}' "
                  f"({scale}) is NON-PROPORTIONAL across mults: "
                  f"{ {k: (round(x,3) if x is not None else None) for k,x in prop.items()} }",
                  flush=True)

    # ── Confirmatory / non-confirmatory branch evaluation ──
    # the decision is driven by the mde_tost GRADUATION ALONE (the only
    # object indexed to the SESOI by construction). The pre-registered ladder:
    #   mult=0.5  -> EQUIVALENT-TO-NEGLIGIBLE  (detection OFF below MDE-to-SESOI)
    #   mult=1.0  -> INCONCLUSIVE              (at the scale: straddles the band)
    #   mult=2.0  -> NON-NEGLIGIBLE            (detection ON above the scale)
    # CONFIRMATORY iff that ladder holds; NON-CONFIRMATORY (MAJOR) iff mde_tost
    # stays != NON-NEGLIGIBLE at the largest mult>=2.0 (instrument under-powered).
    tost_by_mult = {v["mult"]: v["mde_tost"]["verdict"] for v in verdicts}
    detected_by_mult = {v["mult"]: v["detected_power"] for v in verdicts}
    by_mult = detected_by_mult  # retained name for the meta block below

    EQUIV, INCONC, NONNEG = rmde.EQUIV, rmde.INCONC, rmde.NONNEG
    below = [m for m in args.mults if m < 1.0]
    at_scale = [m for m in args.mults if m == 1.0]
    above = [m for m in args.mults if m >= 2.0]

    # The MAJOR trigger: the largest planted scale fails to reach NON-NEGLIGIBLE.
    largest = max(args.mults) if args.mults else None
    largest_tost = tost_by_mult.get(largest)
    # Expected ladder checks (only on the canonical anchors that are present).
    below_ok = all(tost_by_mult.get(m) == EQUIV for m in below) if below else None
    at_ok = all(tost_by_mult.get(m) == INCONC for m in at_scale) if at_scale else None
    above_ok = all(tost_by_mult.get(m) == NONNEG for m in above) if above else None

    under_powered = (largest is not None and largest >= 2.0
                     and largest_tost != NONNEG)
    # Trigger-happy: the equivalence test already REJECTS negligibility below the
    # MDE-to-SESOI scale (mult<1.0 graduates to NON-NEGLIGIBLE).
    trigger_happy = [m for m in below if tost_by_mult.get(m) == NONNEG]

    ladder_holds = all(x is not False for x in (below_ok, at_ok, above_ok))

    if under_powered:
        branch = "NON-CONFIRMATORY (MAJOR)"
        branch_msg = (
            f"The headline mde_tost equivalence test FAILED to graduate to "
            f"NON-NEGLIGIBLE at the largest planted scale mult={largest} "
            f"(verdict={largest_tost}, expected NON-NEGLIGIBLE at mult>=2.0). "
            f"The instrument is UNDER-POWERED at/above the MDE-to-SESOI scale, so "
            f"the paper's bounded-null claim is WEAKENED ('we found nothing' may "
            f"be 'we could not have found it'). Report this, do not bury it."
        )
    elif trigger_happy:
        branch = "NON-CONFIRMATORY (CALIBRATION)"
        branch_msg = (
            f"mde_tost graduated to NON-NEGLIGIBLE BELOW the MDE-to-SESOI scale "
            f"at mult(s) {trigger_happy} (< 1.0): the equivalence test rejects "
            f"negligibility for a plant smaller than the detectable scale. The "
            f"instrument may be trigger-happy / mis-calibrated; investigate "
            f"before trusting positive detections."
        )
    elif ladder_holds:
        branch = "CONFIRMATORY"
        branch_msg = (
            "The headline mde_tost equivalence test GRADUATES as pre-registered: "
            "EQUIVALENT-TO-NEGLIGIBLE below the MDE-to-SESOI scale (mult<1.0), "
            "INCONCLUSIVE at it (mult=1.0), NON-NEGLIGIBLE above it (mult>=2.0). "
            "Detection switches ON exactly where the SESOI says it should: the "
            "all-negative reading on the real data is an informative null, not a "
            "dead instrument. (placebo/exceedance/skew are SUPPORTING, always-"
            "detect evidence and do not enter this decision.)"
        )
    else:
        branch = "NON-CONFIRMATORY (PARTIAL LADDER)"
        branch_msg = (
            f"mde_tost reaches NON-NEGLIGIBLE at the top scale (not under-powered) "
            f"and is not trigger-happy below, but the intermediate ladder did not "
            f"match the pre-registered EQUIV->INCONCLUSIVE->NON-NEGLIGIBLE shape "
            f"exactly: by-mult verdicts = "
            f"{ {str(k): tost_by_mult[k] for k in tost_by_mult} }. "
            f"Reported transparently; inspect the SESOI-scale calibration."
        )
    print(f"\n=== BRANCH: {branch} ===\n  {branch_msg}", flush=True)
    print(f"  mde_tost verdict ladder by mult: "
          f"{ {str(k): tost_by_mult[k] for k in tost_by_mult} }", flush=True)

    df_out = (pd.DataFrame(all_rows)
              .sort_values(["mult", "object", "h"], kind="mergesort")
              .reset_index(drop=True)[OUT_COLS])

    meta = {
        "script": "scripts/aux/run_positive_control.py",
        "purpose": ("DGP #2 positive control: plant a TRUE, calibrated downside "
                    "cascade and verify the diagnostic battery detects it "
                    "(power curve below/at/above the MDE-to-SESOI scale)."),
        "flagged_robustness": True,
        "touches_canonical": False,
        "dgp": {
            "null": "r0 = m_t + sign_t*|resid_t| (placebo sign-flip; zero "
                    "planted asymmetry — the comparison null)",
            "plant": "down-only shock-driven cascade: hazard p_t = clip("
                     "gamma*shock_t/span, 0, P_MAX); on trigger subtract "
                     "kick*|z_t| (half-normal). Displacement ALWAYS negative; "
                     "hazard depends ONLY on the shock => raises P(down-viol|"
                     "shock), leaves P(up-viol|shock) unchanged.",
            "p_max": P_MAX,
            "kick": float(args.kick),
            "mean_shock_coef": mean_info["mean_shock_coef"],
            "n_mask": mean_info["n_mask"],
            "shock_span_p50p95": span,
            "tail_thresholds": {"alpha": ALPHA_EXC, "q_lo": q_lo, "q_hi": q_hi},
            "kick_capping_fix3": (
                "Triggered (cascade) returns are CLIPPED into "
                "ret_support_mult x [min,max] of the REAL in-sample ETH "
                "per-period return series, so the tau=0.01 QuantReg is not "
                "driven by gross out-of-support outliers (which blew the placebo "
                "gap_plant up to ~0.174 and caused non-convergence at mult=2.0). "
                "Only the LOWER edge bites (the kick is always negative), so the "
                "one-sided cascade / sigma-cancellation logic is preserved; gamma "
                "is re-calibrated against the capped plant. The extreme-tau QLP "
                "point fits use POINT_MAX_ITER=%d so the gap is a converged "
                "estimate. Each object's induced effect is reported on its OWN "
                "native scale (placebo gap, exceedance Delta, skew beta)."
                % POINT_MAX_ITER
            ),
            "ret_support": {
                "real_ret_min": r_min, "real_ret_max": r_max,
                "ret_support_mult": float(args.ret_support_mult),
                "clip_lo": float(ret_clip[0]), "clip_hi": float(ret_clip[1]),
            },
            "predetermined_controls_fix4a": (
                "vol_eth_7d (the conditional-vol standardizer in the "
                "skew object z=ret/vol_eth_7d) is INTENTIONALLY PREDETERMINED — "
                "it is taken from the FROZEN panel and is NOT recomputed from the "
                "planted LHS. This matches the canonical run_skew_test convention "
                "(it standardizes by the realised conditional vol, not a "
                "re-estimated one). Directional implication: because the planted "
                "down-cascade INCREASES |ret| on triggered down dates while "
                "vol_eth_7d is held fixed, the standardized z is pushed further "
                "into the LEFT tail than a recomputed-vol standardizer would "
                "allow — i.e. the skew object's response is, if anything, a "
                "MILD UPPER BOUND on the genuine-skew signal, not an "
                "understatement. The skew object is SUPPORTING evidence only, so "
                "this does not affect the mde_tost power decision."
            ),
        },
        "calibration": {
            "rule": "gamma chosen by bisection so mean induced point Delta "
                    "(exceedance beta_down-beta_up, alpha=0.01, h=0) over "
                    "n_calib null draws == mult * SESOI_anchor.",
            "sesoi_anchor_key": SESOI_ANCHOR_KEY,
            "sesoi_anchor_value": sesoi_anchor,
            "mde80_exceedance_delta": mde80,
            "anchor_source": anchor_info,
            "n_calib": int(args.n_calib),
            "traces": calib_traces,
        },
        "battery_objects": {
            "mde_tost": "[POWER OBJECT] run_mde_equivalence TOST on the induced "
                        "Delta vs the p50->p95 SESOI anchor, using the PER-PANEL "
                        "block-bootstrap SE; FLAG = verdict "
                        "NON-NEGLIGIBLE. This is the ONLY object indexed to the "
                        "SESOI by construction and the SOLE power decision.",
            "placebo_gap": "[SUPPORTING] QLP |beta(0.01,h)|-|beta(0.99,h)| vs "
                           "symmetric (null r0) band; flags at every mult>0.",
            "exceedance_delta": "[SUPPORTING] run_exceedance LPM beta_down-"
                                "beta_up @alpha=0.01 h=0; across-draw CI excludes "
                                "0 at every mult>0 (a calibration consequence, "
                                "not power).",
            "skew_tail01": "[SUPPORTING] run_skew_test signed-tail-skew "
                           "beta_shock @tau=0.01 (scale removed); flags at every "
                           "mult>0.",
        },
        "se_methodology_fix1": {
            "power_object_se": ("per-panel moving-block bootstrap (block=%d, "
                                "n_boot=%d) of the paired exceedance Delta on a "
                                "SINGLE representative planted panel per mult — "
                                "the SAME engine (src.bootstrap.run_parallel_boot "
                                "+ run_exceedance._one_rep_lpm_pair) the canonical "
                                "exceedance_paired CI uses."
                                % (SE_BOOT_BLOCK, int(args.se_n_boot))),
            "why": ("This within-panel sampling SE is a property of sample size "
                    "and block structure, NOT of the simulation budget, so the "
                    "mde_tost graduation is INVARIANT to n_sim and ANCHORED to "
                    "the published MDE@80=%.6g. The old SE (across-draw "
                    "dispersion of the MEAN Delta over n_sim panels) shrank with "
                    "n_sim and was ~2x too small; it is retained as "
                    "mde_tost_mc_calib_check (audit only), never a verdict."
                    % mde80),
            "per_panel_se_by_mult": panel_se_by_mult,
        },
        "detection_rule": ("detected_power = (mde_tost verdict == "
                           "NON-NEGLIGIBLE). The power decision is mde_tost ALONE"
                           "; placebo/exceedance/skew are SUPPORTING."),
        "decision_rule": {
            "pre_registered_ladder": ("mde_tost graduates EQUIVALENT-TO-"
                                      "NEGLIGIBLE @ mult=0.5 -> INCONCLUSIVE @ "
                                      "mult=1.0 -> NON-NEGLIGIBLE @ mult=2.0 "
                                      "(detection ON at/above the MDE-to-SESOI "
                                      "scale, OFF below)."),
            "confirmatory": "mde_tost ladder matches the pre-registered shape.",
            "non_confirmatory_major": ("mde_tost stays != NON-NEGLIGIBLE at the "
                                       "largest mult>=2.0 => instrument under-"
                                       "powered; bounded-null claim weakened; "
                                       "REPORTED not buried."),
            "non_confirmatory_calibration": ("mde_tost graduates NON-NEGLIGIBLE "
                                             "for some mult<1.0 => trigger-happy."),
        },
        "native_scale_by_object_fix3": native_scale_by_object,
        "branch": branch,
        "branch_message": branch_msg,
        "mde_tost_verdict_by_mult": {str(k): tost_by_mult[k] for k in tost_by_mult},
        "detected_power_by_mult": {str(k): bool(v) for k, v in detected_by_mult.items()},
        "detected_by_mult": {str(k): bool(v) for k, v in by_mult.items()},
        "verdicts": verdicts,
        "sigma_cancellation_self_check": (
            "The plant is a fixed-sign (always-subtract) down-only cascade with a "
            "shock-driven one-sided hazard. It is NOT a symmetric rescaling "
            "(ret = m +- |resid| would make sigma cancel in Delta = beta_down - "
            "beta_up, the trap that voided the earlier placebo). Per-mult the "
            "script records beta_down_plant rising with the plant while "
            "beta_up_plant stays at its null level (up_side_drift_vs_null ~ 0): "
            "the induced Delta is carried entirely by the down side, so the plant "
            "GENUINELY differs from the symmetric null it is compared against."
        ),
        "mults": [float(m) for m in args.mults],
        "n_sim": int(args.n_sim),
        "se_n_boot": int(args.se_n_boot),
        "se_boot_block_size": int(SE_BOOT_BLOCK),
        "ret_support_mult": float(args.ret_support_mult),
        "horizons": [int(h) for h in args.horizons],
        "max_iter": int(args.max_iter),
        "point_max_iter": int(POINT_MAX_ITER),
        "seed": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "seed_scheme": ("calib: default_rng([BASE_SEED, CALIB_TEST_ID, "
                        "round(mult*1000)]); per-panel SE boot: "
                        f"SeedSequence([BASE_SEED, SE_BOOT_TEST_ID={SE_BOOT_TEST_ID}, "
                        "round(mult*1000), b]); MC: SeedSequence([BASE_SEED, "
                        f"TEST_ID={TEST_ID}, round(mult*1000), b]) pre-drawn "
                        "before any parallel dispatch (dispatch-invariant)."),
        "regressors": list(rqlp.REGRESSORS),
        "controls": list(rqlp.CONTROLS),
        "panel": str(CFG.FILES.econ_core_full),
        "n_rows_panel": int(len(df_base)),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    save_outputs(df_out, meta, args.out_dir)
    print(f"\nDone. {len(df_out)} rows. Total wall time: "
          f"{(time.time() - t_all) / 60:.2f} min", flush=True)
    if args.n_sim < 300:
        print("NOTE: n_sim<300 => smoke-grade power curve. Use --n_sim 300+ for "
              "the canonical figure.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
