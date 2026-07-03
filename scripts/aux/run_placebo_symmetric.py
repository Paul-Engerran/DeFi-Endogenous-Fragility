#!/usr/bin/env python3
"""
run_placebo_symmetric.py — [ROBUSTNESS / backs A3 — does NOT change the main spec]

Carriero-style SYMMETRIC-DENSITY PLACEBO (a reproducible promotion of the
original smoke implementation, which this script supersedes).

Question (A3 / CCM 2024, Carriero et al.)
-----------------------------------------
Is the apparent downside asymmetry in the quantile-LP IRF — a large, negative
beta(tau=0.01) and a left-heavy left-right gap  g_h = |beta_h(0.01)| - |beta_h(0.99)|
— a genuine SKEW response of ETH returns to DeFi liquidations, or merely an
artefact of conditional HETEROSKEDASTICITY (scale) when innovations are in fact
symmetric? Carriero / Caldara-Cascaldi-Garcia-Manchego (CCM 2024) show that a
shock raising the *scale* of a symmetric conditional distribution mechanically
fans the tails out symmetrically, inflating |beta| at BOTH tails. The
discriminating object is therefore not |beta(0.01)| alone (large even under
symmetry) but the *left-right gap*: symmetry implies g_h ~ 0.

Design — TWO DGPs (--dgp), one algebraic discovery
---------------------------------------------------
Only the LHS (ETH returns) is simulated; the RHS (shock + controls) stays REAL.

DESIGN NOTE: the original implementation
intended  ret_t = m_t + sigma_t * e_t  with  e_t = sign_t * |resid_t/sigma_t|,
but the SAME-t magnitude makes sigma_t cancel ALGEBRAICALLY:
    m_t + sigma_t * sign_t * |resid_t| / sigma_t  =  m_t ± |resid_t|.
The "rolling" and "garch" blocks therefore simulated the IDENTICAL DGP
(differing only by RNG stream) — the dual-vol-spec robustness claim was vacuous.
The honest recast is BOTH of the following, now explicit:

  --dgp sign_flip   (DEFAULT — the historical *effective* DGP, now named
        honestly): ret_t = m_t ± |resid_t| with iid Rademacher signs — a
        wild-bootstrap-style sign-flip placebo. Zero skew by construction,
        and it preserves the EXACT empirical conditional-volatility path
        (|resid_t| carries sigma_t) with NO vol model at all — the strongest,
        model-free version of the CCM question. Runs ONCE (vol_models is
        ignored; labelled "empirical").

  --dgp model_scaled  (the originally *intended* dual-spec design): per sim,
        standardised magnitudes are PERMUTED across dates before rescaling,
            ret_t = m_t + sigma_t * sign_t * |resid_{pi(t)} / sigma_{pi(t)}|,
        so sigma_t no longer cancels and the two vol specs become genuinely
        distinct:
          (1) "rolling": shock-driven log-variance OLS (skeptic's channel);
          (2) "garch"  : GARCH(1,1) on the mean residual (arch package),
                         shock-agnostic pure scale model.
        This is the vol-model-dependence robustness check.

The EXACT quantile-LP kernel (run_quantile_lp._fit_one, same REGRESSORS /
CONTROLS / kernel kwargs) is run on each simulated panel, for every requested
(tau, h). We compare, per (vol_model, h):
  - beta_real            : beta(tau) on the REAL returns;
  - beta_placebo_mean    : mean across symmetric sims of beta(tau);
  - placebo_ci_[lo,hi]   : 2.5 / 97.5 pct band of the symmetric-sim beta(tau);
  - gap_real             : |beta_real(0.01)| - |beta_real(0.99)|;
  - gap_placebo (mean)   : mean symmetric-sim gap, with a band for inference.

Interpretation
--------------
  gap_real INSIDE the symmetric-sim gap band  -> asymmetry is a volatility/scale
                                                 artefact (consistent with A3 /
                                                 Carriero / CCM 2024);
  gap_real ABOVE the band                      -> genuine downside-specific skew.
The per-(vol_model,h) verdict is written to placebo_symmetric_meta.json.

OUTPUT
------
  data/econ/placebo_symmetric.csv
    [vol_model, h, tau, beta_real, beta_placebo_mean,
     placebo_ci_lo, placebo_ci_hi, gap_real, gap_placebo]
    (gap_* are the per-(vol_model,h) left-right gap; repeated on each tau row of
     that (vol_model,h) block for convenience.)
  data/econ/placebo_symmetric_draws.csv
    PER-SIM beta draws [vol_model, sim, tau, h, beta] — the raw distribution
    behind the bands (Fig F3 plots the per-sim gap distribution vs gap_real).
  data/econ/placebo_symmetric_meta.json
    run provenance + per-(vol_model,h) gap band & verdict + DGP diagnostics.

CLI
---
    .venv/bin/python scripts/aux/run_placebo_symmetric.py            # canonical-ish defaults
    .venv/bin/python scripts/aux/run_placebo_symmetric.py \
        --n_sim 30 --horizons 0,12 --vol_models rolling \
        --out_dir /tmp/placebo_smoke                                 # SMOKE
    .venv/bin/python scripts/aux/run_placebo_symmetric.py --n_sim 500  # CANONICAL

`--n_sim` default 100 (smoke-grade); use >=500 for the canonical/VM run.
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
import statsmodels.api as sm  # noqa: E402
from src.estimation import build_df_est_raw  # noqa: E402
import run_quantile_lp as rqlp  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Defaults — mirror the canonical pipeline (run_quantile_lp + smoke)
# ──────────────────────────────────────────────────────────────
# Canonical 6-quantile grid (run_quantile_lp.QUANTILES_DEFAULT) PLUS 0.99, the
# upside mirror of 0.01 needed for the left-right gap (0.99 is not in the main
# grid; the smoke fits it on-the-fly, we do the same).
TAUS_DEFAULT: list[float] = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
# The mirror pair whose betas define the discriminating gap statistic.
GAP_LO_TAU: float = 0.01
GAP_HI_TAU: float = 0.99
HORIZONS_DEFAULT: list[int] = [0, 6, 12, 18, 24]
VOL_MODELS_DEFAULT: list[str] = ["rolling", "garch"]

# QuantReg max_iter inside the sim loop. The canonical NB07 uses 20000 (reached
# at tau=0.50); for the many-sim placebo we use a smaller cap for tractability —
# the placebo statistic is a sampling band, not a point estimate, so a tighter
# cap is acceptable and is recorded in the meta. Override with --max_iter.
MAX_ITER_DEFAULT: int = 2000
SEED_DEFAULT: int = 42

# Stable per-vol-model seed keys. The original code derived this key from
# abs(hash(vol_model)) — but Python str hashes are randomised per process
# (PYTHONHASHSEED), so the simulation stream was NOT reproducible across runs.
# Fixed mapping = deterministic stream; pre-fix smoke numbers were
# stream-dependent and are superseded by the canonical run.
VOL_MODEL_SEED_KEY: dict = {"empirical": 0, "rolling": 1, "garch": 2}

DGP_MODES: tuple = ("sign_flip", "model_scaled", "pareto_kappa")
DGP_DEFAULT: str = "sign_flip"

# ──────────────────────────────────────────────────────────────
# pareto_kappa — kappa-LIGHT symmetric tail-thickness sweep (item A2)
# ──────────────────────────────────────────────────────────────
# A SYMMETRIC placebo whose tail thickness is swept by a single shape kappa
# WITHOUT touching skew or variance. The signed innovation is
#     e_t = sign_t * c(kappa) * |resid_t| ** kappa ,    sign_t ~ Rademacher,
# where the scale c(kappa) holds the innovation VARIANCE EXACTLY at the OLS
# mean-residual variance:
#     c(kappa) = sqrt( Var(resid) / mean(|resid| ** (2*kappa)) ).
# Because mean(resid) ~ 0 we take Var(resid) = mean(resid**2). At kappa=1,
# c(1)=1 and the DGP COLLAPSES to the sign_flip null exactly (the transform is
# a strict generalisation of the historical placebo, nesting it at kappa=1) —
# so any (kappa != 1) effect is a pure tail-thickness perturbation at FIXED
# variance and ZERO population skew (the Rademacher sign makes every odd moment
# vanish in expectation). The sweep fattens (kappa>1) or thins (kappa<1) the
# tails; excess kurtosis moves monotonically while variance is pinned — this is
# the discriminating perturbation for "is the null gap invariant to symmetric
# tail thickness?". The transform is genuinely NOT a no-op (verified: excess
# kurtosis sweeps a wide range across the grid while variance is held to ~1e-7
# relative error) — see dgp_meta["kappa_diagnostics"].
KAPPAS_DEFAULT: list[float] = [0.5, 1.0, 1.5, 2.0]
# The null reference: kappa=1 reproduces sign_flip exactly (c(1)=1).
KAPPA_NULL: float = 1.0
# Per-draw sample-skew distribution is recorded over this many extra sign draws
# (cheap: no kernel refit) to honestly disclose that per-draw skew DISPERSION
# grows with kappa even though POPULATION skew stays ~0.
SKEW_PROBE_DRAWS: int = 400
# Power gate: a 2.8 = z_{0.975}+z_{0.80} multiplier maps the per-draw
# gap sd to an MDE@80-style detectable effect (same convention as
# run_subsample_stability MDE@80 = 2.8 * SE). A cell is 'adequately powered'
# only where the band HALF-WIDTH is below the A4/MDE SESOI span.
MDE80_Z: float = 2.8
# A4 / MDE SESOI spans — read from mde_equivalence.csv at runtime; these module
# constants are only the documented fallbacks (strict = p50p95, iqr).
SESOI_STRICT_FALLBACK: float = 0.0010915461253353605   # sesoi_beta p50p95 (strict)
SESOI_IQR_FALLBACK: float = 0.003833798142946502       # sesoi_beta iqr (lenient)

OUT_COLS: list[str] = [
    "dgp", "vol_model", "h", "tau",
    "beta_real", "beta_placebo_mean", "placebo_ci_lo", "placebo_ci_hi",
    "gap_real", "gap_placebo",
]

# pareto_kappa output schema (placebo_kappa.csv). The decision-relevant columns
# (band width, power gate, verdict) sit on every tau row of a (kappa,h) block.
KAPPA_OUT_COLS: list[str] = [
    "dgp", "kappa", "h", "tau",
    "beta_real", "beta_placebo_mean", "placebo_ci_lo", "placebo_ci_hi",
    "gap_real", "gap_band_width", "gap_band_half_width",
    "gap_real_inside_band", "adequately_powered", "verdict",
]


# ──────────────────────────────────────────────────────────────
# Quantile-LP kernel wrapper (reuses run_quantile_lp._fit_one verbatim)
# ──────────────────────────────────────────────────────────────
def _fit_beta(df_one: pd.DataFrame, tau: float, h: int, max_iter: int) -> float:
    """Single beta_shock from the canonical quantile-LP kernel; NaN on failure."""
    r = rqlp._fit_one(
        tau, h, f"cumret_h{h}", df_one, rqlp.REGRESSORS, rqlp.CONTROLS, max_iter
    )
    return np.nan if r is None else float(r["beta_shock"])


def _betas_for(
    df_base: pd.DataFrame,
    ret_full: np.ndarray,
    taus: list[float],
    horizons: list[int],
    max_iter: int,
) -> dict[tuple[float, int], float]:
    """Materialise cumret_h{h} from a returns vector and fit beta for each (tau,h).

    Mirrors the smoke's betas_for(): overwrite ret_eth_perp, rebuild cumret_h{h}
    via the SAME rolling(h+1).sum().shift(-h) convention as build_df_est_raw, then
    fit the canonical kernel.
    """
    d = df_base.copy()
    d["ret_eth_perp"] = ret_full
    for h in horizons:
        if h == 0:
            d[f"cumret_h{h}"] = d["ret_eth_perp"]
        else:
            d[f"cumret_h{h}"] = d["ret_eth_perp"].rolling(h + 1).sum().shift(-h)
    return {
        (tau, h): _fit_beta(d, tau, h, max_iter)
        for tau in taus
        for h in horizons
    }


# ──────────────────────────────────────────────────────────────
# DGP — conditional mean + two volatility specs + symmetric innovations
# ──────────────────────────────────────────────────────────────
def _fit_mean(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """OLS conditional mean ret ~ const + shock + controls.

    Returns (mask_bool_over_all_rows, m_t_on_mask, resid_on_mask, info).
    Verbatim DGP-mean step from the smoke.
    """
    from scipy import stats as _st
    feats = ["shock"] + rqlp.CONTROLS
    mask = df[["ret_eth_perp"] + feats].notna().all(axis=1)
    Xc = sm.add_constant(df.loc[mask, feats].fillna(0.0))
    rr = df.loc[mask, "ret_eth_perp"].to_numpy(float)
    mean_fit = sm.OLS(rr, Xc).fit()
    m_t = np.asarray(mean_fit.predict(Xc))
    resid = rr - m_t
    # OLS residual is mean-zero by construction; assert it so var_resid =
    # mean(resid**2) is the genuine variance (no dead demeaning line needed).
    assert abs(float(np.mean(resid))) < 1e-8, (
        f"OLS mean residual not ~0 (mean={np.mean(resid):.3e})")
    var_resid = float(np.mean(resid ** 2))
    info = {
        "mean_shock_coef": float(mean_fit.params["shock"]),
        "n_mask": int(mask.sum()),
        "var_resid": var_resid,
        "resid_mean": float(np.mean(resid)),
        # Excess kurtosis of the REAL frozen-panel OLS mean residual — the
        # empirical tail benchmark the kappa grid is annotated against.
        "real_resid_excess_kurtosis": float(_st.kurtosis(resid, fisher=True)),
        "real_resid_skew": float(_st.skew(resid)),
    }
    return mask.to_numpy(), m_t, resid, info


def _sigma_rolling(df: pd.DataFrame, mask: np.ndarray, resid: np.ndarray
                   ) -> tuple[np.ndarray, dict]:
    """Spec (1): shock-driven log-variance, sigma_t = sqrt(exp(fitted log resid^2)).

    Identical to the smoke: regress log(resid^2) on const+shock+controls so the
    skeptic's "liquidations raise volatility" channel is built into the scale.
    The empirical rolling window (CFG.ECON.vol_window) is the conceptual basis
    (vol_eth_7d is itself a 168h rolling std and enters as a control), but the
    operational sigma here is the fitted shock-driven conditional std, matching
    the smoke so this spec reproduces it.
    """
    feats = ["shock"] + rqlp.CONTROLS
    Xc = sm.add_constant(df.loc[mask, feats].fillna(0.0))
    logv = sm.OLS(np.log(resid ** 2 + 1e-8), Xc).fit()
    sig_t = np.sqrt(np.exp(np.asarray(logv.predict(Xc))))
    info = {
        "logvar_shock_coef": float(logv.params["shock"]),
        "vol_window": int(CFG.ECON.vol_window),
    }
    return sig_t, info


def _sigma_garch(resid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Spec (2): GARCH(1,1) conditional volatility of the mean residual (arch).

    Zero-mean GARCH(1,1) on the OLS-mean residual; sigma_t = conditional std.
    Independent of the shock by construction — a pure scale model that knows
    nothing about liquidations — so it is a strict skeptic's baseline: if even a
    shock-agnostic heteroskedastic process reproduces gap_real, the asymmetry is
    a scale artefact.
    """
    from arch import arch_model
    am = arch_model(resid, mean="Zero", vol="GARCH", p=1, q=1, dist="normal",
                    rescale=False)
    res = am.fit(disp="off")
    sig_t = np.asarray(res.conditional_volatility, dtype=float)
    info = {
        "garch_omega": float(res.params.get("omega", np.nan)),
        "garch_alpha": float(res.params.get("alpha[1]", np.nan)),
        "garch_beta": float(res.params.get("beta[1]", np.nan)),
        "garch_persistence": float(
            res.params.get("alpha[1]", 0.0) + res.params.get("beta[1]", 0.0)
        ),
    }
    return sig_t, info


# ──────────────────────────────────────────────────────────────
# pareto_kappa helpers — variance-preserving tail-thickness scale + SESOI IO
# ──────────────────────────────────────────────────────────────
def _kappa_scale(absR: np.ndarray, var_resid: float, kappa: float) -> float:
    """c(kappa) = sqrt(Var(resid) / mean(|resid|**(2*kappa))) — holds the
    transformed innovation's variance EXACTLY at var_resid (since the signed
    innovation is sign*c*|resid|**kappa with Rademacher sign, its variance is
    c**2 * mean(|resid|**(2*kappa)) = var_resid). At kappa=1 returns ~1."""
    denom = float(np.mean(absR ** (2.0 * kappa)))
    return float(np.sqrt(var_resid / denom)) if denom > 0 else np.nan


def _kappa_diagnostics(
    absR: np.ndarray, var_resid: float, kappas: list[float],
    real_excess_kurtosis: float, seed: int,
) -> dict:
    """Per-kappa DGP self-check (sigma-check + kappa-grid annotation).

    Records, per kappa: c(kappa); the EXACT variance of the transformed signed
    innovation (sigma-check — must equal var_resid); its excess kurtosis (proves
    the transform differs from the kappa=1 null and is NOT a no-op); and the
    EMPIRICAL per-draw sample-skew distribution over SKEW_PROBE_DRAWS Rademacher
    draws (mean ~0 = population symmetry; sd/p2.5/p97.5 honestly disclose that
    per-draw skew DISPERSION grows with kappa — an earlier informal self-check,
    which claimed the per-draw skew stays within ~1e-2, was false). Annotates
    each kappa against the REAL frozen-panel OLS mean-residual excess kurtosis
    (kappa beyond it = 'stress only')."""
    from scipy import stats as _st
    rng = np.random.default_rng([seed, 7919])   # diagnostics-only stream
    n = absR.size
    out: dict = {}
    for k in kappas:
        c = _kappa_scale(absR, var_resid, k)
        e = c * (absR ** k)                      # magnitude; sign applied below
        skews = np.empty(SKEW_PROBE_DRAWS, dtype=float)
        for j in range(SKEW_PROBE_DRAWS):
            sgn = rng.choice([-1.0, 1.0], size=n)
            skews[j] = float(_st.skew(sgn * e))
        sgn = rng.choice([-1.0, 1.0], size=n)
        inno = sgn * e
        var_inno = float(np.var(inno))
        beyond = bool(k > 1.0 and
                      _st.kurtosis(inno, fisher=True) > real_excess_kurtosis)
        out[f"{k:g}"] = {
            "c_kappa": c,
            "var_inno": var_inno,                # sigma-check: must ~ var_resid
            "var_resid_target": float(var_resid),
            "var_rel_err": float(abs(var_inno - var_resid) / var_resid),
            "excess_kurtosis_inno": float(_st.kurtosis(inno, fisher=True)),
            "is_null_kappa": bool(abs(k - KAPPA_NULL) < 1e-9),
            "per_draw_skew": {
                "mean": float(skews.mean()),     # ~0 => population symmetry
                "sd": float(skews.std(ddof=1)),  # GROWS with kappa (disclosed)
                "p2.5": float(np.percentile(skews, 2.5)),
                "p97.5": float(np.percentile(skews, 97.5)),
                "n_probe_draws": SKEW_PROBE_DRAWS,
            },
            "kappa_grid_label": (
                "null (reproduces sign_flip; c~1)" if abs(k - KAPPA_NULL) < 1e-9
                else ("beyond empirical tail, stress only" if beyond
                      else "within empirical tail regime")
            ),
        }
    return out


def _load_sesoi_spans(in_dir: Path) -> dict:
    """Read the A4/MDE SESOI beta spans from mde_equivalence.csv (strict=p50p95,
    lenient=iqr). Falls back to the locked module constants if the file is
    absent (e.g. minimal smoke env), recording which source was used."""
    csv_path = in_dir / "mde_equivalence.csv"
    if csv_path.exists():
        try:
            md = pd.read_csv(csv_path)
            strict = float(md["sesoi_beta_p50p95"].dropna().iloc[0])
            iqr = float(md["sesoi_beta_iqr"].dropna().iloc[0])
            return {"strict": strict, "iqr": iqr, "source": str(csv_path)}
        except Exception as e:  # noqa: BLE001
            print(f"  WARN reading {csv_path}: {e}; using fallback SESOI.",
                  flush=True)
    return {"strict": SESOI_STRICT_FALLBACK, "iqr": SESOI_IQR_FALLBACK,
            "source": "module fallback (mde_equivalence.csv not found)"}


def _sim_betas_batch(
    df_base: pd.DataFrame,
    base_ret: np.ndarray,
    idx: np.ndarray,
    m_t: np.ndarray,
    sig_t: np.ndarray,
    absE: np.ndarray,
    draws_batch: list,
    taus: list[float],
    horizons: list[int],
    max_iter: int,
    dgp: str,
) -> list[dict]:
    """Fit the QLP kernel on a batch of pre-drawn symmetric panels (picklable).

    Each element of draws_batch is (signs,) for dgp='sign_flip' or
    (signs, perm) for dgp='model_scaled'. All randomness is PRE-DRAWN by the
    caller in a fixed sequential order, so the batching / worker count cannot
    change the numbers — only the wall time.

    sign_flip    : ret_t = m_t + sign_t * |resid_t|          (sigma-free; absE
                   here is |resid_t| itself, sig_t is all-ones)
    model_scaled : ret_t = m_t + sig_t * sign_t * absE[perm]  (permuted
                   standardised magnitudes rescaled by the MODEL's sigma —
                   sigma no longer cancels, the vol specs genuinely differ)
    pareto_kappa : ret_t = m_t + sign_t * absE  where absE is the PRE-SCALED
                   variance-preserving tail-reshaped magnitude
                   c(kappa)*|resid_t|**kappa (caller passes the per-kappa absE;
                   sig_t is all-ones — sigma-free, like sign_flip). At kappa=1
                   absE == |resid_t| so this collapses to sign_flip exactly.
    """
    import warnings as _w
    _w.filterwarnings("ignore")
    out = []
    for draw in draws_batch:
        if dgp in ("sign_flip", "pareto_kappa"):
            (signs,) = draw
            e_scaled = signs * absE
        else:  # model_scaled
            signs, perm = draw
            e_scaled = sig_t * (signs * absE[perm])
        ret_full = base_ret.copy()
        ret_full[idx] = m_t + e_scaled
        out.append(_betas_for(df_base, ret_full, taus, horizons, max_iter))
    return out


def _chunk(seq: list, n_chunks: int) -> list[list]:
    """Split seq into ~equal contiguous chunks (preserving order)."""
    n_chunks = max(1, min(n_chunks, len(seq)))
    size = (len(seq) + n_chunks - 1) // n_chunks
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _simulate_one_vol_model(
    vol_model: str,
    df_base: pd.DataFrame,
    mask: np.ndarray,
    m_t: np.ndarray,
    sig_t: np.ndarray,
    resid: np.ndarray,
    taus: list[float],
    horizons: list[int],
    n_sim: int,
    max_iter: int,
    seed: int,
    real: dict,
    n_jobs: int = 1,
    dgp: str = DGP_DEFAULT,
) -> dict:
    """Run the symmetric-sim loop for one volatility spec under the chosen DGP.

    `real` (the per-(tau,h) betas on the REAL returns) is computed ONCE by the
    caller and shared across vol models — it does not depend on the vol spec.
    Returns dict with per-(tau,h) sim arrays + the real betas.
    """
    idx = np.where(mask)[0]
    base_ret = df_base["ret_eth_perp"].to_numpy(float)
    rng = np.random.default_rng([seed, VOL_MODEL_SEED_KEY[vol_model]])

    if dgp == "sign_flip":
        # sigma-free: |resid_t| keeps the empirical vol path; sig_t unused.
        absE = np.abs(resid)
        sig_used = np.ones_like(resid)
        draws = [(rng.choice([-1.0, 1.0], size=absE.size),) for _ in range(n_sim)]
    else:  # model_scaled
        absE = np.abs(resid / sig_t)            # standardised magnitudes
        sig_used = sig_t
        # Per sim: one sign vector THEN one date-permutation (fixed order so
        # the stream is reproducible and dispatch-invariant).
        draws = []
        for _ in range(n_sim):
            signs = rng.choice([-1.0, 1.0], size=absE.size)
            perm = rng.permutation(absE.size)
            draws.append((signs, perm))

    t0 = time.time()
    eff_jobs = n_jobs if n_jobs > 0 else max(1, (__import__("os").cpu_count() or 2) - 1)
    batches = _chunk(draws, max(1, 4 * eff_jobs))
    if n_jobs == 1:
        chunk_results = []
        done = 0
        for b in batches:
            chunk_results.append(_sim_betas_batch(
                df_base, base_ret, idx, m_t, sig_used, absE, b,
                taus, horizons, max_iter, dgp))
            done += len(b)
            print(f"    [{vol_model}/{dgp}] {done}/{n_sim}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    else:
        from joblib import Parallel, delayed
        chunk_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_sim_betas_batch)(df_base, base_ret, idx, m_t, sig_used,
                                      absE, b, taus, horizons, max_iter, dgp)
            for b in batches
        )
        print(f"    [{vol_model}/{dgp}] {n_sim}/{n_sim}  ({time.time() - t0:.0f}s)",
              flush=True)

    sim: dict[tuple[float, int], list[float]] = {
        (tau, h): [] for tau in taus for h in horizons
    }
    for chunk in chunk_results:
        for b in chunk:
            for k in sim:
                sim[k].append(b[k])
    return {"real": real, "sim": sim}


def _simulate_one_kappa(
    kappa: float,
    df_base: pd.DataFrame,
    mask: np.ndarray,
    m_t: np.ndarray,
    resid: np.ndarray,
    var_resid: float,
    taus: list[float],
    horizons: list[int],
    n_sim: int,
    max_iter: int,
    seed: int,
    real: dict,
    n_jobs: int = 1,
) -> dict:
    """Run the symmetric sign-flip loop at ONE tail-thickness kappa (pareto_kappa
    DGP). The innovation magnitude is the variance-preserving reshaped magnitude
    absE = c(kappa)*|resid|**kappa; signs are Rademacher (sigma-free, like
    sign_flip). At kappa=1 absE == |resid| so this reproduces the sign_flip null
    exactly. Stream keyed by (seed, kappa-index) so each kappa is reproducible
    and dispatch-invariant."""
    idx = np.where(mask)[0]
    base_ret = df_base["ret_eth_perp"].to_numpy(float)
    # Stable per-kappa key: round to avoid float-repr drift in the seed spawn.
    kappa_key = int(round(kappa * 1000))
    rng = np.random.default_rng([seed, kappa_key])

    absR = np.abs(resid)
    c = _kappa_scale(absR, var_resid, kappa)
    absE = c * (absR ** kappa)                  # pre-scaled reshaped magnitude
    sig_used = np.ones_like(resid)
    draws = [(rng.choice([-1.0, 1.0], size=absE.size),) for _ in range(n_sim)]

    t0 = time.time()
    eff_jobs = n_jobs if n_jobs > 0 else max(1, (__import__("os").cpu_count() or 2) - 1)
    batches = _chunk(draws, max(1, 4 * eff_jobs))
    if n_jobs == 1:
        chunk_results = []
        done = 0
        for b in batches:
            chunk_results.append(_sim_betas_batch(
                df_base, base_ret, idx, m_t, sig_used, absE, b,
                taus, horizons, max_iter, "pareto_kappa"))
            done += len(b)
            print(f"    [kappa={kappa:g}] {done}/{n_sim}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    else:
        from joblib import Parallel, delayed
        chunk_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_sim_betas_batch)(df_base, base_ret, idx, m_t, sig_used,
                                      absE, b, taus, horizons, max_iter,
                                      "pareto_kappa")
            for b in batches
        )
        print(f"    [kappa={kappa:g}] {n_sim}/{n_sim}  ({time.time() - t0:.0f}s)",
              flush=True)

    sim: dict[tuple[float, int], list[float]] = {
        (tau, h): [] for tau in taus for h in horizons
    }
    for chunk in chunk_results:
        for b in chunk:
            for k in sim:
                sim[k].append(b[k])
    return {"real": real, "sim": sim, "c_kappa": c}


# ──────────────────────────────────────────────────────────────
# Assemble rows + per-(vol_model,h) gap band
# ──────────────────────────────────────────────────────────────
def _build_rows_and_gaps(
    vol_model: str,
    results: dict,
    taus: list[float],
    horizons: list[int],
) -> tuple[list[dict], dict]:
    """Turn per-(tau,h) sim arrays into output rows + gap meta for one vol model."""
    real = results["real"]
    sim = results["sim"]
    rows: list[dict] = []
    gap_meta: dict = {}

    for h in horizons:
        # Left-right gap statistic on the mirror pair (0.01, 0.99).
        have_pair = (GAP_LO_TAU in taus) and (GAP_HI_TAU in taus)
        if have_pair:
            rb_lo = abs(real[(GAP_LO_TAU, h)])
            rb_hi = abs(real[(GAP_HI_TAU, h)])
            gap_real = rb_lo - rb_hi
            s_lo = np.abs(np.asarray(sim[(GAP_LO_TAU, h)], dtype=float))
            s_hi = np.abs(np.asarray(sim[(GAP_HI_TAU, h)], dtype=float))
            g_sim = s_lo - s_hi
            g_sim = g_sim[~np.isnan(g_sim)]
            gap_placebo = float(np.mean(g_sim)) if g_sim.size else np.nan
            g_lo = float(np.percentile(g_sim, 2.5)) if g_sim.size else np.nan
            g_hi = float(np.percentile(g_sim, 97.5)) if g_sim.size else np.nan
            verdict = (
                "GENUINE asymmetry (gap_real above symmetric band)"
                if (g_sim.size and gap_real > g_hi)
                else "scale artefact (gap_real inside symmetric band)"
            )
            gap_meta[str(h)] = {
                "gap_real": float(gap_real),
                "gap_placebo_mean": gap_placebo,
                "gap_placebo_ci_lo": g_lo,
                "gap_placebo_ci_hi": g_hi,
                "real_abs_beta_lo": float(rb_lo),
                "real_abs_beta_hi": float(rb_hi),
                "verdict": verdict,
            }
        else:
            gap_real = np.nan
            gap_placebo = np.nan

        for tau in taus:
            arr = np.asarray(sim[(tau, h)], dtype=float)
            arr = arr[~np.isnan(arr)]
            rows.append({
                "vol_model": vol_model,
                "h": int(h),
                "tau": float(tau),
                "beta_real": float(real[(tau, h)]),
                "beta_placebo_mean": float(np.mean(arr)) if arr.size else np.nan,
                "placebo_ci_lo": float(np.percentile(arr, 2.5)) if arr.size else np.nan,
                "placebo_ci_hi": float(np.percentile(arr, 97.5)) if arr.size else np.nan,
                "gap_real": float(gap_real),
                "gap_placebo": float(gap_placebo),
            })
    return rows, gap_meta


# ──────────────────────────────────────────────────────────────
# pareto_kappa — per-(kappa,h) band, power gate, power-conditional verdict
# ──────────────────────────────────────────────────────────────
def _build_kappa_rows_and_decision(
    kappa: float,
    results: dict,
    taus: list[float],
    horizons: list[int],
    sesoi: dict,
) -> tuple[list[dict], dict]:
    """Per-(kappa,h) output rows + the CORRECTED decision object.

    The stability/monotonicity diagnostic is RE-TARGETED onto the
    gap-BAND WIDTH (gap_ci_hi - gap_ci_lo) and the per-draw gap sd, NOT
    gap_placebo_mean (which is ~0 everywhere by construction, so any
    monotonicity flag on it fires on MC noise).

    DISCRIMINATING POWER per (kappa,h): the MDE-style gap the band could
    detect = band HALF-WIDTH (and 2.8*gap_sd as the MDE@80 cross-check). The
    'scale artefact' verdict is made CONDITIONAL on adequate power: a cell is
    labelled 'scale artefact (adequately powered)' only where gap_real is INSIDE
    the band AND band_half_width < SESOI span; otherwise 'inconclusive
    (underpowered)'. Per-edge MCSE of the band edges is reported so heavy-tail
    kappa runs can be flagged for raising n_sim."""
    real = results["real"]
    sim = results["sim"]
    rows: list[dict] = []
    decision: dict = {}
    sesoi_strict = sesoi["strict"]
    sesoi_iqr = sesoi["iqr"]
    have_pair = (GAP_LO_TAU in taus) and (GAP_HI_TAU in taus)

    for h in horizons:
        if have_pair:
            rb_lo = abs(real[(GAP_LO_TAU, h)])
            rb_hi = abs(real[(GAP_HI_TAU, h)])
            gap_real = rb_lo - rb_hi
            s_lo = np.abs(np.asarray(sim[(GAP_LO_TAU, h)], dtype=float))
            s_hi = np.abs(np.asarray(sim[(GAP_HI_TAU, h)], dtype=float))
            g_sim = (s_lo - s_hi)
            g_sim = g_sim[~np.isnan(g_sim)]
            n_g = int(g_sim.size)
            gap_placebo_mean = float(np.mean(g_sim)) if n_g else np.nan
            gap_sd = float(np.std(g_sim, ddof=1)) if n_g > 1 else np.nan
            g_lo = float(np.percentile(g_sim, 2.5)) if n_g else np.nan
            g_hi = float(np.percentile(g_sim, 97.5)) if n_g else np.nan
            band_width = (g_hi - g_lo) if n_g else np.nan
            band_half_width = band_width / 2.0 if n_g else np.nan
            mde80_gap = (MDE80_Z * gap_sd) if n_g > 1 else np.nan
            inside_band = bool(n_g and (g_lo <= gap_real <= g_hi))
            # MCSE of the 2.5/97.5 percentile edges (Maritz–Jarrett-style normal
            # approx): se_q ~ sqrt(q(1-q)/n) / f(x_q); we report the cheaper
            # bootstrap-free density-free proxy gap_sd/sqrt(n) for the edges,
            # which bounds raise-n_sim decisions (heavy kappa => larger).
            mcse_edge = (gap_sd / np.sqrt(n_g)) if (n_g > 1) else np.nan

            # power gate: adequately powered iff the band can resolve an
            # effect smaller than the SESOI span (strict first, then lenient).
            powered_strict = bool(n_g and band_half_width < sesoi_strict)
            powered_iqr = bool(n_g and band_half_width < sesoi_iqr)
            if not n_g:
                verdict = "no draws"
                powered = False
            elif inside_band and powered_strict:
                verdict = "scale artefact (adequately powered)"
                powered = True
            elif inside_band and powered_iqr:
                verdict = "scale artefact (adequately powered, lenient SESOI)"
                powered = True
            elif inside_band:
                verdict = "inconclusive (underpowered)"
                powered = False
            else:
                verdict = ("GENUINE asymmetry (gap_real outside symmetric band)"
                           if gap_real > g_hi else
                           "ANOMALY: gap_real below symmetric band")
                powered = bool(powered_strict or powered_iqr)

            decision[str(h)] = {
                "kappa": float(kappa),
                "c_kappa": float(results.get("c_kappa", np.nan)),
                "gap_real": float(gap_real),
                "real_abs_beta_lo": float(rb_lo),
                "real_abs_beta_hi": float(rb_hi),
                # monotonicity object = band width / per-draw sd, NOT mean
                "gap_band_width": float(band_width),
                "gap_band_half_width": float(band_half_width),
                "gap_sd": float(gap_sd),
                "gap_placebo_mean": gap_placebo_mean,   # ~0 by construction
                "gap_ci_lo": g_lo,
                "gap_ci_hi": g_hi,
                "gap_real_inside_band": inside_band,
                # discriminating power
                "mde80_gap": float(mde80_gap) if mde80_gap == mde80_gap else np.nan,
                "sesoi_strict": float(sesoi_strict),
                "sesoi_iqr": float(sesoi_iqr),
                "adequately_powered_strict": powered_strict,
                "adequately_powered_iqr": powered_iqr,
                "adequately_powered": powered,
                "band_edge_mcse": float(mcse_edge) if mcse_edge == mcse_edge else np.nan,
                "n_gap_draws": n_g,
                "verdict": verdict,
            }
        else:
            gap_real = np.nan

        for tau in taus:
            arr = np.asarray(sim[(tau, h)], dtype=float)
            arr = arr[~np.isnan(arr)]
            rows.append({
                "dgp": "pareto_kappa",
                "kappa": float(kappa),
                "h": int(h),
                "tau": float(tau),
                "beta_real": float(real[(tau, h)]),
                "beta_placebo_mean": float(np.mean(arr)) if arr.size else np.nan,
                "placebo_ci_lo": float(np.percentile(arr, 2.5)) if arr.size else np.nan,
                "placebo_ci_hi": float(np.percentile(arr, 97.5)) if arr.size else np.nan,
                "gap_real": float(gap_real),
                "gap_band_width": decision.get(str(h), {}).get("gap_band_width", np.nan),
                "gap_band_half_width": decision.get(str(h), {}).get("gap_band_half_width", np.nan),
                "gap_real_inside_band": decision.get(str(h), {}).get("gap_real_inside_band", False),
                "adequately_powered": decision.get(str(h), {}).get("adequately_powered", False),
                "verdict": decision.get(str(h), {}).get("verdict", "n/a"),
            })
    return rows, decision


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(df_out: pd.DataFrame, df_draws: pd.DataFrame, meta: dict,
                 out_dir: Path, dgp: str = DGP_DEFAULT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Each DGP variant gets its own filename stem so it can never overwrite
    # another's artefacts. pareto_kappa => placebo_kappa.csv (the NEW A2 output).
    if dgp == "sign_flip":
        stem = "placebo_symmetric"
    elif dgp == "model_scaled":
        stem = "placebo_symmetric_model_scaled"
    else:  # pareto_kappa
        stem = "placebo_kappa"
    csv_path = out_dir / f"{stem}.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}", flush=True)
    # Per-sim draws (Fig F3: distribution of the placebo gap vs the real gap).
    draws_path = out_dir / f"{stem}_draws.csv"
    df_draws.to_csv(draws_path, index=False)
    print(f"  wrote {draws_path}", flush=True)
    meta_path = out_dir / f"{stem}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)
    print(f"\n--- {stem}_draws.csv ---", flush=True)
    print(f"shape: {df_draws.shape}", flush=True)
    print("HEAD:", flush=True)
    print(df_draws.head(3).to_string(index=False), flush=True)
    print("TAIL:", flush=True)
    print(df_draws.tail(3).to_string(index=False), flush=True)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _arch_version() -> str | None:
    try:
        import arch
        return str(arch.__version__)
    except ImportError:
        return None


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_strs(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _run_pareto_kappa(args, df_base, mask, m_t, resid, mean_info, real_betas,
                      t_all) -> int:
    """item A2 driver — sweep symmetric tail thickness via kappa at FIXED
    variance, build the per-(kappa,h) power-gated decision, write
    placebo_kappa.csv (+ draws + meta). Surfaces the non-confirmatory branch."""
    var_resid = mean_info["var_resid"]
    real_exkurt = mean_info["real_resid_excess_kurtosis"]
    kappas = list(args.kappas)
    sesoi = _load_sesoi_spans(args.in_dir)

    print(f"\n=== dgp=pareto_kappa  kappas={kappas} ===", flush=True)
    print(f"  var_resid (fixed) = {var_resid:.6e}; real-resid excess "
          f"kurtosis = {real_exkurt:.3f} (empirical tail benchmark)", flush=True)
    print(f"  SESOI spans: strict={sesoi['strict']:.7f}  iqr={sesoi['iqr']:.7f}  "
          f"(source: {sesoi['source']})", flush=True)
    if KAPPA_NULL not in kappas:
        print(f"  NOTE: kappa={KAPPA_NULL} (the sign_flip null) not in grid; "
              "the nesting check is skipped.", flush=True)

    # sigma-check + grid annotation: per-kappa DGP diagnostics.
    absR = np.abs(resid)
    kappa_diag = _kappa_diagnostics(absR, var_resid, kappas, real_exkurt,
                                    args.seed)
    print("  kappa diagnostics (sigma-check + per-draw skew dispersion):",
          flush=True)
    for k in kappas:
        d = kappa_diag[f"{k:g}"]
        print(f"    kappa={k:g}: c={d['c_kappa']:.4e} var_inno={d['var_inno']:.6e} "
              f"(rel_err={d['var_rel_err']:.2e}) exc_kurt={d['excess_kurtosis_inno']:+.2f} "
              f"skew[mean={d['per_draw_skew']['mean']:+.4f} "
              f"sd={d['per_draw_skew']['sd']:.4f}] [{d['kappa_grid_label']}]",
              flush=True)

    all_rows: list[dict] = []
    draws_rows: list[dict] = []
    decisions: dict = {}

    for k in kappas:
        print(f"\n--- kappa = {k:g} ---", flush=True)
        res = _simulate_one_kappa(
            k, df_base, mask, m_t, resid, var_resid,
            args.taus, args.horizons, args.n_sim, args.max_iter, args.seed,
            real=real_betas, n_jobs=args.n_jobs,
        )
        rows, decision = _build_kappa_rows_and_decision(
            k, res, args.taus, args.horizons, sesoi)
        all_rows.extend(rows)
        decisions[f"{k:g}"] = decision
        for (tau, h), betas in res["sim"].items():
            for s, b in enumerate(betas):
                draws_rows.append({"dgp": "pareto_kappa", "kappa": float(k),
                                   "sim": s, "tau": float(tau), "h": int(h),
                                   "beta": float(b)})
        for h in args.horizons:
            dc = decision.get(str(h))
            if dc is None:
                continue
            print(f"  h={h:>2}: gap_real={dc['gap_real']:+.4f}  "
                  f"band[{dc['gap_ci_lo']:+.4f},{dc['gap_ci_hi']:+.4f}] "
                  f"half_w={dc['gap_band_half_width']:.5f} "
                  f"-> {dc['verdict']}", flush=True)

    # ── PRE-PLANNED DECISION/BRANCH: confirmatory vs non-confirmatory ──
    band_widths = {f"{k:g}": [decisions[f"{k:g}"][str(h)]["gap_band_width"]
                              for h in args.horizons
                              if str(h) in decisions[f"{k:g}"]]
                   for k in kappas}
    nonconf_outside = []   # gap_real outside band for some (kappa,h)
    nonconf_underpow = []  # inconclusive (underpowered) at some (kappa,h)
    for k in kappas:
        for h in args.horizons:
            dc = decisions[f"{k:g}"].get(str(h))
            if dc is None:
                continue
            if not dc["gap_real_inside_band"]:
                nonconf_outside.append((k, h, dc["verdict"]))
            elif not dc["adequately_powered"]:
                nonconf_underpow.append((k, h))
    # monotonicity diagnostic: on BAND WIDTH, not the ~0 mean.
    mean_bw_by_kappa = {kk: (float(np.mean(v)) if v else float("nan"))
                        for kk, v in band_widths.items()}
    bw_series = [mean_bw_by_kappa[f"{k:g}"] for k in kappas]
    bw_monotone_increasing = all(
        bw_series[i] <= bw_series[i + 1] + 1e-12
        for i in range(len(bw_series) - 1)
    ) if len(bw_series) > 1 else None

    if nonconf_outside:
        branch = "NON-CONFIRMATORY (gap_real outside band for some kappa)"
        expected = ("a symmetric heavy tail manufactures a downside gap: "
                    "gap_real escapes the symmetric band at "
                    + "; ".join(f"kappa={k:g},h={h}" for k, h, _ in nonconf_outside)
                    + " -> REPORT (the placebo's null is NOT invariant to "
                      "symmetric tail thickness)")
    elif nonconf_underpow:
        branch = "NON-CONFIRMATORY (inconclusive/underpowered at heavy kappa)"
        expected = ("gap_real stays inside the band for all kappa, but the band "
                    "is too wide (half-width >= SESOI) to discriminate at "
                    + "; ".join(f"kappa={k:g},h={h}" for k, h in nonconf_underpow)
                    + " -> REPORT: the placebo's robustness is CONDITIONAL on "
                      "tail calibration (heavy-tail cells are underpowered)")
    else:
        branch = "CONFIRMATORY (gap invariant to symmetric tail thickness)"
        expected = ("for ALL kappa, gap_real is INSIDE the symmetric band AND "
                    "adequately powered (band half-width < SESOI span) -> the "
                    "null left-right gap is invariant to symmetric tail "
                    "thickness; the apparent downside asymmetry is a scale "
                    "artefact that survives the kappa stress test")
    print(f"\nPRE-PLANNED BRANCH -> {branch}", flush=True)
    print(f"  expected_result: {expected}", flush=True)

    df_out = pd.DataFrame(all_rows, columns=KAPPA_OUT_COLS)
    df_out = df_out.sort_values(["kappa", "h", "tau"],
                                kind="mergesort").reset_index(drop=True)

    meta = {
        "script": "scripts/aux/run_placebo_symmetric.py",
        "purpose": ("kappa-light symmetric placebo — variance-preserving "
                    "tail-thickness sweep (item A2; backs A3 robustness)"),
        "dgp": "pareto_kappa",
        "n_sim": int(args.n_sim),
        "kappas": [float(k) for k in kappas],
        "kappa_null": KAPPA_NULL,
        "taus": [float(t) for t in args.taus],
        "horizons": [int(h) for h in args.horizons],
        "gap_pair": {"lo_tau": GAP_LO_TAU, "hi_tau": GAP_HI_TAU},
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "seed_scheme": ("default_rng([seed, round(kappa*1000)]) per kappa; sign "
                        "vectors pre-drawn sequentially before any parallel "
                        "dispatch (dispatch-invariant, reproducible)."),
        "n_rows_panel": int(len(df_base)),
        "mean": mean_info,
        "var_resid_fixed": float(var_resid),
        "real_resid_excess_kurtosis": float(real_exkurt),
        "sesoi": sesoi,
        "power_gate": {
            "rule": ("'scale artefact (adequately powered)' iff gap_real INSIDE "
                     "the symmetric band AND band half-width < SESOI span; else "
                     "'inconclusive (underpowered)'. SESOI strict=p50p95, "
                     "lenient=iqr (from mde_equivalence.csv)."),
            "mde80_multiplier": MDE80_Z,
        },
        "kappa_diagnostics": kappa_diag,
        "decisions": decisions,
        "band_width_by_kappa_mean": mean_bw_by_kappa,
        "band_width_monotone_increasing_in_kappa": bw_monotone_increasing,
        "preplanned_branch": branch,
        "expected_result": expected,
        "nonconfirmatory_outside_band": [
            {"kappa": float(k), "h": int(h), "verdict": v}
            for k, h, v in nonconf_outside],
        "nonconfirmatory_underpowered": [
            {"kappa": float(k), "h": int(h)} for k, h in nonconf_underpow],
        "regressors": list(rqlp.REGRESSORS),
        "controls": list(rqlp.CONTROLS),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "arch_version": _arch_version(),
        "platform": platform.platform(),
    }

    df_draws = pd.DataFrame(draws_rows,
                            columns=["dgp", "kappa", "sim", "tau", "h", "beta"])
    save_outputs(df_out, df_draws, meta, args.out_dir, dgp="pareto_kappa")
    print(f"\nDone (pareto_kappa). {len(df_out)} rows. Total wall time: "
          f"{(time.time() - t_all) / 60:.2f} min", flush=True)
    if args.n_sim < 500:
        print("NOTE: n_sim<500 => smoke-grade band; band edges have larger "
              "MCSE at heavy kappa. Use --n_sim 500 (or more for kappa>=1.5) "
              "for the canonical run.", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--n_sim", type=int, default=100,
                    help="Symmetric panels per vol model. Default 100 (smoke); "
                         ">=500 canonical.")
    ap.add_argument("--horizons", type=_parse_ints, default=HORIZONS_DEFAULT,
                    help=f"Comma-separated. Default: {HORIZONS_DEFAULT}")
    ap.add_argument("--taus", type=_parse_floats, default=TAUS_DEFAULT,
                    help=f"Comma-separated. Default: {TAUS_DEFAULT} "
                         "(must include 0.01 & 0.99 for the gap).")
    ap.add_argument("--vol_models", type=_parse_strs, default=VOL_MODELS_DEFAULT,
                    help="Subset of {rolling,garch}. Default: both. IGNORED "
                         "under --dgp sign_flip (sigma-free; runs once as "
                         "'empirical').")
    ap.add_argument("--dgp", choices=DGP_MODES, default=DGP_DEFAULT,
                    help="'sign_flip' (DEFAULT): model-free Rademacher "
                         "sign-flip placebo, ret = m_t ± |resid_t| — the "
                         "historical effective DGP, named honestly. "
                         "'model_scaled': permuted standardised magnitudes "
                         "rescaled by the model sigma — the genuine "
                         "dual-vol-spec robustness (rolling + garch). "
                         "'pareto_kappa' (item A2): variance-preserving "
                         "symmetric tail-thickness sweep — ret = m_t + "
                         "sign*c(kappa)*|resid|**kappa, kappa swept by --kappas "
                         "(kappa=1 reproduces sign_flip). Writes placebo_kappa.csv.")
    ap.add_argument("--kappas", type=_parse_floats, default=KAPPAS_DEFAULT,
                    help=f"pareto_kappa tail-thickness grid. Default: "
                         f"{KAPPAS_DEFAULT}. kappa=1 = null (reproduces "
                         "sign_flip); kappa=2.0 is beyond the empirical tail "
                         "(stress only). Ignored under other --dgp.")
    ap.add_argument("--in_dir", type=Path, default=ECON_DIR,
                    help="Dir holding mde_equivalence.csv for the SESOI power "
                         "gate (pareto_kappa). Default: data/econ.")
    ap.add_argument("--max_iter", type=int, default=MAX_ITER_DEFAULT,
                    help=f"QuantReg max_iter in sim loop. Default {MAX_ITER_DEFAULT}.")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="1 = sequential (default). -1/-N = joblib loky across "
                         "pre-drawn sim batches — bit-identical results (sign "
                         "vectors are drawn sequentially BEFORE dispatch).")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    for vm in args.vol_models:
        if vm not in ("rolling", "garch"):
            ap.error(f"unknown vol_model '{vm}' (allowed: rolling, garch)")

    print("run_placebo_symmetric (Carriero-style symmetric-density placebo, A3)",
          flush=True)
    print(f"  n_sim={args.n_sim}  vol_models={args.vol_models}", flush=True)
    print(f"  taus={args.taus}", flush=True)
    print(f"  horizons={args.horizons}  max_iter={args.max_iter}  seed={args.seed}",
          flush=True)
    if not ((GAP_LO_TAU in args.taus) and (GAP_HI_TAU in args.taus)):
        print(f"  WARNING: gap needs both {GAP_LO_TAU} & {GAP_HI_TAU} in --taus; "
              "gap_* will be NaN.", flush=True)

    t_all = time.time()
    print("Building estimation sample ...", flush=True)
    # Warmup uses max(CFG.ECON.lp_horizons) internally regardless of `horizons`
    # (build_df_est_raw is hardwired to the full grid), so the sample is the same
    # as the main pipeline; we pass our horizons only to materialise cumret cols.
    df_base = build_df_est_raw(horizons=args.horizons).reset_index(drop=True)
    print(f"  rows={len(df_base):,}", flush=True)

    # DGP mean (shared across vol models).
    mask, m_t, resid, mean_info = _fit_mean(df_base)
    print(f"  DGP mean shock coef = {mean_info['mean_shock_coef']:+.4f} "
          f"(n_mask={mean_info['n_mask']:,})", flush=True)

    # Real-return betas: computed ONCE (they do not depend on the vol model;
    # the original code refit them per vol model — identical numbers, wasted
    # wall time at canonical max_iter).
    print("Real-return betas (shared across vol models) ...", flush=True)
    base_ret_full = df_base["ret_eth_perp"].to_numpy(float)
    real_betas = _betas_for(df_base, base_ret_full, args.taus, args.horizons,
                            args.max_iter)

    # ── pareto_kappa (item A2): variance-preserving tail-thickness sweep ──
    if args.dgp == "pareto_kappa":
        return _run_pareto_kappa(args, df_base, mask, m_t, resid, mean_info,
                                 real_betas, t_all)

    all_rows: list[dict] = []
    draws_rows: list[dict] = []
    dgp_meta: dict = {"mean": mean_info, "dgp": args.dgp,
                      "vol_specs": {}, "gaps": {}}

    if args.dgp == "sign_flip":
        # sigma-free DGP: one pass, no vol model fitted.
        if args.vol_models != VOL_MODELS_DEFAULT:
            print("  NOTE: --vol_models ignored under --dgp sign_flip "
                  "(sigma-free).", flush=True)
        vol_list = ["empirical"]
    else:
        vol_list = args.vol_models

    for vm in vol_list:
        print(f"\n=== vol_model = {vm}  (dgp={args.dgp}) ===", flush=True)
        if vm == "empirical":
            sig_t = np.ones_like(resid)
            vinfo = {"note": "sigma-free sign-flip DGP — ret = m_t ± |resid_t|; "
                             "the empirical conditional-vol path is preserved "
                             "exactly, no vol model is fitted"}
        elif vm == "rolling":
            sig_t, vinfo = _sigma_rolling(df_base, mask, resid)
            print(f"  log-variance shock coef = {vinfo['logvar_shock_coef']:+.3f} "
                  "(>0 => liquidations raise volatility — the skeptic's channel)",
                  flush=True)
        else:  # garch
            try:
                sig_t, vinfo = _sigma_garch(resid)
            except Exception as e:
                print(f"  ERROR fitting GARCH ({e}); skipping vol_model=garch.",
                      flush=True)
                dgp_meta["vol_specs"][vm] = {"error": str(e)}
                continue
            print(f"  GARCH(1,1): alpha={vinfo['garch_alpha']:.3f} "
                  f"beta={vinfo['garch_beta']:.3f} "
                  f"persistence={vinfo['garch_persistence']:.3f}", flush=True)
        dgp_meta["vol_specs"][vm] = vinfo

        res = _simulate_one_vol_model(
            vm, df_base, mask, m_t, sig_t, resid,
            args.taus, args.horizons, args.n_sim, args.max_iter, args.seed,
            real=real_betas, n_jobs=args.n_jobs, dgp=args.dgp,
        )
        rows, gap_meta = _build_rows_and_gaps(vm, res, args.taus, args.horizons)
        for r in rows:
            r["dgp"] = args.dgp
        all_rows.extend(rows)
        dgp_meta["gaps"][vm] = gap_meta

        # Per-sim draws (long format) for the F3 figure / any re-analysis.
        for (tau, h), betas in res["sim"].items():
            for s, b in enumerate(betas):
                draws_rows.append({"dgp": args.dgp, "vol_model": vm, "sim": s,
                                   "tau": float(tau), "h": int(h),
                                   "beta": float(b)})

        # Console verdict summary for this vol model.
        for h in args.horizons:
            gm = gap_meta.get(str(h))
            if gm is None:
                continue
            print(f"  h={h:>2}: gap_real={gm['gap_real']:+.3f}  "
                  f"sym-band[{gm['gap_placebo_ci_lo']:+.3f},"
                  f"{gm['gap_placebo_ci_hi']:+.3f}]  -> {gm['verdict']}",
                  flush=True)

    df_out = pd.DataFrame(all_rows, columns=OUT_COLS)
    df_out = df_out.sort_values(["dgp", "vol_model", "h", "tau"],
                                kind="mergesort").reset_index(drop=True)

    meta = {
        "script": "scripts/aux/run_placebo_symmetric.py",
        "purpose": "Carriero-style symmetric-density placebo (backs A3)",
        "n_sim": int(args.n_sim),
        "dgp": args.dgp,
        "taus": [float(t) for t in args.taus],
        "horizons": [int(h) for h in args.horizons],
        "vol_models": list(vol_list),
        "gap_pair": {"lo_tau": GAP_LO_TAU, "hi_tau": GAP_HI_TAU},
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "n_jobs": int(args.n_jobs),
        "seed_scheme": ("default_rng([seed, VOL_MODEL_SEED_KEY[vol_model]]) with "
                        "VOL_MODEL_SEED_KEY={rolling:1, garch:2}; sign vectors "
                        "pre-drawn sequentially before any parallel dispatch. "
                        "NOTE: pre-2026-06-12 runs used abs(hash(vol_model)) "
                        "which is process-randomised (PYTHONHASHSEED) — those "
                        "streams are NOT reproducible; canonical = this scheme."),
        "n_rows_panel": int(len(df_base)),
        "dgp": dgp_meta,
        "regressors": list(rqlp.REGRESSORS),
        "controls": list(rqlp.CONTROLS),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        # arch is only needed by the garch vol model; do not crash the save
        # path when it is absent (e.g. --dgp sign_flip on a minimal env).
        "arch_version": _arch_version(),
        "platform": platform.platform(),
    }

    df_draws = pd.DataFrame(draws_rows,
                            columns=["dgp", "vol_model", "sim", "tau", "h",
                                     "beta"])
    save_outputs(df_out, df_draws, meta, args.out_dir, dgp=args.dgp)
    print(f"\nDone. {len(df_out)} rows. Total wall time: "
          f"{(time.time() - t_all) / 60:.2f} min", flush=True)
    if args.n_sim < 500:
        print("NOTE: n_sim<500 => smoke-grade band. Use --n_sim 500 for canonical.",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
