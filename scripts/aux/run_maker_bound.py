#!/usr/bin/env python3
"""
run_maker_bound.py — [ROBUSTNESS / FLAGGED — does NOT change the main spec]

MakerDAO COVERAGE-GAP BIAS BOUND, by ACTUAL SENSITIVITY RE-FIT (not an
algebraic OLS-analog bound). Lands in paper §2 (data) or Appendix C.

Question
--------
Dune's Spellbook coverage of MakerDAO / Sky ETH-vault liquidations is
incomplete: the collateral filter `symbol IN ('WETH','ETH','stETH',...)` on
`lending.supply` (DUNE_EXTRACTION_BRIEF.md §B, §E note 4) under-indexes Maker's
`clip`/`flip` auction events on the ETH-A/B/C ilks. The brief states the panel
is therefore a "lower bound on true DeFi liquidation activity". The headline
QLP regressor is

    shock = log_liq_lag1 = ln(1 + L_{t-1}),   L = total_debt_repaid_usd,

so the OBSERVED L_t under-measures true ETH-collateral liquidation volume
L*_t = L_t + M_t with missing-Maker mass M_t >= 0. This script asks ONE
data-coverage question, and answers it by ACTUALLY RE-FITTING the headline QLP:

    Under a WORST-CASE assumption about the missing Maker volume, by how much
    is the headline coefficient beta_shock(tau=0.01, h) (and the interaction
    beta_shock_x_oi_high) actually DISPLACED when the locked QLP equation is
    re-estimated on the inflated panel?

WHY AN ACTUAL RE-FIT, NOT THE OLD ALGEBRAIC BOUND  (sigma-cancellation class)
----------------------------------------------------------------------------
The previous version reported lambda/(1+lambda) (lambda = cov(shock,g)/var(shock),
"collinear worst case") as a *bound on the QLP slope displacement*. That is an
OLS-analog rescaling identity; it is NOT a bound on the displacement of a
QUANTILE-REGRESSION slope at tau=0.01, and it is ANTI-CONSERVATIVE: the actual
re-fit displacement of the INTERACTION coefficient is ~3% at m=0.10, ~8% at
m=0.25, >11% at m=0.50 — already breaching the old script's printed 2.5%/4.5%
"bounds". An algebraic identity on the conditional mean cannot bound a tail
quantile estimator. We therefore DROP the algebraic bound and report the TRUE
displacement obtained by re-fitting the locked equation:

    shock*_t          = ln(1 + (1+m) L_{t-1})         (inflated, per allocation)
    shock_x_oi_high*_t = shock*_t * oi_high_t
    -> rqlp._fit_one(tau=0.01, h, "cumret_h{h}", df_inflated, REGRESSORS, ...)
    -> dbeta(h)            = beta*_shock(0.01,h)            - beta_shock(0.01,h)
       dbeta_interaction(h)= beta*_shock_x_oi_high(0.01,h)  - beta_shock_x_oi(...)

This is READ-ONLY reuse of the canonical kernel (rqlp._fit_one, rqlp.REGRESSORS,
rqlp.CONTROLS), exactly as run_subsample_stability / run_placebo_symmetric do.
It does NOT modify the locked spec and writes only the NEW files below.

WORST-CASE ALLOCATION IS NOT FREE — WE SWEEP IT
-------------------------------------------------------
The missing Maker mass M_total = m * L_total has to be PLACED somewhere. The old
script ASSERTED (never proved) that loading it proportionally onto already-active
hours binds. It does NOT: that allocation is in fact the MILDEST. We re-fit under
several economically-credible allocations of the SAME total missing mass and
report the MAX displacement across them as the true worst case:

  prop_active   inflate every active hour by (1+m): adds m*L_t on active hours
                (the old script's assumed worst case; turns out mildest).
  stress_adj0   place the missing mass on currently-ZERO hours that are
                stress-ADJACENT to an active hour (Maker auctions clearing one
                hour off the indexed events), equal split. This CREATES support
                where there was none and moves the tail slope the most.
  prop_oi       distribute the missing mass over the STRESS SET (active OR
                stress-adjacent) proportional to lagged OI (leverage-weighted
                infill). Maker liquidations co-occur with high leverage; deep-zero
                non-stress hours are never touched (that would be an incoherent
                "Maker volume on a calm hour" allocation).

Every allocation conserves total added mass = m * L_total. The headline number is
SUP over (allocation, h) of |dbeta/beta| — the genuine worst case.

Pre-planned NON-CONFIRMATORY branch
-----------------------------------
The thesis-relevant claim is that the data gap does NOT overturn the headline.
CONFIRMATORY  : at the defensible m (<=0.25), the MAX actual displacement across
                allocations is small (pre-registered threshold:
                SUP |dbeta/beta| < 0.10) AND sign-preserving — the gap cannot
                manufacture or kill the headline. Print the TRUE re-fit number.
NONCONFIRM.   : if SUP |dbeta/beta| >= 0.10 at a defensible m, OR any allocation
                flips a sign, the coverage gap is MATERIAL — REPORTED in §2 as a
                live data-coverage qualification (the headline is gap-sensitive),
                not buried.

sigma-cancellation / construct self-check
-------------------------------------------------
This is now a genuine RE-FIT, so the real risk is NOT "g == 0" (that guarded a
property of the old algebraic shift, the wrong object). The real risks are:
  (i)  the inflation is a no-op (shock* == shock) -> guarded: every m>0 must move
       the regressor on active hours (max |shock*-shock| > 0), AND the re-fit must
       actually change at least one headline coefficient.
  (ii) an algebraic OLS-analog "bound" is mistaken for a QR-slope bound -> we no
       longer report any algebraic bound; only the re-fit displacement is the
       deliverable, and the meta self-check states this explicitly.
The non-degeneracy guard below asserts (i): the inflation is non-trivial AND the
re-fit displacement is non-zero on the headline horizons. (There is no algebraic
bound left to "dominate".)

OUTPUT (data/econ/ — NEW files, never overwrite canonical)
----------------------------------------------------------
  maker_bound.csv
    [m, allocation, h, beta_head, beta_refit, dbeta, rel_dbeta,
     beta_inter_head, beta_inter_refit, dbeta_interaction, rel_dbeta_interaction,
     n_obs]
  maker_bound_meta.json
    method, allocations, defensible-m justification, SUP-over-(alloc,h) headline
    numbers, pre-planned verdict, non-degeneracy guard, run provenance.

Run
---
    .venv/bin/python scripts/aux/run_maker_bound.py \
        --m 0.10,0.25,0.50 --horizons 0 --out_dir /tmp/maker_smoke   # smoke
    .venv/bin/python scripts/aux/run_maker_bound.py --n_jobs 4       # canonical
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
from src.estimation import build_df_est_raw, WARMUP_OI_WINDOW  # noqa: E402
from src.io import load_econ_panel  # noqa: E402
import run_quantile_lp as rqlp  # noqa: E402  (read-only kernel reuse)


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
TAU_HEAD: float = 0.01                  # headline tail quantile
LIQ_USD_COL: str = "liq_usd_total"      # L_t = total_debt_repaid_usd (USD)

# Worst-case missing-Maker share grid (m = M_total / L_total). 0.10 = defensible
# (Maker historically a minority of ETH-collateral lending liquidations within
# the sample); 0.25 = generous; 0.50 = deliberately punitive ("half of ALL true
# ETH-collateral liquidation volume is missing Maker") — a strict upper fence.
M_GRID_DEFAULT: list[float] = [0.10, 0.25, 0.50]
M_DEFENSIBLE: float = 0.25              # m at which the pre-registered verdict is read
HORIZONS_DEFAULT: list[int] = list(range(0, 25))

# Worst-case mass allocations (see module docstring). prop_active is the OLD
# asserted worst case; the others are the alternatives added when the
# algebraic bound was replaced by the actual re-fit.
ALLOCATIONS: list[str] = ["prop_active", "stress_adj0", "prop_oi"]

# Pre-registered confirmatory threshold on SUP |dbeta/beta|.
REL_BOUND_THRESHOLD: float = 0.10

DEFAULT_MAX_ITER: int = 20000

OUT_COLS: list[str] = [
    "m", "allocation", "h",
    "beta_head", "beta_refit", "dbeta", "rel_dbeta",
    "beta_inter_head", "beta_inter_refit",
    "dbeta_interaction", "rel_dbeta_interaction",
    "n_obs",
]


# ──────────────────────────────────────────────────────────────
# Inflated-panel construction (per allocation) — exact log shock
# ──────────────────────────────────────────────────────────────
def _lagged_liq_and_oi(df_est: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """L_{t-1} and OI_{t-1} aligned EXACTLY to df_est's `shock = log_liq.shift(1)`.

    build_df_est_raw sets shock = ln(1+L).shift(1) after slicing the panel at
    `warmup`. We reconstruct L_{t-1} on the SAME rows by lagging the raw panel
    liq column and applying the identical warmup slice, so log1p(L_lag) is
    bit-identical to df_est['shock'] (verified in the non-degeneracy guard).
    """
    df_full = load_econ_panel().sort_values("date").reset_index(drop=True)
    warmup = (max(CFG.ECON.vol_window, WARMUP_OI_WINDOW)
              + max(CFG.ECON.lp_horizons) + 2)
    liq_lag = (df_full[LIQ_USD_COL].shift(1)
               .iloc[warmup:].reset_index(drop=True)
               .fillna(0.0).to_numpy(dtype=float))
    oi_lag = (df_full["oi"].shift(1)
              .iloc[warmup:].reset_index(drop=True)
              .fillna(0.0).to_numpy(dtype=float))
    n = len(df_est)
    return liq_lag[:n], oi_lag[:n]


def inflated_liq(liq_lag: np.ndarray, oi_lag: np.ndarray,
                 m: float, allocation: str) -> np.ndarray:
    """Place the missing Maker mass M = m * sum(L) per `allocation`; return L*.

    Every allocation conserves the SAME total added mass m*sum(L_lag); they
    differ only in WHERE it lands. See module docstring for the economics.
    """
    L = liq_lag.astype(float).copy()
    M_total = m * L.sum()
    active = L > 0
    # stress-adjacent zero hours: zero-liq rows neighbouring an active hour
    adj = np.zeros(len(L), dtype=bool)
    adj[1:] |= active[:-1]
    adj[:-1] |= active[1:]
    sa_zero = adj & (~active)
    stress_set = active | sa_zero

    Lstar = L.copy()
    if allocation == "prop_active":
        # inflate every active hour by (1+m): adds m*L_t on active hours
        Lstar = (1.0 + m) * L
    elif allocation == "stress_adj0":
        # all missing mass on currently-zero stress-adjacent hours, equal split
        n_sa = int(sa_zero.sum())
        if n_sa > 0:
            Lstar[sa_zero] += M_total / n_sa
    elif allocation == "prop_oi":
        # leverage-weighted infill over the stress set (active OR stress-adjacent)
        w = np.where(stress_set, oi_lag, 0.0)
        wsum = w.sum()
        if wsum > 0:
            Lstar = L + M_total * (w / wsum)
    else:
        raise ValueError(f"unknown allocation {allocation!r}")
    return Lstar


def refit_inflated(df_est: pd.DataFrame, Lstar: np.ndarray, h: int,
                   max_iter: int) -> dict | None:
    """Re-fit the LOCKED QLP at (tau=0.01, h) on the inflated panel.

    Rebuilds ONLY the two regressor columns the inflation touches
    (shock, shock_x_oi_high) in a COPY of df_est, then calls the canonical
    kernel rqlp._fit_one verbatim (read-only reuse; spec unchanged).
    """
    dfi = df_est.copy()
    shock_star = np.log1p(Lstar)
    dfi["shock"] = shock_star
    dfi["shock_x_oi_high"] = shock_star * dfi["oi_high"].to_numpy(dtype=float)
    return rqlp._fit_one(TAU_HEAD, h, f"cumret_h{h}", dfi,
                         rqlp.REGRESSORS, rqlp.CONTROLS, max_iter)


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(df_out: pd.DataFrame, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "maker_bound.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}", flush=True)
    meta_path = out_dir / "maker_bound_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)
    print("\n--- maker_bound.csv ---", flush=True)
    print(f"shape: {df_out.shape}", flush=True)
    print("HEAD:", flush=True)
    print(df_out.head().to_string(index=False), flush=True)
    print("TAIL:", flush=True)
    print(df_out.tail().to_string(index=False), flush=True)


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--m", "--m_grid", dest="m_grid", type=_parse_floats,
                    default=M_GRID_DEFAULT,
                    help=f"Missing-Maker shares. Default: {M_GRID_DEFAULT}")
    ap.add_argument("--m_defensible", type=float, default=M_DEFENSIBLE,
                    help="m at which the pre-registered verdict is read.")
    ap.add_argument("--horizons", type=_parse_ints, default=HORIZONS_DEFAULT,
                    help="Comma-separated. Default: 0..24")
    ap.add_argument("--allocations", type=lambda s: [x.strip() for x in s.split(",")],
                    default=ALLOCATIONS,
                    help=f"Mass allocations. Default: {ALLOCATIONS}")
    ap.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER,
                    help="QuantReg max_iter (20000 canonical; lower for smoke).")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="joblib loky workers over (m, allocation, h). "
                         "Use 4 for the canonical run (keeps CPU bounded).")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    print("run_maker_bound: MakerDAO coverage-gap bias, ACTUAL QLP RE-FIT (A5)",
          flush=True)
    print(f"  m_grid={args.m_grid}  m_defensible={args.m_defensible}  "
          f"horizons=[{min(args.horizons)}..{max(args.horizons)}]  "
          f"allocations={args.allocations}", flush=True)
    print(f"  max_iter={args.max_iter}  n_jobs={args.n_jobs}", flush=True)

    t0 = time.time()
    print("Building estimation sample (build_df_est_raw, full panel) ...",
          flush=True)
    df_est = build_df_est_raw(horizons=args.horizons).reset_index(drop=True)
    print(f"  rows={len(df_est):,}", flush=True)

    liq_lag, oi_lag = _lagged_liq_and_oi(df_est)

    # ---- baseline headline betas: re-fit the LOCKED QLP on the OBSERVED panel ----
    # (m=0 anchor; identical to the canonical fit, recomputed here so dbeta is a
    #  like-for-like difference under the SAME kernel settings / max_iter.)
    base: dict = {}
    for h in args.horizons:
        r0 = rqlp._fit_one(TAU_HEAD, h, f"cumret_h{h}", df_est,
                           rqlp.REGRESSORS, rqlp.CONTROLS, args.max_iter)
        base[h] = r0
    print(f"  baseline fits: {sum(v is not None for v in base.values())}"
          f"/{len(args.horizons)}", flush=True)

    # ---- non-degeneracy guard (validates the RIGHT property) -------------
    # The real risk is a no-op inflation or a no-op re-fit, NOT g==0. Assert:
    #   (1) shock reconstruction is exact (log1p(L_lag) == df_est['shock']);
    #   (2) every m>0 actually MOVES the regressor on active hours; AND
    #   (3) the re-fit displacement is nonzero on the headline horizons.
    shock_recon_err = float(np.nanmax(np.abs(
        np.log1p(liq_lag) - df_est["shock"].fillna(0.0).to_numpy(dtype=float))))
    h_guard = min(args.horizons)
    guard_diag: dict = {"shock_recon_max_abs_err": shock_recon_err,
                        "per_m": {}}
    if not (shock_recon_err < 1e-8):
        print(f"  GUARD FAIL: shock reconstruction err {shock_recon_err:.2e} "
              f">= 1e-8 — L_lag is NOT aligned to the headline shock. ABORT.",
              flush=True)
        return 2
    b0_guard = base.get(h_guard)
    for m in args.m_grid:
        if m <= 0:
            continue
        Lstar = inflated_liq(liq_lag, oi_lag, m, "prop_active")
        shock_move = float(np.max(np.abs(np.log1p(Lstar) - np.log1p(liq_lag))))
        refit_moves = False
        if b0_guard is not None:
            rg = refit_inflated(df_est, Lstar, h_guard, args.max_iter)
            if rg is not None:
                refit_moves = (abs(rg["beta_shock"] - b0_guard["beta_shock"]) > 0
                               or abs(rg["beta_interaction"]
                                      - b0_guard["beta_interaction"]) > 0)
        guard_diag["per_m"][str(m)] = {
            "shock_max_move": shock_move,
            "refit_displaces_headline": bool(refit_moves),
        }
        if not (shock_move > 0 and refit_moves):
            print(f"  GUARD FAIL: m={m} no-op (shock_move={shock_move:.2e}, "
                  f"refit_displaces={refit_moves}). The inflation does not move "
                  f"the re-fit — bound is vacuous. ABORT.", flush=True)
            return 2

    # ---- actual re-fit over (m, allocation, h) ---------------------------------
    rows: list[dict] = []
    for m in args.m_grid:
        for alloc in args.allocations:
            Lstar = inflated_liq(liq_lag, oi_lag, m, alloc)
            for h in args.horizons:
                b0 = base.get(h)
                r = refit_inflated(df_est, Lstar, h, args.max_iter)
                if b0 is None or r is None:
                    bh = bi = br = bir = db = dbi = np.nan
                    rel = reli = np.nan
                    nobs = np.nan
                else:
                    bh, bi = b0["beta_shock"], b0["beta_interaction"]
                    br, bir = r["beta_shock"], r["beta_interaction"]
                    db = br - bh
                    dbi = bir - bi
                    rel = abs(db) / abs(bh) if bh != 0 else np.nan
                    reli = abs(dbi) / abs(bi) if bi != 0 else np.nan
                    nobs = int(r["n_obs"])
                rows.append({
                    "m": float(m), "allocation": alloc, "h": int(h),
                    "beta_head": float(bh), "beta_refit": float(br),
                    "dbeta": float(db), "rel_dbeta": float(rel),
                    "beta_inter_head": float(bi), "beta_inter_refit": float(bir),
                    "dbeta_interaction": float(dbi),
                    "rel_dbeta_interaction": float(reli),
                    "n_obs": nobs,
                })
            # per-(m,alloc) SUP over h, for the running log
            sub = [row for row in rows
                   if row["m"] == m and row["allocation"] == alloc]
            sup_s = np.nanmax([row["rel_dbeta"] for row in sub]) if sub else np.nan
            sup_i = (np.nanmax([row["rel_dbeta_interaction"] for row in sub])
                     if sub else np.nan)
            print(f"  m={m:>4}  alloc={alloc:<12}  "
                  f"SUP_h rel_dbeta_shock={sup_s:.4f}  "
                  f"rel_dbeta_inter={sup_i:.4f}", flush=True)

    df_out = (pd.DataFrame(rows)
              .sort_values(["m", "allocation", "h"], kind="mergesort")
              .reset_index(drop=True)[OUT_COLS])

    # ---- SUP-over-(allocation,h) headline numbers + pre-registered verdict ------
    def sup_rel(m: float, col: str) -> float:
        sub = df_out[np.isclose(df_out["m"], m)]
        return float(np.nanmax(sub[col])) if len(sub) else np.nan

    def argsup(m: float, col: str) -> dict:
        sub = df_out[np.isclose(df_out["m"], m)].copy()
        sub = sub[np.isfinite(sub[col])]
        if not len(sub):
            return {}
        r = sub.loc[sub[col].idxmax()]
        return {"allocation": str(r["allocation"]), "h": int(r["h"]),
                "value": float(r[col])}

    sup_by_m_shock = {str(m): sup_rel(m, "rel_dbeta") for m in args.m_grid}
    sup_by_m_inter = {str(m): sup_rel(m, "rel_dbeta_interaction")
                      for m in args.m_grid}
    # The headline worst case = MAX over BOTH coefficients.
    sup_by_m = {str(m): float(np.nanmax([sup_by_m_shock[str(m)],
                                         sup_by_m_inter[str(m)]]))
                for m in args.m_grid}

    on_grid = any(np.isclose(args.m_defensible, args.m_grid))
    sup_def = sup_by_m.get(str(args.m_defensible)) if on_grid else None
    sup_max_punitive = float(np.nanmax(list(sup_by_m.values()))) if sup_by_m else np.nan

    if sup_def is None:
        verdict = "INCONCLUSIVE (m_defensible not on grid)"
        confirmatory = None
    elif sup_def < REL_BOUND_THRESHOLD:
        verdict = (f"CONFIRMATORY: at m={args.m_defensible}, MAX over "
                   f"(allocation,h,{{shock,interaction}}) |dbeta/beta| = "
                   f"{sup_def:.4f} < {REL_BOUND_THRESHOLD} — the Maker coverage "
                   f"gap cannot overturn the headline (re-fit displacement small).")
        confirmatory = True
    else:
        worst_s = argsup(args.m_defensible, "rel_dbeta")
        worst_i = argsup(args.m_defensible, "rel_dbeta_interaction")
        verdict = (f"NON-CONFIRMATORY: at m={args.m_defensible}, MAX over "
                   f"(allocation,h,{{shock,interaction}}) |dbeta/beta| = "
                   f"{sup_def:.4f} >= {REL_BOUND_THRESHOLD} — the Maker coverage "
                   f"gap MATERIALLY moves the headline under a credible "
                   f"allocation. Worst shock: {worst_s}; worst interaction: "
                   f"{worst_i}. REPORT as a live §2 data-coverage qualification "
                   f"(the prop_active allocation is NOT worst case).")
        confirmatory = False

    meta = {
        "script": "scripts/aux/run_maker_bound.py",
        "purpose": ("MakerDAO coverage-gap sensitivity of the headline "
                    "beta_shock(tau=0.01,h) AND the interaction "
                    "beta_shock_x_oi_high, by ACTUAL re-fit of the LOCKED QLP on "
                    "the inflated panel (read-only rqlp._fit_one reuse). Replaces "
                    "the anti-conservative algebraic lambda/(1+lambda) bound."),
        "method": {
            "estimator": "rqlp._fit_one(tau=0.01, h, cumret_h{h}, df_inflated, "
                         "rqlp.REGRESSORS, rqlp.CONTROLS, max_iter)",
            "inflation": ("shock* = ln(1+(1+m)L_{t-1}); "
                          "shock_x_oi_high* = shock* * oi_high; only these two "
                          "regressor columns are rebuilt — spec unchanged."),
            "displacement": ("dbeta = beta*_shock(0.01,h) - beta_shock(0.01,h); "
                             "dbeta_interaction likewise; rel = |dbeta|/|beta|."),
            "baseline": ("m=0 anchor re-fit on the OBSERVED panel under the SAME "
                         "kernel settings/max_iter (like-for-like difference)."),
            "allocations": {
                "prop_active": "inflate every active hour by (1+m) [old asserted "
                               "worst case; empirically the MILDEST].",
                "stress_adj0": "missing mass onto currently-zero stress-adjacent "
                               "hours (equal split); creates tail support.",
                "prop_oi":     "leverage-weighted infill over the stress set "
                               "(active OR stress-adjacent), prop. to lagged OI.",
            },
            "worst_case": "SUP over (allocation, h, {shock, interaction}) of "
                          "|dbeta/beta|; allocation is swept, not asserted.",
        },
        "assumptions": [
            "Maker ETH-vault (clip/flip, ETH-A/B/C ilks) liquidations are the "
            "dominant missing mass; DUNE_EXTRACTION_BRIEF.md §B collateral filter "
            "+ §E note 4 ('lower bound on true DeFi liquidation activity').",
            "m = M_total/L_total is the missing-Maker share of total ETH-collateral "
            "debt repaid. Defensible m<=0.25 (Maker a minority of in-sample "
            "ETH-collateral lending liquidations); m=0.50 is a punitive fence.",
            "The missing mass is conserved across allocations; only its placement "
            "differs. Deep-zero non-stress hours are never assigned Maker volume "
            "(an incoherent 'liquidation on a calm hour'); prop_oi/stress_adj0 keep "
            "mass on the stress set.",
            "NO algebraic slope bound is claimed: an OLS-analog rescaling identity "
            "(lambda/(1+lambda)) does NOT bound a tau=0.01 QR slope and was "
            "anti-conservative; the deliverable is the ACTUAL re-fit displacement.",
        ],
        "m_grid": [float(m) for m in args.m_grid],
        "m_defensible": float(args.m_defensible),
        "allocations": list(args.allocations),
        "tau_headline": TAU_HEAD,
        "max_iter": int(args.max_iter),
        "rel_bound_threshold": REL_BOUND_THRESHOLD,
        "sup_rel_dbeta_shock_by_m": sup_by_m_shock,
        "sup_rel_dbeta_interaction_by_m": sup_by_m_inter,
        "sup_rel_dbeta_combined_by_m": sup_by_m,
        "sup_combined_at_m_defensible": sup_def,
        "sup_combined_max_over_grid": sup_max_punitive,
        "argsup_at_m_defensible": {
            "shock": argsup(args.m_defensible, "rel_dbeta") if on_grid else None,
            "interaction": (argsup(args.m_defensible, "rel_dbeta_interaction")
                            if on_grid else None),
        },
        "verdict": verdict,
        "confirmatory": confirmatory,
        "nondegeneracy_guard": {
            "note": ("Validates the RIGHT property. Risk is a no-op "
                     "inflation or a no-op re-fit (NOT g==0, which guarded the "
                     "wrong object). Asserts: (1) shock reconstruction exact; "
                     "(2) every m>0 moves the regressor on active hours; (3) the "
                     "re-fit displaces a headline coefficient. No algebraic bound "
                     "remains to 'dominate'."),
            "diag": guard_diag,
        },
        "sigma_cancellation_self_check": (
            "The displacement is now a genuine QLP RE-FIT (kernel rqlp._fit_one), "
            "not an algebraic identity, so there is no transformed-DGP null pair "
            "that could cancel. The prior trap — reporting an OLS-analog "
            "lambda/(1+lambda) as a bound on the QR slope — is REMOVED; that "
            "formula was anti-conservative (re-fit displacement exceeds it). The "
            "only degeneracy risk (no-op inflation / no-op re-fit) is guarded "
            "numerically (nondegeneracy_guard)."),
        "panel": str(CFG.FILES.econ_core_full),
        "n_rows_est": int(len(df_est)),
        "n_active_hours": int((liq_lag > 0).sum()),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    save_outputs(df_out, meta, args.out_dir)
    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"Done. Total wall time: {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
