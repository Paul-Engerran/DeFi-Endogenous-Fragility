#!/usr/bin/env python3
"""
run_maker_asymmetry.py — [ROBUSTNESS / FLAGGED — does NOT change the main spec]

A5-Δ — Does the MakerDAO coverage gap move the ASYMMETRY, or only the level?

Companion to run_maker_bound.py. That script showed the LEVEL beta_shock(0.01,h)
is coverage-sensitive at medium horizons under a leverage-/stress-weighted infill
of the missing Maker mass (up to ~68% magnitude attenuation at h=12; the
interaction can flip sign). BUT the paper's thesis is NOT about the level — it is
about the ASYMMETRY Delta = |beta(0.01)| - |beta(0.99)| (down vs up). The level is
the symmetric volatility channel; a symmetric scale effect (whatever its cause,
including a data-coverage gap that widens both tails together) CANCELS in the
down-minus-up difference. So the decisive question is:

    Under the SAME worst-case Maker allocations that move the level a lot, does
    the down-vs-up ASYMMETRY Delta move — or does the displacement cancel?

If Delta is stable where the level is not, the Maker gap is a symmetric scale
perturbation: it cannot manufacture (or erase) downside-specific asymmetry, and
the thesis is robust to the coverage gap by the SAME cancellation that protects
the asymmetry from the volatility channel. If Delta moves materially, that is a
genuine threat and must be reported.

Method (read-only kernel reuse; locked spec untouched)
------------------------------------------------------
Reuses run_maker_bound's inflated_liq() / _lagged_liq_and_oi() VERBATIM (same
mass-conserving allocations prop_active / stress_adj0 / prop_oi). For each
(m, allocation, h) it re-fits the LOCKED QLP at BOTH tails:
    beta_down = beta_shock(tau=0.01, h);  beta_up = beta_shock(tau=0.99, h)
on the inflated panel via rqlp._fit_one, and forms the mirror-pair asymmetry
    gap = |beta_down| - |beta_up|.
Reported per cell: the level displacements (down and up) AND the asymmetry
displacement dgap = gap_refit - gap_base, plus dgap as a fraction of the baseline
level |beta_down_base| (so it is comparable to run_maker_bound's rel_dbeta).

The thesis-relevant readout is the CONTRAST:
    SUP |dgap| / |beta_down_base|   (asymmetry displacement, relative)
  vs
    SUP rel_dbeta_down              (level displacement, relative; from A5).
If the first is much smaller than the second, the level moves but the asymmetry
does not — the coverage gap is a symmetric scale effect (cancellation holds).

Pre-planned branch
------------------
CONFIRMATORY  : SUP |dgap|/|beta_down_base| is small (< REL_GAP_THRESHOLD = 0.10)
                AND much smaller than the level displacement — the Maker gap does
                NOT move the asymmetry; thesis robust (symmetric-scale cancellation).
NON-CONFIRM.  : SUP |dgap|/|beta_down_base| >= 0.10 at a defensible m, OR the gap
                changes sign — the coverage gap moves the asymmetry itself; a
                genuine threat, REPORTED not buried.

OUTPUT (data/econ/ — NEW files, never overwrite canonical)
----------------------------------------------------------
  maker_asymmetry.csv
    [m, allocation, h, beta_down_head, beta_down_refit, rel_dbeta_down,
     beta_up_head, beta_up_refit, rel_dbeta_up,
     gap_head, gap_refit, dgap, dgap_rel_level, n_obs]
  maker_asymmetry_meta.json

Run
---
    .venv/bin/python scripts/aux/run_maker_asymmetry.py \
        --m 0.10,0.25,0.50 --horizons 0,12 --out_dir /tmp/maker_asym_smoke   # smoke
    .venv/bin/python scripts/aux/run_maker_asymmetry.py --n_jobs 4           # canonical
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
from src.estimation import build_df_est_raw  # noqa: E402
import run_quantile_lp as rqlp  # noqa: E402  (read-only kernel reuse)
# Reuse the EXACT mass-allocation + lag-alignment machinery of A5.
from run_maker_bound import (  # noqa: E402
    inflated_liq, _lagged_liq_and_oi, M_GRID_DEFAULT, M_DEFENSIBLE, ALLOCATIONS,
    DEFAULT_MAX_ITER,
)

TAU_DOWN: float = 0.01
TAU_UP: float = 0.99                     # 1%/1% mirror pair (clean symmetric gap)
HORIZONS_DEFAULT: list[int] = [0, 6, 12]
REL_GAP_THRESHOLD: float = 0.10

OUT_COLS: list[str] = [
    "m", "allocation", "h",
    "beta_down_head", "beta_down_refit", "rel_dbeta_down",
    "beta_up_head", "beta_up_refit", "rel_dbeta_up",
    "gap_head", "gap_refit", "dgap", "dgap_rel_level", "n_obs",
]


def _fit_tau(df_est: pd.DataFrame, Lstar: np.ndarray | None, tau: float,
             h: int, max_iter: int) -> float | None:
    """beta_shock(tau, h). If Lstar is None, fit on the OBSERVED panel (baseline);
    else rebuild the two inflated regressor columns and re-fit (read-only kernel)."""
    if Lstar is None:
        df = df_est
    else:
        df = df_est.copy()
        shock_star = np.log1p(Lstar)
        df["shock"] = shock_star
        df["shock_x_oi_high"] = shock_star * df["oi_high"].to_numpy(dtype=float)
    r = rqlp._fit_one(tau, h, f"cumret_h{h}", df,
                      rqlp.REGRESSORS, rqlp.CONTROLS, max_iter)
    return None if r is None else (float(r["beta_shock"]), int(r["n_obs"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--m", "--m_grid", dest="m_grid",
                    type=lambda s: [float(x) for x in s.split(",") if x.strip()],
                    default=M_GRID_DEFAULT)
    ap.add_argument("--m_defensible", type=float, default=M_DEFENSIBLE)
    ap.add_argument("--horizons",
                    type=lambda s: [int(x) for x in s.split(",") if x.strip()],
                    default=HORIZONS_DEFAULT)
    ap.add_argument("--allocations",
                    type=lambda s: [x.strip() for x in s.split(",")],
                    default=ALLOCATIONS)
    ap.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    print("run_maker_asymmetry: does the Maker gap move the ASYMMETRY? (A5-Delta)",
          flush=True)
    print(f"  m_grid={args.m_grid}  horizons={args.horizons}  "
          f"allocations={args.allocations}  max_iter={args.max_iter}", flush=True)
    t0 = time.time()

    df_est = build_df_est_raw(horizons=args.horizons).reset_index(drop=True)
    liq_lag, oi_lag = _lagged_liq_and_oi(df_est)
    print(f"  rows={len(df_est):,}  active_hours={int((liq_lag>0).sum()):,}",
          flush=True)

    # ---- baseline mirror-pair betas + gap on the OBSERVED panel ----
    base: dict = {}
    for h in args.horizons:
        d = _fit_tau(df_est, None, TAU_DOWN, h, args.max_iter)
        u = _fit_tau(df_est, None, TAU_UP, h, args.max_iter)
        if d is None or u is None:
            base[h] = None
        else:
            bd, nobs = d; bu, _ = u
            base[h] = {"bd": bd, "bu": bu, "gap": abs(bd) - abs(bu), "n": nobs}
        if base[h]:
            print(f"  baseline h={h:>2}: beta_down={base[h]['bd']:+.5f} "
                  f"beta_up={base[h]['bu']:+.5f} gap={base[h]['gap']:+.5f}",
                  flush=True)

    # ---- re-fit the mirror pair under each (m, allocation, h) ----
    rows: list[dict] = []
    for m in args.m_grid:
        for alloc in args.allocations:
            Lstar = inflated_liq(liq_lag, oi_lag, m, alloc)
            for h in args.horizons:
                b0 = base.get(h)
                d = _fit_tau(df_est, Lstar, TAU_DOWN, h, args.max_iter)
                u = _fit_tau(df_est, Lstar, TAU_UP, h, args.max_iter)
                if b0 is None or d is None or u is None:
                    row = {k: np.nan for k in OUT_COLS}
                    row.update({"m": float(m), "allocation": alloc, "h": int(h)})
                    rows.append(row); continue
                bdr, nobs = d; bur, _ = u
                bd0, bu0, gap0 = b0["bd"], b0["bu"], b0["gap"]
                gapr = abs(bdr) - abs(bur)
                dgap = gapr - gap0
                rows.append({
                    "m": float(m), "allocation": alloc, "h": int(h),
                    "beta_down_head": bd0, "beta_down_refit": bdr,
                    "rel_dbeta_down": abs(bdr - bd0) / abs(bd0) if bd0 else np.nan,
                    "beta_up_head": bu0, "beta_up_refit": bur,
                    "rel_dbeta_up": abs(bur - bu0) / abs(bu0) if bu0 else np.nan,
                    "gap_head": gap0, "gap_refit": gapr, "dgap": dgap,
                    "dgap_rel_level": abs(dgap) / abs(bd0) if bd0 else np.nan,
                    "n_obs": int(nobs),
                })
            sub = [r for r in rows if r["m"] == m and r["allocation"] == alloc]
            sup_lvl = np.nanmax([r["rel_dbeta_down"] for r in sub]) if sub else np.nan
            sup_gap = np.nanmax([r["dgap_rel_level"] for r in sub]) if sub else np.nan
            print(f"  m={m:>4} alloc={alloc:<12} "
                  f"SUP_h rel_dbeta_down(level)={sup_lvl:.4f}  "
                  f"SUP_h |dgap|/level(asymmetry)={sup_gap:.4f}", flush=True)

    df_out = (pd.DataFrame(rows)
              .sort_values(["m", "allocation", "h"], kind="mergesort")
              .reset_index(drop=True)[OUT_COLS])

    def sup(col: str, m=None) -> float:
        sub = df_out if m is None else df_out[np.isclose(df_out["m"], m)]
        return float(np.nanmax(sub[col])) if len(sub) else np.nan

    sup_gap_def = sup("dgap_rel_level", args.m_defensible)
    sup_lvl_def = sup("rel_dbeta_down", args.m_defensible)
    # sign-flip check on the gap
    sub_def = df_out[np.isclose(df_out["m"], args.m_defensible)]
    gap_sign_flip = bool(((sub_def["gap_head"] * sub_def["gap_refit"]) < 0).any())

    if sup_gap_def < REL_GAP_THRESHOLD and not gap_sign_flip:
        verdict = (f"CONFIRMATORY: at m={args.m_defensible}, SUP asymmetry "
                   f"displacement |dgap|/level = {sup_gap_def:.4f} < "
                   f"{REL_GAP_THRESHOLD} and no gap sign-flip, while the LEVEL "
                   f"moves up to {sup_lvl_def:.4f}. The Maker gap is a symmetric "
                   f"scale perturbation: it moves both tails together and CANCELS "
                   f"in the down-vs-up asymmetry. The thesis (bounded, symmetric) "
                   f"is robust to the coverage gap by the same cancellation that "
                   f"protects it from the volatility channel.")
        confirmatory = True
    else:
        verdict = (f"NON-CONFIRMATORY: at m={args.m_defensible}, SUP asymmetry "
                   f"displacement |dgap|/level = {sup_gap_def:.4f} "
                   f"(sign_flip={gap_sign_flip}) — the Maker coverage gap moves "
                   f"the ASYMMETRY itself, not just the symmetric level. REPORT.")
        confirmatory = False

    meta = {
        "script": "scripts/aux/run_maker_asymmetry.py",
        "purpose": ("Does the MakerDAO coverage gap move the down-vs-up ASYMMETRY "
                    "Delta=|beta(0.01)|-|beta(0.99)|, or only the symmetric level? "
                    "Companion to run_maker_bound.py (which showed the level is "
                    "coverage-sensitive)."),
        "mirror_pair": {"tau_down": TAU_DOWN, "tau_up": TAU_UP},
        "method": ("reuse run_maker_bound.inflated_liq (same mass-conserving "
                   "allocations) + rqlp._fit_one at BOTH tails; gap=|bd|-|bu|; "
                   "dgap=gap_refit-gap_base; locked spec untouched, read-only kernel."),
        "rel_gap_threshold": REL_GAP_THRESHOLD,
        "m_grid": [float(m) for m in args.m_grid],
        "m_defensible": float(args.m_defensible),
        "allocations": list(args.allocations),
        "sup_dgap_rel_level_by_m": {str(m): sup("dgap_rel_level", m)
                                    for m in args.m_grid},
        "sup_rel_dbeta_down_by_m": {str(m): sup("rel_dbeta_down", m)
                                    for m in args.m_grid},
        "sup_rel_dbeta_up_by_m": {str(m): sup("rel_dbeta_up", m)
                                  for m in args.m_grid},
        "sup_dgap_rel_level_at_m_defensible": sup_gap_def,
        "sup_level_displacement_at_m_defensible": sup_lvl_def,
        "gap_sign_flip_at_m_defensible": gap_sign_flip,
        "verdict": verdict,
        "confirmatory": confirmatory,
        "interpretation": ("If asymmetry displacement << level displacement, the "
                           "coverage gap is symmetric and cancels in Delta — the "
                           "same sigma-cancellation that immunizes the asymmetry "
                           "objects against the volatility channel."),
        "panel": str(CFG.FILES.econ_core_full),
        "n_rows_est": int(len(df_est)),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "maker_asymmetry.csv"
    df_out.to_csv(csv_path, index=False)
    with open(args.out_dir / "maker_asymmetry_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  wrote {csv_path}", flush=True)
    print(f"  shape: {df_out.shape}", flush=True)
    print("HEAD:\n" + df_out.head(12).to_string(index=False), flush=True)
    print("\nVERDICT: " + verdict, flush=True)
    print(f"Done. {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
