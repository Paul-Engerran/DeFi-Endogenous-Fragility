#!/usr/bin/env python3
"""
run_maker_exceedance.py — [ROBUSTNESS / FLAGGED — does NOT change the main spec]

A5-Δexc — The DECISIVE coverage-gap check: does the MakerDAO gap move the
THESIS-BOUNDED object — the per-period exceedance asymmetry
Delta = beta_down - beta_up at alpha=0.01 — outside its equivalence bound?

Why this object (not the QLP gap)
---------------------------------
run_maker_asymmetry.py showed the RAW QLP cumulative gap |beta(0.01)|-|beta(0.99)|
is coverage-sensitive at MEDIUM horizons (the down tail moves, the up tail does
not, under a leverage-weighted infill). But that is the SECTION-5 "apparent"
object, already deconstructed. The paper's actual quantitative claim is the
EQUIVALENCE BOUND on the per-period exceedance Delta at alpha=0.01, h=0
(Section 7.3(iii)): baseline Delta = +0.000529, MDE@80 = 0.000942, SESOI(strict,
p50->p95) = 0.0010915, verdict EQUIVALENT-TO-NEGLIGIBLE. THIS is what must be
robust to the Maker gap. We re-fit it under the same three mass allocations and
ask: does |Delta_refit| stay inside the SESOI band?

  CONFIRMATORY  : |Delta_refit| < SESOI for every allocation at the defensible m
                  (h=0). The thesis bound survives the coverage gap.
  NON-CONFIRM.  : |Delta_refit| >= SESOI for some credible allocation at h=0, OR
                  Delta changes sign and exceeds the bound -> the coverage gap
                  can breach the equivalence bound -> REPORT, do not bury.

Method (read-only kernel reuse; locked spec untouched)
------------------------------------------------------
Reuses run_exceedance's per-period machinery VERBATIM (build_sample,
tail_thresholds, add_indicators, _prepare_pair_arrays, fit_point_lpm) and
run_maker_bound's mass allocations (inflated_liq, _lagged_liq_and_oi). The
exceedance indicators D/U depend ONLY on future returns, so the Maker inflation
changes ONLY the regressor matrix (the `shock` and `shock_x_oi_high` columns);
we rebuild those two columns per allocation and re-fit the paired LPM. OLS, fast.

OUTPUT (data/econ/ — NEW files, never overwrite canonical)
----------------------------------------------------------
  maker_exceedance.csv
    [m, allocation, h, alpha, beta_down_base, beta_up_base, delta_base,
     beta_down_refit, beta_up_refit, delta_refit, ddelta, abs_delta_refit,
     sesoi, mde80, within_sesoi, within_mde, n_obs]
  maker_exceedance_meta.json

Run
---
    .venv/bin/python scripts/aux/run_maker_exceedance.py \
        --m 0.10,0.25,0.50 --horizons 0,6,12 --out_dir /tmp/maker_exc_smoke   # smoke
    .venv/bin/python scripts/aux/run_maker_exceedance.py                      # canonical
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from config import CFG, ECON_DIR  # noqa: E402
import run_exceedance as rexc  # noqa: E402  (read-only kernel reuse)
from run_maker_bound import (  # noqa: E402
    inflated_liq, _lagged_liq_and_oi, M_GRID_DEFAULT, M_DEFENSIBLE, ALLOCATIONS,
)

ALPHA_HEAD: float = 0.01
HORIZONS_DEFAULT: list[int] = [0, 6, 12]


def read_sesoi_and_mde() -> tuple[float, float]:
    """Canonical strict SESOI (p50->p95) and MDE@80 for exceedance_delta alpha=0.01,
    read from data/econ/mde_equivalence.csv. Falls back to documented literals."""
    p = ECON_DIR / "mde_equivalence.csv"
    sesoi, mde80 = 0.0010915461253353605, 0.0009422558
    if p.exists():
        mde = pd.read_csv(p)
        sel = mde[(mde["object"] == "exceedance_delta")
                  & (mde["alpha_or_measure"].astype(str)
                     .str.contains("alpha=0.01", regex=False, na=False))]
        if len(sel) >= 1:
            r = sel.iloc[0]
            if "sesoi_beta_p50p95" in r and pd.notna(r["sesoi_beta_p50p95"]):
                sesoi = float(r["sesoi_beta_p50p95"])
            if "mde_80" in r and pd.notna(r["mde_80"]):
                mde80 = float(r["mde_80"])
    return sesoi, mde80


def delta_at(df_est: pd.DataFrame, h: int, q_lo: float, q_hi: float) -> tuple:
    """Paired exceedance Delta = beta_down - beta_up at (alpha, h) on df_est's
    CURRENT regressor columns (so an inflated `shock` flows straight through)."""
    df_est = rexc.add_indicators(df_est, h, q_lo, q_hi)
    yD, yU, X = rexc._prepare_pair_arrays(df_est, h)
    bd = rexc.fit_point_lpm(yD, X)
    bu = rexc.fit_point_lpm(yU, X)
    return bd["beta"], bu["beta"], bd["beta"] - bu["beta"], int(bd["n_obs"])


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
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    print("run_maker_exceedance: does the Maker gap breach the exceedance-Delta "
          "EQUIVALENCE BOUND? (A5-Delta-exc)", flush=True)
    sesoi, mde80 = read_sesoi_and_mde()
    print(f"  alpha={ALPHA_HEAD}  SESOI(strict)={sesoi:.7f}  MDE@80={mde80:.7f}",
          flush=True)
    t0 = time.time()

    # per-period sample (cumulative=False) + unconditional thresholds — VERBATIM
    df_est = rexc.build_sample(args.horizons, cumulative=False).reset_index(drop=True)
    thresholds = rexc.tail_thresholds(df_est, [ALPHA_HEAD])
    q_lo, q_hi = thresholds[ALPHA_HEAD]
    print(f"  rows={len(df_est):,}  q_lo={q_lo:+.4f}  q_hi={q_hi:+.4f}", flush=True)

    liq_lag, oi_lag = _lagged_liq_and_oi(df_est)
    # guard: reconstructed shock matches the panel shock (alignment)
    recon = float(np.nanmax(np.abs(
        np.log1p(liq_lag) - df_est["shock"].fillna(0.0).to_numpy(dtype=float))))
    assert recon < 1e-8, f"shock reconstruction err {recon:.2e} — misaligned"

    # ---- baseline Delta per h (observed panel) ----
    base = {}
    shock0 = df_est["shock"].to_numpy(dtype=float).copy()
    sxoi0 = df_est["shock_x_oi_high"].to_numpy(dtype=float).copy()
    for h in args.horizons:
        bd, bu, d, n = delta_at(df_est, h, q_lo, q_hi)
        base[h] = {"bd": bd, "bu": bu, "delta": d, "n": n}
        print(f"  baseline h={h:>2}: beta_down={bd:+.6f} beta_up={bu:+.6f} "
              f"Delta={d:+.6f}  |Delta|/SESOI={abs(d)/sesoi:.3f}", flush=True)

    # ---- re-fit Delta under each (m, allocation, h) ----
    rows = []
    for m in args.m_grid:
        for alloc in args.allocations:
            Lstar = inflated_liq(liq_lag, oi_lag, m, alloc)
            shock_star = np.log1p(Lstar)
            df_est["shock"] = shock_star
            df_est["shock_x_oi_high"] = (shock_star
                                         * df_est["oi_high"].to_numpy(dtype=float))
            for h in args.horizons:
                bd, bu, d, n = delta_at(df_est, h, q_lo, q_hi)
                b0 = base[h]
                rows.append({
                    "m": float(m), "allocation": alloc, "h": int(h),
                    "alpha": ALPHA_HEAD,
                    "beta_down_base": b0["bd"], "beta_up_base": b0["bu"],
                    "delta_base": b0["delta"],
                    "beta_down_refit": bd, "beta_up_refit": bu,
                    "delta_refit": d, "ddelta": d - b0["delta"],
                    "abs_delta_refit": abs(d),
                    "sesoi": sesoi, "mde80": mde80,
                    "within_sesoi": bool(abs(d) < sesoi),
                    "within_mde": bool(abs(d) < mde80),
                    "n_obs": n,
                })
            # restore observed columns for the next allocation's baseline parity
            df_est["shock"] = shock0
            df_est["shock_x_oi_high"] = sxoi0
            sub = [r for r in rows if r["m"] == m and r["allocation"] == alloc]
            d0 = next(r["delta_refit"] for r in sub if r["h"] == args.horizons[0])
            print(f"  m={m:>4} alloc={alloc:<12} h0: Delta_refit={d0:+.6f} "
                  f"|Delta|/SESOI={abs(d0)/sesoi:.3f} "
                  f"within={'YES' if abs(d0)<sesoi else 'NO'}", flush=True)

    df_out = (pd.DataFrame(rows)
              .sort_values(["m", "allocation", "h"], kind="mergesort")
              .reset_index(drop=True))

    # ---- verdict: focus on h=0 (where the strong bound lives), m<=defensible ----
    h0 = args.horizons[0]
    dsub = df_out[(df_out["h"] == h0)
                  & (df_out["m"] <= args.m_defensible + 1e-9)]
    max_abs_delta = float(dsub["abs_delta_refit"].max())
    all_within = bool(dsub["within_sesoi"].all())
    worst = dsub.loc[dsub["abs_delta_refit"].idxmax()]
    sign_flip = bool(((dsub["delta_base"] * dsub["delta_refit"]) < 0).any())

    if all_within:
        verdict = (f"CONFIRMATORY: at h={h0}, every allocation at m<={args.m_defensible} "
                   f"keeps |Delta_refit| < SESOI={sesoi:.6f} (max |Delta|="
                   f"{max_abs_delta:.6f} = {max_abs_delta/sesoi:.2f}x SESOI, "
                   f"alloc={worst['allocation']}). The exceedance-Delta EQUIVALENCE "
                   f"BOUND survives the Maker coverage gap: the gap moves the "
                   f"SECTION-5 magnitude but NOT the SECTION-7 bounded asymmetry. "
                   f"The thesis is robust; only the medium-horizon magnitude carries "
                   f"a coverage caveat.")
        confirmatory = True
    else:
        verdict = (f"NON-CONFIRMATORY: at h={h0}, some allocation at "
                   f"m<={args.m_defensible} pushes |Delta_refit|={max_abs_delta:.6f} "
                   f">= SESOI={sesoi:.6f} ({max_abs_delta/sesoi:.2f}x; "
                   f"alloc={worst['allocation']}, sign_flip={sign_flip}). The Maker "
                   f"coverage gap can BREACH the equivalence bound — a genuine "
                   f"threat to the headline quantitative claim. REPORT.")
        confirmatory = False

    meta = {
        "script": "scripts/aux/run_maker_exceedance.py",
        "purpose": ("Coverage-gap robustness of the THESIS-bounded object: the "
                    "per-period exceedance Delta=beta_down-beta_up at alpha=0.01. "
                    "Decisive companion to run_maker_asymmetry (QLP gap) and "
                    "run_maker_bound (level)."),
        "object": "exceedance_delta alpha=0.01 (per-period, paired LPM)",
        "sesoi_strict_p50p95": sesoi, "mde80": mde80,
        "baseline_delta_by_h": {str(h): base[h]["delta"] for h in args.horizons},
        "m_grid": [float(m) for m in args.m_grid],
        "m_defensible": float(args.m_defensible),
        "allocations": list(args.allocations),
        "h_bound": h0,
        "max_abs_delta_refit_at_h0_mdef": max_abs_delta,
        "max_abs_delta_over_sesoi": max_abs_delta / sesoi,
        "all_within_sesoi_at_h0_mdef": all_within,
        "sign_flip_at_h0_mdef": sign_flip,
        "verdict": verdict, "confirmatory": confirmatory,
        "interpretation": ("If the bounded exceedance-Delta stays inside the SESOI "
                           "under every credible Maker allocation, the coverage gap "
                           "moves the apparent (Section-5) magnitude but not the "
                           "bounded (Section-7) asymmetry the thesis rests on."),
        "panel": str(CFG.FILES.econ_core_full),
        "n_rows_est": int(len(df_est)),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "maker_exceedance.csv"
    df_out.to_csv(csv_path, index=False)
    with open(args.out_dir / "maker_exceedance_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  wrote {csv_path}  shape={df_out.shape}", flush=True)
    print(df_out[["m", "allocation", "h", "delta_base", "delta_refit",
                  "abs_delta_refit", "within_sesoi"]].to_string(index=False),
          flush=True)
    print("\nVERDICT: " + verdict, flush=True)
    print(f"Done. {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
