#!/usr/bin/env python3
"""
run_btc_placebo.py — [ROBUSTNESS / FLAGGED — does NOT change the main spec]

FULL BTC-OUTCOME PLACEBO of the quantile local projection.

Question
--------
Re-run the ENTIRE quantile-LP with BTC (ret_btc_spot) as the OUTCOME — same
RAW liquidation shock, same interaction, same horizons, same quantile grid —
and compare the BTC profile to the canonical ETH profile. The stakes:

  * If the apparent "downside amplification" signature (β(0.01) ≫ |β(0.50)| at
    impact, deepening with h) shows up on BTC TOO, the effect is NOT
    ETH-DeFi-specific — the shock behaves as a common-crypto stress proxy
    (consistent with global reverse causality). That KILLS any residual
    ETH-specificity claim BUT STRENGTHENS the methodological lesson: the
    naive-QLP artifact pattern is generic, not asset-specific.
  * If the ETH tail profile deepens materially more than BTC's, that is the
    real empirical defense of an ETH-specific component.

WARNING: DISTINCT FROM TEST A (run_robustness_all):
  Test A is a CROSS-ASSET PLACEBO OF THE SHOCK — it uses the BTC-ORTHOGONALISED
  shock (build_df_est_orth, SSIV protocol) on OTHER assets' returns (XRP, DOGE)
  as outcomes, asking "does an orthogonalised liquidation shock move unrelated
  assets?". HERE, BTC is the OUTCOME of the RAW-shock specification — we ask
  "does the headline ETH pattern replicate verbatim on BTC?". Different
  question, different shock definition, different outcome set.

Specification fidelity & the one unavoidable deviation
------------------------------------------------------
The estimation kernel is bit-identical to the main table: the panel comes from
src.estimation.build_df_est_raw (same warmup/shock/interaction), the estimator
is run_quantile_lp._fit_one (same REGRESSORS structure, QR_FIT_KWARGS,
MIN_OBS, max_iter), the LHS cumulative convention is the same
rolling(h+1).sum().shift(-h) applied to ret_btc_spot.

"Same controls" cannot be taken literally: the locked control set contains
ret_btc_spot, which IS the outcome here (at h=0, cumret_btc_h0 == ret_btc_spot
exactly → the regression is degenerate; at h>0 the cumulative LHS still
contains the control's own period-t return → mechanical fit). The faithful
translation is the MIRROR convention (default):

  --control_mode mirror   (DEFAULT) cross-asset control becomes ret_eth_perp.
        ETH main spec :  y=ETH | controls: ret_btc_spot + vol_eth_7d + funding + basis
        BTC placebo   :  y=BTC | controls: ret_eth_perp  + vol_eth_7d + funding + basis
        Each outcome is controlled for the OTHER asset's contemporaneous
        return — the two regressions are symmetric, hence comparable.
  --control_mode drop     sensitivity: no cross-asset return control at all
        (BTC response gross of the common-crypto co-movement).

Everything else (shock, shock_x_oi_high, oi_high, vol_eth_7d, funding_rate,
basis_bps, quantile grid, horizons) is the locked spec verbatim.

Inference note: kernel SEs are reported for completeness but the comparison
object is the β PROFILE (point estimates across τ × h), same reading as the
main table's Figure; tail-τ kernel SEs are unreliable (Test D1) and no new
inference claim is made from this placebo.

Sanity check built in: before fitting BTC, the script re-fits two ETH cells
((0.01, 0) and (0.50, 0)) with the canonical machinery and compares them to
data/econ/quantile_lp_results.csv (if present) — guaranteeing the kernel here
is the same kernel that produced the headline numbers.

OUTPUT (data/econ/)
-------------------
  btc_placebo_results.csv   [tau, h, beta_shock, se_shock, pval_shock,
                             beta_interaction, se_interaction,
                             pval_interaction, n_obs]   (BTC outcome)
  btc_vs_eth_profile.csv    per h: BTC vs ETH β(0.01), β(0.50), tail/median
                            ratio, left-right gap |β(0.01)|−|β(0.99)| (ETH
                            side read from quantile_lp_results.csv +
                            quantile_lp_results_9q.csv when available)
  btc_placebo_meta.json     provenance + control_mode + sanity-check report

Run
---
    .venv/bin/python scripts/aux/run_btc_placebo.py \
        --quantiles 0.01,0.50 --horizons 0,12 --out_dir /tmp/btc_smoke   # smoke
    .venv/bin/python scripts/aux/run_btc_placebo.py --n_jobs -1          # canonical
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from config import CFG, ECON_DIR  # noqa: E402
from src.estimation import build_df_est_raw  # noqa: E402
import run_quantile_lp as rqlp  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
BTC_COL: str = "ret_btc_spot"
ETH_COL: str = "ret_eth_perp"
# Main 6-quantile locked grid + 0.99 (the upside mirror of 0.01, needed for the
# left-right gap; the ETH-side 0.99 is read from quantile_lp_results_9q.csv).
TAUS_DEFAULT: list[float] = [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
CONTROL_MODES: tuple[str, ...] = ("mirror", "drop")
MAX_ITER_DEFAULT: int = 20000          # canonical NB07 value
RATIO_MED_FLOOR: float = 1e-4          # |β(0.50)| below this → ratio reported NaN

OUT_COLS: list[str] = list(rqlp.OUT_COLS)   # identical schema to the main table


def controls_for(mode: str) -> list[str]:
    """Adjusted control list (see module docstring)."""
    base = [c for c in rqlp.CONTROLS if c != BTC_COL]
    if mode == "mirror":
        return [ETH_COL] + base
    return base                            # drop


def add_btc_cumrets(df_est: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Materialise cumret_btc_h{h} with the EXACT build_df_est_raw convention."""
    d = df_est.copy()
    for h in horizons:
        col = f"cumret_btc_h{h}"
        if h == 0:
            d[col] = d[BTC_COL]
        else:
            d[col] = d[BTC_COL].rolling(h + 1).sum().shift(-h)
    return d


# ──────────────────────────────────────────────────────────────
# Sanity check — same kernel as the canonical ETH table
# ──────────────────────────────────────────────────────────────
def eth_kernel_sanity(df_est: pd.DataFrame, max_iter: int) -> dict:
    """Re-fit ETH (0.01, 0) and (0.50, 0) and compare to the canonical CSV."""
    report: dict = {"checked": False}
    canon_path = ECON_DIR / "quantile_lp_results.csv"
    if not canon_path.exists():
        report["note"] = "quantile_lp_results.csv absent; kernel check skipped"
        return report
    canon = pd.read_csv(canon_path)
    checks = []
    for tau in (0.01, 0.50):
        r = rqlp._fit_one(tau, 0, "cumret_h0", df_est, rqlp.REGRESSORS,
                          rqlp.CONTROLS, max_iter)
        row = canon[(canon["tau"] == tau) & (canon["h"] == 0)]
        if r is None or row.empty:
            checks.append({"tau": tau, "status": "unavailable"})
            continue
        ref = float(row["beta_shock"].iloc[0])
        got = float(r["beta_shock"])
        checks.append({
            "tau": tau, "beta_canonical": ref, "beta_refit": got,
            "abs_diff": abs(ref - got),
            "match_1e6": bool(abs(ref - got) < 1e-6),
        })
    report.update({"checked": True, "cells": checks})
    ok = all(c.get("match_1e6", False) for c in checks)
    print(f"  ETH kernel sanity: {'OK' if ok else 'WARNING: MISMATCH'} "
          + " ".join(f"tau={c['tau']}: Δ={c.get('abs_diff', float('nan')):.2e}"
                     for c in checks if 'abs_diff' in c), flush=True)
    return report


# ──────────────────────────────────────────────────────────────
# Profile comparison BTC vs canonical ETH
# ──────────────────────────────────────────────────────────────
def _eth_reference() -> pd.DataFrame | None:
    """Stack the canonical ETH tables (6τ main + 9q for τ=0.99) if present."""
    frames = []
    for name in ("quantile_lp_results.csv", "quantile_lp_results_9q.csv"):
        p = ECON_DIR / name
        if p.exists():
            f = pd.read_csv(p)[["tau", "h", "beta_shock"]]
            f["source"] = name
            frames.append(f)
    if not frames:
        return None
    eth = pd.concat(frames, ignore_index=True)
    # main table wins on duplicated (tau, h)
    eth = eth.drop_duplicates(subset=["tau", "h"], keep="first")
    return eth


def build_profile(df_btc: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Per-horizon BTC-vs-ETH comparison of the headline objects."""
    eth = _eth_reference()

    def _beta(frame: pd.DataFrame, tau: float, h: int) -> float:
        row = frame[(np.isclose(frame["tau"], tau)) & (frame["h"] == h)]
        return float(row["beta_shock"].iloc[0]) if len(row) else np.nan

    rows = []
    for h in horizons:
        b01 = _beta(df_btc, 0.01, h)
        b50 = _beta(df_btc, 0.50, h)
        b99 = _beta(df_btc, 0.99, h)
        row = {
            "h": int(h),
            "btc_beta_001": b01,
            "btc_beta_050": b50,
            "btc_ratio_tail_med": (abs(b01) / abs(b50)
                                   if abs(b50) >= RATIO_MED_FLOOR else np.nan),
            "btc_gap_lr": (abs(b01) - abs(b99)
                           if not (np.isnan(b01) or np.isnan(b99)) else np.nan),
        }
        if eth is not None:
            e01 = _beta(eth, 0.01, h)
            e50 = _beta(eth, 0.50, h)
            e99 = _beta(eth, 0.99, h)
            row.update({
                "eth_beta_001": e01,
                "eth_beta_050": e50,
                "eth_ratio_tail_med": (abs(e01) / abs(e50)
                                       if abs(e50) >= RATIO_MED_FLOOR else np.nan),
                "eth_gap_lr": (abs(e01) - abs(e99)
                               if not (np.isnan(e01) or np.isnan(e99)) else np.nan),
                "beta_001_btc_over_eth": (b01 / e01 if e01 else np.nan),
            })
        rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def _print_csv_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\n--- {name} ---", flush=True)
    print(f"shape: {df.shape}", flush=True)
    print("HEAD:", flush=True)
    print(df.head().to_string(index=False), flush=True)
    print("TAIL:", flush=True)
    print(df.tail().to_string(index=False), flush=True)


def save_outputs(df_btc: pd.DataFrame, df_profile: pd.DataFrame, meta: dict,
                 out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "btc_placebo_results.csv"
    df_btc.to_csv(res_path, index=False)
    print(f"  wrote {res_path}", flush=True)
    prof_path = out_dir / "btc_vs_eth_profile.csv"
    df_profile.to_csv(prof_path, index=False)
    print(f"  wrote {prof_path}", flush=True)
    meta_path = out_dir / "btc_placebo_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)
    _print_csv_summary("btc_placebo_results.csv", df_btc)
    _print_csv_summary("btc_vs_eth_profile.csv", df_profile)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--quantiles", type=_parse_floats, default=TAUS_DEFAULT,
                    help=f"Comma-separated. Default: {TAUS_DEFAULT}")
    ap.add_argument("--horizons", type=_parse_ints,
                    default=list(CFG.ECON.lp_horizons),
                    help="Comma-separated. Default: CFG.ECON.lp_horizons (0..24).")
    ap.add_argument("--control_mode", choices=CONTROL_MODES, default="mirror",
                    help="mirror (DEFAULT): cross-asset control becomes "
                         "ret_eth_perp. drop: no cross-asset return control.")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="1 = sequential; -1/-N = joblib loky over (tau,h).")
    ap.add_argument("--max_iter", type=int, default=MAX_ITER_DEFAULT)
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    print("run_btc_placebo: FULL QLP with BTC as OUTCOME", flush=True)
    print(f"  quantiles={args.quantiles}", flush=True)
    print(f"  horizons n={len(args.horizons)}  control_mode={args.control_mode}  "
          f"n_jobs={args.n_jobs}  max_iter={args.max_iter}", flush=True)

    t0 = time.time()
    print("Building estimation sample (build_df_est_raw) ...", flush=True)
    df_est = build_df_est_raw(horizons=args.horizons).reset_index(drop=True)
    print(f"  rows={len(df_est):,}", flush=True)

    sanity = eth_kernel_sanity(df_est, args.max_iter)

    df_est = add_btc_cumrets(df_est, args.horizons)
    controls = controls_for(args.control_mode)
    regressors = ["shock", "shock_x_oi_high", "oi_high"] + controls
    print(f"  BTC controls = {controls}", flush=True)

    jobs = [(tau, h, f"cumret_btc_h{h}")
            for tau in args.quantiles for h in args.horizons]
    print(f"Fitting BTC table ({len(jobs)} fits) ...", flush=True)
    if args.n_jobs == 1:
        out = [rqlp._fit_one(tau, h, y_col, df_est, regressors, controls,
                             args.max_iter)
               for tau, h, y_col in jobs]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(rqlp._fit_one)(tau, h, y_col, df_est, regressors, controls,
                                   args.max_iter)
            for tau, h, y_col in jobs
        )
    rows = [r for r in out if r is not None]
    print(f"  {len(rows)}/{len(jobs)} fits ok  ({time.time()-t0:.0f}s)", flush=True)
    df_btc = rqlp._to_sorted_df(rows)

    df_profile = build_profile(df_btc, args.horizons)

    # Console verdict preview at the headline horizons.
    for h in (0, 12):
        sub = df_profile[df_profile["h"] == h]
        if len(sub):
            s = sub.iloc[0]
            eth_part = (f"  ETH β01={s.get('eth_beta_001', np.nan):+.3f} "
                        f"ratio={s.get('eth_ratio_tail_med', np.nan):.1f}x"
                        if "eth_beta_001" in sub.columns else "")
            print(f"  h={h:>2}: BTC β01={s['btc_beta_001']:+.3f} "
                  f"ratio={s['btc_ratio_tail_med'] if not np.isnan(s['btc_ratio_tail_med']) else float('nan'):.1f}x"
                  + eth_part, flush=True)

    meta = {
        "script": "scripts/aux/run_btc_placebo.py",
        "purpose": ("Full BTC-outcome placebo of the quantile LP. "
                    "DISTINCT from Test A (orthogonalised-shock placebo on "
                    "XRP/DOGE outcomes): here the RAW-shock spec is rerun with "
                    "BTC as outcome."),
        "control_mode": args.control_mode,
        "controls": controls,
        "regressors": regressors,
        "quantiles": [float(t) for t in args.quantiles],
        "horizons": [int(h) for h in args.horizons],
        "max_iter": int(args.max_iter),
        "outcome_col": BTC_COL,
        "lhs_convention": "cumret_btc_h{h} = rolling(h+1).sum().shift(-h) of ret_btc_spot",
        "estimator": "run_quantile_lp._fit_one (canonical kernel)",
        "kernel_sanity_check": sanity,
        "inference_note": ("kernel SEs reported for completeness; comparison "
                           "object is the β profile (Test D1: tail kernel SEs "
                           "unreliable)"),
        "n_rows_panel": int(len(df_est)),
        "panel": str(CFG.FILES.econ_core_full),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    save_outputs(df_btc, df_profile, meta, args.out_dir)
    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
