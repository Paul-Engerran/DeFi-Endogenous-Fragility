#!/usr/bin/env python3
"""
run_subsample_stability.py — [ROBUSTNESS / FLAGGED — does NOT change the main spec]

SUBSAMPLE STABILITY + LEAVE-OUT-AUGUST-2024.

Question
--------
Are the paper's headline objects driven by one half of the sample or by the
single largest liquidation episode (2024-08-05, max-day ≈ $307M, worst-7d
≈ $336M — one event)? Test F (run_robustness_all) only drops the terra / ftx /
usdc windows; it contains NO temporal split and does NOT leave out Aug-2024.
This script reports, per subsample:

  (a) beta_q01      — the headline QLP profile beta_shock(tau=0.01, h)
                      (point estimates; kernel SEs are unreliable in the tail
                      per Test D1, and a per-subsample QuantReg bootstrap
                      would be prohibitive — the stability reading is the
                      PROFILE, the inference object is (b)),
  (b) delta_exceed  — the thesis null object: paired exceedance asymmetry
                      Delta = beta_down - beta_up at alpha=0.01 (per-period
                      LPM, same-resample MBB 24h paired CI — the
                      run_exceedance machinery verbatim), plus its
                      MDE@80 = 2.8 x SE (SE = CI width / 3.92), so the
                      "bounded null" survives or not per subsample.

Subsamples
----------
  full         — reference (the canonical estimation sample)
  first_half   — observation time t < median date of the estimation sample
  second_half  — t >= median date
  loo_aug2024  — leave-out August-2024 episode: drop every row whose OUTCOME
                 window [t, t + max(horizons)] intersects
                 [--loo_start, --loo_end) (default 2024-08-01 → 2024-08-15);
                 i.e. rows t in [loo_start - max_h hours, loo_end) are dropped
                 so the episode contaminates neither the regressors' rows nor
                 the future-return outcomes of retained rows.

Construction notes
------------------
- Variables (shock, interaction, controls, rolling stats) are built ONCE on
  the full panel by src.estimation.build_df_est_raw — subsampling restricts
  the ESTIMATION ROWS, it does not rebuild the series (no rolling-window
  recomputation on truncated data).
- Tail thresholds (q_lo, q_hi) for the D/U indicators are SUBSAMPLE-SPECIFIC
  empirical quantiles, so each side carries ~alpha mass within the subsample
  (the run_exceedance construction logic, applied per subsample; thresholds
  are recorded in the meta).
- Seed namespace: make_seed_sequences(BASE_SEED, TEST_ID, alpha_int, h, n)
  with per-subsample TEST_IDs in the >=15 slot range (companion registry:
  slots 8/9 reserved, new tests take >=15).

OUTPUT (data/econ/)
-------------------
  subsample_stability.csv   [subsample, object, h, estimate, ci_lo, ci_hi,
                             pval, mde80, n_obs]
  subsample_stability_meta.json

Run
---
    .venv/bin/python scripts/aux/run_subsample_stability.py \
        --horizons 0,12 --n_boot 100 --max_iter 2000 \
        --out_dir /tmp/sub_smoke                       # smoke
    .venv/bin/python scripts/aux/run_subsample_stability.py \
        --n_boot 1000 --max_iter 20000 --n_jobs -1     # canonical
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
from src.bootstrap import make_seed_sequences, run_parallel_boot  # noqa: E402
from src.estimation import build_df_est_raw  # noqa: E402
import run_quantile_lp as rqlp  # noqa: E402
# Reuse the exceedance machinery verbatim (indicators, paired LPM bootstrap).
import run_exceedance as rexc  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
TAU_HEAD: float = 0.01
ALPHA_EXC: float = 0.01
HORIZONS_DEFAULT: list[int] = [0, 1, 3, 6, 12, 18, 24]
LOO_START_DEFAULT: str = "2024-08-01"
LOO_END_DEFAULT: str = "2024-08-15"

BASE_SEED: int = 42
# Companion registry: new tests take slots >= 15.
TEST_IDS: dict = {"full": 15101, "first_half": 15102,
                  "second_half": 15103, "loo_aug2024": 15104}

# Reference SESOI spans (beta units) from the smoke-grade mde_equivalence
# run (data/econ inventory). Used ONLY for the convenience
# verdict columns; the canonical MDE verdict remains run_mde_equivalence's.
SESOI_BETA_STRICT_P50P95: float = 0.001092
SESOI_BETA_IQR: float = 0.003834

OUT_COLS: list[str] = [
    "subsample", "object", "h", "estimate", "ci_lo", "ci_hi",
    "pval", "mde80", "n_obs",
]


def build_subsamples(df_est: pd.DataFrame, horizons: list[int],
                     loo_start: str, loo_end: str) -> dict:
    """{name: row-mask} over df_est (variables already built on full panel)."""
    dates = df_est["date"]
    med = dates.median()
    loo_s = pd.Timestamp(loo_start, tz="UTC") - pd.Timedelta(hours=max(horizons))
    loo_e = pd.Timestamp(loo_end, tz="UTC")
    return {
        "full":        pd.Series(True, index=df_est.index),
        "first_half":  dates < med,
        "second_half": dates >= med,
        "loo_aug2024": ~((dates >= loo_s) & (dates < loo_e)),
    }, {"median_split": str(med), "loo_drop_from": str(loo_s),
        "loo_drop_to": str(loo_e), "loo_backward_extension_hours": max(horizons)}


def run_one_subsample(
    name: str,
    df_sub: pd.DataFrame,
    horizons: list[int],
    n_boot: int,
    block_size: int,
    max_iter: int,
    n_jobs: int,
    ckpt_dir: Path,
) -> tuple[list[dict], dict]:
    """beta_q01 profile + paired exceedance Delta (alpha=0.01) on one subsample."""
    rows: list[dict] = []
    a_int = int(round(ALPHA_EXC * 1000))

    # ---- (a) headline QLP profile beta(tau=0.01, h) — point estimates ----
    for h in horizons:
        r = rqlp._fit_one(TAU_HEAD, h, f"cumret_h{h}", df_sub,
                          rqlp.REGRESSORS, rqlp.CONTROLS, max_iter)
        rows.append({
            "subsample": name, "object": "beta_q01", "h": int(h),
            "estimate": (np.nan if r is None else float(r["beta_shock"])),
            "ci_lo": np.nan, "ci_hi": np.nan, "pval": np.nan, "mde80": np.nan,
            "n_obs": (np.nan if r is None else int(r["n_obs"])),
        })

    # ---- (b) paired exceedance Delta at alpha=0.01 (per-period, MBB paired) ----
    # Subsample-specific unconditional thresholds (each side ~alpha mass).
    q_lo, q_hi = rexc.tail_thresholds(df_sub, [ALPHA_EXC])[ALPHA_EXC]
    df_sub = df_sub.copy()
    for h in horizons:
        # per-period future return (the clean object), then D/U indicators
        df_sub[f"fut_r_h{h}"] = (df_sub["ret_eth_perp"] if h == 0
                                 else df_sub["ret_eth_perp"].shift(-h))
        df_sub = rexc.add_indicators(df_sub, h, q_lo, q_hi)
        yD, yU, Xp = rexc._prepare_pair_arrays(df_sub, h)
        d_point = (rexc.fit_point_lpm(yD, Xp)["beta"]
                   - rexc.fit_point_lpm(yU, Xp)["beta"])
        seeds = make_seed_sequences(BASE_SEED, TEST_IDS[name], a_int, h, n=n_boot)
        boot = run_parallel_boot(
            one_rep_fn=rexc._one_rep_lpm_pair,
            seeds=seeds,
            args_tuple=(yD, yU, Xp, block_size),
            n_jobs=n_jobs,
            batch_size=max(1, n_boot // 4),
            ckpt_path=ckpt_dir,
            out_shape_per_rep=(2,),
            label=f"sub_{name}_h{h}",
        )
        ok = ~np.isnan(boot).any(axis=1)
        deltas = boot[ok, 0] - boot[ok, 1]
        if len(deltas) == 0:
            ci_lo = ci_hi = pval = mde80 = np.nan
        else:
            ci_lo = float(np.percentile(deltas, 2.5))
            ci_hi = float(np.percentile(deltas, 97.5))
            centered = deltas - np.mean(deltas)
            pval = float(np.mean(np.abs(centered) >= np.abs(d_point)))
            se = (ci_hi - ci_lo) / 3.92
            mde80 = float(2.8 * se)
        rows.append({
            "subsample": name, "object": "delta_exceed_a01", "h": int(h),
            "estimate": float(d_point), "ci_lo": ci_lo, "ci_hi": ci_hi,
            "pval": pval, "mde80": mde80, "n_obs": int(len(yD)),
        })
        print(f"  [{name}] h={h:>2}  Delta={d_point:+.5f} "
              f"CI=[{ci_lo:+.5f},{ci_hi:+.5f}] p={pval:.3f} MDE80={mde80:.5f}",
              flush=True)

    sub_meta = {"n_rows": int(len(df_sub)), "q_lo": q_lo, "q_hi": q_hi,
                "date_min": str(df_sub["date"].min()),
                "date_max": str(df_sub["date"].max()),
                "test_id": TEST_IDS[name]}
    return rows, sub_meta


def save_outputs(df_out: pd.DataFrame, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "subsample_stability.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}", flush=True)
    meta_path = out_dir / "subsample_stability_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)
    print("\n--- subsample_stability.csv ---", flush=True)
    print(f"shape: {df_out.shape}", flush=True)
    print("HEAD:", flush=True)
    print(df_out.head().to_string(index=False), flush=True)
    print("TAIL:", flush=True)
    print(df_out.tail().to_string(index=False), flush=True)


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--horizons", type=_parse_ints, default=HORIZONS_DEFAULT,
                    help=f"Comma-separated. Default: {HORIZONS_DEFAULT}")
    ap.add_argument("--n_boot", type=int, default=150,
                    help="Paired-bootstrap reps. 150 smoke / 1000 canonical.")
    ap.add_argument("--block_size", type=int, default=CFG.ECON.block_boot_size)
    ap.add_argument("--max_iter", type=int, default=20000,
                    help="QuantReg max_iter for the beta_q01 points "
                         "(2000 smoke / 20000 canonical).")
    ap.add_argument("--loo_start", type=str, default=LOO_START_DEFAULT)
    ap.add_argument("--loo_end", type=str, default=LOO_END_DEFAULT)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    print("run_subsample_stability: temporal splits + leave-out-Aug-2024",
          flush=True)
    print(f"  horizons={args.horizons}  n_boot={args.n_boot}  "
          f"block={args.block_size}  max_iter={args.max_iter}  "
          f"n_jobs={args.n_jobs}", flush=True)
    print(f"  loo window: {args.loo_start} -> {args.loo_end}", flush=True)

    t0 = time.time()
    print("Building estimation sample (build_df_est_raw, full panel) ...",
          flush=True)
    df_est = build_df_est_raw(horizons=args.horizons).reset_index(drop=True)
    print(f"  rows={len(df_est):,}", flush=True)

    masks, split_meta = build_subsamples(df_est, args.horizons,
                                         args.loo_start, args.loo_end)
    ckpt_dir = args.out_dir / "_subsample_ckpt"

    all_rows: list[dict] = []
    meta_subs: dict = {}
    for name, m in masks.items():
        df_sub = df_est.loc[m].reset_index(drop=True)
        print(f"\n=== subsample = {name}  (n={len(df_sub):,}) ===", flush=True)
        rows, sub_meta = run_one_subsample(
            name, df_sub, args.horizons, args.n_boot, args.block_size,
            args.max_iter, args.n_jobs, ckpt_dir,
        )
        all_rows.extend(rows)
        meta_subs[name] = sub_meta

    df_out = (pd.DataFrame(all_rows)
              .sort_values(["object", "subsample", "h"], kind="mergesort")
              .reset_index(drop=True)[OUT_COLS])

    meta = {
        "script": "scripts/aux/run_subsample_stability.py",
        "purpose": ("Temporal-split + leave-out-Aug-2024 stability of the "
                    "headline beta(0.01,h) profile and the paired exceedance "
                    "Delta at alpha=0.01."),
        "objects": {
            "beta_q01": "QLP beta_shock(0.01, h) point estimates "
                        "(profile reading; no tail kernel inference)",
            "delta_exceed_a01": "per-period LPM paired Delta, MBB block CI "
                                "(run_exceedance machinery; subsample-specific "
                                "thresholds)",
        },
        "splits": split_meta,
        "subsamples": meta_subs,
        "alpha": ALPHA_EXC, "tau_headline": TAU_HEAD,
        "horizons": [int(h) for h in args.horizons],
        "n_boot": int(args.n_boot), "block_size": int(args.block_size),
        "max_iter": int(args.max_iter), "seed": BASE_SEED,
        "seed_namespace": {
            "scheme": "make_seed_sequences(BASE_SEED, TEST_ID, alpha_int, h, n)",
            "test_ids": TEST_IDS,
        },
        "sesoi_reference_spans_beta": {
            "strict_p50p95": SESOI_BETA_STRICT_P50P95, "iqr": SESOI_BETA_IQR,
            "note": "smoke-grade reference; canonical verdicts via "
                    "run_mde_equivalence",
        },
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
