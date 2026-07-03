#!/usr/bin/env python3
"""
run_block_sensitivity.py — [ROBUSTNESS / FLAGGED — does NOT change the main spec]

BLOCK-SIZE SENSITIVITY of the tail inference (the reserved Test-I slot
covers exactly this).

Question
--------
Every tail CI in the paper — and therefore the MDE bound itself
(SE = CI width / 3.92) — is a moving-block bootstrap with block = 24h
(CFG.ECON.block_boot_size, justified by daily seasonality and the Hall-1995
n^(1/3) ≈ 34 rule). A referee will ask: does the inference move at 12 / 36 /
48h blocks? This script re-runs the MDE-critical object — the PAIRED
exceedance asymmetry Delta = beta_down - beta_up at alpha = 0.01 (per-period
LPM, run_exceedance machinery verbatim) — plus the per-side LPM betas, across
block sizes, and reports the CI / MDE@80 per block.

Validation property (by construction)
-------------------------------------
The seed namespace reuses run_exceedance's EXACT TEST_IDs
(make_seed_sequences(42, TEST_ID, alpha_int, h, n)), so the block=24 rows
must REPRODUCE exceedance_results.csv / exceedance_paired.csv to the last
digit (same draws, same block) — a built-in cross-check that this script runs
the same machinery. Other block sizes share the same per-rep seed stream and
vary ONLY the block length.

OUTPUT (data/econ/)
-------------------
  block_sensitivity.csv   [block_size, alpha, h, object, estimate,
                           ci_lo, ci_hi, pval, mde80, n_obs]
       object in {delta_paired, beta_down, beta_up}
  block_sensitivity_meta.json

Run
---
    .venv/bin/python scripts/aux/run_block_sensitivity.py \
        --blocks 12,24 --horizons 0,12 --n_boot 60 \
        --out_dir /tmp/blk_smoke                         # smoke
    .venv/bin/python scripts/aux/run_block_sensitivity.py \
        --n_boot 1000 --n_jobs -1                        # canonical
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
sys.path.insert(0, str(HERE))

from config import CFG, ECON_DIR  # noqa: E402
from src.bootstrap import make_seed_sequences, run_parallel_boot, summarize  # noqa: E402
import run_exceedance as rexc  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
ALPHA: float = 0.01                      # the MDE-critical tail level
BLOCKS_DEFAULT: list[int] = [12, 24, 36, 48]
HORIZONS_DEFAULT: list[int] = [0, 1, 3, 6, 12, 24]

OUT_COLS: list[str] = [
    "block_size", "alpha", "h", "object", "estimate",
    "ci_lo", "ci_hi", "pval", "mde80", "n_obs",
]


def run(
    horizons: list[int],
    blocks: list[int],
    n_boot: int,
    n_jobs: int,
    ckpt_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    print("Building estimation sample (run_exceedance.build_sample) ...",
          flush=True)
    df_est = rexc.build_sample(horizons, cumulative=False)
    q_lo, q_hi = rexc.tail_thresholds(df_est, [ALPHA])[ALPHA]
    print(f"  rows={len(df_est):,}  thresholds q_lo={q_lo:+.4f} q_hi={q_hi:+.4f}",
          flush=True)
    a_int = int(round(ALPHA * 1000))

    rows: list[dict] = []
    for h in horizons:
        df_est = rexc.add_indicators(df_est, h, q_lo, q_hi)
        yD, yU, Xp = rexc._prepare_pair_arrays(df_est, h)
        bD = rexc.fit_point_lpm(yD, Xp)
        bU = rexc.fit_point_lpm(yU, Xp)
        d_point = bD["beta"] - bU["beta"]

        for block in blocks:
            # SAME seed namespace as run_exceedance -> block=24 must reproduce
            # exceedance_results/paired verbatim (built-in cross-check).
            for side, yv, tid, pt in (("beta_down", yD, rexc.TEST_ID_DOWN, bD),
                                      ("beta_up", yU, rexc.TEST_ID_UP, bU)):
                seeds = make_seed_sequences(rexc.BASE_SEED, tid, a_int, h,
                                            n=n_boot)
                boot = run_parallel_boot(
                    one_rep_fn=rexc._one_rep_lpm,
                    seeds=seeds,
                    args_tuple=(yv, Xp, block),
                    n_jobs=n_jobs,
                    batch_size=max(1, n_boot // 4),
                    ckpt_path=ckpt_dir,
                    out_shape_per_rep=(),
                    label=f"blk{block}_{side}_h{h}",
                )
                bs = summarize(boot)
                rows.append({
                    "block_size": block, "alpha": ALPHA, "h": int(h),
                    "object": side, "estimate": pt["beta"],
                    "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                    "pval": np.nan, "mde80": np.nan, "n_obs": pt["n_obs"],
                })

            seeds_p = make_seed_sequences(rexc.BASE_SEED, rexc.TEST_ID_PAIRED,
                                          a_int, h, n=n_boot)
            boot_pair = run_parallel_boot(
                one_rep_fn=rexc._one_rep_lpm_pair,
                seeds=seeds_p,
                args_tuple=(yD, yU, Xp, block),
                n_jobs=n_jobs,
                batch_size=max(1, n_boot // 4),
                ckpt_path=ckpt_dir,
                out_shape_per_rep=(2,),
                label=f"blk{block}_pair_h{h}",
            )
            ok = ~np.isnan(boot_pair).any(axis=1)
            deltas = boot_pair[ok, 0] - boot_pair[ok, 1]
            if len(deltas) == 0:
                ci_lo = ci_hi = pval = mde80 = np.nan
            else:
                ci_lo = float(np.percentile(deltas, 2.5))
                ci_hi = float(np.percentile(deltas, 97.5))
                centered = deltas - np.mean(deltas)
                pval = float(np.mean(np.abs(centered) >= np.abs(d_point)))
                mde80 = float(2.8 * (ci_hi - ci_lo) / 3.92)
            rows.append({
                "block_size": block, "alpha": ALPHA, "h": int(h),
                "object": "delta_paired", "estimate": float(d_point),
                "ci_lo": ci_lo, "ci_hi": ci_hi, "pval": pval, "mde80": mde80,
                "n_obs": int(len(yD)),
            })
            print(f"  h={h:>2} block={block:>2}  Delta={d_point:+.5f} "
                  f"CI=[{ci_lo:+.5f},{ci_hi:+.5f}]  MDE80={mde80:.5f}",
                  flush=True)

    df_out = (pd.DataFrame(rows)
              .sort_values(["object", "h", "block_size"], kind="mergesort")
              .reset_index(drop=True)[OUT_COLS])
    meta = {
        "script": "scripts/aux/run_block_sensitivity.py",
        "purpose": ("Block-size sensitivity (12/24/36/48h) of the alpha=0.01 "
                    "exceedance tail inference — the MDE-critical objects."),
        "alpha": ALPHA, "horizons": [int(h) for h in horizons],
        "blocks": [int(b) for b in blocks], "n_boot": int(n_boot),
        "seed": rexc.BASE_SEED,
        "seed_namespace": ("run_exceedance TEST_IDs verbatim -> block=24 rows "
                           "reproduce exceedance_results/paired exactly"),
        "thresholds": {"q_lo": q_lo, "q_hi": q_hi},
        "estimator": "per-period LPM, 7 NB07 regressors (run_exceedance verbatim)",
        "mde80_def": "2.8 * (ci_hi - ci_lo) / 3.92",
        "panel": str(CFG.FILES.econ_core_full),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    return df_out, meta


def save_outputs(df_out: pd.DataFrame, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "block_sensitivity.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}", flush=True)
    meta_path = out_dir / "block_sensitivity_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)
    print("\n--- block_sensitivity.csv ---", flush=True)
    print(f"shape: {df_out.shape}", flush=True)
    print("HEAD:", flush=True)
    print(df_out.head().to_string(index=False), flush=True)
    print("TAIL:", flush=True)
    print(df_out.tail().to_string(index=False), flush=True)


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--blocks", type=_parse_ints, default=BLOCKS_DEFAULT,
                    help=f"Comma-separated block lengths. Default: {BLOCKS_DEFAULT}")
    ap.add_argument("--horizons", type=_parse_ints, default=HORIZONS_DEFAULT,
                    help=f"Comma-separated. Default: {HORIZONS_DEFAULT}")
    ap.add_argument("--n_boot", type=int, default=150,
                    help="150 smoke / 1000 canonical.")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    print("run_block_sensitivity: MBB block-length sensitivity",
          flush=True)
    print(f"  blocks={args.blocks}  horizons={args.horizons}  "
          f"n_boot={args.n_boot}  n_jobs={args.n_jobs}", flush=True)

    t0 = time.time()
    ckpt_dir = args.out_dir / "_blocksens_ckpt"
    df_out, meta = run(args.horizons, args.blocks, args.n_boot,
                       args.n_jobs, ckpt_dir)
    save_outputs(df_out, meta, args.out_dir)
    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
