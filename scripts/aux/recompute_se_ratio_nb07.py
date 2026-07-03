#!/usr/bin/env python3
"""
recompute_se_ratio_nb07.py — [POST-PROCESSOR / FLAGGED — arithmetic only]

SAME-SPEC kernel-vs-bootstrap SE ratio (fixes the Test-D1 confound).

Problem
-------
Test D1 (run_robustness_all) justifies the paper's "bootstrap CIs at tail
quantiles" rule with a ratio se_bootstrap / se_kernel in [2.2, 3.5] at
tau=0.01. But its se_bootstrap comes from Test B (ORTH shock, 5 regressors,
vol-NORMALISED LHS ret_eth_std) while its se_kernel comes from the NB07 main
table (RAW shock, 7 regressors, LHS in raw %): the ratio mixes the LHS scale,
the control set, and the shock definition. The clean comparison uses **Test M**
(the bootstrap on the EXACT NB07 spec — robustness_bootstrap_nb07_spec_fast.csv),
which exists precisely for same-object comparisons.

This script is pure arithmetic on two existing canonical CSVs (no re-run):

    se_boot_M  = (ci_hi - ci_lo) / 3.92          per h, tau = 0.01
    ratio      = se_boot_M / se_kernel(NB07)     same spec, same units, same beta

OUTPUT (data/econ/)
-------------------
  se_ratio_nb07.csv  [tau, h, beta_kernel, se_kernel, beta_boot_mean,
                      se_boot_M, ratio]
  (companion provenance fields embedded as header comment in the meta json)
  se_ratio_nb07_meta.json

Run
---
    .venv/bin/python scripts/aux/recompute_se_ratio_nb07.py
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from config import ECON_DIR  # noqa: E402

TAU_M: float = 0.01   # Test M is the tau=0.01 NB07-spec bootstrap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    ap.add_argument("--kernel_csv", type=Path,
                    default=ECON_DIR / "quantile_lp_results.csv")
    ap.add_argument("--boot_csv", type=Path,
                    default=ECON_DIR / "robustness_bootstrap_nb07_spec_fast.csv")
    args = ap.parse_args()

    kern = pd.read_csv(args.kernel_csv)
    boot = pd.read_csv(args.boot_csv)

    kern = kern[np.isclose(kern["tau"], TAU_M)][["h", "beta_shock", "se_shock"]]
    rows = []
    for _, b in boot.iterrows():
        h = int(b["h"])
        k = kern[kern["h"] == h]
        if k.empty:
            continue
        se_kernel = float(k["se_shock"].iloc[0])
        se_boot = float((b["ci_hi"] - b["ci_lo"]) / 3.92)
        rows.append({
            "tau": TAU_M, "h": h,
            "beta_kernel": float(k["beta_shock"].iloc[0]),
            "se_kernel": se_kernel,
            "beta_boot_mean": float(b["mean"]),
            "se_boot_M": se_boot,
            "ratio": se_boot / se_kernel if se_kernel > 0 else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("h").reset_index(drop=True)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "se_ratio_nb07.csv"
    df.to_csv(csv_path, index=False)
    meta = {
        "script": "scripts/aux/recompute_se_ratio_nb07.py",
        "purpose": ("Same-spec (main-table specification) kernel-vs-bootstrap SE ratio at "
                    "tau=0.01 — replaces the confounded D1 ratio (Test B: ORTH "
                    "shock, 5 regressors, vol-normalised LHS) as the citable "
                    "justification for bootstrap CIs at tail quantiles."),
        "inputs": {"kernel": str(args.kernel_csv), "bootstrap_M": str(args.boot_csv)},
        "se_boot_def": "(ci_hi - ci_lo) / 3.92 on the Test-M percentile CI",
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    with open(out_dir / "se_ratio_nb07_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {csv_path}", flush=True)
    print(f"  wrote {out_dir / 'se_ratio_nb07_meta.json'}", flush=True)

    print("\n--- se_ratio_nb07.csv ---")
    print(f"shape: {df.shape}")
    print("HEAD:")
    print(df.head().to_string(index=False))
    print("TAIL:")
    print(df.tail().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
