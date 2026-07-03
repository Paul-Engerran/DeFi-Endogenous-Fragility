#!/usr/bin/env python3
"""compute_tau_justification.py — auxiliary, hors `make all`.

Re-derives `tau_choice_justification.csv` from the main quantile-LP results.
This artefact was originally produced by a manual analysis step.

It is NOT consumed by any active notebook in the pipeline. It is shipped
as an auxiliary script for two reasons:

1. Reversibility: if the LaTeX draft (in the canonical repo
   Defi_endogenous_fragility) cites |β(τ,h)| / |β(0.50,h)| ratios, this
   script lets the author regenerate the CSV in one command without
   reopening the replication package.
2. Audit trail: documents the exact derivation from quantile_lp_results.csv
   so the legacy artefact's lineage is reproducible.

This script is OUT-OF-SCOPE for `make all` and `make smoke`. It must be
invoked explicitly:

    python scripts/aux/compute_tau_justification.py

Inputs
------
data/econ/quantile_lp_results.csv  (produced by run_quantile_lp.py)

Output
------
data/econ/tau_choice_justification.csv  (12 rows × 4 columns)

Columns: tau, h, beta_shock, ratio_vs_median = |β(τ,h)| / |β(0.50,h)|

Default scope: τ ∈ {0.01, 0.05}, h ∈ {0, 1, 3, 6, 12, 24}.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from config import CFG, ECON_DIR  # noqa: E402


DEFAULT_TAUS: list[float] = [0.01, 0.05]
DEFAULT_HORIZONS: list[int] = [0, 1, 3, 6, 12, 24]
MEDIAN_TAU: float = 0.50


def compute_ratios(
    df_main: pd.DataFrame,
    taus: list[float],
    horizons: list[int],
    median_tau: float = MEDIAN_TAU,
) -> pd.DataFrame:
    """Compute |β(τ,h)| / |β(median_tau, h)| from a quantile-LP results frame.

    Parameters
    ----------
    df_main : DataFrame with columns ['tau', 'h', 'beta_shock', ...]
    taus : list of float
        Tail quantiles to report (τ ∈ {0.01, 0.05}).
    horizons : list of int
    median_tau : float, default 0.50

    Returns
    -------
    DataFrame with columns ['tau', 'h', 'beta_shock', 'ratio_vs_median'].
    """
    rows = []
    for tau in taus:
        for h in horizons:
            sel_tail = df_main[
                (df_main["tau"] == tau) & (df_main["h"] == h)
            ]
            sel_med = df_main[
                (df_main["tau"] == median_tau) & (df_main["h"] == h)
            ]
            if sel_tail.empty or sel_med.empty:
                print(
                    f"  WARN: missing (tau={tau}, h={h}) or median row; "
                    f"skipping.",
                    file=sys.stderr,
                )
                continue
            b_tail = float(sel_tail.iloc[0]["beta_shock"])
            b_med = float(sel_med.iloc[0]["beta_shock"])
            if abs(b_med) < 1e-12:
                ratio = np.nan
            else:
                ratio = abs(b_tail) / abs(b_med)
            rows.append({
                "tau":             tau,
                "h":               h,
                "beta_shock":      b_tail,
                "ratio_vs_median": ratio,
            })
    return pd.DataFrame(rows, columns=["tau", "h", "beta_shock", "ratio_vs_median"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--main_csv", type=Path,
        default=ECON_DIR / "quantile_lp_results.csv",
        help="Input CSV (default: data/econ/quantile_lp_results.csv).",
    )
    ap.add_argument(
        "--out_csv", type=Path,
        default=ECON_DIR / "tau_choice_justification.csv",
        help="Output CSV (default: data/econ/tau_choice_justification.csv).",
    )
    ap.add_argument(
        "--taus", type=str, default=",".join(str(t) for t in DEFAULT_TAUS),
        help=f"Comma-separated tail quantiles (default: {DEFAULT_TAUS}).",
    )
    ap.add_argument(
        "--horizons", type=str,
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help=f"Comma-separated horizons (default: {DEFAULT_HORIZONS}).",
    )
    args = ap.parse_args()

    if not args.main_csv.exists():
        raise FileNotFoundError(
            f"{args.main_csv} not found. Run `make estimation` first."
        )

    print(f"Reading: {args.main_csv}", flush=True)
    df_main = pd.read_csv(args.main_csv)

    taus = [float(x) for x in args.taus.split(",")]
    horizons = [int(x) for x in args.horizons.split(",")]
    df_out = compute_ratios(df_main, taus, horizons)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"\nWrote: {args.out_csv}  ({len(df_out)} rows)", flush=True)
    print("\n" + df_out.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
