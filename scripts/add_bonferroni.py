"""
Bonferroni multiple-testing correction for the main quantile-LP table (tab:qlp).

Reads canonical CSV (READ ONLY), produces enriched CSV with two Bonferroni
columns:
  - p_bonf_table : global table family (m = 12 cells)
  - p_bonf_tail  : tail-only family, paired tail-vs-centre contrast (m = 6 h)

No re-run of pipeline. Pure arithmetic on existing p-values.
Author: Paul Engerran.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_IN = ROOT / "data" / "econ" / "quantile_lp_results.csv"
CSV_OUT = ROOT / "data" / "econ" / "quantile_lp_results_with_bonferroni.csv"

# Sub-grid for the main quantile-LP table (tail vs centre, 6 representative horizons)
TABLE_TAUS = [0.01, 0.50]
TABLE_HORIZONS = [0, 1, 3, 6, 12, 24]

# Family sizes for the two Bonferroni corrections
N_TABLE_CELLS = len(TABLE_TAUS) * len(TABLE_HORIZONS)  # = 12  (global table family)
N_PAIRED_CONTRASTS = len(TABLE_HORIZONS)               # = 6   (tail-only family)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    df = pd.read_csv(CSV_IN)
    print(f"Loaded canonical CSV: {CSV_IN}")
    print(f"  shape = {df.shape}, cols = {df.columns.tolist()}")
    print(f"  taus  = {sorted(df['tau'].unique())}")
    print(f"  h     = {sorted(df['h'].unique())}")

    # Sub-grid of the main quantile-LP table
    mask = df["tau"].isin(TABLE_TAUS) & df["h"].isin(TABLE_HORIZONS)
    df_table = df.loc[mask].copy().sort_values(["tau", "h"]).reset_index(drop=True)
    assert len(df_table) == N_TABLE_CELLS, (
        f"Expected {N_TABLE_CELLS} rows in main-table sub-grid, got {len(df_table)}"
    )

    # Global table family, m = 12
    df_table["pval_shock_bonf_table"] = (
        df_table["pval_shock"] * N_TABLE_CELLS
    ).clip(upper=1.0)
    df_table["pval_interaction_bonf_table"] = (
        df_table["pval_interaction"] * N_TABLE_CELLS
    ).clip(upper=1.0)

    # Tail-only family (τ=0.01 over 6 horizons), m = 6
    # NaN at τ=0.50 (this correction is meaningful only for the tail leg)
    df_table["pval_shock_bonf_tail"] = df_table.apply(
        lambda r: min(1.0, r["pval_shock"] * N_PAIRED_CONTRASTS)
        if r["tau"] == 0.01 else float("nan"),
        axis=1,
    )
    df_table["pval_interaction_bonf_tail"] = df_table.apply(
        lambda r: min(1.0, r["pval_interaction"] * N_PAIRED_CONTRASTS)
        if r["tau"] == 0.01 else float("nan"),
        axis=1,
    )

    # Write enriched CSV (NEW file, do not touch canonical)
    df_table.to_csv(CSV_OUT, index=False)
    print(f"\nWrote enriched CSV: {CSV_OUT}")
    print(f"  shape = {df_table.shape}")

    # Pretty-print the full enriched table
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\nEnriched main-table sub-grid (12 rows):")
    cols_show = [
        "tau", "h", "beta_shock", "pval_shock",
        "pval_shock_bonf_table", "pval_shock_bonf_tail",
        "beta_interaction", "pval_interaction",
        "pval_interaction_bonf_table", "pval_interaction_bonf_tail",
    ]
    print(df_table[cols_show].to_string(index=False))

    # Survival report
    alpha = 0.05
    def survive(col: str) -> int:
        return int((df_table[col] < alpha).sum())

    print(f"\n--- Survival report (p < {alpha}) ---")
    print(f"  pval_shock (uncorrected)              : "
          f"{survive('pval_shock')}/12 cells survive")
    print(f"  pval_shock_bonf_table  (table family) : "
          f"{survive('pval_shock_bonf_table')}/12 cells survive")
    print(f"  pval_shock_bonf_tail   (tail family)  : "
          f"{survive('pval_shock_bonf_tail')}/6 tail cells survive")
    print(f"  pval_interaction (uncorrected)              : "
          f"{survive('pval_interaction')}/12 cells survive")
    print(f"  pval_interaction_bonf_table  (table family) : "
          f"{survive('pval_interaction_bonf_table')}/12 cells survive")
    print(f"  pval_interaction_bonf_tail   (tail family)  : "
          f"{survive('pval_interaction_bonf_tail')}/6 tail cells survive")

    print(f"\n--- SHA256 ---")
    print(f"  canonical : {sha256_of(CSV_IN)}")
    print(f"  enriched  : {sha256_of(CSV_OUT)}")


if __name__ == "__main__":
    main()
