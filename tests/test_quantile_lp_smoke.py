"""Mini-smoke for `run_quantile_lp.compute_main` on a 2x2 grid (~2 s)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_compute_main_2x2(panel):
    """Run the main estimation on a tiny 2-tau × 2-h grid and check
    output schema. This is the unit-level mirror of `make smoke`.
    """
    pytest.importorskip("statsmodels")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from src.estimation import build_df_est_raw

    # Re-import as a module so we don't hit the script's __main__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_quantile_lp",
        str(Path(__file__).resolve().parent.parent / "scripts" / "run_quantile_lp.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df_est = build_df_est_raw(horizons=[0, 1])
    df_main = mod.compute_main(
        df_est, quantiles=[0.01, 0.50], horizons=[0, 1],
        n_jobs=1, max_iter=5000,
    )

    assert df_main.shape == (4, 9)  # 2 tau × 2 h, 9 contracted cols
    expected_cols = [
        "tau", "h", "beta_shock", "se_shock", "pval_shock",
        "beta_interaction", "se_interaction", "pval_interaction", "n_obs",
    ]
    assert list(df_main.columns) == expected_cols
    # Tau 0.01 at h=0 should give the headline negative β around -0.032
    row = df_main[(df_main["tau"] == 0.01) & (df_main["h"] == 0)].iloc[0]
    assert row["beta_shock"] < 0, "Expected negative β̂_0(0.01)"
    assert abs(row["beta_shock"] - (-0.032173)) < 1e-3, (
        f"β̂_0(0.01) drifted: got {row['beta_shock']}, expected ≈ -0.032"
    )
