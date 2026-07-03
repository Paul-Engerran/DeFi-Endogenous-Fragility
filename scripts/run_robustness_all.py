#!/usr/bin/env python3
"""
run_robustness_all.py — Unified CLI for all NB08 robustness tests.

Factorises the eight tests of `08_robustness.ipynb` (A, B, C, D1, D2,
E, F, G) into a single parallel, seeded, checkpointed script.

CHANGELOG vs run_bootstrap.py
─────────────────────────────
NEW
- Single CLI for tests A, B, C, D1, D2, E, F, G.
- Shared primitives: `_load_econ_panel`, `build_df_est_orth`,
  `build_df_est_raw`, `prepare_arrays`, `run_parallel_boot`, `summarize`,
  `summarize_pair`. Zero duplication across tests.
- SeedSequence namespace widened to 4 levels: [base_seed, test_id, h, b]
  (vs [base_seed, h, b] in run_bootstrap.py). Guarantees that B and E
  draw independent sequences even with the same base seed.
- Test E parallelised on the same model as B: `_one_rep_pair` returns
  (β_τ01, β_τ50) fitted on the SAME block resample (paired estimates).
- Test D1 reads `robustness_bootstrap_fast.csv` (seeded, reproducible)
  instead of the legacy `robustness_bootstrap.csv`. Difference is sub-bp
  by CLT at N_BOOT = 1000.
- Test G — block bootstrap CI for the interaction coefficient
  δ̂_h(0.01) (shock_x_oi at τ=0.01), 5 horizons. Mirrors Test B with
  shock_col_idx = 2 instead of 1.

TESTS COVERED — CSV produced
- A  → data/econ/robustness_placebo_fast.csv (+ .parquet)
- B  → data/econ/robustness_bootstrap_fast.csv
- C  → data/econ/robustness_sensitivity.csv         [snapshot legacy first]
- D1 → data/econ/se_comparison_kernel_bootstrap.csv
- D2 → data/econ/ols_lp_hac_benchmark.csv           [snapshot legacy first]
- E  → data/econ/quantile_monotonicity_test_fast.csv
- F  → data/econ/robustness_subperiods_fast.csv
- G  → data/econ/quantile_interaction_bootstrap_fast.csv
- J  → data/econ/robustness_funding_regime_fast.csv
       (alternative leverage-regime proxy `funding_high` vs `oi_high`;
       deterministic kernel SE, ~5 s)
- K  → data/econ/shock_definition_comparison.csv
       (β̂(shock) under RAW vs ORTH shock definitions at
       τ∈{0.01,0.50}, h∈{0,1,3,6,12,24}; deterministic kernel SE)
- L  → data/econ/robustness_bootstrap_raw_lhs_fast.csv
       (Test B refit on RAW LHS = cumret_h on ret_eth_perp, ORTH shock
       preserved; intermediate arm, the gap to the main RAW point at h=0
       persists)
- M  → data/econ/robustness_bootstrap_nb07_spec_fast.csv
       (Test B refit on the main spec = RAW shock + full 7-regressor
       set [shock, shock_x_oi_high, oi_high, ret_btc_spot, vol_eth_7d,
       funding_rate, basis_bps] + raw LHS cumret_h. Resolves the Fig 3
       point↔CI scale identity; bootstrap on the exact main-table
       parameter.)
- N  → data/econ/robustness_bootstrap_table_4_1_fast.csv
       (Test M extended to 4 tail quantiles τ∈{0.01,0.05,0.95,0.99}
       on the main spec. Provides bootstrap CIs to replace
       kernel Hall-Sheather SE in the paper's main quantile-LP
       table (tab:qlp) at the 4 tail τ.
       Kernel SE retained for τ=0.50 (centre) where density is well
       estimated. Output: 20 rows = 5 horizons × 4 τ, 8 cols.
       Cf. Chernozhukov et al. 2016 Extremal QR overview;
       Fitzenberger 1998 MBB for QR; Test D1 ratio SE_boot/SE_kernel
       ∈ [2.2, 3.5] at τ=0.01.)

Shock definitions
- Orthogonalised shock [resid(log_liq ~ ret_btc_spot + ret_btc_lag1).shift(1)]
  is used by A (placebo) and B (block bootstrap).
- Raw shock [log_liq.shift(1)] is used by C (sensitivity), D2 (OLS-LP HAC),
  E (quantile monotonicity), F (sub-period exclusions).
  The dual definition is deliberate; the paper's inference section documents it.

Usage
-----
    python run_robustness_all.py --tests all --n_boot 1000 --n_jobs -1 --seed 42
    python run_robustness_all.py --tests A --n_boot 10 --n_jobs 2     # smoke test
    python run_robustness_all.py --tests B,D1 --n_boot 1000 --seed 42
    python run_robustness_all.py --tests G --n_boot 1000 --seed 42
    python run_robustness_all.py --tests J                            # deterministic, ~5 s
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Maximum number of iterations")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from config import CFG, ECON_DIR, REPORTS_DIR  # noqa: E402

import statsmodels.api as sm  # noqa: E402
from statsmodels.regression.linear_model import OLS  # noqa: E402
from statsmodels.regression.quantile_regression import QuantReg  # noqa: E402

# ── Shared primitives (src/) ──
from src.io import load_econ_panel as _load_econ_panel, load_spot as _load_spot  # noqa: E402
from src.estimation import (                                                       # noqa: E402
    build_df_est_orth, build_df_est_raw, prepare_arrays,
    _warmup as _compute_warmup,
    MIN_OBS_QR,
)
from src.bootstrap import (                                                        # noqa: E402
    make_seed_sequences, summarize, summarize_pair, run_parallel_boot,
    one_rep_scalar as _one_rep_scalar, one_rep_pair as _one_rep_pair,
)


# ──────────────────────────────────────────────────────────────
# Constants — test namespace (SeedSequence) and output file map
# ──────────────────────────────────────────────────────────────
TEST_IDS: dict[str, int] = {
    "A": 0, "B": 1, "C": 2, "D1": 3, "D2": 4, "E": 5, "F": 6, "G": 7,
    "J": 10,   # slot 10; 8/9 reserved for future H/I tests
    "K": 11,   # shock-definition comparison RAW vs ORTH
    "L": 12,   # Test B refit on RAW LHS (intermediate arm)
    "M": 13,   # Test B refit on the main spec (RAW shock + 7 controls)
    "N": 14,   # Test M extended to 4 tail τ ∈ {0.01,0.05,0.95,0.99}
}
ALL_TESTS: list[str] = ["A", "B", "C", "D1", "D2", "E", "F", "G", "J", "K", "L", "M", "N"]

OUTPUT_FILES: dict[str, str] = {
    "A":  "robustness_placebo_fast.csv",
    "B":  "robustness_bootstrap_fast.csv",
    "C":  "robustness_sensitivity.csv",
    "D1": "se_comparison_kernel_bootstrap.csv",
    "D2": "ols_lp_hac_benchmark.csv",
    "E":  "quantile_monotonicity_test_fast.csv",
    "F":  "robustness_subperiods_fast.csv",
    "G":  "quantile_interaction_bootstrap_fast.csv",
    "J":  "robustness_funding_regime_fast.csv",
    "K":  "shock_definition_comparison.csv",
    "L":  "robustness_bootstrap_raw_lhs_fast.csv",
    "M":  "robustness_bootstrap_nb07_spec_fast.csv",
    "N":  "robustness_bootstrap_table_4_1_fast.csv",
}

# Block bootstrap parameters (match NB08 cells B2, E1)
BLOCK_SIZE = CFG.ECON.block_boot_size  # 24
TAU_BOOT = 0.01
MAX_ITER_BOOT = 3000
MAX_ITER_POINT = 5000


# ══════════════════════════════════════════════════════════════
# Test A — Cross-asset placebo (deterministic, kernel SE)
# ══════════════════════════════════════════════════════════════
def run_test_A(args: argparse.Namespace) -> Path:
    """
    Test A — Cross-asset placebo with orthogonalised shock.

    Deterministic (no bootstrap). Reproduces NB08 cells A1–A4.
    Writes both CSV and parquet.

    WARNING: Overwrites existing CSV at ECON_DIR / 'robustness_placebo.csv' is
    NOT an issue (new output uses '_fast' suffix).
    """
    t0 = time.time()
    print("[A] placebo cross-asset (orthogonalised shock)", flush=True)

    test_horizons = [0, 1, 3, 6, 12, 24]
    test_quantiles = [0.01, 0.05, 0.50]
    # Note: vol_eth_7d is intentionally dropped from the control set
    # for Test A. The placebo regresses cumulative returns of multiple
    # assets (ETH, BTC, XRP, DOGE) on the same shock; including an
    # ETH-specific volatility control would be inconsistent for the
    # non-ETH placebo arms. Convention follows Adrian, Boyarchenko, and
    # Giannone (2019, AER) on asset-specific controls in vulnerable-
    # growth-style cross-asset placebos.
    controls = ["ret_btc_spot", "funding_rate", "basis_bps"]
    assets = {
        "ETH":  "ret_eth_std",
        "BTC":  "ret_btc_std",
        "XRP":  "ret_xrp_std",
        "DOGE": "ret_doge_std",
    }

    df_est = build_df_est_orth(
        horizons=test_horizons,
        assets=assets,
        add_shock_x_oi=False,
        merge_placebos=True,
    )

    rows = []
    for asset_name in assets:
        for tau in test_quantiles:
            for h in test_horizons:
                y_col = f"cumret_{asset_name}_h{h}"
                regressors = ["shock"] + controls
                mask = df_est[[y_col, "shock"] + controls].notna().all(axis=1)
                y = df_est.loc[mask, y_col]
                X = sm.add_constant(df_est.loc[mask, regressors].fillna(0))
                if len(y) < 500:
                    continue
                try:
                    res = QuantReg(y, X).fit(
                        q=tau, vcov="robust", kernel="epa",
                        bandwidth="hsheather", max_iter=5000,
                    )
                    rows.append({
                        "asset": asset_name, "tau": tau, "h": h,
                        "beta_shock": res.params.get("shock", np.nan),
                        "se_shock":   res.bse.get("shock", np.nan),
                        "pval_shock": res.pvalues.get("shock", np.nan),
                        "n_obs": int(res.nobs),
                    })
                except Exception as e:
                    print(f"  warn {asset_name} tau={tau} h={h}: {e}", flush=True)
        print(f"  {asset_name} done", flush=True)

    df_placebo = pd.DataFrame(rows)
    out_csv = args.out_dir / OUTPUT_FILES["A"]
    out_parquet = out_csv.with_suffix(".parquet")
    df_placebo.to_csv(out_csv, index=False)
    df_placebo.to_parquet(out_parquet, index=False, engine="pyarrow")
    print(f"  wrote {out_csv}  (n={len(df_placebo)}, {(time.time()-t0):.1f}s)", flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test B — Block bootstrap
# ══════════════════════════════════════════════════════════════
def run_test_B(args: argparse.Namespace) -> Path:
    """
    Test B — Block bootstrap for β(shock) at τ=0.01, 5 horizons.

    Parallel, checkpointed. Reproduces NB08 cells B1–B2.
    """
    t0 = time.time()
    print(f"[B] block bootstrap  n_boot={args.n_boot}  n_jobs={args.n_jobs}", flush=True)

    boot_horizons = [0, 3, 6, 12, 24]
    boot_regressors = ["shock", "shock_x_oi", "oi_high", "funding_rate", "basis_bps"]
    shock_col_idx = 1 + boot_regressors.index("shock")

    df_est = build_df_est_orth(
        horizons=boot_horizons,
        assets={"ETH": "ret_eth_std"},
        add_shock_x_oi=True,
        merge_placebos=False,
    )

    ckpt_root = args.ckpt_dir / "test_B"
    results: dict[int, dict] = {}
    for h in boot_horizons:
        th = time.time()
        y, X = prepare_arrays(df_est, f"cumret_ETH_h{h}", boot_regressors)
        seeds = make_seed_sequences(args.seed, TEST_IDS["B"], h, n=args.n_boot)
        betas = run_parallel_boot(
            one_rep_fn=_one_rep_scalar,
            seeds=seeds,
            args_tuple=(y, X, BLOCK_SIZE, TAU_BOOT, shock_col_idx),
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            ckpt_path=ckpt_root,
            out_shape_per_rep=(),
            label=f"h{h:02d}",
        )
        if args.raw_dir is not None:
            args.raw_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.raw_dir / f"B_betas_h{h}.npy", betas)
        results[h] = summarize(betas)
        print(f"  h={h:>2} done in {(time.time()-th)/60:.2f} min  "
              f"mean={results[h]['mean']:+.4f}  "
              f"CI=[{results[h]['ci_lo']:+.4f}, {results[h]['ci_hi']:+.4f}]",
              flush=True)

    boot_df = pd.DataFrame(results).T
    boot_df.index.name = "h"
    boot_df = boot_df[["mean", "median", "ci_lo", "ci_hi", "n_success", "pct_negative"]]
    out_csv = args.out_dir / OUTPUT_FILES["B"]
    boot_df.to_csv(out_csv)
    print(f"  wrote {out_csv}  ({(time.time()-t0)/60:.2f} min)", flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test C — Sensitivity to specification choices (deterministic)
# ══════════════════════════════════════════════════════════════
def run_test_C(args: argparse.Namespace) -> Path:
    """
    Test C — Sensitivity: OI thresholds {70, 80, 90} + collateral variant.

    Deterministic. Reproduces NB08 cell 12. Uses the RAW shock
    (log_liq.shift(1)), NOT the orthogonalised shock — preserved as in
    the original notebook.

    WARNING: Overwrites ECON_DIR / 'robustness_sensitivity.csv' (no _fast suffix);
    snapshot the legacy CSV before the first run if comparison is needed.
    """
    t0 = time.time()
    print("[C] sensitivity (raw shock)", flush=True)

    df_c = _load_econ_panel()
    tau = 0.01
    warmup_c = _compute_warmup(CFG.ECON.lp_horizons)
    results = []

    def _fit_row(test_name: str, shock_series, interaction_series, oi_flag_series):
        y = df_c["ret_eth_perp"].iloc[warmup_c:]
        X = sm.add_constant(pd.DataFrame({
            "shock":        shock_series.iloc[warmup_c:],
            "shock_x_oi":   interaction_series.iloc[warmup_c:],
            "oi_high":      oi_flag_series.iloc[warmup_c:],
            "ret_btc_spot": df_c["ret_btc_spot"].iloc[warmup_c:],
            "vol_eth_7d":   df_c["vol_eth_7d"].iloc[warmup_c:],
            "funding_rate": df_c["funding_rate"].iloc[warmup_c:],
            "basis_bps":    df_c["basis_bps"].iloc[warmup_c:],
        }).fillna(0))
        mask = y.notna()
        try:
            res = QuantReg(y[mask], X[mask]).fit(
                q=tau, vcov="robust", kernel="epa",
                bandwidth="hsheather", max_iter=5000,
            )
            results.append({
                "test": test_name,
                "beta_shock":    res.params.get("shock", np.nan),
                "pval_shock":    res.pvalues.get("shock", np.nan),
                "beta_interact": res.params.get("shock_x_oi", np.nan),
                "se_interact":   res.bse.get("shock_x_oi", np.nan),
                "pval_interact": res.pvalues.get("shock_x_oi", np.nan),
            })
        except Exception as e:
            print(f"  warn {test_name}: {e}", flush=True)

    oi_roll_rank = df_c["oi"].rolling(720).rank(pct=True)
    shock = df_c["log_liq"].shift(1)
    for oi_pctile in (70, 80, 90):
        oi_flag = (oi_roll_rank > (oi_pctile / 100)).astype(int)
        _fit_row(f"OI_P{oi_pctile}", shock, shock * oi_flag, oi_flag)

    shock_coll = np.log1p(df_c["liq_usd_collateral"]).shift(1)
    oi_flag_80 = (oi_roll_rank > 0.80).astype(int)
    _fit_row("Collateral_USD", shock_coll, shock_coll * oi_flag_80, oi_flag_80)

    df_sens = pd.DataFrame(results)
    out_csv = args.out_dir / OUTPUT_FILES["C"]
    df_sens.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  (n={len(df_sens)}, {(time.time()-t0):.1f}s)", flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test D1 — Kernel SE vs Bootstrap SE (arithmetic from CSVs)
# ══════════════════════════════════════════════════════════════
def run_test_D1(args: argparse.Namespace, b_csv_path: Path | None) -> Path:
    """
    Test D1 — Kernel-vs-bootstrap SE comparison.

    If Test B ran in this call, `b_csv_path` is its returned Path. Otherwise
    this function falls back to `args.out_dir / 'robustness_bootstrap_fast.csv'`
    with a warning, and to the legacy `robustness_bootstrap.csv` as last resort.
    """
    t0 = time.time()
    print("[D1] kernel vs bootstrap SE", flush=True)

    if b_csv_path is None:
        fast = args.out_dir / OUTPUT_FILES["B"]
        legacy = args.out_dir / "robustness_bootstrap.csv"
        if fast.exists():
            b_csv_path = fast
            print(f"  WARN: Using existing bootstrap CSV at {fast}; "
                  f"for reproducibility rerun with --tests B,D1", file=sys.stderr)
        elif legacy.exists():
            b_csv_path = legacy
            print(f"  WARN: Using existing legacy CSV at {legacy}; "
                  f"for reproducibility rerun with --tests B,D1", file=sys.stderr)
        else:
            raise FileNotFoundError(
                f"D1 needs bootstrap output; neither {fast} nor {legacy} exists. "
                f"Run with --tests B,D1."
            )

    main_path = args.out_dir / "quantile_lp_results.csv"
    if not main_path.exists():
        raise FileNotFoundError(
            f"D1 needs {main_path} (produced by notebook 07); not found."
        )

    df_main = pd.read_csv(main_path)
    boot_df = pd.read_csv(b_csv_path)
    # Accept both index=h and column=h layouts
    if "h" not in boot_df.columns:
        boot_df = boot_df.rename(columns={boot_df.columns[0]: "h"})

    rows = []
    for _, brow in boot_df.iterrows():
        h = int(brow["h"])
        m = df_main[(df_main["tau"] == 0.01) & (df_main["h"] == h)]
        if m.empty:
            continue
        m = m.iloc[0]
        beta = m["beta_shock"]
        se_kernel = m["se_shock"]
        ci_lo, ci_hi = brow["ci_lo"], brow["ci_hi"]
        se_boot = (ci_hi - ci_lo) / (2 * 1.96)
        ratio = se_boot / se_kernel if se_kernel > 0 else np.nan
        rows.append({
            "h": h, "beta": beta, "se_kernel": se_kernel, "se_boot": se_boot,
            "ratio": ratio, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "zero_in_ci": bool(ci_lo <= 0 <= ci_hi),
        })

    df_d1 = pd.DataFrame(rows)
    out_csv = args.out_dir / OUTPUT_FILES["D1"]
    df_d1.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  (n={len(df_d1)}, {(time.time()-t0):.1f}s)", flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test D2 — OLS local projections + Newey-West HAC (deterministic)
# ══════════════════════════════════════════════════════════════
def run_test_D2(args: argparse.Namespace) -> Path:
    """
    Test D2 — OLS-LP with Newey-West HAC, 6 horizons.

    Deterministic. Reproduces NB08 cell 16. Uses the RAW shock.

    WARNING: Overwrites ECON_DIR / 'ols_lp_hac_benchmark.csv' (no _fast suffix);
    snapshot the legacy CSV before the first run if comparison is needed.
    """
    t0 = time.time()
    print("[D2] OLS-LP + Newey-West HAC (raw shock)", flush=True)

    ols_horizons = [0, 1, 3, 6, 12, 24]
    df_est = build_df_est_raw(horizons=ols_horizons)

    ols_controls = ["ret_btc_spot", "vol_eth_7d", "funding_rate", "basis_bps"]
    ols_regressors = ["shock", "shock_x_oi_high", "oi_high"] + ols_controls
    nw_lags = CFG.ECON.nw_lags

    rows = []
    for h in ols_horizons:
        y_col = f"cumret_h{h}"
        mask = df_est[[y_col, "shock"] + ols_controls].notna().all(axis=1)
        y = df_est.loc[mask, y_col]
        X = sm.add_constant(df_est.loc[mask, ols_regressors].fillna(0))

        res_ols = OLS(y, X).fit()
        actual_lags = max(h + 1, nw_lags)
        res_hac = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": actual_lags})

        rows.append({
            "h": h,
            "beta_shock_ols": res_hac.params.get("shock", np.nan),
            "se_ols":         res_ols.bse.get("shock", np.nan),
            "se_hac":         res_hac.bse.get("shock", np.nan),
            "pval_hac":       res_hac.pvalues.get("shock", np.nan),
            "ratio_hac_ols":  (res_hac.bse.get("shock", np.nan)
                               / res_ols.bse.get("shock", np.nan))
                              if res_ols.bse.get("shock", 0) > 0 else np.nan,
        })

    df_d2 = pd.DataFrame(rows)
    out_csv = args.out_dir / OUTPUT_FILES["D2"]
    df_d2.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  (n={len(df_d2)}, {(time.time()-t0):.1f}s)", flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test E — Quantile monotonicity (paired bootstrap)
# ══════════════════════════════════════════════════════════════
def run_test_E(args: argparse.Namespace) -> Path:
    """
    Test E — Monotonicity test Δ_h = β(τ=0.01, h) - β(τ=0.50, h).

    Parallel, checkpointed. Reproduces NB08 cell 19. Critical constraint:
    both τ=0.01 and τ=0.50 are fitted on the SAME block resample per
    replication (via `_one_rep_pair`), making Δ a paired estimate.

    p-value:
        centered = deltas - mean(deltas)
        p = mean(|centered| >= |delta_point|)
    """
    t0 = time.time()
    print(f"[E] paired monotonicity bootstrap  n_boot={args.n_boot}  "
          f"n_jobs={args.n_jobs}", flush=True)

    mono_horizons = [0, 1, 3, 6, 12, 24]
    df_est = build_df_est_raw(horizons=mono_horizons)

    controls = ["ret_btc_spot", "vol_eth_7d", "funding_rate", "basis_bps"]
    regressors = ["shock", "shock_x_oi_high", "oi_high"] + controls
    shock_col_idx = 1 + regressors.index("shock")
    taus = (0.01, 0.50)

    ckpt_root = args.ckpt_dir / "test_E"
    rows = []
    for h in mono_horizons:
        th = time.time()
        y, X = prepare_arrays(df_est, f"cumret_h{h}", regressors)

        # Deterministic point estimates (no bootstrap) — NB08 cell 19
        try:
            b01_pt = float(QuantReg(y, X).fit(q=0.01, max_iter=MAX_ITER_POINT)
                           .params[shock_col_idx])
            b50_pt = float(QuantReg(y, X).fit(q=0.50, max_iter=MAX_ITER_POINT)
                           .params[shock_col_idx])
            delta_point = b01_pt - b50_pt
        except Exception as e:
            print(f"  warn point h={h}: {e}", flush=True)
            delta_point = np.nan

        seeds = make_seed_sequences(args.seed, TEST_IDS["E"], h, n=args.n_boot)
        arr = run_parallel_boot(
            one_rep_fn=_one_rep_pair,
            seeds=seeds,
            args_tuple=(y, X, BLOCK_SIZE, taus, shock_col_idx),
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            ckpt_path=ckpt_root,
            out_shape_per_rep=(2,),
            label=f"h{h:02d}",
        )
        if args.raw_dir is not None:
            args.raw_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.raw_dir / f"E_pairs_h{h}.npy", arr)

        stats = summarize_pair(arr, delta_point)
        stats["h"] = h
        rows.append(stats)
        print(f"  h={h:>2} done in {(time.time()-th)/60:.2f} min  "
              f"Δ={delta_point:+.4f}  "
              f"CI=[{stats['delta_ci_lo']:+.4f}, {stats['delta_ci_hi']:+.4f}]  "
              f"p={stats['delta_pval']:.4f}", flush=True)

    cols = ["h",
            "beta01_mean", "beta01_median", "beta01_ci_lo", "beta01_ci_hi",
            "beta01_pct_negative",
            "beta50_mean", "beta50_median", "beta50_ci_lo", "beta50_ci_hi",
            "beta50_pct_negative",
            "delta_point", "delta_mean", "delta_median",
            "delta_ci_lo", "delta_ci_hi", "delta_pval", "n_boot"]
    df_e = pd.DataFrame(rows)[cols]
    out_csv = args.out_dir / OUTPUT_FILES["E"]
    df_e.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  ({(time.time()-t0)/60:.2f} min)", flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test F — Sub-period exclusions (deterministic, kernel SE)
# ══════════════════════════════════════════════════════════════
def run_test_F(args: argparse.Namespace) -> Path:
    """
    Test F — Sub-period robustness: drop Terra / FTX / USDC windows.

    Deterministic. Reproduces NB08 cell 22. Uses the RAW shock.
    """
    t0 = time.time()
    print("[F] sub-period exclusions (raw shock)", flush=True)

    horizons_sp = [0, 1, 6, 12, 24]
    df_est = build_df_est_raw(horizons=horizons_sp)

    controls = ["ret_btc_spot", "vol_eth_7d", "funding_rate", "basis_bps"]
    regressors = ["shock", "shock_x_oi_high", "oi_high"] + controls
    taus = [0.01, 0.50]
    exclusion_windows = {
        "none":  (None, None),
        "terra": ("2022-04-20", "2022-06-20"),
        "ftx":   ("2022-10-20", "2022-12-20"),
        "usdc":  ("2023-02-20", "2023-04-20"),
    }

    rows = []
    for period, (start, end) in exclusion_windows.items():
        if start is None:
            df_sub = df_est
        else:
            mask_excl = (
                (df_est["date"] >= pd.Timestamp(start, tz="UTC"))
                & (df_est["date"] <= pd.Timestamp(end, tz="UTC"))
            )
            df_sub = df_est.loc[~mask_excl]

        for tau in taus:
            for h in horizons_sp:
                y_col = f"cumret_h{h}"
                mask = df_sub[[y_col, "shock"] + controls].notna().all(axis=1)
                y = df_sub.loc[mask, y_col]
                X = sm.add_constant(df_sub.loc[mask, regressors].fillna(0))
                if len(y) < 500:
                    continue
                try:
                    res = QuantReg(y, X).fit(
                        q=tau, vcov="robust", kernel="epa",
                        bandwidth="hsheather", max_iter=20000,
                    )
                    rows.append({
                        "period_dropped": period, "tau": tau, "h": h,
                        "beta": res.params.get("shock", np.nan),
                        "se":   res.bse.get("shock", np.nan),
                        "pval": res.pvalues.get("shock", np.nan),
                        "n_obs": int(res.nobs),
                    })
                except Exception as e:
                    print(f"  warn period={period} tau={tau} h={h}: {e}", flush=True)
        print(f"  {period} done", flush=True)

    df_f = (pd.DataFrame(rows)
            [["period_dropped", "tau", "h", "beta", "se", "pval", "n_obs"]]
            .sort_values(["period_dropped", "tau", "h"])
            .reset_index(drop=True))
    out_csv = args.out_dir / OUTPUT_FILES["F"]
    df_f.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  (n={len(df_f)}, {(time.time()-t0):.1f}s)", flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test G — Block bootstrap on δ̂_h(0.01) (interaction coefficient)
# ══════════════════════════════════════════════════════════════
def run_test_G(args: argparse.Namespace) -> Path:
    """
    Test G — Block bootstrap for δ̂(shock_x_oi) at τ=0.01, 5 horizons.

    Parallel, checkpointed. Mirror of Test B with `shock_col_idx = 2`
    (constant + position of `shock_x_oi` in BOOT_REGRESSORS = 1) instead
    of 1, isolating the interaction coefficient that captures the
    leverage-cycle amplification in the high-OI regime.

    Provides bootstrap inference for δ̂_h, previously addressed only
    by kernel SE in `run_quantile_lp.py`.
    """
    t0 = time.time()
    print(f"[G] block bootstrap (interaction)  n_boot={args.n_boot}  "
          f"n_jobs={args.n_jobs}", flush=True)

    boot_horizons = [0, 3, 6, 12, 24]
    boot_regressors = ["shock", "shock_x_oi", "oi_high", "funding_rate", "basis_bps"]
    shock_col_idx = 1 + boot_regressors.index("shock_x_oi")

    df_est = build_df_est_orth(
        horizons=boot_horizons,
        assets={"ETH": "ret_eth_std"},
        add_shock_x_oi=True,
        merge_placebos=False,
    )

    ckpt_root = args.ckpt_dir / "test_G"
    results: dict[int, dict] = {}
    for h in boot_horizons:
        th = time.time()
        y, X = prepare_arrays(df_est, f"cumret_ETH_h{h}", boot_regressors)
        seeds = make_seed_sequences(args.seed, TEST_IDS["G"], h, n=args.n_boot)
        betas = run_parallel_boot(
            one_rep_fn=_one_rep_scalar,
            seeds=seeds,
            args_tuple=(y, X, BLOCK_SIZE, TAU_BOOT, shock_col_idx),
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            ckpt_path=ckpt_root,
            out_shape_per_rep=(),
            label=f"h{h:02d}",
        )
        if args.raw_dir is not None:
            args.raw_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.raw_dir / f"G_betas_h{h}.npy", betas)
        results[h] = summarize(betas)
        print(f"  h={h:>2} done in {(time.time()-th)/60:.2f} min  "
              f"mean={results[h]['mean']:+.4f}  "
              f"CI=[{results[h]['ci_lo']:+.4f}, {results[h]['ci_hi']:+.4f}]",
              flush=True)

    boot_df = pd.DataFrame(results).T
    boot_df.index.name = "h"
    boot_df = boot_df[["mean", "median", "ci_lo", "ci_hi", "n_success", "pct_negative"]]
    out_csv = args.out_dir / OUTPUT_FILES["G"]
    boot_df.to_csv(out_csv)
    print(f"  wrote {out_csv}  ({(time.time()-t0)/60:.2f} min)", flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test J — Funding regime as alternative leverage-stress proxy
# ══════════════════════════════════════════════════════════════
def run_test_J(args: argparse.Namespace) -> Path:
    """
    Test J — Sensitivity to the leverage-regime proxy.

    Re-estimates the main quantile-LP coefficients at τ=0.01,
    h ∈ {0,1,3,6,12,24} using `funding_high` (funding-rate upper-quintile
    rolling indicator) instead of `oi_high` as the leverage-regime
    indicator, and reports both regimes side-by-side. Deterministic,
    kernel SE. Mirrors the spec of Tests C/F (raw shock, kernel SE) — not
    the bootstrap of Test B — so it is fast (~5 s) and reproducible
    without `--seed`.

    Output schema (12 rows × 8 cols; 6 horizons × 2 regimes):
        regime, tau, h, beta_shock, se_shock,
        beta_interact, se_interact, pval_interact
    where regime ∈ {"oi_high", "funding_high"}.

    Rationale: convergence of the two regimes corroborates the
    OI-as-leverage proxy
    interpretation of the main spec; divergence opens an additional
    angle (`oi_high` ~ crowding/notional; `funding_high` ~ price-of-
    leverage / long-only stress).

    Note on collinearity: `funding_rate` (continuous) is kept in
    `base_controls` while `funding_high` (binary regime indicator) enters
    via the interaction. The two are not redundant — the indicator
    measures regime occupancy at the 720h-rolling P80 threshold, the
    continuous control absorbs level effects.
    """
    t0 = time.time()
    print("[J] alternative leverage regime (funding_high vs oi_high)", flush=True)

    horizons_j = [0, 1, 3, 6, 12, 24]
    df_est = build_df_est_raw(horizons=horizons_j)
    # build_df_est_raw already adds `shock` and `shock_x_oi_high`. We add
    # the parallel interaction for the funding regime (no .fillna(0),
    # matching the build_df_est_raw convention for shock_x_oi_high).
    df_est["shock_x_funding_high"] = df_est["shock"] * df_est["funding_high"]

    tau = 0.01
    base_controls = ["ret_btc_spot", "vol_eth_7d", "funding_rate", "basis_bps"]
    results = []

    for regime, interact_col, flag_col in [
        ("oi_high",      "shock_x_oi_high",      "oi_high"),
        ("funding_high", "shock_x_funding_high", "funding_high"),
    ]:
        for h in horizons_j:
            y_col = f"cumret_h{h}"
            regressors = ["shock", interact_col, flag_col] + base_controls
            mask = df_est[[y_col, "shock"] + base_controls].notna().all(axis=1)
            y = df_est.loc[mask, y_col]
            X = sm.add_constant(df_est.loc[mask, regressors].fillna(0))
            if len(y) < MIN_OBS_QR:
                continue
            try:
                res = QuantReg(y, X).fit(
                    q=tau, vcov="robust", kernel="epa",
                    bandwidth="hsheather", max_iter=20000,
                )
                results.append({
                    "regime":         regime,
                    "tau":            tau,
                    "h":              h,
                    "beta_shock":     res.params.get("shock", np.nan),
                    "se_shock":       res.bse.get("shock", np.nan),
                    "beta_interact":  res.params.get(interact_col, np.nan),
                    "se_interact":    res.bse.get(interact_col, np.nan),
                    "pval_interact":  res.pvalues.get(interact_col, np.nan),
                })
            except Exception as e:
                print(f"  warn regime={regime} h={h}: {e}", flush=True)
        print(f"  {regime} done", flush=True)

    df_j = pd.DataFrame(results)
    out_csv = args.out_dir / OUTPUT_FILES["J"]
    df_j.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  (n={len(df_j)}, {(time.time()-t0):.1f}s)", flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test K — Shock definition robustness (RAW vs ORTH)
# ══════════════════════════════════════════════════════════════
def run_test_K(args: argparse.Namespace) -> Path:
    """
    Test K — Sensitivity to shock definition.

    Re-estimates β̂(shock) at τ ∈ {0.01, 0.50}, h ∈ {0,1,3,6,12,24}
    under two shock definitions, holding the control set fixed:
      - RAW  = log_liq.shift(1)                          (cf. NB07 main spec)
      - ORTH = OLS-resid(log_liq ~ ret_btc_spot + ret_btc_lag1).shift(1)
                                                         (cf. Tests A/B/G)

    Same controls in both arms: [ret_btc_spot, vol_eth_7d,
    funding_rate, basis_bps]. Deterministic, kernel SE (Epanechnikov +
    Hall-Sheather), max_iter=20000.

    LHS note (acknowledged unit asymmetry): the RAW arm uses
    cumret_h{h} based on
    `ret_eth_perp` (the main-table LHS), while the ORTH arm uses
    cumret_ETH_h{h} based on `ret_eth_std` (the placebo/bootstrap
    LHS). Magnitudes are therefore on different scales; comparison
    is *qualitative* (sign, IRF shape) rather than purely numerical
    ratio. Both arms are reported so the reader can judge.

    Output schema (24 rows × 8 cols):
        shock_def, tau, h, beta_shock, se_shock, pval_shock, n_obs,
        iter_limit_hit
    where shock_def ∈ {"RAW", "ORTH"}.

    Rationale: the battery uses two shock definitions; Test K
    quantifies sensitivity to that choice (see the paper's methods
    section).
    """
    t0 = time.time()
    print("[K] shock definition comparison (RAW vs ORTH)", flush=True)

    horizons_k = [0, 1, 3, 6, 12, 24]
    taus_k = [0.01, 0.50]
    base_controls = ["ret_btc_spot", "vol_eth_7d", "funding_rate", "basis_bps"]

    # RAW path: build_df_est_raw exports `shock` = log_liq.shift(1)
    # and cumret_h{h} (based on ret_eth_perp, matching NB07 main).
    df_raw = build_df_est_raw(horizons=horizons_k)
    # ORTH path: build_df_est_orth exports `shock` = OLS-resid.shift(1)
    # and cumret_ETH_h{h} (based on ret_eth_std, matching Tests A/B/G).
    df_orth = build_df_est_orth(
        horizons=horizons_k,
        assets={"ETH": "ret_eth_std"},
        add_shock_x_oi=False,
        merge_placebos=False,
    )

    results = []
    sanity_h0 = {}  # for the post-write summary block

    for shock_def, df_est, ycol_prefix in [
        ("RAW",  df_raw,  "cumret_h"),
        ("ORTH", df_orth, "cumret_ETH_h"),
    ]:
        for tau in taus_k:
            for h in horizons_k:
                y_col = f"{ycol_prefix}{h}"
                regressors = ["shock"] + base_controls
                mask = df_est[[y_col, "shock"] + base_controls].notna().all(axis=1)
                y = df_est.loc[mask, y_col]
                X = sm.add_constant(df_est.loc[mask, regressors].fillna(0))
                if len(y) < MIN_OBS_QR:
                    continue

                iter_limit_hit = False
                try:
                    import warnings as _w
                    with _w.catch_warnings(record=True) as wlist:
                        _w.simplefilter("always")
                        res = QuantReg(y, X).fit(
                            q=tau, vcov="robust", kernel="epa",
                            bandwidth="hsheather", max_iter=20000,
                        )
                        iter_limit_hit = any(
                            "Maximum number of iterations" in str(w.message)
                            for w in wlist
                        )
                    row = {
                        "shock_def":      shock_def,
                        "tau":            tau,
                        "h":              h,
                        "beta_shock":     float(res.params.get("shock", np.nan)),
                        "se_shock":       float(res.bse.get("shock", np.nan)),
                        "pval_shock":     float(res.pvalues.get("shock", np.nan)),
                        "n_obs":          int(res.nobs),
                        "iter_limit_hit": bool(iter_limit_hit),
                    }
                    results.append(row)
                    if h == 0:
                        sanity_h0[(shock_def, tau)] = row["beta_shock"]
                except Exception as e:
                    print(f"  warn shock_def={shock_def} tau={tau} h={h}: {e}",
                          flush=True)
        print(f"  {shock_def} done", flush=True)

    df_k = (pd.DataFrame(results)
            [["shock_def", "tau", "h", "beta_shock", "se_shock",
              "pval_shock", "n_obs", "iter_limit_hit"]]
            .sort_values(["shock_def", "tau", "h"])
            .reset_index(drop=True))
    out_csv = args.out_dir / OUTPUT_FILES["K"]
    df_k.to_csv(out_csv, index=False)

    # ── Sanity-check block (intuitive RAW↔ORTH at h=0) ───────────
    def _fmt(v: float | None) -> str:
        return f"{v: .4f}" if v is not None and np.isfinite(v) else "   NA  "

    print("  Sanity check (RAW vs ORTH at h=0):", flush=True)
    for tau in taus_k:
        b_raw  = sanity_h0.get(("RAW",  tau))
        b_orth = sanity_h0.get(("ORTH", tau))
        ratio = (b_orth / b_raw) if (b_raw not in (None, 0)
                                     and b_orth is not None
                                     and np.isfinite(b_raw) and b_raw != 0
                                     and np.isfinite(b_orth)) else None
        ratio_str = f"  (ratio ORTH/RAW = {ratio:.2f})" if ratio is not None else ""
        print(f"    τ={tau:.2f} RAW  β ={_fmt(b_raw)}", flush=True)
        print(f"    τ={tau:.2f} ORTH β ={_fmt(b_orth)}{ratio_str}", flush=True)
    print(f"  wrote {out_csv}  (n={len(df_k)}, {(time.time()-t0):.1f}s)", flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test L — Block bootstrap (ORTH shock, RAW LHS) for Fig 3 unit-fix
# ══════════════════════════════════════════════════════════════
def run_test_L(args: argparse.Namespace) -> Path:
    """
    Test L — Bootstrap CI on the unit-comparable arm.

    Mirror of Test B but LHS = cumret_h{h} on ret_eth_perp (raw return,
    main-table units) instead of cumret_ETH_h{h} on ret_eth_std
    (vol-normalised). Shock is still ORTH (residualised log_liq lagged) —
    cohérent avec Test B/G's identifying object. Purpose: produce
    bootstrap CIs that are unit-comparable to the NB07 main-table point
    estimates so that Fig 3 can plot point and CI on a
    single scale without a caption note (avoids a natural referee
    objection on unit-comparability).

    5 horizons {0, 3, 6, 12, 24}, τ=0.01, same regressor set as Test B
    (shock, shock_x_oi, oi_high, funding_rate, basis_bps).

    Output (CSV indexed by h, 5 rows × 6 cols):
        mean, median, ci_lo, ci_hi, n_success, pct_negative
    """
    t0 = time.time()
    print(f"[L] block bootstrap (ORTH shock, RAW LHS)  "
          f"n_boot={args.n_boot}  n_jobs={args.n_jobs}", flush=True)

    boot_horizons = [0, 3, 6, 12, 24]
    boot_regressors = ["shock", "shock_x_oi", "oi_high",
                       "funding_rate", "basis_bps"]
    shock_col_idx = 1 + boot_regressors.index("shock")

    # ORTH shock construction (same as Test B) — but skip the
    # vol-normalised cumret materialisation; we'll add cumret_h{h} on
    # the raw ret_eth_perp below to keep the LHS on the main-table scale.
    df_est = build_df_est_orth(
        horizons=boot_horizons,
        assets=None,                # ← skip cumret_ETH_h{h} materialisation
        add_shock_x_oi=True,
        merge_placebos=False,
    )
    # Materialise cumret_h{h} on RAW ret_eth_perp (main-table LHS).
    # Mirrors the loop inside build_df_est_raw, so Test L's β̂ are
    # directly comparable to NB07 main and to Test K's RAW arm.
    for h in boot_horizons:
        col = f"cumret_h{h}"
        if h == 0:
            df_est[col] = df_est["ret_eth_perp"]
        else:
            df_est[col] = (df_est["ret_eth_perp"]
                           .rolling(h + 1).sum().shift(-h))

    ckpt_root = args.ckpt_dir / "test_L"
    results: dict[int, dict] = {}
    for h in boot_horizons:
        th = time.time()
        y, X = prepare_arrays(df_est, f"cumret_h{h}", boot_regressors)
        seeds = make_seed_sequences(args.seed, TEST_IDS["L"], h,
                                    n=args.n_boot)
        betas = run_parallel_boot(
            one_rep_fn=_one_rep_scalar,
            seeds=seeds,
            args_tuple=(y, X, BLOCK_SIZE, TAU_BOOT, shock_col_idx),
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            ckpt_path=ckpt_root,
            out_shape_per_rep=(),
            label=f"h{h:02d}",
        )
        if args.raw_dir is not None:
            args.raw_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.raw_dir / f"L_betas_h{h}.npy", betas)
        results[h] = summarize(betas)
        print(f"  h={h:>2} done in {(time.time()-th)/60:.2f} min  "
              f"mean={results[h]['mean']:+.4f}  "
              f"CI=[{results[h]['ci_lo']:+.4f}, "
              f"{results[h]['ci_hi']:+.4f}]", flush=True)

    boot_df = pd.DataFrame(results).T
    boot_df.index.name = "h"
    boot_df = boot_df[["mean", "median", "ci_lo", "ci_hi",
                       "n_success", "pct_negative"]]
    out_csv = args.out_dir / OUTPUT_FILES["L"]
    boot_df.to_csv(out_csv)
    print(f"  wrote {out_csv}  ({(time.time()-t0)/60:.2f} min)",
          flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test M — Block bootstrap on the NB07 EXACT spec
# ══════════════════════════════════════════════════════════════
def run_test_M(args: argparse.Namespace) -> Path:
    """
    Test M — Bootstrap CI on the exact NB07 main-table specification.

    Mirror of Test B but with two changes that align the spec to NB07:
      1. Shock = RAW (log_liq.shift(1)) via build_df_est_raw — same
         as NB07 main + Tests C/D2/E/F/J.
      2. Regressors = the full 7-variable NB07 set
         ["shock", "shock_x_oi_high", "oi_high", "ret_btc_spot",
          "vol_eth_7d", "funding_rate", "basis_bps"]
         (copied verbatim from run_quantile_lp.REGRESSORS); Tests
         B/G/L use a reduced 5-variable set that misses ret_btc_spot
         and vol_eth_7d.
    LHS unchanged from Test L: cumret_h{h} on ret_eth_perp (raw).

    Rationale: the
    factor-4.3× discrepancy between Test L (β̂≈-0.137 at h=0) and
    NB07 main (β̂=-0.0322) is entirely explained by the control set,
    not by the shock definition. With the full NB07 controls,
    RAW (β̂=-0.0322) and ORTH (β̂=-0.0332) agree within 3%; with the
    reduced Test-B control set, both jump to β̂≈-0.14. Test M provides
    bootstrap CIs on the exact NB07 RAW main-table coefficient so
    that Figure 3 plots point and 95% band on the same statistical
    object, used for Fig 3 in place of Test L.

    5 horizons {0, 3, 6, 12, 24}, τ=0.01, n_boot from CLI args, same
    block size as Test B (BLOCK_SIZE=24).

    Output (CSV indexed by h, 5 rows × 6 cols):
        mean, median, ci_lo, ci_hi, n_success, pct_negative
    """
    t0 = time.time()
    print(f"[M] block bootstrap (NB07 EXACT spec: RAW shock + 7 controls)  "
          f"n_boot={args.n_boot}  n_jobs={args.n_jobs}", flush=True)

    boot_horizons = [0, 3, 6, 12, 24]
    # Copied verbatim from run_quantile_lp.REGRESSORS — keep in sync
    # if NB07 main spec ever changes.
    nb07_regressors = ["shock", "shock_x_oi_high", "oi_high",
                       "ret_btc_spot", "vol_eth_7d",
                       "funding_rate", "basis_bps"]
    shock_col_idx = 1 + nb07_regressors.index("shock")   # = 1

    # RAW shock + cumret_h{h} on ret_eth_perp materialised by
    # build_df_est_raw (same primitive used by NB07).
    df_est = build_df_est_raw(horizons=boot_horizons)

    ckpt_root = args.ckpt_dir / "test_M"
    results: dict[int, dict] = {}
    for h in boot_horizons:
        th = time.time()
        y, X = prepare_arrays(df_est, f"cumret_h{h}", nb07_regressors)
        seeds = make_seed_sequences(args.seed, TEST_IDS["M"], h,
                                    n=args.n_boot)
        betas = run_parallel_boot(
            one_rep_fn=_one_rep_scalar,
            seeds=seeds,
            args_tuple=(y, X, BLOCK_SIZE, TAU_BOOT, shock_col_idx),
            n_jobs=args.n_jobs,
            batch_size=args.batch_size,
            ckpt_path=ckpt_root,
            out_shape_per_rep=(),
            label=f"h{h:02d}",
        )
        if args.raw_dir is not None:
            args.raw_dir.mkdir(parents=True, exist_ok=True)
            np.save(args.raw_dir / f"M_betas_h{h}.npy", betas)
        results[h] = summarize(betas)
        print(f"  h={h:>2} done in {(time.time()-th)/60:.2f} min  "
              f"mean={results[h]['mean']:+.4f}  "
              f"CI=[{results[h]['ci_lo']:+.4f}, "
              f"{results[h]['ci_hi']:+.4f}]", flush=True)

    boot_df = pd.DataFrame(results).T
    boot_df.index.name = "h"
    boot_df = boot_df[["mean", "median", "ci_lo", "ci_hi",
                       "n_success", "pct_negative"]]
    out_csv = args.out_dir / OUTPUT_FILES["M"]
    boot_df.to_csv(out_csv)
    print(f"  wrote {out_csv}  ({(time.time()-t0)/60:.2f} min)",
          flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# Test N — Block bootstrap on NB07 spec at τ ∈ {0.01, 0.05, 0.95, 0.99}
# ══════════════════════════════════════════════════════════════
def run_test_N(args: argparse.Namespace) -> Path:
    """
    Test N — Bootstrap CIs at 4 tail quantiles on the NB07 exact spec.

    Extends Test M to 4 quantiles τ∈{0.01,0.05,0.95,0.99} so
    that the paper's main quantile-LP table (tab:qlp) can report
    bootstrap 95% CIs at the four tail τ
    instead of kernel Hall-Sheather SE. The kernel SE at the median
    (τ=0.50) remains valid because the conditional density is well
    estimated there; at the tails, Test D1 reports the ratio
    SE_boot/SE_kernel ∈ [2.2, 3.5] at τ=0.01 across horizons, so the
    kernel SE underestimates uncertainty by a factor 2-3 at the tail.
    Cf. Chernozhukov et al. (2016) *Extremal Quantile Regression: An
    Overview* and Fitzenberger (1998) *Moving Blocks Bootstrap for QR*.

    Same spec as Test M:
      - Shock = RAW (log_liq.shift(1)) via build_df_est_raw
      - Regressors = full 7-variable NB07 set (verbatim from
        run_quantile_lp.REGRESSORS)
      - LHS = cumret_h{h} on ret_eth_perp (raw)
      - block_size = BLOCK_SIZE = 24, n_boot from CLI

    5-level SeedSequence namespace: [base_seed, test_id, tau_idx, h, b]
    where tau_idx∈{0,1,2,3} indexes {0.01,0.05,0.95,0.99}. Guarantees
    independence across τ for the same (test_id, h, b).

    Output (CSV, 20 rows = 5 horizons × 4 τ, 8 cols):
        tau, h, mean, median, ci_lo, ci_hi, n_success, pct_negative

    Wall-time: ~20-25 min on 16 vCPU at n_boot=1000; ~2-4 min on
    2-core Mac at n_boot=50 (smoke).
    """
    t0 = time.time()
    print(f"[N] block bootstrap multi-tau (NB07 EXACT spec) "
          f"n_boot={args.n_boot}  n_jobs={args.n_jobs}", flush=True)

    boot_horizons = [0, 3, 6, 12, 24]
    boot_taus = [0.01, 0.05, 0.95, 0.99]
    # Copied verbatim from run_quantile_lp.REGRESSORS — keep in sync
    # if NB07 main spec ever changes (identical to Test M).
    nb07_regressors = ["shock", "shock_x_oi_high", "oi_high",
                       "ret_btc_spot", "vol_eth_7d",
                       "funding_rate", "basis_bps"]
    shock_col_idx = 1 + nb07_regressors.index("shock")   # = 1

    # RAW shock + cumret_h{h} on ret_eth_perp materialised by
    # build_df_est_raw (same primitive used by NB07 and Test M).
    df_est = build_df_est_raw(horizons=boot_horizons)

    ckpt_root = args.ckpt_dir / "test_N"
    results: list[dict] = []

    for tau_idx, tau in enumerate(boot_taus):
        print(f"  tau={tau} (index {tau_idx})", flush=True)
        for h in boot_horizons:
            th = time.time()
            y, X = prepare_arrays(df_est, f"cumret_h{h}", nb07_regressors)
            # 5-level seed namespace: [base_seed, test_id, tau_idx, h, b]
            seeds = make_seed_sequences(
                args.seed, TEST_IDS["N"], tau_idx, h, n=args.n_boot,
            )
            ckpt_subdir = ckpt_root / f"tau{tau:.2f}"
            betas = run_parallel_boot(
                one_rep_fn=_one_rep_scalar,
                seeds=seeds,
                args_tuple=(y, X, BLOCK_SIZE, tau, shock_col_idx),
                n_jobs=args.n_jobs,
                batch_size=args.batch_size,
                ckpt_path=ckpt_subdir,
                out_shape_per_rep=(),
                label=f"tau{tau:.2f}_h{h:02d}",
            )
            if args.raw_dir is not None:
                args.raw_dir.mkdir(parents=True, exist_ok=True)
                np.save(args.raw_dir / f"N_betas_tau{tau:.2f}_h{h}.npy",
                        betas)
            stats = summarize(betas)
            stats["tau"] = tau
            stats["h"] = h
            results.append(stats)
            print(f"    h={h:>2} done in {(time.time()-th)/60:.2f} min  "
                  f"mean={stats['mean']:+.4f}  "
                  f"CI=[{stats['ci_lo']:+.4f}, {stats['ci_hi']:+.4f}]",
                  flush=True)

    df_n = pd.DataFrame(results)
    df_n = df_n[["tau", "h", "mean", "median", "ci_lo", "ci_hi",
                 "n_success", "pct_negative"]].sort_values(
        ["tau", "h"]).reset_index(drop=True)
    out_csv = args.out_dir / OUTPUT_FILES["N"]
    df_n.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  ({(time.time()-t0)/60:.2f} min, "
          f"n={len(df_n)})", flush=True)
    return out_csv


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def _parse_tests(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(ALL_TESTS)
    requested = [t.strip().upper() for t in raw.split(",") if t.strip()]
    unknown = [t for t in requested if t not in TEST_IDS]
    if unknown:
        raise SystemExit(f"Unknown test(s): {unknown}. Valid: {ALL_TESTS} or 'all'.")
    return [t for t in ALL_TESTS if t in requested]  # canonical order


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tests", type=str, default="all",
                    help="Comma-separated list (e.g. 'A,B,E') or 'all'.")
    ap.add_argument("--n_boot", type=int, default=1000,
                    help="Bootstrap replications for tests B and E.")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=100)
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    ap.add_argument("--ckpt_dir", type=Path, default=ECON_DIR / "_robust_ckpt")
    ap.add_argument("--raw_dir", type=Path, default=None,
                    help="If set, saves per-horizon bootstrap arrays as .npy")
    args = ap.parse_args()

    tests = _parse_tests(args.tests)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_robustness_all: tests={tests}  seed={args.seed}  "
          f"n_boot={args.n_boot}  n_jobs={args.n_jobs}", flush=True)

    t0 = time.time()
    outputs: dict[str, Path] = {}
    dispatch = {
        "A":  run_test_A,
        "B":  run_test_B,
        "C":  run_test_C,
        "D2": run_test_D2,
        "E":  run_test_E,
        "F":  run_test_F,
        "G":  run_test_G,
        "J":  run_test_J,
        "K":  run_test_K,
        "L":  run_test_L,
        "M":  run_test_M,
        "N":  run_test_N,
    }
    for t in tests:
        if t == "D1":
            outputs[t] = run_test_D1(args, outputs.get("B"))
        else:
            outputs[t] = dispatch[t](args)

    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.2f} min", flush=True)
    for t, p in outputs.items():
        print(f"  [{t}] → {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
