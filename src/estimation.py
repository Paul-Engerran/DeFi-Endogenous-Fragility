"""Estimation-sample builders + QuantReg kernel kwargs + control lists.

This module consolidates the build_df_est_* family and prepare_arrays
that were duplicated/specialised across v1 scripts:

- run_robustness_all.build_df_est_orth (canonical, fully parameterised)
- run_robustness_all.build_df_est_raw  (canonical)
- run_robustness_all.prepare_arrays
- run_quantile_lp.py: imported build_df_est_raw via `from run_robustness_all`
- run_bootstrap.build_df_est           (specialised orth, ETH-only)
- run_bootstrap.prepare_arrays         (specialised, signature (df, h))

Functions
---------
- build_df_est_orth(horizons, assets=None, add_shock_x_oi=False,
                    merge_placebos=False) -> DataFrame
- build_df_est_raw(horizons) -> DataFrame
- prepare_arrays(df_est, y_col, regressors) -> (y, X)

Constants
---------
- CONTROLS_BASELINE     : 4 baseline controls
- BOOT_REGRESSORS       : 5 regressors for Test B / legacy bootstrap
- QR_KERNEL_KWARGS      : QuantReg kernel SE kwargs (vcov/kernel/bandwidth)
- WARMUP_OI_WINDOW = 720
- MIN_OBS_QR       = 500
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

from src.io import load_econ_panel, load_spot


# ── Constants (single source of truth across scripts) ──
CONTROLS_BASELINE: list[str] = [
    "ret_btc_spot", "vol_eth_7d", "funding_rate", "basis_bps",
]
BOOT_REGRESSORS: list[str] = [
    "shock", "shock_x_oi", "oi_high", "funding_rate", "basis_bps",
]
QR_KERNEL_KWARGS: dict = dict(
    vcov="robust",
    kernel="epa",
    bandwidth="hsheather",
)
WARMUP_OI_WINDOW: int = 720    # rolling OI z-score / rank window (NB03 verbatim)
MIN_OBS_QR: int = 500          # minimum observations to attempt QuantReg


def _warmup(horizons: list[int]) -> int:
    """warmup = max(vol_window, 720) + max(horizons) + 2.

    Verbatim from NB08 (cells A2/B2) and run_robustness_all.build_df_est_*.
    Lazy CFG import to keep src/ side-effect-free at import time.
    """
    from config import CFG
    return max(CFG.ECON.vol_window, WARMUP_OI_WINDOW) + max(horizons) + 2


def build_df_est_orth(
    horizons: list[int],
    assets: dict[str, str] | None = None,
    add_shock_x_oi: bool = False,
    merge_placebos: bool = False,
) -> pd.DataFrame:
    """Build estimation sample with orthogonalised shock.

    Replaces:
        run_robustness_all.build_df_est_orth (canonical, this is a verbatim port)
        run_bootstrap.build_df_est           (specialised wrapper, replaced
                                              by calling this with the right
                                              kwargs)

    The shock is the residuals of OLS(log_liq ~ ret_btc_spot + ret_btc_lag1)
    lagged by 1. Used by Tests A (placebo) and B (block bootstrap) of NB08.

    Parameters
    ----------
    horizons : list of int
        Cumulative-return horizons to materialise via cumret_{asset}_h{h}.
    assets : dict[asset_name, vol_normalized_ret_column], optional
        If given, materialises `cumret_{asset}_h{h}` for every (asset, h).
        If None, no cumret columns are added (caller does it).
    add_shock_x_oi : bool, default False
        If True, adds `shock_x_oi = shock.fillna(0) * oi_high` (Test B).
    merge_placebos : bool, default False
        If True, merges XRP and DOGE spot data and computes `ret_{asset}`
        and `ret_{asset}_std` for each (Test A).

    Returns
    -------
    df_est : DataFrame
        Panel sliced after warmup, with `shock`, optional shock_x_oi,
        optional placebo asset returns, and optional cumret columns.
    """
    from config import CFG
    df = load_econ_panel()

    if merge_placebos:
        df = df.merge(load_spot("xrp"), on="date", how="left")
        df = df.merge(load_spot("doge"), on="date", how="left")
        vol_window = CFG.ECON.vol_window
        for asset in ("xrp", "doge"):
            df[f"ret_{asset}"] = np.log(df[f"close_{asset}"]).diff() * 100
            vol = df[f"ret_{asset}"].rolling(vol_window).std()
            df[f"ret_{asset}_std"] = df[f"ret_{asset}"] / vol.replace(0, np.nan)

    # Ensure sorted order before warmup slice (row order from parquet is not guaranteed)
    df = df.sort_values("date").reset_index(drop=True)
    df_est = df.iloc[_warmup(horizons):].copy()

    df_est["ret_btc_lag1"] = df_est["ret_btc_spot"].shift(1)
    X_orth = sm.add_constant(df_est[["ret_btc_spot", "ret_btc_lag1"]].fillna(0))
    y_orth = df_est["log_liq"].fillna(0)
    mask_orth = X_orth.notna().all(axis=1) & y_orth.notna()
    orth = OLS(y_orth[mask_orth], X_orth[mask_orth]).fit()
    df_est["shock_orth"] = np.nan
    df_est.loc[mask_orth, "shock_orth"] = orth.resid
    df_est["shock"] = df_est["shock_orth"].shift(1)

    if assets is not None:
        for asset_name, ret_col in assets.items():
            for h in horizons:
                col = f"cumret_{asset_name}_h{h}"
                if h == 0:
                    df_est[col] = df_est[ret_col]
                else:
                    df_est[col] = df_est[ret_col].rolling(h + 1).sum().shift(-h)

    if add_shock_x_oi:
        df_est["shock_x_oi"] = df_est["shock"].fillna(0) * df_est["oi_high"]

    return df_est


def build_df_est_raw(horizons: list[int]) -> pd.DataFrame:
    """Build estimation sample with raw shock (log_liq.shift(1)).

    Replaces:
        run_robustness_all.build_df_est_raw (canonical, verbatim port)
        run_quantile_lp.py: previously imported via run_robustness_all

    Used by NB07 (main quantile-LP table), Tests D2/E/F of NB08.
    Adds:
        - shock = log_liq.shift(1)
        - shock_x_oi_high = shock * oi_high   (no .fillna(0))
        - cumret_h{h} for each h in horizons

    Parameters
    ----------
    horizons : list of int
        Horizons to materialise. The warmup uses max(CFG.ECON.lp_horizons)
        regardless of `horizons`, matching NB08 verbatim.

    Returns
    -------
    df_est : DataFrame after warmup slice.
    """
    from config import CFG
    df = load_econ_panel()
    df["shock"] = df["log_liq"].shift(1)
    df["shock_x_oi_high"] = df["shock"] * df["oi_high"]

    warmup = (max(CFG.ECON.vol_window, WARMUP_OI_WINDOW)
              + max(CFG.ECON.lp_horizons) + 2)
    df_est = df.iloc[warmup:].copy()

    for h in horizons:
        col = f"cumret_h{h}"
        if h == 0:
            df_est[col] = df_est["ret_eth_perp"]
        else:
            df_est[col] = df_est["ret_eth_perp"].rolling(h + 1).sum().shift(-h)
    return df_est


def prepare_arrays(
    df_est: pd.DataFrame,
    y_col: str,
    regressors: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build NaN-free NumPy y and X (with intercept).

    Replaces:
        run_robustness_all.prepare_arrays   (canonical, verbatim port)
        run_bootstrap.prepare_arrays        (specialised — was (df_est, h)
                                             with implicit BOOT_REGRESSORS)

    Parameters
    ----------
    df_est : DataFrame
        Estimation sample (output of build_df_est_*).
    y_col : str
        Name of the dependent-variable column.
    regressors : list[str]
        Names of the regressor columns (a constant column will be
        prepended in the returned X).

    Returns
    -------
    y : 1-D float64 array (no NaN)
    X : 2-D float64 array of shape (n, 1 + len(regressors)) (no NaN)
        Column 0 is the constant; columns 1..k follow `regressors` order.
    """
    cols = [y_col] + regressors
    clean = df_est.loc[df_est[cols].notna().all(axis=1), cols].reset_index(drop=True)
    y = clean[y_col].to_numpy(dtype=np.float64)
    X = np.column_stack([
        np.ones(len(clean), dtype=np.float64),
        clean[regressors].to_numpy(dtype=np.float64),
    ])
    return y, X
