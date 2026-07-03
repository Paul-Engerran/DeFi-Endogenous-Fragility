"""Tests for `src.estimation`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_warmup_formula():
    """warmup = max(vol_window=168, WARMUP_OI_WINDOW=720) + max(horizons) + 2.

    For horizons=[0..24], expected = max(168,720)+24+2 = 746.
    Locked by NB03 + NB08 contract. A drift here moves N_after_warmup.
    """
    from src.estimation import _warmup
    assert _warmup([0, 24]) == 746
    assert _warmup([0]) == 722
    assert _warmup(list(range(0, 25))) == 746


def test_constants_unchanged():
    """The 5 constants in src.estimation are part of a locked
    bit-for-bit contract.
    """
    from src.estimation import (
        CONTROLS_BASELINE, BOOT_REGRESSORS, QR_KERNEL_KWARGS,
        WARMUP_OI_WINDOW, MIN_OBS_QR,
    )
    assert CONTROLS_BASELINE == [
        "ret_btc_spot", "vol_eth_7d", "funding_rate", "basis_bps",
    ]
    assert BOOT_REGRESSORS == [
        "shock", "shock_x_oi", "oi_high", "funding_rate", "basis_bps",
    ]
    assert QR_KERNEL_KWARGS == {
        "vcov": "robust", "kernel": "epa", "bandwidth": "hsheather",
    }
    assert WARMUP_OI_WINDOW == 720
    assert MIN_OBS_QR == 500


def test_prepare_arrays_shape_and_no_nan(panel):
    """`prepare_arrays` must return finite NaN-free y, X with the
    constant in column 0.
    """
    from src.estimation import prepare_arrays

    df = panel.iloc[800:1500].copy()  # post-warmup slice
    df["shock"] = df["log_liq"].shift(1)
    df["cumret_h0"] = df["ret_eth_perp"]
    y, X = prepare_arrays(df, "cumret_h0", ["shock", "ret_btc_spot"])

    assert y.ndim == 1
    assert X.ndim == 2
    assert X.shape[1] == 3, "Expected (const, shock, ret_btc_spot)"
    assert y.shape[0] == X.shape[0]
    assert np.isfinite(y).all()
    assert np.isfinite(X).all()
    # Column 0 is the constant
    assert np.allclose(X[:, 0], 1.0)


def test_build_df_est_raw_has_required_columns(panel):
    """Output of `build_df_est_raw` must contain `shock`,
    `shock_x_oi_high`, and `cumret_h0` (NB07 contract).

    Skip if the on-disk panel is unavailable.
    """
    pytest.importorskip("statsmodels")
    from src.estimation import build_df_est_raw
    df = build_df_est_raw(horizons=[0, 1])
    for col in ["shock", "shock_x_oi_high", "cumret_h0"]:
        assert col in df.columns, f"Missing column {col!r} in build_df_est_raw"
    # Post-warmup slice should give 41328 - 746 = 40582 rows
    assert len(df) == 40582, f"Expected 40582 rows post-warmup, got {len(df)}"
