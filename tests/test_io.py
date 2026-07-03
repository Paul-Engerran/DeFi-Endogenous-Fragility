"""Tests for `src.io`."""
from __future__ import annotations

import pandas as pd
import pytest


def test_load_econ_panel_shape(panel):
    """Panel must be exactly (41,328, 27). `funding_high` is the
    alternative leverage-stress regime indicator used by Test J.

    A regression on shape would silently break NB07/NB08 downstream.
    """
    assert panel.shape == (41328, 27), (
        f"Panel shape regression: expected (41328, 27), got {panel.shape}"
    )


def test_panel_date_utc_monotonic(panel):
    """`date` column must be UTC-localised, monotonic increasing,
    and start at 2021-03-15 00:00:00+00:00.
    """
    assert "date" in panel.columns
    assert panel["date"].is_monotonic_increasing, "date column not monotonic"
    assert str(panel["date"].dt.tz) == "UTC", "date column not UTC"
    expected_start = pd.Timestamp("2021-03-15T00:00:00Z")
    assert panel["date"].iloc[0] == expected_start
    expected_end_excl = pd.Timestamp("2025-12-01T00:00:00Z")
    assert panel["date"].iloc[-1] == expected_end_excl - pd.Timedelta(hours=1)


def test_panel_zero_missing_on_key_columns(panel):
    """Per DATA_STATUS §9, the 5 raw key columns have zero NaN.
    `ret_eth_perp` and other first-difference returns have NaN at
    row 0 by construction (first-diff initial condition); not tested
    here. The 3 DeFi columns are zero-filled by design after
    `run_defi_merge.py`.

    A regression on these columns would invalidate every robustness
    test downstream.
    """
    key_cols = [
        "close_perp", "oi", "funding_rate",
        "liq_usd_total", "liq_usd_collateral",
    ]
    for col in key_cols:
        assert panel[col].notna().all(), f"NaN regression in column {col!r}"


def test_panel_returns_first_row_nan_then_finite(panel):
    """First-difference returns: NaN at row 0 only; finite afterwards.
    Locks the warmup boundary expected by `_warmup` formula.
    """
    for col in ["ret_eth_perp", "ret_btc_spot", "ret_eth_spot"]:
        assert pd.isna(panel[col].iloc[0]), (
            f"{col} should be NaN at row 0 (first-diff)"
        )
        # After row 0, all values must be finite
        non_first = panel[col].iloc[1:]
        assert non_first.notna().all(), (
            f"Unexpected NaN in {col} past row 0"
        )
