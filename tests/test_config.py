"""Smoke tests for `config.py` — locks the SST (single source of truth)."""
from __future__ import annotations

import pandas as pd


def test_config_temporal_window():
    """Window is locked at 41,328 hours = 4 years 9 months."""
    from config import CFG, START_UTC, END_UTC_EXCL
    assert START_UTC == pd.Timestamp("2021-03-15T00:00:00Z")
    assert END_UTC_EXCL == pd.Timestamp("2025-12-01T00:00:00Z")
    expected_n = int((END_UTC_EXCL - START_UTC) / pd.Timedelta(hours=1))
    assert expected_n == 41328


def test_config_econ_quantile_grid():
    """CFG.ECON.quantiles is the FULL 9-value grid (appendix);
    `run_quantile_lp.QUANTILES_DEFAULT` is the 6-value main subset.
    The two are intentionally different.
    """
    from config import CFG
    assert CFG.ECON.quantiles == [
        0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99
    ]
    assert CFG.ECON.lp_horizons == list(range(0, 25))
    assert CFG.ECON.block_boot_size == 24
    assert CFG.ECON.lp_n_boot == 1000


def test_config_econ_thresholds():
    """Stress and OI thresholds are locked."""
    from config import CFG
    assert CFG.ECON.stress_pctile == 95
    assert CFG.ECON.high_oi_pctile == 80
    assert CFG.ECON.vol_window == 168
