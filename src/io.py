"""I/O helpers for loading parquet artefacts on the canonical UTC grid.

This module consolidates three near-identical helpers that were inlined in
the v1 scripts:

- run_data_prep._load_parquet  (cell NB02 08922d96)
- run_core_panel._load_utc     (cell NB03 06f2275d)
- run_robustness_all._load_econ_panel + _load_spot

Functions
---------
- load_utc_parquet(path, columns=None) -> DataFrame
- load_econ_panel() -> DataFrame
- load_spot(name)   -> DataFrame  (name in {btc, eth, xrp, doge})
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_utc_parquet(
    path: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a parquet, cast `date` to UTC, optionally restrict columns.

    Replaces:
        run_data_prep._load_parquet (which had a never-used cols_rename arg)
        run_core_panel._load_utc

    Parameters
    ----------
    path : Path
        Absolute path to a parquet file with a `date` column.
    columns : list[str] | None
        If given, restricts the read to these columns (passed to
        pyarrow). The `date` column is always included implicitly if
        listed; this helper does not auto-add it.

    Returns
    -------
    DataFrame with `date` cast to UTC pandas Timestamp.
    """
    df = pd.read_parquet(path, engine="pyarrow", columns=columns)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def load_econ_panel() -> pd.DataFrame:
    """Load the canonical post-DeFi hourly panel and sort by date.

    Replaces:
        run_robustness_all._load_econ_panel
        run_bootstrap.build_df_est (lines 1-3)

    The path is read from CFG.FILES.econ_core_full at call time (lazy
    import to avoid circulars).

    Returns
    -------
    DataFrame with 27 columns (per the run_defi_merge contract),
    sorted by date ascending, index reset.
    """
    from config import CFG  # lazy import: src/ stays import-cycle-free
    df = load_utc_parquet(CFG.FILES.econ_core_full)
    return df.sort_values("date").reset_index(drop=True)


def load_spot(name: str) -> pd.DataFrame:
    """Load a CCData spot feed and return [date, close_<name>].

    Replaces:
        run_robustness_all._load_spot

    Parameters
    ----------
    name : str
        One of {"btc", "eth", "xrp", "doge"}. The path is constructed
        as CFG.ROOT/data/normalized/spot/{name}_ccdata_1h.parquet.

    Returns
    -------
    DataFrame with two columns: ['date', f'close_{name}'].
    """
    from config import CFG  # lazy import
    path = CFG.ROOT / "data" / "normalized" / "spot" / f"{name}_ccdata_1h.parquet"
    s = load_utc_parquet(path)
    return s[["date", "close"]].rename(columns={"close": f"close_{name}"})
