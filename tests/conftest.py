"""Shared pytest fixtures for the defi-endogenous-fragility tests.

Tests run from the project root and depend on `src/` and `config.py`.
The `sys.path` insertion mirrors the same pattern used by every
`scripts/run_*.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def econ_dir(project_root: Path) -> Path:
    return project_root / "data" / "econ"


@pytest.fixture(scope="session")
def has_full_panel(econ_dir: Path) -> bool:
    """True iff `econ_core_full_1h.parquet` is present.

    Some unit tests need the full panel (heavy fixture); they are
    skipped when running on CI without the data archive.
    """
    return (econ_dir / "econ_core_full_1h.parquet").exists()


@pytest.fixture(scope="session")
def panel(has_full_panel: bool, econ_dir: Path):
    """Lazy load the canonical post-DeFi panel; skip if absent."""
    if not has_full_panel:
        pytest.skip("data/econ/econ_core_full_1h.parquet not present")
    import pandas as pd
    return pd.read_parquet(econ_dir / "econ_core_full_1h.parquet")
