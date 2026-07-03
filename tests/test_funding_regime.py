"""Smoke tests for the `funding_high` panel column.

`funding_high` is the alternative leverage-stress regime indicator,
consumed by `run_test_J` in `scripts/run_robustness_all.py`. By construction it
mirrors `oi_high`:

    funding_high = 1{funding_rate > P80 rolling 720h}  ∈ {0, 1}

These tests are skipped when the canonical post-DeFi panel is absent
(typical of CI on a fresh clone without the data archive).
"""
from __future__ import annotations

import pytest


def test_funding_high_present(panel):
    """`funding_high` must be in the full panel (after the schema widening
    that added `funding_high`)."""
    assert "funding_high" in panel.columns, list(panel.columns)


def test_funding_high_dtype(panel):
    """`funding_high` is encoded as int64 (matches `oi_high`)."""
    assert panel["funding_high"].dtype.kind == "i", panel["funding_high"].dtype


def test_funding_high_binary(panel):
    """`funding_high` ∈ {0, 1} only."""
    uniq = set(int(v) for v in panel["funding_high"].unique())
    assert uniq <= {0, 1}, uniq


def test_funding_high_share_is_nondegenerate(panel):
    """Over the post-warmup window the share of `funding_high == 1`
    must be a non-degenerate proportion: high enough to be a usable
    regime signal (not all-zero) and low enough to remain a tail-regime
    indicator (not all-one).

    The naive theoretical target P(rank > 0.80) = 0.20 holds only under
    a stationary marginal. With trending or persistent series (OI
    upward over 2021–2025; funding-rate clustered post-2022) the
    realised share drifts substantially from 0.20, which is expected
    and not a bug. We only assert a wide sanity range here.
    """
    from src.estimation import WARMUP_OI_WINDOW
    sub = panel.iloc[WARMUP_OI_WINDOW:]
    share = sub["funding_high"].mean()
    assert 0.05 <= share <= 0.45, f"funding_high share = {share:.4f}"


def test_oi_high_unchanged_by_q1_widening(panel):
    """Sanity: the `oi_high` indicator must still have a non-degenerate
    share after the schema widening (the addition of `funding_high` must
    not perturb pre-existing derived columns).

    Same wide sanity range as `funding_high` — see the docstring of
    `test_funding_high_share_is_nondegenerate` for why we do NOT pin
    the share at ≈0.20.
    """
    from src.estimation import WARMUP_OI_WINDOW
    sub = panel.iloc[WARMUP_OI_WINDOW:]
    share = sub["oi_high"].mean()
    assert 0.05 <= share <= 0.45, f"oi_high share = {share:.4f}"


def test_funding_high_and_oi_high_are_distinct(panel):
    """`funding_high` and `oi_high` must capture different regime
    occupancies — otherwise Test J would be vacuous. We require
    Hamming distance > 5% (i.e. ≥2,000 disagreeing hours).
    """
    disagree = (panel["funding_high"] != panel["oi_high"]).mean()
    assert disagree > 0.05, f"funding_high vs oi_high disagree on only {disagree:.2%}"


def test_panel_has_27_cols(panel):
    """After the schema widening that added `funding_high`: the full
    panel is 27 cols (was 26).

    Locks the schema contract documented in `docs/DATA_STATUS.md` §9 and
    in the module docstring of `scripts/run_defi_merge.py`.
    """
    assert panel.shape[1] == 27, panel.shape


def test_run_test_J_registered():
    """`J` must be in TEST_IDS, ALL_TESTS, OUTPUT_FILES, and `run_test_J`
    must be importable from `scripts/run_robustness_all`."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_robustness_all as rra

    assert rra.TEST_IDS.get("J") == 10
    assert "J" in rra.ALL_TESTS
    assert rra.OUTPUT_FILES.get("J") == "robustness_funding_regime_fast.csv"
    assert callable(rra.run_test_J)
