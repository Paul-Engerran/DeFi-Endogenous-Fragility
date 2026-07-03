"""Tests for `src.bootstrap` — unit tests locking the seeded block-bootstrap contract."""
from __future__ import annotations

import numpy as np
import pytest


def test_seed_independence_across_test_ids():
    """The 4-level SeedSequence `[base_seed, test_id, h, b]` must produce
    INDEPENDENT first-draw indices when only `test_id` differs.

    This locks the 4-level namespace: at fixed (base_seed, h, b), Tests B
    and E (test_id 1 and 5) MUST draw different residuals.
    """
    from src.bootstrap import make_seed_sequences

    seeds_b = make_seed_sequences(42, 1, 0, n=10)  # test_id=1 (Test B)
    seeds_e = make_seed_sequences(42, 5, 0, n=10)  # test_id=5 (Test E)

    rng_b = np.random.default_rng(seeds_b[0])
    rng_e = np.random.default_rng(seeds_e[0])
    draws_b = rng_b.integers(0, 1000, size=20)
    draws_e = rng_e.integers(0, 1000, size=20)

    assert not np.array_equal(draws_b, draws_e), (
        "Tests B and E produce identical bootstrap draws — SeedSequence "
        "namespace not 4-level"
    )


def test_seed_reproducibility_at_fixed_key():
    """Same key → same draws. Locks the determinism of the bootstrap."""
    from src.bootstrap import make_seed_sequences

    seeds_a = make_seed_sequences(42, 1, 0, n=5)
    seeds_b = make_seed_sequences(42, 1, 0, n=5)
    rng_a = np.random.default_rng(seeds_a[2])
    rng_b = np.random.default_rng(seeds_b[2])
    assert np.array_equal(
        rng_a.integers(0, 1000, size=10),
        rng_b.integers(0, 1000, size=10),
    )


def test_summarize_handles_empty_array():
    """Regression test: `summarize` of an empty/all-NaN array must
    return NaN-filled dict, not raise. Locks v1 bug fix.
    """
    from src.bootstrap import summarize
    result = summarize(np.array([]))
    assert result["n_success"] == 0.0
    assert np.isnan(result["mean"])
    assert np.isnan(result["median"])

    result = summarize(np.array([np.nan, np.nan, np.nan]))
    assert result["n_success"] == 0.0


def test_summarize_basic_stats():
    """Standard sanity check on `summarize`."""
    from src.bootstrap import summarize
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = summarize(arr)
    assert result["mean"] == 3.0
    assert result["median"] == 3.0
    assert result["n_success"] == 5.0
    assert result["pct_negative"] == 0.0


def test_one_rep_scalar_returns_finite_or_nan():
    """`one_rep_scalar` on a small synthetic problem must return either
    a finite float (success) or np.nan (graceful failure), never raise.
    """
    from src.bootstrap import one_rep_scalar
    rng_seed = np.random.SeedSequence([42, 0, 0])
    n = 200
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    y = X[:, 1] * 0.5 + X[:, 2] * (-0.3) + rng.standard_normal(n) * 0.1

    out = one_rep_scalar(rng_seed, y, X, block_size=24, tau=0.5,
                         shock_col_idx=1)
    assert np.isfinite(out) or np.isnan(out)


def test_one_rep_scalar_block_size_assertion():
    """If panel is smaller than block_size, must raise ValueError
    (regression on the v2 fix that replaced silent NaN with explicit
    error — see src/bootstrap.py line ~115).
    """
    from src.bootstrap import one_rep_scalar
    rng_seed = np.random.SeedSequence([42, 0, 0])
    y = np.zeros(10)
    X = np.column_stack([np.ones(10), np.ones(10)])
    with pytest.raises(ValueError, match="smaller than block_size"):
        one_rep_scalar(rng_seed, y, X, block_size=24, tau=0.5,
                       shock_col_idx=1)
