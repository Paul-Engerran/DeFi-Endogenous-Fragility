"""Block-bootstrap primitives shared across robustness scripts.

This module consolidates the bootstrap engine that was duplicated between
run_robustness_all.py and (legacy) run_bootstrap.py.

Functions
---------
- make_seed_sequences(*key, n)    : per-rep SeedSequences keyed by tuple
- one_rep_scalar(seed, y, X, ...) : one block-rep, scalar output
- one_rep_pair(seed, y, X, ...)   : one block-rep, paired (β_τ1, β_τ2)
- summarize(arr)                  : standard summary of bootstrap array
- summarize_pair(arr, delta_pt)   : paired summary for Test E
- run_parallel_boot(...)          : joblib loky driver with checkpointing

Constants
---------
- MAX_ITER_BOOT  = 3000   (QuantReg.fit max_iter inside workers)
- MAX_ITER_POINT = 5000   (QuantReg.fit max_iter for non-bootstrap point ests)
- TAU_BOOT       = 0.01   (default tail quantile for Test B)

Note on seed namespace
----------------------
The canonical scheme is 4-level [base_seed, test_id, h, b], which
guarantees independence across tests sharing a base seed.
`make_seed_sequences` accepts an arbitrary tuple:
    seeds = make_seed_sequences(base_seed, test_id, h, n=n_boot)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg


# ── Constants ──
MAX_ITER_BOOT: int = 3000
MAX_ITER_POINT: int = 5000
TAU_BOOT: float = 0.01


def make_seed_sequences(
    *key: int,
    n: int,
) -> list[np.random.SeedSequence]:
    """Per-replication SeedSequences keyed by an arbitrary integer tuple.

    Replaces:
        run_robustness_all.make_seed_sequences (4-level call site)
        run_bootstrap inline list comp        (3-level call site)

    Usage:
        # 4-level (run_robustness_all canonical scheme):
        seeds = make_seed_sequences(base_seed, test_id, h, n=n_boot)
        # 3-level (legacy):
        seeds = make_seed_sequences(base_seed, h, n=n_boot)
    """
    return [np.random.SeedSequence(list(key) + [b]) for b in range(n)]


def one_rep_scalar(
    seed_state: np.random.SeedSequence,
    y: np.ndarray,
    X: np.ndarray,
    block_size: int,
    tau: float,
    shock_col_idx: int,
) -> float:
    """One block-bootstrap replication; returns a single coefficient.

    Replaces:
        run_robustness_all._one_rep_scalar  (canonical)
        run_bootstrap._one_rep              (3-level seeds variant)

    Parameters
    ----------
    seed_state : np.random.SeedSequence
        Per-replication seed (deterministic regardless of worker).
    y, X : NumPy arrays
        Output of estimation.prepare_arrays(...).
    block_size : int
        Block length in rows (typically CFG.ECON.block_boot_size = 24).
    tau : float
        Quantile to fit.
    shock_col_idx : int
        Index in X of the coefficient to extract (1 + position in
        regressors for a const-prefixed X).

    Returns
    -------
    float : β at `shock_col_idx`, or np.nan on QuantReg failure.
    """
    import warnings as _w
    _w.filterwarnings("ignore")
    rng = np.random.default_rng(seed_state)
    n = len(y)
    if n < block_size:
        raise ValueError(
            f"Panel size n={n} smaller than block_size={block_size}. "
            f"Cannot perform block bootstrap. Check warmup truncation upstream."
        )
    n_blocks = n // block_size
    block_starts = rng.integers(0, n - block_size, size=n_blocks)
    idx = (block_starts[:, None] + np.arange(block_size)[None, :]).ravel()
    idx = idx[idx < n]
    try:
        res = QuantReg(y[idx], X[idx]).fit(q=tau, max_iter=MAX_ITER_BOOT)
        return float(res.params[shock_col_idx])
    except Exception:
        return np.nan


def one_rep_pair(
    seed_state: np.random.SeedSequence,
    y: np.ndarray,
    X: np.ndarray,
    block_size: int,
    taus: tuple[float, float],
    shock_col_idx: int,
) -> np.ndarray:
    """One block-bootstrap replication, shared resample across two τ values.

    Replaces:
        run_robustness_all._one_rep_pair (Test E, monotonicity)

    Returns a length-2 array (β at taus[0], β at taus[1]) fitted on the
    SAME y[idx], X[idx]. Required so that Δ = β01 - β50 is a paired
    estimate (NB08 cell 19).
    """
    import warnings as _w
    _w.filterwarnings("ignore")
    rng = np.random.default_rng(seed_state)
    n = len(y)
    if n < block_size:
        raise ValueError(
            f"Panel size n={n} smaller than block_size={block_size}. "
            f"Cannot perform block bootstrap. Check warmup truncation upstream."
        )
    n_blocks = n // block_size
    block_starts = rng.integers(0, n - block_size, size=n_blocks)
    idx = (block_starts[:, None] + np.arange(block_size)[None, :]).ravel()
    idx = idx[idx < n]
    out = np.full(2, np.nan, dtype=np.float64)
    for i, tau in enumerate(taus):
        try:
            res = QuantReg(y[idx], X[idx]).fit(q=tau, max_iter=MAX_ITER_BOOT)
            out[i] = float(res.params[shock_col_idx])
        except Exception:
            pass
    return out


def summarize(arr: np.ndarray) -> dict[str, float]:
    """Standard summary of a 1-D bootstrap array (NaNs dropped).

    Replaces:
        run_robustness_all.summarize  (canonical, handles len==0)
        run_bootstrap.summarize       (does not handle len==0; this version
                                       is a strict superset)
    """
    v = arr[~np.isnan(arr)]
    if len(v) == 0:
        return {
            "mean":         np.nan,
            "median":       np.nan,
            "ci_lo":        np.nan,
            "ci_hi":        np.nan,
            "n_success":    0.0,
            "pct_negative": np.nan,
        }
    return {
        "mean":         float(np.mean(v)),
        "median":       float(np.median(v)),
        "ci_lo":        float(np.percentile(v, 2.5)),
        "ci_hi":        float(np.percentile(v, 97.5)),
        "n_success":    float(len(v)),
        "pct_negative": float(100 * np.mean(v < 0)),
    }


def summarize_pair(
    arr: np.ndarray,
    delta_point: float,
) -> dict[str, float]:
    """Summary for Test E. `arr` is shape (n_boot, 2) with cols (β01, β50).

    Replaces:
        run_robustness_all.summarize_pair (verbatim port)

    Returns prefixed stats for β01, β50, and the *paired* difference
    Δ = β01 - β50. Δ's p-value replicates NB08 cell 19 exactly:
        centered = deltas - mean(deltas)
        p = mean(|centered| >= |delta_point|)
    """
    mask = ~np.isnan(arr).any(axis=1)
    clean = arr[mask]
    if len(clean) == 0:
        nan = float("nan")
        out = {
            f"{p}_{k}": nan
            for p in ("beta01", "beta50")
            for k in ("mean", "median", "ci_lo", "ci_hi", "pct_negative")
        }
        out.update({
            "delta_mean": nan, "delta_median": nan,
            "delta_ci_lo": nan, "delta_ci_hi": nan,
            "delta_point": delta_point, "delta_pval": nan, "n_boot": 0.0,
        })
        return out

    b01 = clean[:, 0]
    b50 = clean[:, 1]
    deltas = b01 - b50  # paired difference on same block resample

    out: dict[str, float] = {}
    for prefix, v in (("beta01", b01), ("beta50", b50)):
        out[f"{prefix}_mean"]         = float(np.mean(v))
        out[f"{prefix}_median"]       = float(np.median(v))
        out[f"{prefix}_ci_lo"]        = float(np.percentile(v, 2.5))
        out[f"{prefix}_ci_hi"]        = float(np.percentile(v, 97.5))
        out[f"{prefix}_pct_negative"] = float(100 * np.mean(v < 0))

    centered = deltas - np.mean(deltas)
    pval = float(np.mean(np.abs(centered) >= np.abs(delta_point)))
    out["delta_mean"]   = float(np.mean(deltas))
    out["delta_median"] = float(np.median(deltas))
    out["delta_ci_lo"]  = float(np.percentile(deltas, 2.5))
    out["delta_ci_hi"]  = float(np.percentile(deltas, 97.5))
    out["delta_point"]  = float(delta_point)
    out["delta_pval"]   = pval
    out["n_boot"]       = float(len(clean))
    return out


def run_parallel_boot(
    one_rep_fn: Callable,
    seeds: list[np.random.SeedSequence],
    args_tuple: tuple,
    n_jobs: int,
    batch_size: int,
    ckpt_path: Path,
    out_shape_per_rep: tuple[int, ...],
    label: str,
) -> np.ndarray:
    """Run one_rep_fn(seed, *args_tuple) over seeds with batch checkpointing.

    Replaces:
        run_robustness_all.run_parallel_boot (canonical, verbatim port)
        run_bootstrap.bootstrap_horizon      (specialised; legacy script
                                              keeps its own wrapper for
                                              traceability)

    Parameters
    ----------
    one_rep_fn : callable returning a float or 1-D ndarray
    args_tuple : tuple
        Positional args passed to one_rep_fn after `seed`.
    ckpt_path : Path
        Directory for per-batch `.npy` checkpoints.
    out_shape_per_rep : tuple
        () for scalar output, (k,) for vector output.
    label : str
        Short string included in progress prints + checkpoint filenames.

    Returns
    -------
    np.ndarray of shape (n_boot, *out_shape_per_rep), with NaN-padded
    rows for any replication where `one_rep_fn` returned NaN.
    """
    n_boot = len(seeds)
    full_shape = (n_boot,) + out_shape_per_rep
    out = np.full(full_shape, np.nan, dtype=np.float64)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    n_batches = (n_boot + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, n_boot, batch_size)):
        end = min(start + batch_size, n_boot)
        ck = ckpt_path / f"ckpt_{label}_b{batch_idx:03d}.npy"
        if ck.exists():
            chunk = np.load(ck)
            if chunk.shape[0] == end - start:
                out[start:end] = chunk
                continue

        if n_jobs == 1:
            chunk_list = [
                one_rep_fn(seeds[b], *args_tuple) for b in range(start, end)
            ]
        else:
            from joblib import Parallel, delayed
            chunk_list = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(one_rep_fn)(seeds[b], *args_tuple)
                for b in range(start, end)
            )
        chunk = np.asarray(chunk_list, dtype=np.float64)
        np.save(ck, chunk)
        out[start:end] = chunk
        print(
            f"  {label}: batch {batch_idx+1}/{n_batches}  ({end}/{n_boot})",
            flush=True,
        )
    return out
