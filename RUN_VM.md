# Running the diagnostic battery on a VM

The paper's main results (the quantile-LP IRF, the nine-quantile appendix grid,
the Bonferroni family, and robustness Tests A to G) ship pre-computed in
`data/econ/` and are reproduced by `make estimation && make robustness && make
paper` from the analysis panel (see [`README.md`](README.md), Path A). They are
**not** re-run here.

This guide covers the **extended diagnostic battery** (`make canonical`): the
auxiliary scripts under `scripts/aux/` that produce the deconstruction and
robustness exhibits (placebo, dual permutation nulls, exceedance symmetry,
BTC-outcome placebo, out-of-sample test, subsample and block-size sensitivity,
equivalence bounds). It runs in several hours on a 16-vCPU VM and is optional.

## 1. Fast vs. full

`make canonical` (= `canonical-fast`) computes every number the paper reports
from these diagnostics. `make canonical-full` additionally fills the full 0..24
horizon grid and the τ=0.50 placebo columns, for figure-grade smooth curves
only.

| | `make canonical` (fast) | `make canonical-full` |
|---|---|---|
| Core steps (1 to 13) | identical | identical |
| Permutation nulls (14, 15) | h ∈ {0,1,2,3,4,6,8,12,18,24} | h 0..24 |
| Placebo, sign-flip (16) | τ ∈ {0.01,0.05,0.95,0.99} × h ∈ {0,1,3,6,12,18,24} | 7τ × h 0..24 |
| Placebo, model-scaled (17) | fast grid × {rolling, garch} | full grid × {rolling, garch} |
| Wall time (16 vCPU) | ~6 to 10 h | ~15 to 35 h |

The two placebo DGPs are distinct: `sign_flip` is a model-free Rademacher
sign-flip that preserves the empirical volatility path (the primary placebo);
`model_scaled` permutes standardised magnitudes rescaled by a fitted σ_t
(rolling or GARCH), the volatility-model-dependence robustness. Both parallelise
internally (`--n_jobs -1`) with results bit-identical to sequential (draws are
pre-generated before dispatch).

## 2. Prerequisites

- An **x86_64 (amd64)** VM. The pinned environment was resolved and tested only
  on x86_64; arm64 / Apple Silicon is **not** tested (the lockfile contains
  `appnope==0.1.4`; see [`README.md`](README.md)).
- Python 3.12 (the lockfile was resolved on 3.12.2; any 3.12.x point release is
  fine).
- A clean virtual environment with the pinned dependencies (`arch==7.2.0` is
  pinned in both `requirements.txt` and `requirements-frozen.txt`):

  ```bash
  python3.12 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements-frozen.txt
  ```

- Verify the toolchain before launching:

  ```bash
  python -c "import sys, numpy, pandas, statsmodels, scipy, arch, joblib; \
    print(sys.version.split()[0], 'arch', arch.__version__)"
  # expect: 3.12.x  arch 7.2.0   (numpy 1.26.4 / pandas 2.2.2 / statsmodels 0.14.2)
  ```

- `make` and a POSIX shell (standard on a Linux VM).

## 3. Inputs

The battery reads the shipped analysis panel and the main estimation outputs:

```
data/econ/econ_core_full_1h.parquet                # input panel
data/econ/quantile_lp_results.csv                  # consumed by steps 2, 3, 10
data/econ/quantile_lp_results_9q.csv               # consumed by step 10
data/econ/robustness_bootstrap_nb07_spec_fast.csv  # consumed by step 3
```

Uploading the whole `data/econ/` directory is fine; run `make clean-canonical`
for a guaranteed cold start.

## 4. Run

```bash
. .venv/bin/activate
make clean-canonical                            # optional cold start
nohup make canonical > canonical.log 2>&1 &     # or: make canonical-full
tail -f canonical.log
```

Each step echoes a `[k/17]` marker (`grep '>>>' canonical.log` shows the current
step). Bootstraps and simulations are SeedSequence-keyed and pre-drawn, so
`--n_jobs -1` is reproducible across core counts and across runs. Memory is not
a constraint (the panel is ~41k rows; 32 GB is ample). Steps 14 to 17 (the two
permutation nulls and the two placebo DGPs) are the long poles and run last so
every quick artefact lands early.

## 5. Per-step runtime (16 vCPU, order of magnitude)

| Step | Script | Wall time |
|---|---|---|
| 1 | descriptive_stats | < 1 min |
| 2 | tau_justification | < 1 min |
| 3 | se_ratio_nb07 (post-processor) | < 1 min |
| 4 | exceedance (per-period, full h) | ~20 to 35 min |
| 5 | exceedance --cumulative (reduced h) | ~10 to 15 min |
| 6 | vol_response | ~10 to 20 min |
| 7 | skew_test | ~5 to 15 min |
| 8 | mde_equivalence (post-processor) | < 2 min |
| 9 | size_ratio | < 2 min |
| 10 | btc_placebo (7τ × 25h, parallel) | ~10 to 25 min |
| 11 | oos_predictive (rolling-origin, 2 benchmarks) | ~40 to 90 min |
| 12 | subsample_stability (4 subsamples) | ~20 to 45 min |
| 13 | block_sensitivity (4 blocks × 6h) | ~30 to 60 min |
| 14 | pure-null circular_shift (500 seeds, parallel) | fast ~40 to 90 min · full ~1.5 to 3 h |
| 15 | pure-null innov_shuffle (500 seeds, parallel) | fast ~40 to 90 min · full ~1.5 to 3 h |
| 16 | placebo sign_flip (500 sims, parallel) | fast ~45 min to 1.5 h · full ~4 to 8 h |
| 17 | placebo model_scaled (500 sims × 2 vol models, parallel) | fast ~1.5 to 3 h · full ~7 to 17 h |

## 6. Expected outputs (`data/econ/`)

| Step | CSV(s) | Companion meta |
|---|---|---|
| 1 | `descriptive_stats.csv` + `.json` | `descriptive_stats_meta.json` |
| 2 | `tau_choice_justification.csv` | — |
| 3 | `se_ratio_nb07.csv` | `se_ratio_nb07_meta.json` |
| 4 | `exceedance_results.csv`, `exceedance_paired.csv` | `exceedance_meta.json` |
| 5 | `exceedance_results_cumulative.csv`, `exceedance_paired_cumulative.csv` | `exceedance_meta_cumulative.json` |
| 6 | `vol_response.csv` | `vol_response_meta.json` |
| 7 | `skew_test.csv` | `skew_test_meta.json` |
| 8 | `mde_equivalence.csv` | `mde_equivalence_meta.json` |
| 9 | `size_ratio.csv` | `size_ratio_meta.json` |
| 10 | `btc_placebo_results.csv`, `btc_vs_eth_profile.csv` | `btc_placebo_meta.json` |
| 11 | `oos_predictive.csv` | `oos_predictive_meta.json` |
| 12 | `subsample_stability.csv` | `subsample_stability_meta.json` |
| 13 | `block_sensitivity.csv` | `block_sensitivity_meta.json` |
| 14 | `pure_null_circular_shift_by_horizon.csv` | `pure_null_circular_shift_meta.json` |
| 15 | `pure_null_innov_shuffle_by_horizon.csv` | `pure_null_innov_shuffle_meta.json` |
| 16 | `placebo_symmetric.csv`, `placebo_symmetric_draws.csv` | `placebo_symmetric_meta.json` |
| 17 | `placebo_symmetric_model_scaled.csv`, `placebo_symmetric_model_scaled_draws.csv` | `placebo_symmetric_model_scaled_meta.json` |

Collect the output CSVs, the meta JSONs, and `canonical.log` (a few MB total).
Checkpoint directories (`_exceedance_ckpt/`, `_vol_ckpt/`, `_subsample_ckpt/`,
`_blocksens_ckpt/`) can be deleted afterwards; `make clean-cache` keeps them out.
