"""Shared primitives for the defi-endogenous-fragility replication package.

This package contains the functions and constants shared across
run_data_prep.py, run_core_panel.py, run_defi_merge.py,
run_quantile_lp.py, and run_robustness_all.py.

Modules
-------
- src.io          : parquet loaders + spot helpers
- src.estimation  : df_est builders (orth + raw), prepare_arrays,
                    QR_KERNEL_KWARGS, CONTROLS_BASELINE, BOOT_REGRESSORS
- src.bootstrap   : block-bootstrap primitives, seed sequences,
                    summary helpers, parallel driver
"""
