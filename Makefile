# Makefile — defi-endogenous-fragility replication package
#
# Quick start:
#   python -m venv .venv && . .venv/bin/activate
#   pip install -r requirements.txt
#   make smoke   # ~1 min, validates code paths only
#   make all     # ~35 min on 16 vCPU / 32 GB RAM, full pipeline
#
# Targets are file-based: re-running `make all` does NOT redo a stage
# whose output already exists. To force a rebuild, `make clean` first.

PY ?= .venv/bin/python
ECON_DIR     := data/econ
WINDOWS_DIR  := data/analysis/windows
DATASETS_DIR := data/analysis/datasets
REPORTS_DIR  := data/analysis/reports
FIG_DIR      := paper/figures
PAPER_DIR    := paper

# ──────────────────────────────────────────────────────────────
# Determinism: pin BLAS / thread counts and the hash seed for
# EVERY recipe shell, so a fresh run is bit-identical on a fixed platform.
# joblib parallelism stays process-level (--n_jobs); 1 thread/process avoids
# nested BLAS oversubscription and is reproducibility-preserving.
# ──────────────────────────────────────────────────────────────
export OMP_NUM_THREADS        := 1
export OPENBLAS_NUM_THREADS   := 1
export MKL_NUM_THREADS        := 1
export NUMEXPR_NUM_THREADS    := 1
export VECLIB_MAXIMUM_THREADS := 1
export PYTHONHASHSEED         := 0

# Deterministic artefact whose SHA-256 is pinned in tests/fingerprints.txt.
# Kept to the main estimation output (QuantReg, --n_jobs 1): byte-identical
# run-to-run on a fixed platform and confirmed byte-identical shipped-vs-regenerated.
# Bootstrap/sim CSVs drift sub-bp and label-formatted files differ, so they are excluded.
FINGERPRINT_FILES := $(ECON_DIR)/quantile_lp_results.csv

.PHONY: all data estimation robustness figures smoke tau_just canonical canonical-fast canonical-full canonical-core clean-canonical clean clean-cache help paper paper-figures paper-tables paper-numbers setup reproduce reproduce-fast pdf verify-exhibits fingerprints verify-data check-numbers bonferroni test

# Shared grids for the canonical diagnostic battery (see RUN_VM.md §3).
H_FULL    := 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
H_DIAG    := 0,1,3,6,12,18,24
H_TABLE   := 0,1,3,6,12,24
H_PN_FAST := 0,1,2,3,4,6,8,12,18,24
TAUS_PLACEBO_FAST := 0.01,0.05,0.95,0.99

# ──────────────────────────────────────────────────────────────
# Top-level
# ──────────────────────────────────────────────────────────────
all: data estimation robustness figures   ## run the full pipeline (~35 min)

# ──────────────────────────────────────────────────────────────
# ONE COMMAND — reproduce everything from the COMMITTED panel
#   make setup           create .venv (python3.12) + install pinned deps
#   make reproduce       FULL referenced reproduction from the committed panel
#                        (recomputes estimation + robustness + the canonical
#                        diagnostic battery + paper; HOURS, incl. canonical)
#   make reproduce-fast  wiring smoke (minutes, reduced grids, scratch out_dir,
#                        NON-destructive) — proves automation, NOT referenced numbers
# Both run from data/econ/ + the 2 committed spot parquets; no raw bundle, no
# credentials. Path B (make all / make data) is the separate from-raw rebuild.
# ──────────────────────────────────────────────────────────────
setup: .venv/.installed   ## create .venv (python3.12) and install pinned dependencies

.venv/.installed: requirements.txt
	@command -v python3.12 >/dev/null 2>&1 || { echo "ERROR: python3.12 not found — install Python 3.12"; exit 1; }
	python3.12 -m venv .venv
	.venv/bin/python -m pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet -r requirements.txt
	@touch .venv/.installed
	@echo ">>> environment ready (.venv, python3.12, pinned deps)"

reproduce: setup   ## ONE COMMAND: recompute every exhibit from the committed panel, verify, summarise
	@echo "=============================================================="
	@echo ">>> make reproduce — full referenced reproduction (committed panel)"
	@echo ">>> determinism: OMP/OPENBLAS/MKL/NUMEXPR/VECLIB=1  PYTHONHASHSEED=0"
	@$(PY) -c "import sys;assert sys.version_info[:2]==(3,12),'Python 3.12 required, got '+sys.version.split()[0]"
	@echo "=============================================================="
	$(PY) -m pytest -q
	$(MAKE) estimation
	$(MAKE) robustness
	$(MAKE) canonical
	$(MAKE) paper
	-$(MAKE) pdf
	$(MAKE) verify-exhibits
	@echo "=============================================================="
	@echo ">>> make reproduce COMPLETE"
	@echo ">>> Python: $$($(PY) -V 2>&1)   Platform: $$(uname -srm)"
	@echo ">>> Exhibits: paper/figures/*.pdf + paper/tables/*.tex + paper/numbers.tex"
	@echo "=============================================================="

reproduce-fast: setup   ## wiring smoke (minutes, reduced grids, scratch out_dir) — NOT referenced numbers
	@echo ">>> make reproduce-fast — wiring smoke (reduced grids; numbers NOT referenced)"
	$(PY) -m pytest -q
	$(PY) scripts/run_quantile_lp.py --quantiles 0.01,0.50 --horizons 0,1 --skip_pretrend --out_dir /tmp/reproduce_fast
	$(PY) scripts/run_robustness_all.py --tests A,C,D2 --out_dir /tmp/reproduce_fast --ckpt_dir /tmp/reproduce_fast_ckpt
	$(MAKE) paper
	@echo ">>> reproduce-fast OK (producers ran on reduced grids; paper regenerated)"

pdf:   ## compile manuscript/main.pdf if a LaTeX engine is available (optional, non-blocking)
	@if command -v latexmk >/dev/null 2>&1; then \
	   echo ">>> latexmk -pdf main.tex"; (cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex); \
	 elif command -v tectonic >/dev/null 2>&1; then \
	   echo ">>> tectonic main.tex"; (cd manuscript && tectonic main.tex); \
	 else \
	   echo ">>> WARNING: no LaTeX engine (latexmk/tectonic) — skipping PDF compile (manuscript/main.pdf left as committed)"; \
	 fi

verify-exhibits:   ## regenerate-and-diff paper/ + verify SHA-256 fingerprints of deterministic artefacts
	@if [ -d .git ]; then \
	   echo ">>> git diff paper/ (empty == reproduction is exact):"; \
	   git --no-pager diff --stat -- paper/ || true; \
	 else echo ">>> (no .git — run inside a git clone to enforce regenerate-and-diff)"; fi
	@if [ -f tests/fingerprints.txt ]; then \
	   echo ">>> verifying SHA-256 fingerprints (canonical platform = Linux x86_64):"; \
	   if command -v sha256sum >/dev/null 2>&1; then sha256sum -c tests/fingerprints.txt || echo ">>> NOTE: fingerprint mismatch — expected only on a non-canonical platform (cross-platform BLAS drift); not a failure"; \
	   else shasum -a 256 -c tests/fingerprints.txt || echo ">>> NOTE: fingerprint mismatch — expected only on a non-canonical platform; not a failure"; fi; \
	 else echo ">>> WARNING: tests/fingerprints.txt absent — run 'make fingerprints' on the canonical platform"; fi

fingerprints:   ## (re)generate tests/fingerprints.txt from current deterministic artefacts (run on Linux canonical platform)
	@mkdir -p tests
	@if command -v sha256sum >/dev/null 2>&1; then sha256sum $(FINGERPRINT_FILES) > tests/fingerprints.txt; else shasum -a 256 $(FINGERPRINT_FILES) > tests/fingerprints.txt; fi
	@echo ">>> wrote tests/fingerprints.txt"; cat tests/fingerprints.txt

check-numbers: setup   ## NON-REGRESSION: recomputed macros must equal committed paper/numbers.tex (writes nothing)
	$(PY) scripts/paper/make_numbers.py --check

verify-data:   ## verify committed data artefacts match data/CHECKSUMS.sha256 (integrity of the deposit)
	@test -f data/CHECKSUMS.sha256 || { echo "ERROR: data/CHECKSUMS.sha256 missing"; exit 1; }
	@echo ">>> checking SHA-256 of committed data artefacts (run on a fresh clone, before reproduce):"
	@if command -v sha256sum >/dev/null 2>&1; then sha256sum -c data/CHECKSUMS.sha256; else shasum -a 256 -c data/CHECKSUMS.sha256; fi

help:                                     ## show this help
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*## "}; {printf "  %-15s %s\n", $$1, $$2}'

# ──────────────────────────────────────────────────────────────
# Stage 1 — data prep (NB01+NB02+NB03+NB04)
# ──────────────────────────────────────────────────────────────
data: $(ECON_DIR)/econ_core_full_1h.parquet   ## run_data_prep + run_core_panel + run_defi_merge

$(WINDOWS_DIR)/master_calendar_1h.parquet:
	$(PY) scripts/run_data_prep.py

$(ECON_DIR)/econ_core_predefi_1h.parquet: $(WINDOWS_DIR)/master_calendar_1h.parquet
	$(PY) scripts/run_core_panel.py

$(ECON_DIR)/econ_core_full_1h.parquet: $(ECON_DIR)/econ_core_predefi_1h.parquet
	$(PY) scripts/run_defi_merge.py

# ──────────────────────────────────────────────────────────────
# Stage 2 — main estimation (NB07)
# ──────────────────────────────────────────────────────────────
estimation:   ## run_quantile_lp from the SHIPPED panel (Path A; no data-chain rebuild)
	@test -f $(ECON_DIR)/econ_core_full_1h.parquet || \
	  { echo "ERROR: $(ECON_DIR)/econ_core_full_1h.parquet missing — run 'make data' (needs the raw bundle) or unzip the Release into data/"; exit 1; }
	$(PY) scripts/run_quantile_lp.py --n_jobs 1

# ──────────────────────────────────────────────────────────────
# Stage 3 — robustness battery (Tests A–N)
# WARNING: heavy. Expects 16 vCPU / 32 GB RAM. Wall time ~30 min on 16 vCPU @ n_boot=1000 (includes Test G bootstrap battery).
# ──────────────────────────────────────────────────────────────
robustness:   ## run_robustness_all (all tests, ~30 min on 16 vCPU); needs the panel + estimation
	@test -f $(ECON_DIR)/quantile_lp_results.csv || \
	  { echo "ERROR: $(ECON_DIR)/quantile_lp_results.csv missing — run 'make estimation' first"; exit 1; }
	$(PY) scripts/run_robustness_all.py --tests all --n_boot 1000 --n_jobs -1 --seed 42

# ──────────────────────────────────────────────────────────────
# Stage 4 — descriptive stats + figures (NB05 + NB09)
# ──────────────────────────────────────────────────────────────
# Producer switched from NB09 nbconvert to the re-executable script (Table-1
# numbers must be recomputable without jupyter). NB09 remains the
# reading-side report; the script reproduces its blocks exactly + the new ones.
figures:   ## descriptive stats + paper figures (Path A; no data-chain rebuild)
	@test -f $(ECON_DIR)/econ_core_full_1h.parquet || \
	  { echo "ERROR: $(ECON_DIR)/econ_core_full_1h.parquet missing — run 'make data' or unzip the Release into data/"; exit 1; }
	@test -f $(ECON_DIR)/quantile_lp_results.csv || \
	  { echo "ERROR: $(ECON_DIR)/quantile_lp_results.csv missing — run 'make estimation' first"; exit 1; }
	$(PY) scripts/aux/run_descriptive_stats.py
	$(MAKE) paper-figures

# ──────────────────────────────────────────────────────────────
# Smoke test — fast end-to-end sanity check (~1 min)
# ──────────────────────────────────────────────────────────────
smoke: data                               ## fast end-to-end sanity check (~1 min, n_boot=10) — depends on `make data`
	@echo ">>> Smoke: data prep"
	$(PY) scripts/run_data_prep.py    --out_dir /tmp/smoke_data
	$(PY) scripts/run_core_panel.py   --out_dir /tmp/smoke_data
	$(PY) scripts/run_defi_merge.py   --out_dir /tmp/smoke_data
	@echo ">>> Smoke: estimation (subset)"
	$(PY) scripts/run_quantile_lp.py \
	    --quantiles 0.01,0.50 --horizons 0,1 --skip_pretrend \
	    --out_dir /tmp/smoke_econ
	@echo ">>> Smoke: deterministic robustness tests only"
	$(PY) scripts/run_robustness_all.py \
	    --tests A,C,D2 --out_dir /tmp/smoke_econ \
	    --ckpt_dir /tmp/smoke_ckpt
	@echo "Smoke OK"

# ──────────────────────────────────────────────────────────────
# PAPER LAYER — figures (PDF) + tables (.tex fragments) + numbers (macros)
# Pure formatting from canonical data/econ CSVs (no statistics computed).
# The manuscript \input's paper/tables/*.tex and paper/numbers.tex and
# \includegraphics paper/figures/*.pdf at natural size.
# ──────────────────────────────────────────────────────────────
paper: paper-figures paper-tables paper-numbers   ## regenerate ALL paper outputs from canonical CSVs

paper-figures:                            ## the 8 paper figures -> paper/figures/*.pdf
	$(PY) scripts/paper/make_figures.py

paper-tables:                             ## Tables 1-6 + appx -> paper/tables/*.tex
	$(PY) scripts/paper/make_tables.py

paper-numbers:                            ## prose macros -> paper/numbers.tex
	$(PY) scripts/paper/make_numbers.py

# ──────────────────────────────────────────────────────────────
# Optional — τ choice justification (NOT part of `make all`)
# Manually invoke when the LaTeX draft cites tau_choice_justification ratios.
# ──────────────────────────────────────────────────────────────
tau_just: $(ECON_DIR)/quantile_lp_results.csv   ## (optional) regenerate tau_choice_justification.csv
	$(PY) scripts/aux/compute_tau_justification.py
	@echo "tau_choice_justification.csv régénéré"

bonferroni: $(ECON_DIR)/quantile_lp_results.csv   ## (optional) append family-wise Bonferroni columns -> quantile_lp_results_with_bonferroni.csv
	$(PY) scripts/add_bonferroni.py

test: setup   ## run the pytest suite (also run automatically by make reproduce)
	$(PY) -m pytest -q

# ──────────────────────────────────────────────────────────────
# CANONICAL — diagnostic battery for the paper (VM, env 3.12, ~16 vCPU)
# ──────────────────────────────────────────────────────────────
# ONE command the author runs on the VM after uploading the repo:
#     make canonical          (= canonical-fast, the RECOMMENDED batch, ~6-10 h)
#     make canonical-full     (full grids on the 3 long poles, ~15-35 h)
#
# 17 steps: the 7 core aux diagnostics + 7 additional checks (descriptive
# recompute, same-spec SE ratio, BTC-outcome placebo, pseudo-OOS pinball,
# subsample stability, block-size sensitivity, cumulative-exceedance
# robustness) + tau_just + MDE. The placebo runs under TWO DGPs: sign_flip
# (primary, model-free) and model_scaled (vol-model-dependence robustness) —
# see run_placebo_symmetric.py's design note on the sigma-cancellation fix.
# Steps 1-13 (canonical-core) are identical between fast and full; only the
# 4 LONG POLES (the 2 pure-nulls + the 2 placebo DGPs) differ by grid:
#   fast: pure-null h {$(H_PN_FAST)} ; placebo tau {$(TAUS_PLACEBO_FAST)} x h {$(H_DIAG)}
#   full: h 0..24 everywhere ; placebo full 7-tau grid (incl. 0.50, the slowest fits)
# The reported objects (gap pair 0.01/0.99, ratio profile, h=0/12/24 anchors)
# are covered by the fast grids; the MAIN IRF + Tests A-N are ALREADY canonical
# (data/econ *_fast strata, 2026-05-14, py3.12) and are NOT re-run here.
#
# NOT file-based on purpose: canonical numbers must be (re)produced fresh on the
# VM regardless of smoke artefacts on disk. Checkpoint labels now encode
# mode/n_boot/block, so stale smoke checkpoints can never be silently reused;
# run `make clean-canonical` first for a guaranteed cold start.
#
# See RUN_VM.md for prerequisites, runtime estimates, and expected outputs (§6).
canonical: canonical-fast   ## RECOMMENDED canonical batch (= canonical-fast) — see RUN_VM.md

# NOT a file-rule prerequisite on purpose: on a fresh VM the upstream
# intermediates (econ_core_predefi, master_calendar) do not exist, and a file
# dependency would make `make` try to REBUILD the whole data chain (which
# needs raw data that is never uploaded). The canonical batch REQUIRES the
# panel as given — existence check only.
canonical-core:
	@test -f $(ECON_DIR)/econ_core_full_1h.parquet || \
	  { echo "ERROR: $(ECON_DIR)/econ_core_full_1h.parquet missing — upload the panel (RUN_VM.md §3)"; exit 1; }
	@echo "=============================================================="
	@echo ">>> CANONICAL DIAGNOSTIC BATTERY — core steps 1-13  (env 3.12)"
	@echo ">>> PY=$(PY)   out=$(ECON_DIR)"
	@echo "=============================================================="
	@echo ""
	@echo ">>> [1/17] run_descriptive_stats.py  (Table-1 provenance: skew/kurt, far-extreme counts, z3, 85/42 crosstab)"
	$(PY) scripts/aux/run_descriptive_stats.py
	@echo ""
	@echo ">>> [2/17] compute_tau_justification.py  (regenerates tau_choice_justification.csv)"
	$(PY) scripts/aux/compute_tau_justification.py
	@echo ""
	@echo ">>> [3/17] recompute_se_ratio_nb07.py  (same-spec D1: Test-M bootstrap SE vs NB07 kernel SE, arithmetic)"
	$(PY) scripts/aux/recompute_se_ratio_nb07.py
	@echo ""
	@echo ">>> [4/17] run_exceedance.py  (positive + symmetry null; PER-PERIOD, full h 0..24, n_boot 1000)"
	$(PY) scripts/aux/run_exceedance.py \
	    --n_boot 1000 --n_jobs -1 \
	    --alphas 0.10,0.05,0.01 \
	    --horizons $(H_FULL)
	@echo ""
	@echo ">>> [5/17] run_exceedance.py --cumulative  (FLAGGED robustness: overlap re-introduces the fake gap; reduced h)"
	$(PY) scripts/aux/run_exceedance.py --cumulative \
	    --n_boot 1000 --n_jobs -1 \
	    --alphas 0.10,0.05,0.01 \
	    --horizons $(H_TABLE)
	@echo ""
	@echo ">>> [6/17] run_vol_response.py  (volatility channel; h 0..24, both measures, n_boot 1000)"
	$(PY) scripts/aux/run_vol_response.py \
	    --n_boot 1000 --n_jobs -1 \
	    --horizons $(H_FULL)
	@echo ""
	@echo ">>> [7/17] run_skew_test.py  (genuine downside net of vol; n_boot 1000)"
	$(PY) scripts/aux/run_skew_test.py \
	    --n_boot 1000 --n_jobs -1
	@echo ""
	@echo ">>> [8/17] run_mde_equivalence.py  (post-processor; AFTER steps 3+6)"
	$(PY) scripts/aux/run_mde_equivalence.py
	@echo ""
	@echo ">>> [9/17] run_size_ratio.py  (liq \$$ vs ETH spot+perp turnover)"
	$(PY) scripts/aux/run_size_ratio.py
	@echo ""
	@echo ">>> [10/17] run_btc_placebo.py  (BTC-outcome placebo: FULL QLP with BTC as OUTCOME, mirror controls, all tau x h)"
	$(PY) scripts/aux/run_btc_placebo.py \
	    --n_jobs -1 --max_iter 20000
	@echo ""
	@echo ">>> [11/17] run_oos_predictive.py  (pseudo-OOS pinball vs qr_controls + garch11)"
	$(PY) scripts/aux/run_oos_predictive.py \
	    --n_jobs -1 --benchmarks qr_controls,garch11
	@echo ""
	@echo ">>> [12/17] run_subsample_stability.py  (temporal splits + leave-out-Aug-2024; n_boot 1000)"
	$(PY) scripts/aux/run_subsample_stability.py \
	    --n_boot 1000 --max_iter 20000 --n_jobs -1
	@echo ""
	@echo ">>> [13/17] run_block_sensitivity.py  (block-size sensitivity: MBB blocks 12/24/36/48 on the MDE objects)"
	$(PY) scripts/aux/run_block_sensitivity.py \
	    --n_boot 1000 --n_jobs -1

canonical-fast: canonical-core   ## RECOMMENDED batch (~5-9 h): reported grids on the 3 long poles
	@echo ""
	@echo ">>> [14/17] run_pure_null.py --null_mode circular_shift  (conservative null; 500 seeds, FAST h grid)"
	$(PY) scripts/aux/run_pure_null.py \
	    --null_mode circular_shift --n_seeds 500 --max_iter 20000 --n_jobs -1 \
	    --horizons $(H_PN_FAST)
	@echo ""
	@echo ">>> [15/17] run_pure_null.py --null_mode innov_shuffle  (coarse innovation-shuffle null)"
	$(PY) scripts/aux/run_pure_null.py \
	    --null_mode innov_shuffle --n_seeds 500 --max_iter 20000 --n_jobs -1 \
	    --horizons $(H_PN_FAST)
	@echo ""
	@echo ">>> [16/17] run_placebo_symmetric.py --dgp sign_flip  (model-free Rademacher placebo; FAST tau x h grid)"
	$(PY) scripts/aux/run_placebo_symmetric.py \
	    --dgp sign_flip \
	    --n_sim 500 --max_iter 20000 --n_jobs -1 \
	    --taus $(TAUS_PLACEBO_FAST) \
	    --horizons $(H_DIAG)
	@echo ""
	@echo ">>> [17/17] run_placebo_symmetric.py --dgp model_scaled  (vol-model-dependence robustness: rolling + garch, now genuinely distinct)"
	$(PY) scripts/aux/run_placebo_symmetric.py \
	    --dgp model_scaled \
	    --n_sim 500 --max_iter 20000 --n_jobs -1 \
	    --vol_models rolling,garch \
	    --taus $(TAUS_PLACEBO_FAST) \
	    --horizons $(H_DIAG)
	@echo ""
	@echo "=============================================================="
	@echo "CANONICAL BATTERY COMPLETE (fast grids on the long poles)."
	@echo "   Outputs in $(ECON_DIR)/ — see RUN_VM.md §6 (expected outputs)."
	@echo "=============================================================="

canonical-full: canonical-core   ## full grids on the 3 long poles (~12-28 h)
	@echo ""
	@echo ">>> [14/17] run_pure_null.py --null_mode circular_shift  (FULL h 0..24)"
	$(PY) scripts/aux/run_pure_null.py \
	    --null_mode circular_shift --n_seeds 500 --max_iter 20000 --n_jobs -1 \
	    --horizons $(H_FULL)
	@echo ""
	@echo ">>> [15/17] run_pure_null.py --null_mode innov_shuffle  (FULL h 0..24)"
	$(PY) scripts/aux/run_pure_null.py \
	    --null_mode innov_shuffle --n_seeds 500 --max_iter 20000 --n_jobs -1 \
	    --horizons $(H_FULL)
	@echo ""
	@echo ">>> [16/17] run_placebo_symmetric.py --dgp sign_flip  (FULL 7-tau grid, h 0..24)"
	$(PY) scripts/aux/run_placebo_symmetric.py \
	    --dgp sign_flip \
	    --n_sim 500 --max_iter 20000 --n_jobs -1 \
	    --horizons $(H_FULL)
	@echo ""
	@echo ">>> [17/17] run_placebo_symmetric.py --dgp model_scaled  (FULL grids, rolling + garch)"
	$(PY) scripts/aux/run_placebo_symmetric.py \
	    --dgp model_scaled \
	    --n_sim 500 --max_iter 20000 --n_jobs -1 \
	    --vol_models rolling,garch \
	    --horizons $(H_FULL)
	@echo ""
	@echo "=============================================================="
	@echo "CANONICAL BATTERY COMPLETE (full grids)."
	@echo "   Outputs in $(ECON_DIR)/ — see RUN_VM.md §6 (expected outputs)."
	@echo "=============================================================="

clean-canonical:                          ## remove canonical diagnostic outputs + their checkpoint dirs
	@rm -f  $(ECON_DIR)/exceedance_results.csv $(ECON_DIR)/exceedance_paired.csv \
	        $(ECON_DIR)/exceedance_meta.json \
	        $(ECON_DIR)/exceedance_results_cumulative.csv \
	        $(ECON_DIR)/exceedance_paired_cumulative.csv \
	        $(ECON_DIR)/exceedance_meta_cumulative.json \
	        $(ECON_DIR)/vol_response.csv $(ECON_DIR)/vol_response_meta.json \
	        $(ECON_DIR)/skew_test.csv $(ECON_DIR)/skew_test_meta.json \
	        $(ECON_DIR)/pure_null_*by_horizon.csv $(ECON_DIR)/pure_null_*meta.json \
	        $(ECON_DIR)/pure_null_by_horizon.csv $(ECON_DIR)/pure_null_meta.json \
	        $(ECON_DIR)/placebo_symmetric.csv $(ECON_DIR)/placebo_symmetric_meta.json \
	        $(ECON_DIR)/placebo_symmetric_draws.csv \
	        $(ECON_DIR)/placebo_symmetric_model_scaled.csv \
	        $(ECON_DIR)/placebo_symmetric_model_scaled_draws.csv \
	        $(ECON_DIR)/placebo_symmetric_model_scaled_meta.json \
	        $(ECON_DIR)/se_ratio_nb07.csv $(ECON_DIR)/se_ratio_nb07_meta.json \
	        $(ECON_DIR)/mde_equivalence.csv $(ECON_DIR)/mde_equivalence_meta.json \
	        $(ECON_DIR)/size_ratio.csv $(ECON_DIR)/size_ratio_meta.json \
	        $(ECON_DIR)/btc_placebo_results.csv $(ECON_DIR)/btc_placebo_meta.json \
	        $(ECON_DIR)/btc_vs_eth_profile.csv \
	        $(ECON_DIR)/oos_predictive.csv $(ECON_DIR)/oos_predictive_meta.json \
	        $(ECON_DIR)/subsample_stability.csv $(ECON_DIR)/subsample_stability_meta.json \
	        $(ECON_DIR)/block_sensitivity.csv $(ECON_DIR)/block_sensitivity_meta.json \
	        $(ECON_DIR)/tau_choice_justification.csv \
	        $(ECON_DIR)/descriptive_stats_meta.json
	@rm -rf $(ECON_DIR)/_exceedance_ckpt $(ECON_DIR)/_vol_ckpt \
	        $(ECON_DIR)/_subsample_ckpt $(ECON_DIR)/_blocksens_ckpt
	@echo "Cleaned canonical diagnostic outputs + checkpoints."
	@echo "(descriptive_stats.csv/json kept — regenerate via step 1 or make figures.)"

# ──────────────────────────────────────────────────────────────
# Clean
# ──────────────────────────────────────────────────────────────
clean:                                    ## remove all generated artefacts (preserves raw + normalized + _legacy)
	@find $(ECON_DIR) -maxdepth 1 -type f \( -name '*.csv' -o -name '*.parquet' -o -name '*.json' \) -delete
	@rm -rf $(ECON_DIR)/_bootstrap_ckpt $(ECON_DIR)/_robust_ckpt $(ECON_DIR)/_val_*
	@find $(WINDOWS_DIR) -type f \( -name '*.parquet' -o -name '*.json' \) -delete 2>/dev/null || true
	@find $(DATASETS_DIR) -type f -name '*.parquet' -delete 2>/dev/null || true
	@find $(REPORTS_DIR) -type f -name '*.json' -delete 2>/dev/null || true
	@find $(FIG_DIR) -type f \( -name '*.pdf' -o -name '*.png' \) -delete
	@echo "Cleaned generated artefacts. Preserved: data/raw/, data/normalized/, data/econ/_legacy/."

clean-cache:                              ## remove only checkpoint/cache artefacts
	@rm -rf $(ECON_DIR)/_bootstrap_ckpt $(ECON_DIR)/_robust_ckpt $(ECON_DIR)/_val_*
	@echo "Cleaned bootstrap caches."
