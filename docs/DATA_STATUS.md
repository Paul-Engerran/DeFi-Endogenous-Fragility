# Data status and codebook

**Paper:** *When Volatility Masquerades as Fragility: DeFi Liquidations, Quantile Local Projections, and the Tails of Ethereum Returns*
**Author:** Paul Engerran
**Repository:** <https://github.com/Paul-Engerran/DeFi-Endogenous-Fragility>

This document gives the per-source data-availability statement, the pipeline
data flow, and the column-level codebook for the analysis panel. It complements
the top-level [`README.md`](../README.md) (overview and reproduction steps) and
the methodological appendix [`DUNE_EXTRACTION_BRIEF.md`](./DUNE_EXTRACTION_BRIEF.md)
(filters and exclusions for the DeFi feed).

---

## Data availability statement (AEA / Vilhuber 2020 format)

The data come from five publicly accessible sources:

1. **Bybit ETH/USDT perpetual derivatives** (OHLCV, open interest, funding
   rate; hourly; 2021-03-15 to 2025-11-30). Public REST API
   `https://api.bybit.com`, no API key. Terms of use:
   <https://www.bybit.com/en/help-center/article/Terms-of-Service>. Public
   market data carries no documented redistribution restriction.
2. **Binance ETH/USDT futures** (OHLCV, hourly; cross-venue diagnostic only).
   Public REST API `https://fapi.binance.com`, no API key. Terms of use:
   <https://www.binance.com/en/terms>.
3. **Coinbase ETH spot** (hourly; data-quality cross-check only). Public REST
   API `https://api.exchange.coinbase.com`, no API key. Terms of use:
   <https://www.coinbase.com/legal/user_agreement>.
4. **CCData (CryptoCompare) BTC, ETH, XRP, DOGE spot CCCAGG** (hourly). API
   endpoint `https://min-api.cryptocompare.com`, **free-tier API key required**
   (environment variable `CCDATA_API_KEY`). Terms of use: see the API terms
   on the provider's site <https://developers.ccdata.io/> (CCData, formerly
   CryptoCompare; the legacy `cryptocompare.com/api-terms-of-use` deep-link is
   retired). Verify the live terms page at acquisition time.
5. **Dune Analytics DeFi liquidations** (event-level, aggregated to hourly
   buckets). Open-source Spellbook tables `lending.borrow` and `lending.supply`.
   Query ID `6912877`, re-executable via the Dune API or web interface; SQL
   reproduced in [`dune_queries/liquidations_6912877.sql`](../dune_queries/liquidations_6912877.sql).

All five sources were publicly accessible at no monetary cost as of the last
extraction (March 2026). The package redistributes only **derived hourly
aggregates** (the analysis panels and result artefacts under `data/econ/`),
never raw venue payloads. None of the providers imposed a documented
redistribution restriction on those derived aggregates at the time of
extraction. A reviewer can replicate access to the original sources at no cost
by following §6; the only credential is a free CCData key.

**Code availability.** All analysis code is released under the MIT License
([`LICENSE`](../LICENSE)). The package bundles the source code, the pinned
dependency list (`requirements.txt`), the full transitive closure
(`requirements-frozen.txt`), reproduction instructions (`README.md`), and the
unit tests (`tests/`).

**Computation.** From the committed analysis panel (plus the two committed spot
placebo parquets used by robustness Test A), estimation plus robustness
plus paper-exhibit regeneration runs in about 40 minutes on 16 vCPU / 32 GB RAM
(Path A, §6). A full rebuild from raw inputs is longer and optional (Path B).

**Random seed.** Bootstrap-based statistics use `--seed 42` with a four-level
SeedSequence namespace `[base_seed, test_id, h, b]` for test-level independence.

**Replication contact.** Open an issue on the GitHub repository.

---

## 1. Source data overview

| Source | Asset / variable | Frequency | Window | Access |
|---|---|---|---|---|
| Bybit REST API | ETHUSDT perp klines, OI, funding | 1-hour | 2021-03-15 to 2025-11-30 | Public, no API key |
| Binance Futures API | ETHUSDT perp (diagnostic only) | 1-hour | 2021-01-01 to 2025-11-30 | Public, no API key |
| Coinbase API | ETH spot benchmark | 1-hour | 2021-03-15 to 2025-11-30 | Public, no API key |
| CCData (CryptoCompare) | BTC, ETH, XRP, DOGE spot CCCAGG | 1-hour | 2021-03-15 to 2025-11-30 | Free API key |
| Dune Analytics | DeFi liquidations (cross-EVM) | event to hourly | 2021-03-15 to 2025-11-30 | Free Dune account; query `6912877` |

Period covered by the paper: **2021-03-15 00:00 UTC to 2025-12-01 00:00 UTC
(exclusive)**, hourly grid, **41,328 hours**. Timestamps label the *start* of
the bucket; intervals are `[start, end_excl)`. The canonical declaration is in
`config.py`.

> **What ships, and where.** The committed package contains the analysis-ready
> panels and result artefacts under `data/econ/`, PLUS two normalized spot
> parquets read directly by the analysis layer:
> `data/normalized/spot/xrp_ccdata_1h.parquet` and `…/doge_ccdata_1h.parquet`
> (cross-asset placebo, robustness Test A; ~2.4 MB). Every OTHER raw and
> normalized input is **not committed**; those are distributed as the
> **GitHub Release v2.0.0** asset (needed only for a from-raw rebuild, §6 Path B).
> Except where a row is explicitly marked "committed", the "Canonical file
> (Release)" rows below refer to that bundle.

### 1.1 Bybit, ETH perpetual derivatives (primary venue)

| Field | Value |
|---|---|
| Variables | OHLCV (klines), open interest, funding rate |
| Endpoint | `https://api.bybit.com` (linear perpetuals, symbol `ETHUSDT`) |
| API key | None |
| Re-acquisition | Documented Bybit API procedure in `scripts/data_download/README.md` |
| Canonical file (Release) | `data/normalized/cex/bybit/{klines,funding,open_interest}_1h.parquet` |
| Rows | 41,328 per file, 0 missing on key columns |
| Access date | March 2026 |

Bybit is the primary venue based on the cross-venue diagnostic in
`run_data_prep.py` (Stage 2): return correlation with Binance ≈ 0.9989, mean
absolute spread ≈ 1.98 bps.

### 1.2 Binance, ETH futures (diagnostic only)

| Field | Value |
|---|---|
| Variables | OHLCV |
| Endpoint | `https://fapi.binance.com` (symbol `ETHUSDT`) |
| API key | None |
| Canonical file (Release) | `data/normalized/cex/binance/binance_futures_ethusdt_1h_normalized.parquet` |
| Role | Used only by `run_data_prep.py` Stage 2 for the cross-venue spread / correlation check. **Not used in any econometric specification.** |
| Window | 2021-01-01 to 2025-11-30 (broader than the paper window, sliced down) |

### 1.3 CCData (CryptoCompare), spot benchmarks and placebo assets

| Field | Value |
|---|---|
| Variables | Close price (CCCAGG aggregate) |
| Assets | BTC, ETH (benchmarks); XRP, DOGE (placebo, robustness Test A) |
| Endpoint | `https://min-api.cryptocompare.com/data/v2/histohour` |
| API key | **Free tier required**, register at <https://www.cryptocompare.com> |
| Environment variable | `CCDATA_API_KEY`, read via `os.environ.get("CCDATA_API_KEY", "")`. Required only to re-download from scratch. |
| Re-acquisition | Documented CCData API procedure in `scripts/data_download/README.md` |
| Canonical file, committed | `data/normalized/spot/{xrp,doge}_ccdata_1h.parquet` (read directly by run_robustness_all.py Test A) |
| Canonical file (Release) | `data/normalized/spot/{btc,eth}_ccdata_1h.parquet` (consumed upstream by run_core_panel.py; already embedded in the committed panel) |
| Rows | 41,328 per file, 0 missing |

### 1.4 Coinbase, ETH spot (data-quality audit only)

| Field | Value |
|---|---|
| Endpoint | `https://api.exchange.coinbase.com` |
| API key | None |
| Canonical file (Release) | `data/normalized/benchmarks/coinbase/candles_repaired.parquet` |
| Role | Data-quality cross-check; not used in any econometric specification |

### 1.5 Dune Analytics, DeFi liquidations

| Field | Value |
|---|---|
| Variables | `total_debt_repaid_usd`, `total_collateral_seized_usd`, `n_liquidations` |
| Frequency | 1-hour buckets (hours with zero liquidations are absent from the raw CSV; filled with 0 in `run_defi_merge.py`) |
| Engine | Trino (DuneSQL) |
| Spellbook tables | `lending.borrow`, `lending.supply` |
| Query ID | `6912877`, self-contained, re-executable from any Dune account (no external materialized-view dependency) |
| Re-execution | `dune.get_latest_result(6912877)` via the Dune API, or directly in the Dune web interface |
| SQL source | `dune_queries/liquidations_6912877.sql` |
| Canonical file (Release) | `data/raw/defi/defi_liquidations_1h_clean.csv` (10,976 hours with at least one qualifying liquidation) |
| Monetary cost | Free |
| Access date | March 2026 |
| Methodological appendix | `DUNE_EXTRACTION_BRIEF.md` (filters, exclusions, variable definitions) |

The query covers all EVM chains and lending protocols indexed in Dune's
Spellbook at extraction time, with a collateral-side filter restricting to
ETH-like assets (`WETH`, `ETH`, `stETH`, `wstETH`, `rETH`, `cbETH`, `sfrxETH`,
`ETHx`). See `DUNE_EXTRACTION_BRIEF.md` for the full chain / protocol list and
the dust / exploit exclusions.

> **Re-extraction caveat.** Dune Spellbook tables are community-maintained and
> rebuilt over time, so a fresh re-run of query 6912877 is not guaranteed to be
> bit-identical to the committed panel.

---

## 2. Data licences and provenance

| Source | Licence / terms | Redistribution in this package |
|---|---|---|
| Bybit | Public market data; no redistribution restriction identified | Derived aggregates only (normalized parquet in the Release bundle) |
| Binance | Public market data; no redistribution restriction identified | Derived aggregates only |
| Coinbase | Public market data; no redistribution restriction identified | Derived aggregates only |
| CCData | Free-tier API; redistribution of derived data permitted under CCData terms | Derived aggregates only (raw payloads not retained) |
| Dune Analytics / Spellbook | Open-source Spellbook licence; on-chain data is public domain | Cleaned hourly CSV in the Release bundle |

**Redistribution policy.** All five sources are publicly accessible at no cost
and, at the time of extraction (March 2026), impose no redistribution
restriction on the derived hourly aggregates used here. The committed package
ships the processed `data/econ/` artefacts plus the two normalized spot placebo
parquets (`data/normalized/spot/{xrp,doge}_ccdata_1h.parquet`) that the analysis
layer reads directly. All other normalized parquets and the cleaned Dune CSV
travel in the GitHub Release bundle, so a from-raw rebuild needs no third-party
re-download. Raw payloads (`pages.jsonl`, manifest JSON,
QA reports) are not committed; they are regenerable from the documented Dune
query and download procedures.

**Code licence.** MIT License, see [`LICENSE`](../LICENSE).

**Citation conventions for the data sources.**

- Bybit Exchange. *ETH/USDT Perpetual: OHLCV, Open Interest, Funding Rate.* 1-hour frequency. Accessed March 2026 via public REST API (`https://api.bybit.com`).
- Binance Exchange. *ETH/USDT Futures: OHLCV.* 1-hour frequency. Accessed March 2026 via public REST API (`https://fapi.binance.com`).
- CCData (CryptoCompare). *BTC, ETH, XRP, DOGE, CCCAGG Hourly OHLCV.* Accessed March 2026 via CCData API (`https://min-api.cryptocompare.com`).
- Dune Analytics / Spellbook contributors. *DeFi Liquidations, ETH-Collateralized Positions, EVM Chains.* Query ID `6912877`. Accessed March 2026 via `https://dune.com`.

---

## 3. Pipeline data flow

Each script reads upstream artefacts and writes a small contracted set of
files. The `Makefile` chains them. Steps 1-3 read the normalized inputs from the
Release bundle and run only under Path B. From the committed data/econ/ panel
plus the two committed spot placebo parquets, a reviewer runs steps 4 onward
directly (Path A); the Path A Make targets existence-guard on the shipped panel
and do not re-trigger steps 1-3.

| Step | Script | Reads | Produces |
|---|---|---|---|
| 1 | `scripts/run_data_prep.py` | Bybit klines / OI / funding, Binance futures | `data/analysis/windows/master_calendar_1h.parquet`, `window_metadata.json`, `data/analysis/datasets/cex_diagnostics_1h.parquet`, QA JSONs |
| 2 | `scripts/run_core_panel.py` | master calendar, Bybit klines / OI / funding, CCData BTC / ETH spot | `data/econ/econ_core_predefi_1h.parquet` (22 cols × 41,328 rows), QA JSON |
| 3 | `scripts/run_defi_merge.py` | pre-DeFi panel, `data/raw/defi/defi_liquidations_1h_clean.csv` | `data/econ/econ_core_full_1h.parquet` (27 cols × 41,328 rows), QA JSONs |
| 4 | `scripts/run_quantile_lp.py` | full panel | `data/econ/quantile_lp_results.{csv,parquet}`, `pretrend_results.csv`, `quantile_lp_meta.json` |
| 5 | `scripts/run_robustness_all.py` | full panel, quantile-LP results, committed CCData XRP/DOGE spot parquets data/normalized/spot/{xrp,doge}_ccdata_1h.parquet (Test A) | `robustness_placebo_fast.{csv,parquet}` (A), `robustness_bootstrap_fast.csv` (B), `robustness_sensitivity.csv` (C), `se_comparison_kernel_bootstrap.csv` (D1), `ols_lp_hac_benchmark.csv` (D2), `quantile_monotonicity_test_fast.csv` (E), `robustness_subperiods_fast.csv` (F), `quantile_interaction_bootstrap_fast.csv` (G), plus the Tests J–N CSVs — `--tests all` writes 13 CSVs (Tests A–N) |
| 6 | `scripts/aux/run_descriptive_stats.py` | full panel | `data/econ/descriptive_stats.{csv,json}` |
| 7 | `scripts/paper/make_figures.py` | full panel, quantile-LP results, robustness CSVs | `paper/figures/{fig1,fig2,fig3,fig4,fig5,fig7,fig8,fig9}.pdf` |

The eight read-side report notebooks consume the artefacts at each step and
produce no canonical output; they are diagnostic companions for reviewers.

---

## 4. Hardware and runtime

### 4.1 Software

| Component | Version pinned in `requirements.txt` | Notes |
|---|---|---|
| Python | 3.12 | validated on 3.12.2 (macOS x86_64) and 3.12.3 (Linux x86_64); 3.13 untested |
| pandas | 2.2.2 | |
| numpy | 1.26.4 | |
| pyarrow | 14.0.2 | |
| statsmodels | 0.14.2 | |
| scipy | 1.13.1 | |
| matplotlib | 3.10.9 | |
| joblib | 1.4.2 | |
| arch | 7.2.0 | GARCH placebo / optional OOS benchmark |
| jupyter / ipykernel | 1.1.1 / 7.2.0 | report notebooks |
| requests | 2.33.1 | data download |

Full pinned list: [`requirements.txt`](../requirements.txt). Full transitive
closure (audit trail, includes platform-specific extras such as `appnope`):
[`requirements-frozen.txt`](../requirements-frozen.txt).

**Canonical run platform.** Linux 6.8 x86_64 / glibc 2.39 / Python 3.12.3. The
artefacts in `data/econ/` and `paper/figures/` are the outputs of this run; run
provenance is preserved in `data/econ/quantile_lp_meta.json`.

**Secondary platform.** macOS x86_64 (Intel), Python 3.12.2. Apple Silicon is
untested. Panel construction and the main quantile-LP point estimates reproduce
bit-for-bit on macOS. Bootstrap statistics drift below one basis point across
platforms at fixed `--seed 42` (BLAS / LAPACK and joblib scheduling variance);
sign and statistical significance are unchanged at every horizon.

### 4.2 Hardware and wall time

| Target | Default flags | 16 vCPU / 32 GB | 8 vCPU / 16 GB |
|---|---|---|---|
| `make smoke` | `--tests A,C,D2 --quantiles 0.01,0.50 --horizons 0,1` | ~1 min | ~1 min |
| `make data` | data prep + core panel + defi merge | ~15 s | ~15 s |
| `make estimation` | `--n_jobs 1` (sequential, bit-for-bit) | ~5 min | ~5 min |
| `make robustness` | `--tests all --n_boot 1000 --n_jobs -1` | ~30 min | ~50 min (down-tune `--n_jobs`) |
| `make paper` | figures + table fragments + numbers.tex | ~2 min | ~3 min |
| `make all` | full pipeline (needs the Release bundle, Path B) | **~35 min** | **~60 min** |

**Memory.** ≥ 16 GB RAM suffices for all targets except `make robustness` with
`--n_jobs -1` on many-core / small-per-core machines (it instantiates the full
panel per worker); 32 GB recommended there.

**Smaller machines.** The bootstrap is the only core-scaling target. Cap
`--n_jobs` and reduce `--batch_size`, e.g. on a 4-core / 16 GB machine:

```bash
python scripts/run_robustness_all.py --tests all --n_boot 1000 \
       --n_jobs 4 --batch_size 50 --seed 42
```

### 4.3 Reproducibility flags

- `--seed 42` is the canonical seed for `run_robustness_all.py`. The four-level
  SeedSequence scheme `[base_seed, test_id, h, b]` keeps tests independent at a
  shared base seed.
- `--n_jobs 1` is canonical for `run_quantile_lp.py` (bit-for-bit). Runs with
  `--n_jobs > 1` are numerically equivalent within QuantReg / BLAS tolerance but
  not bit-for-bit.

---

## 5. Output artefacts

### 5.1 Result tables consumed by the paper

| Paper element | Produced by | File |
|---|---|---|
| Table 1 (descriptive) | `scripts/aux/run_descriptive_stats.py` | `data/econ/descriptive_stats.{csv,json}` |
| Table 2 (main) | `scripts/run_quantile_lp.py` | `data/econ/quantile_lp_results.{csv,parquet}` |
| Table 3 (subsample stability) | `scripts/aux/run_subsample_stability.py` | `data/econ/subsample_stability.csv` |
| Table 4 (exceedance / skew / MDE) | `scripts/aux/run_exceedance.py`, `run_skew_test.py`, `run_mde_equivalence.py` | `data/econ/exceedance_paired.csv`, `skew_test.csv`, `mde_equivalence.csv` |
| Table 5 (size ratio) | `scripts/aux/run_size_ratio.py` | `data/econ/size_ratio.csv` |
| Table 6 (out-of-sample) | `scripts/aux/run_oos_predictive.py` | `data/econ/oos_predictive.csv` |
| Block-bootstrap CIs (Fig. 3 band) | `scripts/run_robustness_all.py --tests M` | `data/econ/robustness_bootstrap_nb07_spec_fast.csv` |
| Cross-asset placebo (Test A) | `scripts/run_robustness_all.py --tests A` | `data/econ/robustness_placebo_fast.{csv,parquet}` |
| Main-spec block bootstrap (Test B) | `scripts/run_robustness_all.py --tests B` | `data/econ/robustness_bootstrap_fast.csv` |
| Sensitivity battery (Test C) | `scripts/run_robustness_all.py --tests C` | `data/econ/robustness_sensitivity.csv` |
| Subperiod stability (Test F) | `scripts/run_robustness_all.py --tests F` | `data/econ/robustness_subperiods_fast.csv` |
| Kernel-vs-bootstrap SE | `scripts/run_robustness_all.py --tests D1` | `data/econ/se_comparison_kernel_bootstrap.csv` |
| OLS-LP HAC benchmark | `scripts/run_robustness_all.py --tests D2` | `data/econ/ols_lp_hac_benchmark.csv` |
| Quantile monotonicity (Δ) | `scripts/run_robustness_all.py --tests E` | `data/econ/quantile_monotonicity_test_fast.csv` |
| Pre-trend (h ∈ {−2, −1}) | `scripts/run_quantile_lp.py` | `data/econ/pretrend_results.csv` |

Appendix-only: `data/econ/quantile_lp_results_9q.csv` and
`pretrend_results_9q.csv`, generated separately via `run_quantile_lp.py` with
`--quantiles 0.01,0.05,...,0.99`. Not part of `make all`. The exact
exhibit-to-code mapping (figures and table fragments) is in
[`README.md`](../README.md) §7.

### 5.2 Figures

| Figure | File | Generator | Data inputs |
|---|---|---|---|
| Fig. 1 | `paper/figures/fig1_liquidations_timeseries.pdf` | `make_figures.py::fig1_liquidations_timeseries` | `econ_core_full_1h.parquet` |
| Fig. 2 | `paper/figures/fig2_return_distribution.pdf` | `make_figures.py::fig2_return_distribution` | `econ_core_full_1h.parquet` |
| Fig. 3 | `paper/figures/fig3_qlp_irf.pdf` | `make_figures.py::fig3_qlp_irf` | `quantile_lp_results.csv`, `robustness_bootstrap_nb07_spec_fast.csv` |
| Fig. 4 | `paper/figures/fig4_placebo_gap_distribution.pdf` | `make_figures.py::fig4_placebo_gap_distribution` | `placebo_symmetric_draws.csv`, `placebo_symmetric.csv` |
| Fig. 5 (file `fig9`) | `paper/figures/fig9_pure_null_dual.pdf` | `make_figures.py::fig9_pure_null_dual` | `pure_null_circular_shift_by_horizon.csv`, `pure_null_innov_shuffle_by_horizon.csv` |
| Fig. 6 (file `fig8`) | `paper/figures/fig8_btc_vs_eth.pdf` | `make_figures.py::fig8_btc_vs_eth` | `btc_vs_eth_profile.csv` |
| Fig. 7 (file `fig5`) | `paper/figures/fig5_exceedance_symmetry.pdf` | `make_figures.py::fig5_exceedance_symmetry` | `exceedance_results.csv`, `exceedance_paired.csv`, `exceedance_paired_cumulative.csv` |
| Fig. 8 (file `fig7`) | `paper/figures/fig7_mde_equivalence.pdf` | `make_figures.py::fig7_mde_equivalence` | `exceedance_paired.csv`, `skew_test.csv`, `mde_equivalence.csv` |

Figure numbers are the ones PRINTED in the compiled manuscript (order of
appearance); the `figN` file names are stable artifact identifiers and differ
for the last four.

The eight PDF figures are committed for offline inspection.

### 5.3 QA / audit JSON reports

The five `data/analysis/reports/*.json` audit reports are produced by the
panel-construction scripts (Path B) and are **not committed** — `data/analysis/`
is created at runtime by `make data`. They are listed here for provenance and
are regenerated on a from-raw rebuild:

- `data/analysis/reports/calendar_qa.json`, `run_data_prep.py` Stage 1 (Path B)
- `data/analysis/reports/diagnostics_qa.json`, `run_data_prep.py` Stage 2 (Path B)
- `data/analysis/reports/econ_core_predefi_qa.json`, `run_core_panel.py` (Path B)
- `data/analysis/reports/defi_merge_qa.json`, `run_defi_merge.py` (Path B)
- `data/analysis/reports/stationarity_adf.json`, `run_defi_merge.py` (8 ADF tests, Path B)

Committed (ships in the repo under `data/econ/`):

- `data/econ/quantile_lp_meta.json`, `run_quantile_lp.py` (shock distribution + run provenance)

---

## 6. Reproducibility

### 6.1 Reproduce from the committed panel (Path A, recommended)

The committed `data/econ/` panel — together with the two committed spot placebo
parquets `data/normalized/spot/{xrp,doge}_ccdata_1h.parquet` (Test A) — makes
estimation, robustness and the paper exhibits runnable from a fresh clone, with
no raw-data bundle and no credentials:

```bash
git clone https://github.com/Paul-Engerran/DeFi-Endogenous-Fragility.git
cd DeFi-Endogenous-Fragility
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make estimation     # ~5 min  → quantile_lp_results.{csv,parquet}
make robustness     # ~30 min → robustness_*.csv, se_comparison_*, quantile_*_fast.csv, …
make paper          # ~2 min  → paper/figures/*.pdf, paper/tables/*.tex, numbers.tex
```

The `placebo_*` / `exceedance_*` diagnostics are committed pre-computed under
`data/econ/` and are regenerated only by the multi-hour `make canonical` target.

The unit suite (`pytest`) runs the data-independent tests anywhere. Full
reproduction instructions, including the smaller-machine variants, are in
[`README.md`](../README.md) §6.

### 6.2 Rebuild from raw data (Path B, optional)

A full rebuild of the analysis panel needs the raw/normalized data bundle — the
GitHub Release asset **v2.0.0**, unzipped at the repository root so it lands in
data/raw/… and data/normalized/… exactly as config.py expects (proper directory
names, no `__MACOSX`, no ` copie` suffix). A free `CCDATA_API_KEY` is needed
only if you re-download spot data instead of using the bundle. Then:

```bash
export CCDATA_API_KEY=...   # only if re-downloading spot
make smoke                  # ~1 min end-to-end check on a scratch directory
make all                    # ~35 min, data → estimation → robustness → figures
```

The Dune CSV (`data/raw/defi/defi_liquidations_1h_clean.csv`) is regenerated by
re-executing query `6912877` on Dune; the SQL is in
`dune_queries/liquidations_6912877.sql` and documented in
`DUNE_EXTRACTION_BRIEF.md`. The per-source re-download procedure is in
`scripts/data_download/README.md`.

---

## 7. Repository and licence

- **Repository:** <https://github.com/Paul-Engerran/DeFi-Endogenous-Fragility>
- **Licence (code, documentation, artefacts):** MIT, see [`LICENSE`](../LICENSE).
- **Author:** Paul Engerran (`pyproject.toml`, `LICENSE`, `CITATION.cff`).

---

## 8. Known data anomalies and limitations

These are properties of the data and the measurement, documented for
transparency. They do not affect the sign or significance of the headline
results.

- **Dune Cartesian-product correction.** `lending.borrow` and `lending.supply`
  are pre-aggregated per `(tx_hash, blockchain)` before joining, which prevents
  fan-out inflation of the debt total. See `DUNE_EXTRACTION_BRIEF.md`.
- **Collateral-symbol scope.** The ETH-like whitelist (`WETH`, `ETH`, `stETH`,
  `wstETH`, `rETH`, `cbETH`, `sfrxETH`, `ETHx`) predates the liquid-restaking
  generation (weETH, ezETH, and similar), which is therefore excluded. This is a
  one-sided under-count of the shock, concentrated in 2024-2025. The manuscript
  data section discusses and bounds it.
- **Three late-2025 hours with anomalous collateral/debt ratios**, caused by
  Dune `prices.usd` artefacts on illiquid tokens; economically invisible in the
  primary variable `total_debt_repaid_usd`. See `DUNE_EXTRACTION_BRIEF.md`.
- **September 2025 dust cluster** (~75,000 micro-liquidations on Aave/Polygon)
  removed by the `ABS(amount_usd) > 5` filter on the debt side. See
  `DUNE_EXTRACTION_BRIEF.md`.
- **UwuLend and Euler V1 exploit windows excluded.** See `DUNE_EXTRACTION_BRIEF.md`.
- **Bybit / Binance return correlation = 0.9989** (not 0.9999+): minor
  microstructure differences; mean absolute spread ≈ 1.98 bps; does not affect
  the choice of Bybit as primary venue.
- **Placebo assets (XRP, DOGE) show larger β at long horizons at τ = 0.01.**
  This reflects their higher tail-beta to broad crypto conditions, not exposure
  to the DeFi channel; the placebo is informative mainly at short horizons
  (h = 0 to 3).
- **QuantReg IterationLimitWarning at τ = 0.50.** Convergence is verified
  post-hoc by inspecting coefficient stability across the quantile grid.

---

## 9. Final panel codebook

**File:** `data/econ/econ_core_full_1h.parquet`
**Dimensions:** 41,328 rows × 27 columns
**Frequency:** hourly, UTC
**Window:** 2021-03-15 00:00 UTC to 2025-12-01 00:00 UTC (exclusive; the last
bucket starts 2025-11-30 23:00 UTC)
**Missing values on key columns:** 0 (after the `run_defi_merge.py` left join
and `fillna(0)` on the three DeFi columns, by design)

### 9.1 Identifier

| Column | Type | Description |
|---|---|---|
| `date` | datetime64[UTC] | Bucket start timestamp. Convention `[start, end_excl)`; the timestamp labels the beginning of the hour. |

### 9.2 Prices and returns

| Column | Unit | Source | Description |
|---|---|---|---|
| `close_perp` | USD | Bybit | ETH/USDT perpetual close, end of hour |
| `ret_eth_perp` | % (×100) | derived | `ln(close_perp_t / close_perp_{t−1}) × 100` |
| `close_btc_spot` | USD | CCData | BTC/USD spot close (CCCAGG) |
| `ret_btc_spot` | % (×100) | derived | log-return on BTC spot |
| `close_eth_spot` | USD | CCData | ETH/USD spot close (CCCAGG) |
| `ret_eth_spot` | % (×100) | derived | log-return on ETH spot |

### 9.3 Leverage and market-structure proxies

| Column | Unit | Source | Description |
|---|---|---|---|
| `oi` | ETH | Bybit | open interest in native ETH units |
| `d_oi` | ETH | derived | first difference, `oi_t − oi_{t−1}` |
| `oi_zscore` | n/a | derived | `(oi − μ₇₂₀) / σ₇₂₀` (720-hour rolling) |
| `oi_high` | 0/1 | derived | 1 if `oi` > 80th rolling percentile (720h window) |
| `funding_high` | 0/1 | derived | 1 if `funding_rate` > 80th rolling percentile (720h window). Alternative leverage-stress proxy, same threshold and window as `oi_high` for direct comparability. |
| `oi_vol_ratio` | n/a | derived | `oi / MA₂₄(volume_perp)` (crowding proxy) |
| `funding_rate` | rate / 8h | Bybit | perpetual funding rate (positive = longs pay shorts) |
| `basis_bps` | bps | derived | `(close_perp − close_eth_spot) / close_eth_spot × 10⁴` |

### 9.4 Volatility

| Column | Unit | Description |
|---|---|---|
| `vol_eth_7d` | % | rolling 168-hour std of `ret_eth_perp` |
| `vol_btc_7d` | % | rolling 168-hour std of `ret_btc_spot` |
| `ret_eth_std` | n/a | `ret_eth_perp / vol_eth_7d` (volatility-normalized return) |
| `ret_btc_std` | n/a | `ret_btc_spot / vol_btc_7d` |

### 9.5 DeFi liquidations

| Column | Unit | Description |
|---|---|---|
| `liq_usd_total` | USD | hourly sum of debt forcibly repaid on ETH-collateralized DeFi positions; zero-filled on hours without recorded liquidations. Source: Dune query `6912877`. |
| `liq_usd_collateral` | USD | hourly sum of ETH-like collateral seized. Robustness variable. |
| `n_liquidations` | count (float64) | number of debt-side liquidation event legs. Informational only. |
| `log_liq` | n/a | `ln(1 + liq_usd_total)`, primary regressor in all specifications |
| `log_liq_lag1` | n/a | `log_liq` lagged one hour |
| `liq_stress` | 0/1 | 1 if `liq_usd_total` > P95 of non-zero hours (≈ \$261,577) |
| `shock_x_oi` | n/a | `log_liq_lag1 × oi_high`, the leverage-cycle interaction term |

### 9.6 Volume

| Column | Unit | Source | Description |
|---|---|---|---|
| `volume_perp` | USD | Bybit | ETH/USDT perpetual hourly volume |

---

## 10. QA quick reference

| Stage | Check | Result |
|---|---|---|
| `run_data_prep.py` Stage 1 | 41,328 rows, uniform 1-hour spacing, UTC, no gaps | PASS |
| `run_data_prep.py` Stage 2 | 0 missing on Bybit and Binance close prices | PASS |
| `run_core_panel.py` | 0 missing on 5 key columns (`close_perp`, `oi`, `funding_rate`, `close_btc_spot`, `close_eth_spot`) | PASS |
| `run_defi_merge.py` | no duplicate hours in the DeFi CSV; non-negative liquidation volumes; 8 ADF tests | PASS |
| `run_quantile_lp.py` | all (τ × h) cells fitted (150 main + 12 pre-trend); meta JSON carries run provenance | PASS |
| `run_robustness_all.py` | 13 CSVs produced (`--tests all`, Tests A to N); checkpoints in `data/econ/_robust_ckpt/`; deterministic at fixed `--seed` | PASS |

---

For the methodological appendix on the DeFi liquidations dataset, see
[`DUNE_EXTRACTION_BRIEF.md`](./DUNE_EXTRACTION_BRIEF.md). For the overview and
reproduction steps, see [`README.md`](../README.md).
