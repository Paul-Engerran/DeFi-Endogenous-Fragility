# DATA_STATUS : Replication Package Documentation

**Paper**: *Endogenous Market Fragility: DeFi Liquidations and Tail Risk in ETH Returns*
**Repository**: https://github.com/Paul-Engerran/DeFi-Endogenous-Fragility
**Last updated**: March 2026

---

## 0. Overview

This package constructs a 41,328-hour panel (March 2021 – November 2025) from five independent data sources and uses it to estimate Quantile Local Projections across 6 quantiles and 25 hourly horizons. The pipeline runs in seven notebooks executed sequentially. All paths are managed centrally by `config.py`; no hardcoded paths appear in any notebook.

**Execution order:**
```
01_calendar → 02_diagnostics → 03_core_panel → 04_defi_merge → 07_quantile_lp → 08_robustness → 05_figures
```

A complete data archive (all raw and normalized parquet/CSV files needed to run notebooks 01–08 without re-downloading) is available as a GitHub Release attached to this repository.

---

## 1. Data Availability Statements

### 1.1 Bybit API : ETH perpetual derivatives (primary CEX venue)

| | |
|---|---|
| **Variables** | OHLCV (klines), Open Interest, Funding Rate |
| **Frequency** | 1-hour |
| **Window** | 2021-03-15 → 2025-11-30 |
| **Access** | Public REST API - no account or API key required |
| **Endpoint** | `https://api.bybit.com` (linear perpetuals, symbol `ETHUSDT`) |
| **Download notebooks** | `notebooks/download_notebook/Perp/bybit_*.ipynb` |
| **Normalized files** | `data/normalized/cex/bybit/klines_1h.parquet`, `funding_1h.parquet`, `open_interest_1h.parquet` |
| **Rows** | 41,328 per file |
| **Missing values** | 0 on all key columns |
| **Access date** | March 2026 |
| **Redistribution** | Bybit public data; no redistribution restrictions identified |

Bybit is selected as the primary venue based on the cross-venue diagnostic in `02_diagnostics.ipynb`: return correlation with Binance = 0.9990, mean absolute spread = 1.98 bps.

### 1.2 Binance API : ETH futures (diagnostic / secondary)

| | |
|---|---|
| **Variables** | OHLCV, Funding Rate |
| **Frequency** | 1-hour |
| **Window** | 2021-01-01 → 2025-11-30 (43,080 rows - broader window than study period) |
| **Access** | Public REST API - no account or API key required |
| **Endpoint** | `https://fapi.binance.com` (symbol `ETHUSDT`) |
| **Download notebooks** | `notebooks/download_notebook/Perp/binance_*.ipynb` |
| **Normalized file** | `data/normalized/cex/binance/binance_futures_ethusdt_1h_normalized.parquet` |
| **Role** | Used exclusively in `02_diagnostics.ipynb` to validate Bybit as primary venue. Not used in any econometric specification. |
| **Access date** | March 2026 |

### 1.3 CCData (CryptoCompare) : Spot benchmarks and placebo assets

| | |
|---|---|
| **Variables** | Close price (CCCAGG aggregate) |
| **Assets** | BTC, ETH (benchmarks); XRP, DOGE (placebo) |
| **Frequency** | 1-hour |
| **Window** | 2021-03-15 → 2025-11-30 |
| **Access** | Free API key required : registration at `https://www.ccdata.io` |
| **Download notebooks** | `notebooks/download_notebook/spot/spot_ccdata.ipynb` (BTC, ETH); `download_placebo_spot.ipynb` (XRP, DOGE) |
| **Normalized files** | `data/normalized/spot/btc_ccdata_1h.parquet`, `eth_ccdata_1h.parquet` |
| **Rows** | 41,328 per file |
| **Missing values** | 0 |
| **Monetary cost** | Free tier sufficient for this dataset |
| **Access date** | March 2026 |

### 1.4 Dune Analytics : DeFi liquidations

| | |
|---|---|
| **Variables** | `total_debt_repaid_usd`, `total_collateral_seized_usd`, `n_liquidations` |
| **Frequency** | 1-hour (hours with zero liquidations absent, filled with 0 in `04_defi_merge.ipynb`) |
| **Window** | 2021-03-15 → 2025-11-30 |
| **Access** | Free Dune account : `https://dune.com` |
| **Account** | `paul_engerran` |
| **Query ID** | `6912877` (self-contained, no external dependency) |
| **Re-execution** | `dune.get_latest_result(6912877)` via Dune API, or directly from the Dune interface |
| **SQL source** | Also provided in `dune_queries/` directory |
| **Delivered file** | `data/raw/defi/defi_liquidations_1h_clean.csv` (10,976 rows) |
| **Monetary cost** | Free |
| **Access date** | March 2026 |
| **Full documentation** | See `DUNE_EXTRACTION_BRIEF.md` for query architecture, exclusions, and variable definitions |

> **Note on redistribution**: all five data sources are publicly accessible at no cost and with no redistribution restrictions identified at the time of extraction. The processed parquet files are included in the GitHub Release data archive.

---

## 2. Computational Requirements

### Software

| Component | Version used | Minimum |
|-----------|-------------|---------|
| Python | 3.12.2 | 3.10+ |
| pandas | - | 2.0+ |
| numpy | - | 1.24+ |
| pyarrow | - | 12.0+ |
| statsmodels | - | 0.14+ |
| scikit-learn | - | 1.3+ |
| matplotlib | - | 3.7+ |
| jupyter | - | any recent |

Full dependency list: `requirements.txt` in the repository root.

**Setup:**
```bash
git clone https://github.com/Paul-Engerran/DeFi-Endogenous-Fragility.git
cd DeFi-Endogenous-Fragility
pip install -r requirements.txt
python config.py        # creates all output directories
```

### Hardware and runtime

| Notebook | Approximate runtime | Notes |
|----------|--------------------|----|
| `01_calendar.ipynb` | < 1 min | I/O only |
| `02_diagnostics.ipynb` | < 1 min | I/O + summary stats |
| `03_core_panel.ipynb` | < 1 min | I/O + transformations |
| `04_defi_merge.ipynb` | < 1 min | I/O + merge |
| `07_quantile_lp.ipynb` | 15–40 min | QLP estimation + 1,000-rep bootstrap |
| `08_robustness.ipynb` | 10–20 min | 4 sensitivity specs + placebo |
| `05_figures.ipynb` | 2–5 min | Figure generation |

Tested on: MacBook Pro, Apple M-series, 16 GB RAM. The bootstrap in `07_quantile_lp.ipynb` is the binding constraint; runtime scales approximately linearly with `CFG.ECON.lp_n_boot`.

---

## 3. Pipeline : Reproduction Instructions

### Step 1 : Raw data

If starting from scratch (raw data not provided), collect each source in this order:

1. **Bybit**: run notebooks in `notebooks/download_notebook/Perp/` (no credentials required)
2. **Binance**: same directory (no credentials required)
3. **CCData spot**: run `spot_ccdata.ipynb` (free API key required)
4. **CCData placebo**: run `download_placebo_spot.ipynb` (same key)
5. **Dune**: re-execute query `6912877` and export CSV to `data/raw/defi/defi_liquidations_1h_clean.csv`

If using the data archive from the GitHub Release, skip to Step 2.

### Step 2 : Analysis pipeline

Run notebooks in this exact order from the `notebooks/` directory:

| Step | Notebook | Reads | Produces |
|------|----------|-------|----------|
| 1 | `01_calendar.ipynb` | Bybit klines, funding, OI | `master_calendar_1h.parquet`, `window_metadata.json` |
| 2 | `02_diagnostics.ipynb` | Calendar + Bybit + Binance | `cex_diagnostics_1h.parquet` *(diagnostic only)* |
| 3 | `03_core_panel.ipynb` | Calendar + Bybit + CCData spot | `econ_core_predefi_1h.parquet` |
| 4 | `04_defi_merge.ipynb` | Pre-DeFi panel + DeFi CSV | `econ_core_full_1h.parquet` *(analysis-ready panel)* |
| 5 | `07_quantile_lp.ipynb` | Full panel | `quantile_lp_results.csv` |
| 6 | `08_robustness.ipynb` | Full panel + QLP results | `robustness_sensitivity.csv`, `robustness_bootstrap.csv`, `robustness_placebo.csv` |
| 7 | `05_figures.ipynb` | Full panel + all result CSVs | PDF figures in `paper/figures/` |

All paths resolve automatically via `config.py`. No manual path editing is required.

---

## 4. Final Panel Codebook

**File**: `data/econ/econ_core_full_1h.parquet`
**Dimensions**: 41,328 rows × 26 columns
**Frequency**: hourly, UTC
**Window**: 2021-03-15 00:00 UTC → 2025-11-30 23:00 UTC (inclusive)
**Missing values on key columns**: 0

### 4.1 Identifiers

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime64[UTC] | Bucket start timestamp. Convention: `[start, end_excl)` — timestamp labels the beginning of the hour. |

### 4.2 Prices and returns

| Column | Unit | Source | Description |
|--------|------|--------|-------------|
| `close_perp` | USD | Bybit | ETH/USDT perpetual close price, end of hour |
| `ret_eth_perp` | % (×100) | Derived | Hourly log-return on ETH perpetual: `ln(close_perp_t / close_perp_{t-1}) × 100` |
| `close_btc_spot` | USD | CCData | BTC/USD spot close price (CCCAGG) |
| `ret_btc_spot` | % (×100) | Derived | Hourly log-return on BTC spot |
| `close_eth_spot` | USD | CCData | ETH/USD spot close price (CCCAGG) |
| `ret_eth_spot` | % (×100) | Derived | Hourly log-return on ETH spot |

### 4.3 Leverage and market structure proxies

| Column | Unit | Source | Description |
|--------|------|--------|-------------|
| `oi` | ETH | Bybit | Open Interest in native ETH units : total notional of outstanding perpetual contracts |
| `d_oi` | ETH | Derived | First difference of OI: `oi_t − oi_{t-1}` |
| `oi_zscore` | - | Derived | Rolling z-score of OI: `(oi − μ₇₂₀) / σ₇₂₀` (720-hour = 30-day rolling window) |
| `oi_high` | 0/1 | Derived | Leverage regime indicator: 1 if OI exceeds its 80th rolling percentile (720-hour window), 0 otherwise |
| `oi_vol_ratio` | - | Derived | OI-to-volume ratio: `oi / MA₂₄(volume_perp)` : crowding proxy |
| `funding_rate` | rate/8h | Bybit | Perpetual funding rate : positive = longs pay shorts |
| `basis_bps` | bps | Derived | Perp–spot basis: `(close_perp − close_eth_spot) / close_eth_spot × 10,000` |

### 4.4 Volatility

| Column | Unit | Description |
|--------|------|-------------|
| `vol_eth_7d` | % | Rolling 7-day (168-hour) standard deviation of `ret_eth_perp` |
| `vol_btc_7d` | % | Rolling 7-day standard deviation of `ret_btc_spot` |
| `ret_eth_std` | - | Volatility-normalized ETH return: `ret_eth_perp / vol_eth_7d` |
| `ret_btc_std` | - | Volatility-normalized BTC return: `ret_btc_spot / vol_btc_7d` |

### 4.5 DeFi liquidations

| Column | Unit | Description |
|--------|------|-------------|
| `liq_usd_total` | USD | Hourly sum of debt forcibly repaid on ETH-collateralized DeFi positions. Zero-filled for hours without liquidations. Source: Dune query `6912877`. |
| `liq_usd_collateral` | USD | Hourly sum of ETH-like collateral seized. Robustness variable. |
| `n_liquidations` | count | Number of debt-side liquidation events. Not used as primary variable (dust cluster artifact, see `DUNE_EXTRACTION_BRIEF.md`). |
| `log_liq` | - | Log-transformation: `ln(1 + liq_usd_total)` : primary regressor in all specifications |
| `log_liq_lag1` | - | `log_liq` lagged one hour : used in local projections |
| `liq_stress` | 0/1 | 1 if `liq_usd_total` exceeds $261,577 (95th percentile of non-zero hours) |
| `shock_x_oi` | - | Interaction term: `log_liq_lag1 × oi_high` : tests leverage cycle amplification |

### 4.6 Volume

| Column | Unit | Source | Description |
|--------|------|--------|-------------|
| `volume_perp` | USD | Bybit | ETH/USDT perpetual traded volume, hourly |

---

## 5. Output Map

### 5.1 Result files

| File | Produced by | Dimensions | Description |
|------|-------------|-----------|-------------|
| `quantile_lp_results.csv` | `07_quantile_lp.ipynb` | 150 × 9 | β(τ, h) for 6 quantiles × 25 horizons, with SE, p-value, interaction term, N |
| `robustness_sensitivity.csv` | `08_robustness.ipynb` | 4 × 5 | β under 4 alternative specifications (OI threshold P70/P80/P90, collateral variable) |
| `robustness_bootstrap.csv` | `08_robustness.ipynb` | 5 × 7 | Block bootstrap CI at h = 0, 3, 6, 12, 24 (1,000 replications, 24-hour blocks) |
| `robustness_placebo.csv` | `08_robustness.ipynb` | 72 × 7 | β at τ = 0.01, 0.50 for ETH, BTC, XRP, DOGE across 6 horizons |

### 5.2 Figure map

| Figure | File | Notebook | Data inputs |
|--------|------|----------|-------------|
| Fig. 1 - Time series | `fig1_timeseries.pdf` | `05_figures.ipynb` | `econ_core_full_1h.parquet` |
| Fig. 2 - Return distribution | `fig2_distribution.pdf` | `05_figures.ipynb` | `econ_core_full_1h.parquet` |
| Fig. 3 - IRF by quantile *(central figure)* | `fig3_irf_quantiles.pdf` | `05_figures.ipynb` | `quantile_lp_results.csv` + `robustness_bootstrap.csv` |
| Fig. 4 - Cross-asset placebo | `fig4_placebo_crossasset.pdf` | `05_figures.ipynb` | `robustness_placebo.csv` |
| Fig. 5 - Sensitivity dot-plot | `fig5_sensitivity.pdf` | `05_figures.ipynb` | `robustness_sensitivity.csv` |
| Fig. A1 - Full β(τ, h) heatmap | `figA1_heatmap.pdf` | `05_figures.ipynb` | `quantile_lp_results.csv` |
| Fig. A2 - Liq/volume ratio | `figA2_liq_volume_ratio.pdf` | `05_figures.ipynb` | `econ_core_full_1h.parquet` |

---

## 6. QA Log and Known Issues

### 6.1 QA checks passed at each pipeline stage

| Stage | Check | Result |
|-------|-------|--------|
| `01_calendar` | 41,328 rows, uniform 1-hour spacing, UTC, no gaps | ✅ PASS |
| `02_diagnostics` | 0 missing on Bybit and Binance close prices | ✅ PASS |
| `03_core_panel` | 0 missing on all 5 key columns in full and core window | ✅ PASS |
| `04_defi_merge` | No duplicate hours in DeFi CSV; non-negativity of liquidation volumes | ✅ PASS |
| CSV audit | All documented statistics match CSV to the cent | ✅ PASS |

### 6.2 Known issues and anomalies

**Issue 1 : Cartesian product in initial Dune query (corrected)**
The original join between `lending.borrow` and `lending.supply` at the event level produced N×M rows for multi-position bundled transactions. Corrected by pre-aggregating each table by `(tx_hash, blockchain)` before joining. Impact: −14.1% on 2024 totals, −30.5% on 2025 totals, < 0.25% for 2021–2023. Pearson correlation between old and corrected series: 0.976. Coefficients differ by < 3% in log-space.

**Issue 2 : Three hours with anomalous collateral/debt ratio**
Three hours in October–November 2025 show `total_collateral_seized_usd / total_debt_repaid_usd` ratios exceeding 100,000×, caused by debt values below $100 matched to multi-million collateral valuations of illiquid tokens, a known Dune `prices.usd` artifact. These hours are economically invisible in the primary variable `total_debt_repaid_usd` (values: $57, $87, $44) and do not affect any result. The mean collateral/debt ratio across all hours is distorted to 143× by these three observations; the median is 1.0508, consistent with the expected ~5% liquidation bonus.

**Issue 3 : Binance return correlation slightly below 0.999**
The Bybit–Binance return correlation is 0.9990 (not 0.999+ as might be expected for the same underlying asset). This is consistent with minor microstructure differences between venues and does not affect the choice of Bybit as primary venue. The mean absolute spread is 1.98 bps.

**Issue 4 : Placebo assets show larger β at long horizons (τ = 0.01)**
XRP and DOGE exhibit more negative β than ETH at horizons h = 12 and h = 24, τ = 0.01 (XRP: −0.29, DOGE: −0.30 vs ETH: −0.21). This is not evidence that XRP and DOGE are more exposed to the DeFi liquidation channel than ETH; rather, it reflects their structurally higher tail-beta relative to broad crypto market conditions. The placebo test is therefore informative primarily at short horizons (h = 0–3), where the direct mechanical channel dominates and ETH shows greater sensitivity than non-collateral assets. This interpretation is discussed in `08_robustness.ipynb` and in the paper's robustness section.

**Issue 5 : Dust cluster, September 2025**
Approximately 75,000 micro-liquidation events on September 24–25, 2025 (Aave/Polygon, < $0.02 per position) inflate `n_liquidations` to 654,986 in a single hour. Removed by the `ABS(amount_usd) > 5` filter in the Dune query. Does not affect `total_debt_repaid_usd`. Full documentation in `DUNE_EXTRACTION_BRIEF.md`, Section D.1.

**Issue 6 : Bootstrap convergence warnings at τ = 0.01**
Quantile regression at τ = 0.01 with `max_iter = 10,000` (conquer package) produced convergence warnings on approximately 3–5% of bootstrap replications. Verified against two alternative configurations (statsmodels, max_iter = 20,000; 50 replications). Central estimates differ by < 3% at key horizons. Confidence intervals reported are conservative. Full documentation in `07_quantile_lp.ipynb`.

---

## 7. Citation and License

### Data citations

- **Bybit**: Bybit Exchange. *ETH/USDT Perpetual: OHLCV, Open Interest, Funding Rate.* 1-hour frequency. Accessed March 2026 via public REST API (`https://api.bybit.com`).
- **Binance**: Binance Exchange. *ETH/USDT Futures: OHLCV.* 1-hour frequency. Accessed March 2026 via public REST API (`https://fapi.binance.com`).
- **CCData**: CCData (CryptoCompare). *BTC, ETH, XRP, DOGE — CCCAGG Hourly OHLCV.* Accessed March 2026 via CCData API (`https://data-api.ccdata.io`).
- **Dune Analytics**: Dune Analytics / Spellbook contributors. *DeFi Liquidations — ETH-Collateralized Positions, EVM Chains.* Query ID 6912877, account `paul_engerran`. Accessed March 2026 via `https://dune.com`.

### Code license
Code is provided for academic replication purposes. See `LICENSE` in the repository root.

### AI disclosure
AI tools (Claude, Anthropic) were used as a research assistant during this project for the following tasks: code debugging, pipeline architecture design, econometric methodology discussion, and drafting assistance. All data collection decisions, methodological choices, econometric specifications, interpretation of results, and final writing were performed and validated by the author. The author takes full responsibility for the content and conclusions of this paper.
