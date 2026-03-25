# Endogenous Market Fragility: DeFi Liquidations and Tail Risk in ETH Returns

## Overview

This repository contains the data pipeline, econometric analysis, and replication materials for an empirical study of how forced deleveraging in DeFi lending protocols is associated with extreme price movements in Ethereum.

**Key finding:** DeFi liquidation shocks are associated with significantly larger negative returns in the left tail (1st percentile) of ETH's return distribution, while having near-zero effect at the median. This asymmetry is consistent with the endogenous amplification mechanism predicted by the leverage cycle theory (Geanakoplos, 2010; Brunnermeier & Pedersen, 2009).

**Method:** Quantile Local Projections (Jordà, 2005 × Koenker & Bassett, 1978) estimated across 6 quantiles and 25 hourly horizons, with block bootstrap inference and cross-asset placebo tests.

**Data:** 41,328 hourly observations (March 2021 – November 2025) merging CEX derivatives data (Bybit, Binance), spot benchmarks (CCData), and on-chain DeFi liquidation volumes (Dune Analytics).

---

## Repository Structure

```
Research_paper_leverage/
├── config.py                        # Central configuration (paths, parameters, windows)
├── requirements.txt                 # Python dependencies
│
├── notebooks/
│   ├── 01_calendar.ipynb            # Master hourly calendar (temporal reference grid)
│   ├── 02_diagnostics.ipynb         # Cross-venue spread analysis (Bybit vs Binance)
│   ├── 03_core_panel.ipynb          # Pre-DeFi econometric panel (CEX + spot)
│   ├── 04_defi_merge.ipynb          # Merge DeFi liquidations → final panel
│   ├── 05_figures.ipynb             # All figures for the paper
│   ├── 07_quantile_lp.ipynb         # Main specification: Quantile Local Projections
│   ├── 08_robustness.ipynb          # Placebo, bootstrap, sensitivity tests
│   └── download_notebook/           # Data extraction scripts
│       ├── Perp/                    # Bybit (klines, funding, OI) + Binance futures
│       ├── spot/                    # CCData (BTC, ETH, XRP, DOGE)
│       └── archive/                 # Legacy notebooks (not in pipeline)
│
├── data/
│   ├── raw/                         # Immutable raw extractions with manifests & QA
│   ├── normalized/                  # Cleaned hourly parquet files
│   ├── analysis/                    # Intermediate outputs (calendar, diagnostics, QA)
│   └── econ/                        # Final econometric datasets and results
│
├── paper/
│   └── figures/                     # PDF figures for LaTeX
│
└── dune_queries/                    # Versioned SQL for DeFi data extraction
```

---

## Reproduction

### Prerequisites

Python 3.10+ (tested with 3.12) and the packages listed in `requirements.txt`. An Anaconda environment or a standard pip virtual environment both work.

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/Research_paper_leverage.git
cd Research_paper_leverage
pip install -r requirements.txt
python config.py  # Creates the directory structure
```

### Data

Raw data files are not included in this repository. To reproduce the full pipeline from scratch:

1. **CEX derivatives (Bybit, Binance):** Run the notebooks in `notebooks/download_notebook/Perp/`. Bybit and Binance public APIs, no key required.
2. **Spot benchmarks (BTC, ETH):** Run `notebooks/download_notebook/spot/spot_ccdata.ipynb`. Requires a free CCData (CryptoCompare) API key.
3. **Placebo assets (XRP, DOGE):** Run `notebooks/download_notebook/spot/download_placebo_spot.ipynb`. Same CCData API key.
4. **DeFi liquidations:** Run the aggregation query on [Dune Analytics](https://dune.com/) (free account). The methodology and filtering logic are documented in `DUNE_EXTRACTION_BRIEF.md`.

A complete data archive (all parquet and CSV files needed to run the analysis notebooks without re-downloading) is available as a GitHub Release attached to this repository.

### Pipeline execution order

```
01_calendar → 02_diagnostics → 03_core_panel → 04_defi_merge → 07_quantile_lp → 08_robustness → 05_figures
```

Each notebook reads from files produced by the previous ones. All paths are managed centrally by `config.py` — no hardcoded paths in any notebook.

---

## Data Sources

| Source | Variables | Frequency | Window |
|--------|-----------|-----------|--------|
| Bybit API | ETH perp: OHLCV, Open Interest, Funding Rate | 1h | 2021-03-15 — 2025-12-01 |
| Binance API | ETH futures: OHLCV, Funding Rate | 1h | 2021-03-15 — 2025-12-01 |
| CCData (CryptoCompare) | BTC, ETH, XRP, DOGE spot prices (CCCAGG) | 1h | 2021-03-15 — 2025-12-01 |
| Dune Analytics | DeFi liquidations (Aave, Compound, Maker, Spark, Morpho) | 1h | 2021-03-15 — 2025-12-01 |

### DeFi liquidation scope

The DeFi liquidation variable measures forced deleveraging on ETH-collateralized lending positions across all EVM chains covered by the Dune spellbook (Ethereum, Arbitrum, Optimism, Base, Polygon, and others). The primary variable `total_debt_repaid_usd` captures the aggregate USD value of debt repaid through forced liquidation per hour — a direct measure of the mechanical reduction in outstanding leverage.

**Collateral filter:** ETH and liquid staking derivatives only (WETH, stETH, wstETH, rETH, cbETH, sfrxETH, ETHx).

**Exclusions:** UwuLend post-hack (June 10, 2024 onward), Euler V1 hack window (March 13 – April 12, 2023), and dust positions below $10. All exclusions are documented and justified in the extraction brief.

---

## Key Results

### Distributional properties

The unconditional distribution of hourly ETH log-returns exhibits an excess kurtosis of 17.4 and negative skewness of −0.62 (N = 41,327), confirming substantial departure from the Gaussian benchmark.

### Quantile Local Projections: β(shock) by quantile and horizon

| Horizon | τ = 0.01 | τ = 0.05 | τ = 0.50 | τ = 0.95 |
|---------|----------|----------|----------|----------|
| h = 0   | −0.032   | −0.019   | +0.002   | +0.018   |
| h = 6   | −0.160   | −0.079   | +0.014   | +0.042   |
| h = 12  | −0.245   | −0.102   | +0.032   | +0.079   |
| h = 24  | −0.214   | −0.117   | +0.038   | +0.117   |

The negative β at τ = 0.01 indicates that DeFi liquidation shocks are associated with significantly deeper left-tail returns. The near-zero β at τ = 0.50 confirms that median returns are unaffected, consistent with the "dormant risk" prediction of the leverage cycle theory.

### Interaction with leverage (OI regime)

The interaction term shock × OI_high is significant at τ = 0.01 (β = −0.022, p = 0.022), indicating that the amplification effect is stronger when aggregate leverage is elevated. This interaction is not significant at the median or right tail.

### Robustness

**Sensitivity to specification choices:** β(shock) at τ = 0.01 ranges from −0.145 to −0.152 across four alternative specifications (OI threshold at P70/P80/P90, collateral seized instead of debt repaid). Range of 0.006, all significant at p < 0.001.

**Block bootstrap (1,000 replications, 24h blocks):** The 95% confidence interval excludes zero at h = 0 [−0.120, −0.041], h = 3 [−0.250, −0.037], and h = 12 [−0.505, −0.033]. At h = 24, the CI includes zero [−0.342, +0.064], indicating the effect attenuates at longer horizons.

**Cross-asset placebo (orthogonalized shock):** At short horizons (h = 0 to h = 3), ETH shows greater sensitivity than non-collateral assets (XRP, DOGE), consistent with the direct mechanical channel. The ETH/BTC sensitivity ratio is 1.6x at h = 12.

---

## Methodology

The econometric framework combines two established tools in a way that, to our knowledge, is novel in the DeFi literature.

**Quantile regression** (Koenker & Bassett, 1978) allows estimation at specific percentiles of the return distribution rather than the conditional mean, capturing the asymmetric, regime-dependent nature of the liquidation effect.

**Local projections** (Jordà, 2005) estimate impulse response functions without specifying a VAR system, providing a flexible and robust way to trace the dynamic response of returns to a liquidation shock across horizons h = 0 to 24 hours.

The combined **Quantile Local Projection** estimates β(τ, h) — the response of quantile τ of the cumulative return distribution at horizon h to a one-unit increase in the (lagged, log-transformed) liquidation shock, controlling for BTC returns, funding rate, and basis spread.

The liquidation shock is orthogonalized with respect to BTC returns in the cross-asset placebo test to isolate the DeFi-specific component (R² of log_liq on BTC returns = 0.077, meaning 92.3% of liquidation variation is DeFi-specific).

---

## References

- Brunnermeier, M. K., & Pedersen, L. H. (2009). Market Liquidity and Funding Liquidity. *Review of Financial Studies*, 22(6), 2201–2238.
- Geanakoplos, J. (2010). The Leverage Cycle. *NBER Macroeconomics Annual*, 24(1), 1–66.
- Heimbach, L., & Huang, W. (2024). DeFi Leverage. *BIS Working Papers* No. 1171.
- Jordà, Ò. (2005). Estimation and Inference of Impulse Responses by Local Projections. *American Economic Review*, 95(1), 161–182.
- Koenker, R., & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33–50.
- Lehar, A., & Parlour, C. A. (2022). Systemic Fragility in Decentralized Markets. *BIS Working Papers* No. 1062.
- Financial Stability Board (2023). *The Financial Stability Risks of Decentralised Finance*.
- Warmuz, J., Chaudhary, A., & Pinna, A. (2022). Toxic Liquidation Spirals. Working Paper.

---

## AI Disclosure
AI tools (Claude, Anthropic) were used for code development assistance and methodology discussion. All research decisions and interpretations are the author's own.

---

## License

This project is academic research. Code is provided for replication purposes.

## Contact

Paul Engerran - paul.engerran@gmail.com
