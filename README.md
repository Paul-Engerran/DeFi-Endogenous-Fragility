# Endogenous Market Fragility: DeFi Liquidations and Tail Risk in ETH Returns

## Overview

This repository contains the data pipeline, econometric analysis, and replication materials for an empirical study of how forced deleveraging in DeFi lending protocols is associated with extreme price movements in Ethereum.

**Key finding:** DeFi liquidation shocks are associated with significantly larger negative returns in the left tail (1st percentile) of ETH's return distribution, while having near-zero effect at the median. This asymmetry is consistent with the endogenous amplification mechanism predicted by the leverage cycle theory (Geanakoplos, 2010; Brunnermeier & Pedersen, 2009).

**Method:** Quantile Local Projections (Jordà, 2005 × Koenker & Bassett, 1978) estimated across 6 quantiles and 25 hourly horizons, with block bootstrap inference and cross-asset placebo tests.

**Data:** 41,328 hourly observations (March 2021 – November 2025) merging CEX derivatives data (Bybit, Binance), spot benchmarks (CCData), and on-chain DeFi liquidation volumes (Dune Analytics).

---

## Repository Structure

```
DeFi-Endogenous-Fragility/
├── config.py                        # Central configuration (paths, parameters, windows)
├── requirements.txt                 # Python dependencies
├── README.md
├── DATA_STATUS.md                   # Full replication package documentation
├── DUNE_EXTRACTION_BRIEF.md         # DeFi liquidations dataset — methodology & query
│
├── notebooks/
│   ├── 01_calendar.ipynb            # Master hourly calendar
│   ├── 02_diagnostics.ipynb         # Cross-venue spread analysis (Bybit vs Binance)
│   ├── 03_core_panel.ipynb          # Pre-DeFi econometric panel
│   ├── 04_defi_merge.ipynb          # Merge DeFi liquidations → final panel
│   ├── 05_figures.ipynb             # All figures for the paper
│   ├── 07_quantile_lp.ipynb         # Main specification: Quantile Local Projections
│   ├── 08_robustness.ipynb          # Placebo, bootstrap, sensitivity tests
│   └── download_notebook/
│       ├── Perp/                    # Bybit klines, funding, OI + Binance futures
│       ├── spot/                    # CCData spot (BTC, ETH, XRP, DOGE)
│       └── archive/                 # Legacy notebooks (not in pipeline)
│
├── data/                            # Not tracked — created by pipeline (see DATA_STATUS.md)
│   ├── raw/                         # ↳ Immutable API extractions
│   ├── normalized/                  # ↳ Cleaned hourly parquet files
│   ├── analysis/                    # ↳ Intermediate outputs (calendar, diagnostics, QA)
│   └── econ/                        # ↳ Final econometric datasets and results
│
├── paper/
│   └── figures/                     # PDF figures for LaTeX
│
└── dune_queries/
    └── liquidations_6912877.sql     # DeFi liquidation query (Dune account: paul_engerran)
```

---

## Reproduction

### Prerequisites

Python 3.10+ (tested with 3.12) and the packages listed in `requirements.txt`. An Anaconda environment or a standard pip virtual environment both work.

### Installation

```bash
git clone https://github.com/Paul-Engerran/DeFi-Endogenous-Fragility.git
cd DeFi-Endogenous-Fragility
pip install -r requirements.txt
python config.py  # Creates the directory structure
```

### Data

Raw data files are not included in this repository. To reproduce the full pipeline from scratch:

1. **CEX derivatives (Bybit, Binance):** Run the notebooks in `notebooks/download_notebook/Perp/`. Bybit and Binance public APIs, no key required.
2. **Spot benchmarks (BTC, ETH):** Run `notebooks/download_notebook/spot/spot_ccdata.ipynb`. Requires a free CCData (CryptoCompare) API key.
3. **Placebo assets (XRP, DOGE):** Run `notebooks/download_notebook/spot/download_placebo_spot.ipynb`. Same CCData API key.
4. **DeFi liquidations:** Re-execute query `6912877` on [Dune Analytics](https://dune.com/) (free account, `paul_engerran`). Full methodology in `DUNE_EXTRACTION_BRIEF.md`.

A complete data archive (all parquet and CSV files needed to run the analysis notebooks without re-downloading) is available as a **GitHub Release** attached to this repository.

### Pipeline execution order

```
01_calendar → 02_diagnostics → 03_core_panel → 04_defi_merge → 07_quantile_lp → 08_robustness → 05_figures
```

Each notebook reads from files produced by the previous ones. All paths are managed centrally by `config.py` — no hardcoded paths in any notebook. Full input/output mapping in `DATA_STATUS.md`.

---

## Data Sources

| Source | Variables | Frequency | Window |
|--------|-----------|-----------|--------|
| Bybit API | ETH perp: OHLCV, Open Interest, Funding Rate | 1h | 2021-03-15 → 2025-12-01 |
| Binance API | ETH futures: OHLCV, Funding Rate (diagnostic only) | 1h | 2021-01-01 → 2025-12-01 |
| CCData (CryptoCompare) | BTC, ETH, XRP, DOGE spot prices (CCCAGG) | 1h | 2021-03-15 → 2025-12-01 |
| Dune Analytics | DeFi liquidations (Aave, Compound, Spark, Morpho, and others) | 1h | 2021-03-15 → 2025-12-01 |

### DeFi liquidation scope

The DeFi liquidation variable measures forced deleveraging on ETH-collateralized lending positions across all EVM chains covered by the Dune Spellbook (Ethereum, Arbitrum, Optimism, Base, Polygon, and others). The primary variable `total_debt_repaid_usd` captures the aggregate USD value of debt repaid through forced liquidation per hour — a direct measure of the mechanical reduction in outstanding leverage.

**Collateral filter:** ETH and liquid staking derivatives only (WETH, ETH, stETH, wstETH, rETH, cbETH, sfrxETH, ETHx).

**Exclusions:** UwuLend post-hack (June 10, 2024 onward), Euler V1 hack window (March 13 – April 13, 2023), and dust positions below $5. All exclusions are documented and justified in `DUNE_EXTRACTION_BRIEF.md`.

---

## Key Results

### Distributional properties

The unconditional distribution of hourly ETH log-returns exhibits an excess kurtosis of 17.4 and negative skewness of −0.62 (N = 41,327), confirming substantial departure from the Gaussian benchmark.

### Quantile Local Projections: β(shock) by quantile and horizon

| Horizon | τ = 0.01 | τ = 0.05 | τ = 0.50 | τ = 0.95 |
|---------|----------|----------|----------|----------|
| h = 0   | −0.032   | −0.019   | +0.002   | +0.018   |
| h = 6   | −0.159   | −0.079   | +0.014   | +0.041   |
| h = 12  | −0.241   | −0.102   | +0.033   | +0.077   |
| h = 24  | −0.213   | −0.116   | +0.038   | +0.116   |

The negative β at τ = 0.01 indicates that DeFi liquidation shocks are associated with significantly deeper left-tail returns. The near-zero β at τ = 0.50 confirms that median returns are unaffected, consistent with the "dormant risk" prediction of the leverage cycle theory.

### Interaction with leverage (OI regime)

The interaction term shock × OI_high is significant at τ = 0.01 (β = −0.022, p = 0.021), indicating that the amplification effect is stronger when aggregate leverage is elevated. This interaction is not significant at the median or right tail, and is sensitive to the choice of OI threshold (see robustness section).

### Robustness

**Sensitivity to specification choices:** β(shock) at τ = 0.01 ranges from −0.145 to −0.152 across four alternative specifications (OI threshold at P70/P80/P90, collateral seized instead of debt repaid). All significant at p < 0.001.

**Block bootstrap (1,000 replications, 24h blocks):** The 95% confidence interval excludes zero at h = 0 [−0.121, −0.044], h = 3 [−0.245, −0.043], and h = 12 [−0.493, −0.027]. At h = 24, the CI includes zero [−0.336, +0.065], indicating the effect attenuates at longer horizons.

**Cross-asset placebo:** At short horizons (h = 0 to h = 3), ETH shows greater sensitivity than non-collateral assets (XRP, DOGE), consistent with the direct mechanical channel. The placebo comparison is most informative at short horizons; at longer horizons, XRP and DOGE exhibit larger β, reflecting their structurally higher tail-beta rather than DeFi exposure.

---

## Methodology

The econometric framework combines two established tools:

**Quantile regression** (Koenker & Bassett, 1978) allows estimation at specific percentiles of the return distribution rather than the conditional mean, capturing the asymmetric, tail-specific nature of the liquidation effect.

**Local projections** (Jordà, 2005) estimate impulse response functions without specifying a VAR system, tracing the dynamic response of returns to a liquidation shock across horizons h = 0 to 24 hours.

The combined **Quantile Local Projection** estimates β(τ, h) — the response of quantile τ of the return distribution at horizon h to a one-unit increase in the lagged log-transformed liquidation shock, controlling for BTC returns, funding rate, and basis spread.

---

## Documentation

| File | Description |
|------|-------------|
| `DATA_STATUS.md` | Full replication package documentation: data availability statements, codebook, pipeline map, QA log |
| `DUNE_EXTRACTION_BRIEF.md` | DeFi liquidations dataset: query architecture, exclusions, variable definitions, validation |
| `dune_queries/liquidations_6912877.sql` | SQL source for the Dune extraction query |

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

AI tools (Claude, Anthropic) were used as a research assistant during this project for the following tasks: code debugging, pipeline architecture design, econometric methodology discussion, and drafting assistance. All data collection decisions, methodological choices, econometric specifications, interpretation of results, and final writing were performed and validated by the author. The author takes full responsibility for the content and conclusions of this paper.

---

## License

This project is academic research. Code is provided for replication purposes.

## Contact

Paul Engerran — paul.engerran@gmail.com
