# Methodological Appendix — DeFi Liquidations Dataset

## A. Data source

The DeFi liquidation data are extracted from Dune Analytics using the curated Spellbook tables `lending.borrow` and `lending.supply`, maintained by Dune and the open-source community. These tables normalize on-chain liquidation events across multiple protocols and EVM chains into a uniform schema.

- **Spellbook repository**: `duneanalytics/spellbook` (GitHub)
- **Tables used**: `lending.borrow`, `lending.supply`
- **Query engine**: Trino (DuneSQL)
- **Dune account**: `como_55`
- **Query ID**: `6912877` (self-contained — no external materialized view dependency)
- **Extraction date**: March 2026

The query can be re-executed directly from the Dune Analytics interface or via the API (`dune.get_latest_result(6912877)`). Results may differ marginally from future executions if Spellbook is updated. The SQL source is also provided in the `dune_queries/` directory of this repository.

---

## B. Scope

### Time window
- Start: `2021-03-15 00:00 UTC` (inclusive)
- End: `2025-12-01 00:00 UTC` (exclusive)

This window is identical to the master calendar produced in `01_calendar.ipynb`.

### Chains covered
All EVM chains available in Spellbook at extraction time: Ethereum mainnet, Arbitrum, Optimism, Base, Polygon, Avalanche (avalanche_c), Gnosis, zkSync, Scroll, Linea, BNB Chain, Fantom, Celo, Sonic, Unichain, and others.

**Justification**: the ETH price is unified cross-chain through arbitrage. A liquidation of wrapped ETH on Arbitrum exerts the same directional pressure on the spot price as one on Ethereum mainnet. One implication is a structural upward trend in aggregate liquidation volumes over the study period as L2 coverage in Spellbook expands — a limitation discussed in Section E.

### Protocols covered
All lending protocols indexed in Spellbook at extraction time, including Aave v1/v2/v3 and derivatives, Compound v2/v3, Spark, Morpho, and others. Coverage is not exhaustive of all on-chain lending activity; protocols not yet indexed are excluded. Morpho Blue (launched late 2023) is partially covered at the start of its period, with coverage improving through 2024.

### Collateral filter
Only liquidation transactions in which the seized collateral was an ETH-like asset are retained. The filter is applied in `lending.supply` on the `symbol` column, retaining: `WETH`, `ETH`, `stETH`, `wstETH`, `rETH`, `cbETH`, `sfrxETH`, `ETHx`.

**Justification**: the paper's hypothesis concerns the ETH leverage cycle specifically. Liquidations where collateral is BTC, stablecoins, or other tokens are not mechanically linked to ETH price dynamics.

**Implementation note**: the ETH filter is applied to `lending.supply` (collateral side). The `INNER JOIN` between the two CTEs means that a debt repayment event is retained only if it matches a transaction that also seized ETH-like collateral — effectively filtering the debt side by ETH-collateral scope without a redundant symbol filter on `lending.borrow`.

---

## C. Query architecture

### Structure

The query uses two CTEs that pre-aggregate by transaction before joining, then aggregates to hourly frequency.

```sql
WITH borrow_per_tx AS (
  SELECT
    tx_hash, blockchain, block_time, project, version,
    SUM(ABS(amount_usd)) AS debt_amount_usd,
    COUNT(*)             AS n_debt_events
  FROM lending.borrow
  WHERE transaction_type = 'borrow_liquidation'
    AND block_time >= TIMESTAMP '2021-03-15'
    AND block_time <  TIMESTAMP '2025-12-01'
    AND ABS(amount_usd) > 5
    AND NOT (project = 'uwulend' AND block_time >= TIMESTAMP '2024-06-10')
    AND NOT (project = 'euler'
             AND block_time >= TIMESTAMP '2023-03-13'
             AND block_time <  TIMESTAMP '2023-04-13')
  GROUP BY 1, 2, 3, 4, 5
),
supply_per_tx AS (
  SELECT
    tx_hash, blockchain,
    SUM(ABS(amount_usd)) AS collateral_amount_usd,
    COUNT(*)             AS n_supply_events
  FROM lending.supply
  WHERE transaction_type IN ('deposit_liquidation', 'supply_liquidation')
    AND block_time >= TIMESTAMP '2021-03-15'
    AND block_time <  TIMESTAMP '2025-12-01'
    AND symbol IN ('WETH','ETH','stETH','wstETH','rETH','cbETH','sfrxETH','ETHx')
  GROUP BY 1, 2
)
SELECT
  DATE_TRUNC('hour', b.block_time) AS date,
  SUM(b.debt_amount_usd)           AS total_debt_repaid_usd,
  SUM(s.collateral_amount_usd)     AS total_collateral_seized_usd,
  SUM(b.n_debt_events)             AS n_liquidations
FROM borrow_per_tx b
INNER JOIN supply_per_tx s
  ON  b.tx_hash    = s.tx_hash
  AND b.blockchain = s.blockchain
GROUP BY 1
ORDER BY 1
```

### Why pre-aggregation before the join

A direct event-level join between `lending.borrow` and `lending.supply` on `(tx_hash, blockchain)` produces a Cartesian product when a transaction contains multiple debt events and multiple collateral events — a pattern common in multi-position bundled liquidation bots operating on Aave v3. In such cases, N debt events × M collateral events produce N×M rows, inflating both totals by the wrong factor. Pre-aggregating each table to one row per `(tx_hash, blockchain)` before joining eliminates this by construction.

---

## D. Exclusions and filters

### D.1 — Dust positions (`ABS(amount_usd) > 5`)

Applied in `borrow_per_tx` before aggregation. A cluster of approximately 75,000 micro-liquidation events was observed on September 24–25, 2025, generating up to 654,986 events in a single hour for a total volume of approximately $650k. Transaction-level analysis identifies Aave positions on Polygon with debt and collateral below $0.02, associated with bot activity following Aave's Chainlink SVR integration in March 2025, which altered the profitability of small liquidations on L2s.

The $5 threshold removes these positions without affecting economically significant liquidations. The filter is applied on the debt side only; no corresponding filter is applied to `lending.supply`. This asymmetry is intentional and harmless: DeFi lending protocols require over-collateralization, so a debt position of $5 corresponds to at least $6–9 of collateral by protocol design (Aave v3 WETH LTV ~80%). A collateral-side value below the debt-side threshold is therefore structurally impossible.

The variable `n_liquidations` is strongly inflated by this cluster and is not used as a primary variable in the econometric analysis. **Robustness**: results are robust to alternative thresholds of $1, $10, and $100.

### D.2 — UwuLend post-hack (`project = 'uwulend' AND block_time >= '2024-06-10'`)

UwuLend was exploited on June 10, 2024. A single hour shows a liquidation volume of approximately $50.5 billion — physically impossible given that Aave's total TVL has never exceeded ~$60B. This results from exploit-related on-chain movements combined with pricing errors in Dune's `prices.usd` for illiquid tokens involved in the attack. UwuLend liquidations prior to June 10, 2024 are retained.

### D.3 — Euler V1 hack window (`project = 'euler' AND block_time >= '2023-03-13' AND < '2023-04-13'`)

Verification confirms that the only Euler-labeled activity present in the dataset post-dates April 2023 and corresponds to Euler v2, a fully independent protocol launched in 2024–2025. The exclusion filter applies exclusively to the March–April 2023 window and does not affect Euler v2 activity.

---

## E. Variables and descriptive statistics

### Variable definitions

| Column | Definition | Role |
|--------|------------|------|
| `date` | Hourly bucket start, UTC | Join key with CEX panel |
| `total_debt_repaid_usd` | `SUM(ABS(debt_amount_usd))` per hour | **Primary variable** |
| `total_collateral_seized_usd` | `SUM(ABS(collateral_amount_usd))` per hour | Robustness variable |
| `n_liquidations` | `COUNT(*)` of debt events per hour | Informational only |

**Primary variable**: `total_debt_repaid_usd` measures the volume of debt forcibly extinguished per hour — a direct measure of forced deleveraging in the sense of Geanakoplos (2010) and Brunnermeier & Pedersen (2009). Liquidations are triggered mechanically by smart contracts when the collateral/debt ratio crosses a protocol-defined threshold, independently of any decision by the borrower. Robustness tests confirm that substituting `total_collateral_seized_usd` does not materially alter results (see `08_robustness.ipynb`, Test C).

**Sign convention**: Dune records protocol outflows as negative. `ABS()` is applied before summation. Verification: the median ratio `total_collateral_seized_usd / total_debt_repaid_usd` is 1.0508, consistent with Aave v2's standard liquidation bonus of ~5%.

### Derived variables (computed in `04_defi_merge.ipynb`)

| Column | Definition |
|--------|------------|
| `log_liq` | `ln(1 + liq_usd_total)` — primary regressor in all regressions |
| `log_liq_lag1` | `log_liq` lagged one hour — used in local projections |
| `liq_stress` | 1 if `liq_usd_total` exceeds P95 of non-zero hours ($261,577) |
| `shock_x_oi` | `log_liq_lag1 × oi_high` — interaction term for leverage cycle test |

### Descriptive statistics

| Statistic | Value |
|-----------|-------|
| Total hours in panel | 41,328 |
| Hours with liquidations | 10,976 (26.6%) |
| Hours with zero liquidations | 30,352 (73.4%) |
| Mean — all hours | $61,008 |
| Mean — non-zero hours | $229,712 |
| Median — non-zero hours | $399 |
| 95th percentile — non-zero | $261,577 |
| 99th percentile — non-zero | $3,775,109 |
| Maximum (single hour) | $187,641,971 |
| Cumulative total 2021–2025 | $2.52 billion |
| Stress regime hours (P95) | 549 (1.3%) |

The extreme divergence between mean ($229k) and median ($399) for non-zero hours reflects a heavily right-skewed distribution: the majority of liquidation hours are low-volume micro-events, while aggregate dollar volume is concentrated in a small number of high-stress episodes. This motivates both the log-transformation `ln(1 + L_t)` and the separate stress-regime indicator.

**Annual totals:**

| Year | Total USD |
|------|-----------|
| 2021 | $378,040,491 |
| 2022 | $395,167,149 |
| 2023 | $91,442,786  |
| 2024 | $669,851,938 |
| 2025 | $986,816,176 |

### Validation spot-check

The five largest single-hour events correspond to independently documented market stress episodes:

| Date (UTC) | Volume | Context |
|------------|--------|---------|
| 2024-08-05 01:00 | $187.6M | Yen carry trade unwind |
| 2025-02-03 01:00 | $130.4M | ETH selloff below $2,150 |
| 2025-10-10 21:00 | $111.4M | Sharp ETH price drop, October 2025 |
| 2025-02-03 02:00 | $103.1M | Continuation of February 3 episode |
| 2024-04-13 20:00 | $60.0M | April 2024 macro repricing |

Additional cross-checks against May 2021, Terra/Luna (June 2022), FTX (November 2022), and DeepSeek (February 2025) confirm that liquidation spikes align with independently documented stress dates. See `04_defi_merge.ipynb`, section "Known Event Cross-Check."

### Known limitations

**1. Structural trend from expanding L2 coverage.** Aggregate liquidation volumes increase mechanically over the study period as additional chains are indexed in Spellbook. The log-transformation attenuates but does not eliminate this effect.

**2. Liquidator behavior unobservable.** Whether liquidators immediately resell seized ETH collateral is not directly observable. `total_collateral_seized_usd` is a reasonable proxy for selling pressure only under the assumption that liquidators sell promptly to realize the liquidation bonus.

**3. Pricing errors in `prices.usd`.** Three hours in the dataset (October–November 2025) show `total_collateral_seized_usd / total_debt_repaid_usd` ratios exceeding 100,000×, caused by near-zero debt values ($57–$87) matched to multi-million collateral valuations of illiquid tokens — a known Dune pricing artifact. These three hours are economically invisible in the primary variable `total_debt_repaid_usd`. More broadly, price errors during stress periods may cause systematic underestimation of liquidation volumes in the hours of greatest analytical interest.

**4. Incomplete protocol coverage.** Protocols not indexed in Spellbook at extraction time are absent. This constitutes a lower bound on true DeFi liquidation activity.

---

## F. Delivered file

**Filename**: `defi_liquidations_1h_clean.csv`

- 10,976 rows — hours with at least one qualifying liquidation
- Coverage: `2021-03-15 05:00 UTC` → `2025-11-30 23:00 UTC`
- 26.6% of the 41,328 hours in the full study window
- Hours with zero liquidations are absent; they are filled with zeros via LEFT JOIN on the master calendar in `04_defi_merge.ipynb`
