# DATA_STATUS (as of 2026-02-21)

## Target frequency / timezone
- Frequency: 1h
- Timezone: UTC
- Intended analysis window: [2021-03-15 00:00:00Z, 2025-12-01 00:00:00Z) (chosen to match Bybit coverage)

---

## CEX: Bybit (ETHUSDT linear)
### 1) Klines 1h
- Source: Bybit API (linear, interval=60)
- Window: [2021-03-15 00:00Z, 2025-12-01 00:00Z)
- Expected hours: 41,328
- After dedup: 41,328
- Missing close: 0
- Files: raw CSV/parquet + pages.jsonl + manifest + qa_report; normalized parquet produced
(see manifest + QA)

### 2) Funding (settlement ~8h) -> hourly series
- Source: Bybit funding API
- Raw settlements after dedup: 5,166 (≈ 41,328 / 8)
- Hourly construction: hourly grid + forward-fill between settlements
- Missing funding (hourly series): 0
- Range: min -0.0034595 ; max 0.00173118

### 3) Open Interest 1h
- Source: Bybit OI API (intervalTime=1h)
- Expected hours: 41,328
- After dedup: 41,328
- Missing OI: 0
- Range: min 23,189.6 ; max 1,263,195.04

---

## CEX: Binance Futures (ETHUSDT)
### 1) Klines 1h
- Window: [2021-01-01 00:00Z, 2025-12-01 00:00Z)
- Rows: 43,080 (note: longer than Bybit window)
- Notes: filtered to [start, end) to avoid boundary off-by-one

### 2) Funding (settlement ~8h)
- Window: [2021-01-01 00:00Z, 2025-12-01 00:00Z)
- Rows: 5,385 (≈ 43,080 / 8)
- Note: fundingTime_raw preserved; fundingTime bucketed to hour using round('h')

### 3) Open Interest
- No free historical endpoint available. Snapshot collected but NOT USED in analysis.

---

## Benchmark: Coinbase candles (ETH)
- Repair process implemented.
- Remaining missing hours after refetch: 8
  - 2023-03-04 18:00–20:00 UTC (3h)
  - 2025-10-25 16:00–20:00 UTC (5h)
- Action: decide whether to (a) leave as missing, (b) fill from other venue, or (c) drop those hours from all sources.

---

## DeFi liquidations (BLOCKING)
- Not yet extracted/encoded.
- Planned source: Dune Analytics (protocol-level extracts + hourly aggregation).
- Required deliverables:
  - dune_queries/*.sql (versioned)
  - raw exports (csv/parquet) with extracted_at_utc
  - normalization (USD definition) + data dictionary
  - protocol coverage table and parameter-change notes

---

## Next steps (short)
1) Build master UTC calendar for analysis window and join all CEX/benchmark sources onto it.
2) Produce analysis dataset: analysis/datasets/cex_panel_1h.parquet (+ missing flags).
3) Implement standard QA report (coverage, missing map, duplicates) for the merged dataset.
4) Add DeFi extraction and merge step once Dune queries are finalized.