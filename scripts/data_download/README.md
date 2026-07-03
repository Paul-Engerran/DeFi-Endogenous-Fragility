# Raw data acquisition

The analysis-ready panel (`data/econ/`) ships with the package, together with
two normalized spot parquets (`data/normalized/spot/{xrp,doge}_ccdata_1h.parquet`)
that the cross-asset placebo (Test A) reads directly. Reproducing the estimation,
robustness, and paper exhibits therefore needs no raw bundle (top-level
[`README.md`](../../README.md), Path A). This note documents how the **raw and
normalized inputs** were acquired, for a full from-raw rebuild (Path B).

## Obtaining the raw / normalized bundle

The raw venue payloads and the bulk of the normalized parquets are not committed;
they are distributed as a versioned **GitHub Release asset**. The only exceptions
are two normalized spot files — `data/normalized/spot/xrp_ccdata_1h.parquet` and
`…/doge_ccdata_1h.parquet` (~2.4 MB) — which ship in the repo because the
cross-asset placebo (Test A) reads them directly. The bundle is published as
`data_archive_v2.0.0.zip` on the project's GitHub Releases page
(https://github.com/Paul-Engerran/DeFi-Endogenous-Fragility/releases). v2.0.0
supersedes v1.0, which extracted to non-standard paths.

Download the v2.0.0 data archive from the GitHub Release and unzip it at the
repository root; it expands directly into `data/raw/...` and `data/normalized/...`
(the layout `config.py` expects — no `__MACOSX`, no ` copie` suffix, no manual
renaming). Then run `make data` (or `make all`).

## Re-acquiring from source

All sources are public and documented; only CCData requires a free key.

The Release bundle additionally ships the **acquisition scripts themselves**
under `downloaders/` (next to `data/` in the archive): `download_bybit.py`
(`--dataset klines|funding|open_interest|all`), `download_binance.py`,
`download_coinbase.py`, and `download_ccdata_spot.py` (`--symbols btc,eth,xrp,doge`;
requires `CCDATA_API_KEY`), sharing `downloaders/_common.py`. Each supports
`--check` (dry-run, no network), `--start/--end` (short-slice validation), and
`--out_root`; every dataset directory receives `pages.jsonl`, `manifest.json`
(SHA-256), and `qa_report.json` provenance. Run order and the validation
protocol are documented in `downloaders/README.md`. Validated 2026-07-02: a
one-week re-download reproduced the shipped normalized parquets exactly
(Bybit x3 and Coinbase byte-identical; Binance within float epsilon).

| Asset class | Provider | Endpoint | Credential |
|---|---|---|---|
| ETH perp klines, funding, open interest | Bybit | `https://api.bybit.com` (linear `ETHUSDT`) | none |
| ETH futures (diagnostic only) | Binance | `https://fapi.binance.com` (`ETHUSDT`) | none |
| ETH spot (data-quality audit only) | Coinbase | `https://api.exchange.coinbase.com` | none |
| BTC / ETH / XRP / DOGE spot (CCCAGG) | CCData | `https://min-api.cryptocompare.com/data/v2/histohour` | `CCDATA_API_KEY` (free) |
| DeFi liquidations | Dune Analytics | query `6912877` (see `dune_queries/`) | free Dune account |

```bash
export CCDATA_API_KEY="..."   # free key from cryptocompare.com; spot data only
```

For the DeFi feed, re-execute Dune query `6912877` (SQL in
[`dune_queries/`](../../dune_queries/)) and export the result to
`data/raw/defi/defi_liquidations_1h_clean.csv`. The full filter and exclusion
logic is documented in
[`docs/DUNE_EXTRACTION_BRIEF.md`](../../docs/DUNE_EXTRACTION_BRIEF.md).

Once `data/raw/` and `data/normalized/` are populated, the five build scripts in
`scripts/` consume them to construct the analysis panel in `data/econ/`.

API keys must be supplied via the environment; none are stored in this package.
