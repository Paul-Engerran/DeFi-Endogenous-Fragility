# When Volatility Masquerades as Fragility: DeFi Liquidations, Quantile Local Projections, and the Tails of Ethereum Returns

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/1169412894.svg)](https://doi.org/10.5281/zenodo.21230381)

Replication package for the working paper *"When Volatility Masquerades as
Fragility: DeFi Liquidations, Quantile Local Projections, and the Tails of
Ethereum Returns"* by **Paul Engerran**.

This README follows the [Social Science Data Editors template](https://social-science-data-editors.github.io/template_README/).
It documents the data, code, and exact steps that reproduce every number,
table, and figure in the paper.

---

## 1. Overview

The leverage-cycle literature predicts that forced deleveraging (liquidations)
should amplify *downside* tail risk. This package tests that prediction for
Ethereum using quantile local projections of hourly ETH returns on a DeFi
liquidation shock, over an hourly panel spanning 2021-03-15 → 2025-11-30 (UTC).

A naive specification appears to confirm the prediction: the response of the
lower return quantiles to a liquidation shock is larger than the median
response and deepens with the horizon. A battery of diagnostics overturns that
reading. A symmetric sign-flip placebo reproduces the entire left-versus-right
tail gap; the response is equally strong in the *upper* tail; the same
signature appears when Bitcoin is the outcome; and permutation nulls show the
long-horizon coefficient is genuine rather than a mechanical artefact of
overlapping cumulative returns. Net of realized volatility, an equivalence
test bounds the downside-*specific* asymmetry below an economically negligible
threshold. What survives is a **symmetric volatility channel**: liquidations
carry incremental out-of-sample predictive content for *both* moderate return
tails at short horizons, though not a substitute for a dedicated volatility model.
The portable methodological caution: *in quantile local projections on
overlapping cumulative returns, symmetric heteroskedasticity can masquerade as
downside fragility.*

The register is **descriptive and predictive, never causal**: at short
horizons the liquidation-return link is mechanically bidirectional, and the
paper makes no causal claim.

**Specification (locked).** Estimand: the quantile-LP coefficient
`β_h(τ)` of the conditional ETH return quantile on the liquidation shock.
Shock: `L_{t-1} = ln(1 + liq_usd_{t-1})`, the one-hour-lagged log USD
liquidation flow (an orthogonalized variant, residual on BTC returns, is used
only for the cross-asset placebo). Quantiles `τ ∈ {0.01, 0.05, 0.10, 0.50,
0.90, 0.95}` (a nine-quantile grid is reported in the appendix); horizons
`h ∈ {0, …, 24}`. Interaction `L_{t-1} × oi_high_t` carries the leverage-cycle
amplification hypothesis (it is a detectability **null**). Inference: kernel
standard errors at central quantiles; moving-block bootstrap (24-hour blocks,
1,000 replications, seed 42) at the tails.

> **Total compute.** From the shipped analysis panel and the two committed spot
> benchmarks, the full estimation +
> robustness + paper-exhibit regeneration runs in **~40 minutes on a 16-vCPU /
> 32 GB machine** (Path A below). A from-raw rebuild and the extended
> diagnostic battery are longer and optional (Paths B and C).

**Where to start.** Most readers want **Path A** in §6: install the
environment, then `make estimation && make robustness && make paper` from the
committed analysis panel in `data/econ/` plus the two committed spot benchmarks
in `data/normalized/spot/` — no raw-data download is required for Path A.

---

## 2. Data Availability and Provenance

### 2.1 Rights statement

All inputs are drawn from publicly accessible providers. The package
redistributes only **derived hourly aggregates** (the analysis panels and
result artefacts under `data/econ/`), never raw venue payloads. The author
certifies the right to redistribute these derived artefacts under the MIT
License (see [`LICENSE`](LICENSE)); the underlying data remain governed by each
provider's terms of use. A formal, journal-format data-availability statement
with full per-source detail is in [`docs/DATA_STATUS.md`](docs/DATA_STATUS.md).

### 2.2 What is in this package vs. what is regenerable

| | Contents | In this repository? |
|---|---|---|
| **Analysis panels + results** | `data/econ/`: two analysis-ready hourly panels plus ~75 result/robustness CSV+JSON artefacts | **Yes** (≈ 15 MB, committed) |
| **Normalized spot benchmarks** | `data/normalized/spot/xrp_ccdata_1h.parquet`, `…/doge_ccdata_1h.parquet`: only normalized inputs the analysis layer reads directly (cross-asset placebo, Test A) | **Yes** (~2.4 MB, committed) |
| **Other raw + normalized inputs** | `data/raw/`, rest of `data/normalized/` (cex, benchmarks, btc/eth spot): raw venue payloads and normalized parquets | **No**, GitHub Release asset **v2.0.0** (unzips at repo root into `data/raw/` + `data/normalized/`; see §6, Path B) |

Because the raw venue payloads and the bulk of the normalized inputs are not
committed, the from-raw panel-building targets (`make data`, `make all`,
`make smoke`) require the Release bundle first. Two normalized spot benchmarks
(`data/normalized/spot/{xrp,doge}_ccdata_1h.parquet`) are the exception: they
are committed because the cross-asset placebo (Test A) reads them directly.
Together with the shipped `data/econ/` panel they make the estimation,
robustness and paper layers (**Path A**) run directly from a fresh clone with no
bundle — those targets guard on the presence of the shipped panel rather than
rebuilding it.

### 2.3 Sources

| # | Source (role) | Provider | Access | Credential | Terms |
|---|---|---|---|---|---|
| 1 | DeFi liquidations (shock) | Dune Analytics, Spellbook `lending.borrow` / `lending.supply` | Re-run query **6912877** via the Dune web UI or API; SQL committed in [`dune_queries/`](dune_queries/) | Free Dune account | Open-source Spellbook; on-chain data public domain |
| 2 | ETH/BTC perpetuals (returns, OI, funding) | Bybit | Public REST `api.bybit.com` | None | Public market data |
| 3 | BTC/ETH/XRP/DOGE spot (benchmarks, placebos) | CCData (CryptoCompare) | `min-api.cryptocompare.com` | **Free API key** via env var `CCDATA_API_KEY` | Free-tier terms; redistribution of derived data permitted |
| 4 | ETH futures (cross-venue diagnostic only) | Binance | Public `fapi.binance.com` | None | Public market data |
| 5 | ETH spot (data-quality audit only) | Coinbase | Public `api.exchange.coinbase.com` | None | Public market data |

Sources 4 and 5 enter no econometric specification; they support a one-off
cross-venue diagnostic and a data-quality audit. The DeFi feed is restricted to
ETH-like collateral and excludes two exploit windows; full filter logic is in
[`dune_queries/liquidations_6912877.sql`](dune_queries/liquidations_6912877.sql)
and [`docs/DUNE_EXTRACTION_BRIEF.md`](docs/DUNE_EXTRACTION_BRIEF.md).

> **Access cost and limitations.** All sources are free. Bybit/Binance/Coinbase
> need no credentials; CCData needs a free-tier key. Dune Spellbook tables are
> community-maintained and may drift marginally as they are rebuilt, so a fresh
> re-extraction of query 6912877 is not guaranteed to be bit-identical to the
> committed panel; the access date is March 2026.

---

## 3. Dataset list

The committed analysis artefacts live in `data/econ/`. In addition to
`data/econ/`, two normalized spot benchmarks are committed under
`data/normalized/spot/` (`xrp/doge_ccdata_1h.parquet`) because the cross-asset
placebo (Test A) reads them directly. Column-level codebooks
for the two panels are in [`docs/DATA_STATUS.md`](docs/DATA_STATUS.md) §1 and
[`manuscript/appendix/C_data_construction.tex`](manuscript/appendix/C_data_construction.tex).

| File (in `data/econ/`) | Rows × cols | Provided? | Description |
|---|---|---|---|
| `econ_core_full_1h.parquet` | 41,328 × 27 | Yes | Analysis panel: returns, market-state controls, and the merged liquidation shock (`liq_usd_total`, `log_liq`, `log_liq_lag1`, `oi_high`, …) |
| `econ_core_predefi_1h.parquet` | 41,328 × 22 | Yes | Pre-merge panel (returns + market state, before the DeFi shock is joined) |
| `quantile_lp_results.csv` | varies | Yes | Main quantile-LP estimates `β_h(τ)`, `δ_h(τ)`, SEs |
| `robustness_*.csv`, `placebo_*.csv`, `exceedance_*.csv`, `pure_null_*.csv`, `oos_predictive.csv`, `mde_*.csv`, `skew_test.csv`, `size_ratio.csv`, `descriptive_stats.csv`, … | varies | Yes | ~75 result/robustness artefacts feeding the tables and figures (see §7) |

Raw venue payloads and most normalized parquets are **not** committed (see §2.2);
they are distributed as a GitHub Release asset and are required only for a
from-raw rebuild (Path B). The exception is the two normalized spot benchmarks
`data/normalized/spot/{xrp,doge}_ccdata_1h.parquet`, committed because the
cross-asset placebo (Test A) in Path A reads them directly.

---

## 4. Computational requirements

- **Language / interpreter.** Python 3.12 (validated on 3.12.2, macOS x86_64,
  and 3.12.3, Linux x86_64). Python 3.13+ is untested; Apple Silicon is
  untested.
- **Operating system.** Linux x86_64 (kernel 6.8, glibc 2.39) is the canonical
  platform. The committed `data/econ/` artefacts were produced there. macOS
  x86_64 reproduces panel construction and the main point estimates bit-for-bit;
  bootstrap statistics drift below one basis point across platforms at fixed
  `--seed 42` (BLAS / joblib worker variance) and never move a sign or shift a
  confidence interval across zero.
- **Dependencies.** Pinned in [`requirements.txt`](requirements.txt) (direct)
  and [`requirements-frozen.txt`](requirements-frozen.txt) (full transitive
  closure). Key pins: `pandas==2.2.2`, `numpy==1.26.4`, `pyarrow==14.0.2`,
  `statsmodels==0.14.2`, `scipy==1.13.1`, `matplotlib==3.10.9`, `joblib==1.4.2`,
  `arch==7.2.0`. No editable install is required (`pyproject.toml` is a minimal
  metadata file; scripts resolve `src/` and `config.py` on `sys.path`).
- **Randomness.** All stochastic steps are seeded; the canonical seed is **42**
  (`run_robustness_all.py --seed 42`). The estimation defaults to `--n_jobs 1`
  for bit-for-bit reproducibility.
- **Determinism policy.** `make reproduce` exports single-thread BLAS
  (`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`) and `PYTHONHASHSEED=0` for
  every step. On the canonical platform a fresh run is **byte-identical** for the
  deterministic numeric artefacts: the main estimation output
  (`data/econ/quantile_lp_results.csv`) is pinned by SHA-256 in
  [`tests/fingerprints.txt`](tests/fingerprints.txt) and enforced by the CI
  `determinism` job (two-pass) and by `make verify-exhibits`. Cross-platform,
  bootstrap/simulation statistics drift below one basis point (never moving a
  sign or shifting a CI across zero). The `*_meta.json` sidecars embed a
  `run_timestamp_utc` and are **provenance-only**, excluded from byte-identity
  checks. Integrity of the shipped data artefacts is checked by `make verify-data`
  against [`data/CHECKSUMS.sha256`](data/CHECKSUMS.sha256).
- **Memory / CPU / disk.** ≥ 16 GB RAM for estimation and figures; ≥ 32 GB
  recommended for the bootstrap battery at `--n_jobs -1`; 16 vCPU for the quoted
  wall times; ≈ 150 MB disk for code + committed artefacts.
- **Environment variable.** `CCDATA_API_KEY` is needed **only** for re-downloading
  raw spot data (Path B); it is read from the environment and never stored. Path A
  needs no API key.
- **Network.** Required only to install dependencies and, for Path B, to
  re-acquire raw data.

---

## 5. Description of programs and code

```
DeFi-Endogenous-Fragility/
├── README.md                  this file
├── LICENSE                    MIT
├── CITATION.cff               machine-readable citation metadata
├── Makefile                   build orchestration (see §6)
├── config.py                  single source of truth (paths, parameters)
├── pyproject.toml             project metadata (no editable install needed)
├── requirements.txt           pinned direct dependencies
├── requirements-frozen.txt    full transitive closure (audit trail)
├── RUN_VM.md                  guide for the canonical diagnostic battery on a VM
│
├── docs/
│   ├── DATA_STATUS.md         data-availability statement, per-source detail, codebook
│   └── DUNE_EXTRACTION_BRIEF.md   methodological appendix on the DeFi feed
│
├── dune_queries/
│   └── liquidations_6912877.sql   provenance SQL for the DeFi liquidation feed
│
├── src/                       shared primitives
│   ├── io.py                  parquet loaders
│   ├── estimation.py          design-matrix builders, control sets, QR kernel kwargs
│   └── bootstrap.py           moving-block bootstrap engine
│
├── scripts/                   the build pipeline (run in order)
│   ├── run_data_prep.py       master calendar + cross-venue diagnostics
│   ├── run_core_panel.py      → data/econ/econ_core_predefi_1h.parquet
│   ├── run_defi_merge.py      → data/econ/econ_core_full_1h.parquet
│   ├── run_quantile_lp.py     main estimation → quantile_lp_results.{csv,parquet}
│   ├── run_robustness_all.py  robustness battery (Tests A to N) → robustness_*.csv
│   ├── add_bonferroni.py   appends family-wise Bonferroni columns
│   ├── paper/                 paper layer (formatting only, no statistics)
│   │   ├── make_figures.py    → paper/figures/*.pdf
│   │   ├── make_tables.py     → paper/tables/*.tex
│   │   └── make_numbers.py    → paper/numbers.tex (the macros the manuscript cites)
│   ├── aux/                   auxiliary estimators feeding individual exhibits
│   └── data_download/         notes for re-acquiring raw data (Path B)
│
├── notebooks/                 8 read-side report notebooks (consume pipeline outputs)
├── tests/                     pytest suite (unit + determinism checks)
│
├── data/econ/                 committed analysis panels + result artefacts (§3)
├── data/normalized/spot/      two committed CCData spot parquets (XRP, DOGE) — Test A
│   ├── xrp_ccdata_1h.parquet
│   └── doge_ccdata_1h.parquet
│
├── paper/
│   ├── figures/               8 generated PDF figures
│   ├── tables/                8 generated .tex table fragments
│   └── numbers.tex            generated in-text macros
│
└── manuscript/                LaTeX source of the paper
    ├── main.tex, preamble.tex
    ├── sections/              00_abstract … 10_conclusion
    ├── appendix/              A_robustness, B_inference, C_data_construction
    └── references.bib
```

**Execution order.** Panel construction (run_data_prep → run_core_panel →
run_defi_merge) is **Path B only** and rebuilds `data/econ/` from the
raw/normalized bundle. The analysis layer — run_quantile_lp → run_robustness_all
→ `scripts/paper/` — is **Path A**: its Make targets guard on the *existence* of
the shipped `data/econ/` panel and do NOT trigger panel construction, so
`make estimation && make robustness && make paper` runs from a fresh clone with
no raw data. Path B targets (`make data`/`make all`) remain file-based; use
`make clean` to force a from-raw rebuild. Each script's full CLI is in its module
docstring (`python scripts/<name>.py --help`).

> **Note.** `manuscript/references.bib` is shipped pre-generated and is treated
> as a static artefact; it is not regenerated by the replication pipeline.

---

## 6. Instructions to Replicators

### Install the environment

```bash
git clone https://github.com/Paul-Engerran/DeFi-Endogenous-Fragility.git
cd DeFi-Endogenous-Fragility
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Path A: reproduce from the shipped analysis panel (recommended)

Everything here runs from artefacts committed to the repo — the `data/econ/`
panel plus the two committed spot parquets in `data/normalized/spot/` (xrp/doge)
feeding the cross-asset placebo (Test A). No raw data, no Release bundle, no
credentials.

```bash
# (optional) run the test suite (data-independent tests run anywhere)
pytest

# 1. Main estimation: quantile-LP coefficients         (~5 min, --n_jobs 1)
make estimation        # → data/econ/quantile_lp_results.{csv,parquet}

# 2. Robustness battery (--tests all, Tests A to N)        (~30 min on 16 vCPU)
make robustness        # → data/econ/robustness_*.csv, se_comparison_*, quantile_*_fast.csv, …

# 3. Paper layer: figures, table fragments, numbers.tex (~2 min)
make paper             # → paper/figures/*.pdf, paper/tables/*.tex, paper/numbers.tex
```

The `placebo_*` / `exceedance_*` diagnostics and the other auxiliary artefacts
consumed by Tables 3–6 and printed Figures 4–8 ship pre-computed in
`data/econ/` and are regenerated only by the multi-hour `make canonical` target.

The manuscript consumes `paper/figures/`, `paper/tables/`, and
`paper/numbers.tex` by relative path; no number is typed by hand. Build the PDF
by compiling `manuscript/main.tex` with a LaTeX engine of your choice
(`make pdf` autodetects latexmk or Tectonic; the committed `manuscript/main.pdf`
was compiled with **Tectonic 0.15.0**).

### Path B: full rebuild from raw data (optional)

This regenerates the analysis panel itself. It requires the raw+normalized
bundle published as GitHub Release **v2.0.0** (the v2.0.0 archive unzips at the
repository root and populates `data/raw/` and `data/normalized/` at the exact
paths `config.py` expects — do not use the older v1.0 asset, whose internal
layout differs). Optionally set a free `CCDATA_API_KEY` only if you prefer to
re-download spot data from CCData instead of using the bundle.

```bash
export CCDATA_API_KEY=...        # only if re-downloading spot from CCData
make smoke                       # Path B only; depends on `make data`, so needs the v2.0.0 bundle; quick end-to-end check on a scratch dir (~1 min)
make all                         # data → estimation → robustness → figures (~35 min, 16 vCPU)
```

### Path C: canonical diagnostic battery (optional, VM)

The extended diagnostic battery (`make canonical`) is intended for a dedicated
VM and takes several hours. See [`RUN_VM.md`](RUN_VM.md) for the procedure and
wall-time ranges.

### Smaller machines

The bootstrap battery is the binding constraint. Lower `--n_jobs` and
`--batch_size`, e.g. on 8 vCPU:

```bash
make estimation
python scripts/run_robustness_all.py --tests all --n_boot 1000 --n_jobs 8
make paper
```

The bootstrap engine checkpoints per batch (`data/econ/_robust_ckpt/`); a run
that is interrupted resumes without recomputing completed batches.

---

## 7. List of figures, tables, and programs

Every exhibit is regenerated by the paper layer (`make paper`). Figures are
written by `scripts/paper/make_figures.py` (one function per figure) and tables
by `scripts/paper/make_tables.py` (booktabs fragments wrapped by the
manuscript). In-text numbers come from `scripts/paper/make_numbers.py` →
`paper/numbers.tex`. Figure file names (`fig1`…`fig9`, the set skips `fig6`)
are stable artifact identifiers; the numbers LaTeX prints in the compiled paper
follow order of appearance and differ for the last four. The table below is
ordered by the printed number.

| Exhibit | Generator (`script::function`) | Input data (`data/econ/…`) | Manuscript section |
|---|---|---|---|
| Fig. 1, liquidations time series | `make_figures.py::fig1_liquidations_timeseries` | `econ_core_full_1h.parquet` | §2 Data |
| Fig. 2, return distribution | `make_figures.py::fig2_return_distribution` | `econ_core_full_1h.parquet` | §3 Stylized facts |
| Fig. 3, quantile-LP IRF | `make_figures.py::fig3_qlp_irf` | `quantile_lp_results.csv`, `robustness_bootstrap_nb07_spec_fast.csv` | §5 The apparent result |
| Fig. 4, placebo gap distribution | `make_figures.py::fig4_placebo_gap_distribution` | `placebo_symmetric_draws.csv`, `placebo_symmetric.csv` | §6 Deconstruction |
| Fig. 5, dual permutation nulls (file `fig9`) | `make_figures.py::fig9_pure_null_dual` | `pure_null_circular_shift_by_horizon.csv`, `pure_null_innov_shuffle_by_horizon.csv` | §6 Deconstruction |
| Fig. 6, BTC vs. ETH (file `fig8`) | `make_figures.py::fig8_btc_vs_eth` | `btc_vs_eth_profile.csv` | §6 Deconstruction |
| Fig. 7, exceedance symmetry (file `fig5`) | `make_figures.py::fig5_exceedance_symmetry` | `exceedance_results.csv`, `exceedance_paired.csv`, `exceedance_paired_cumulative.csv` | §7 What survives |
| Fig. 8, equivalence (MDE) (file `fig7`) | `make_figures.py::fig7_mde_equivalence` | `exceedance_paired.csv`, `skew_test.csv`, `mde_equivalence.csv` | §7 What survives |
| Table 1, descriptives | `make_tables.py::tab1_descriptives` | `descriptive_stats.csv` | §2 Data |
| Table 2, main quantile-LP | `make_tables.py::tab2_qlp_main` | `quantile_lp_results.csv` | §5 The apparent result |
| Table 3, subsample stability | `make_tables.py::tab3_subsample` | `subsample_stability.csv` | §7 What survives |
| Table 4, exceedance / skew / MDE | `make_tables.py::tab4_exceedance_skew_mde` | `exceedance_paired.csv`, `skew_test.csv`, `mde_equivalence.csv` | §7 What survives |
| Table 5, size ratio | `make_tables.py::tab5_size_ratio` | `size_ratio.csv` | §8 Mechanisms |
| Table 6, out-of-sample | `make_tables.py::tab6_oos` | `oos_predictive.csv` | §7 What survives |
| Table A1, block sensitivity | `make_tables.py::tabA1_block_sensitivity` | `block_sensitivity.csv` | App. B Inference |
| Table A2, SE ratio | `make_tables.py::tabA2_se_ratio` | `se_ratio_nb07.csv` | App. B Inference |

The robustness inputs above (`robustness_*`, `placebo_*`, `exceedance_*`,
`pure_null_*`, `oos_predictive`, `mde_*`, `skew_test`, `size_ratio`,
`subsample_stability`, `block_sensitivity`, `se_ratio_nb07`) are produced by
`run_robustness_all.py --tests all` and the `scripts/aux/` estimators; the
descriptive panel comes from `scripts/aux/run_descriptive_stats.py`. The
cross-asset placebo (Test A) additionally reads the two committed spot parquets
`data/normalized/spot/{xrp,doge}_ccdata_1h.parquet`; these are the only
normalized inputs the analysis layer reads directly (see §2.2/§5). An
extended nine-quantile grid is available via
`run_quantile_lp.py --quantiles 0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99
--out_dir /tmp/qlp_9q` (the `--out_dir` matters: without it the run would
overwrite the committed main-results CSV; the committed `*_9q.csv` files are
that run's outputs renamed).

---

## 8. References

Data providers: Bybit, Binance, Coinbase, CCData (CryptoCompare), and the Dune
Analytics Spellbook contributors (`lending.borrow` / `lending.supply`; DeFi
liquidations across EVM chains, query ID 6912877). Literature is cited in the
manuscript ([`manuscript/references.bib`](manuscript/references.bib)).

### How to cite

Citation metadata is in [`CITATION.cff`](CITATION.cff). BibTeX:

```bibtex
@unpublished{engerran2026defi,
  author = {Engerran, Paul},
  title  = {When Volatility Masquerades as Fragility: DeFi Liquidations,
            Quantile Local Projections, and the Tails of Ethereum Returns},
  year   = {2026},
  note   = {Working paper. Replication package:
            \url{https://github.com/Paul-Engerran/DeFi-Endogenous-Fragility}}
}
```

---

## Use of generative AI

Large language models (Anthropic Claude, Opus and Fable generations, 2026),
used under the author's direction through an agentic coding interface,
assisted throughout this project: literature and bibliography work; proposal
of diagnostic and robustness methods and of candidate interpretations;
analysis and packaging code; execution of the author's deterministic
pipeline; the verification and reproducibility infrastructure; and targeted
manuscript passages. The research question, the locked specification, the
data construction, and all modelling and interpretive decisions are the
author's, and no reported quantity was estimated by a model. Every reported
number regenerates from the deposited data with `make reproduce`, and
`make check-numbers` confirms that all in-text numerical macros are
unchanged, so correctness rests on reproduction rather than on trust in the
assistance. Citations were checked against primary sources; several
fabricated or mis-attributed references were caught and removed during
verification. The author reviewed and approved all content and assumes sole
responsibility. The full declaration appears in the manuscript, in the
unnumbered section before the references.

---

## License and contact

Code, documentation, generated figures, and processed data artefacts are
released under the MIT License ([`LICENSE`](LICENSE)). Raw data remain governed
by their providers' terms (§2). For questions or reproduction issues, please
open an issue on the
[GitHub repository](https://github.com/Paul-Engerran/DeFi-Endogenous-Fragility).
