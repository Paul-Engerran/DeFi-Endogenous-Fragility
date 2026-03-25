"""
config.py — Single source of truth for the project.
Place this file at the ROOT of Research_paper_leverage/.

Usage in any notebook:
    import sys; sys.path.insert(0, "..")   # adjust depth as needed
    from config import CFG
    CFG.ensure_dirs()
"""

from pathlib import Path
import pandas as pd

# ──────────────────────────────────────────────────────────────
# 1. PROJECT ROOT  (this file lives at root)
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────
# 2. TEMPORAL SCOPE
# ──────────────────────────────────────────────────────────────
START_UTC    = pd.Timestamp("2021-03-15T00:00:00Z")
END_UTC_EXCL = pd.Timestamp("2025-12-01T00:00:00Z")
FREQ         = "1h"
# Convention: [start, end_excl)  — timestamp = bucket-start

# ──────────────────────────────────────────────────────────────
# 3. DIRECTORY TREE
# ──────────────────────────────────────────────────────────────
DATA_DIR          = PROJECT_ROOT / "data"

# --- Raw (immutable after download) ---
RAW_DIR           = DATA_DIR / "raw"
RAW_CEX_BYBIT     = RAW_DIR / "cex" / "bybit"
RAW_CEX_BINANCE   = RAW_DIR / "cex" / "binance"
RAW_BENCHMARKS    = RAW_DIR / "benchmarks" / "coinbase"
RAW_DEFI          = RAW_DIR / "defi"

# --- Normalized (cleaned, hourly, parquet) ---
NORM_DIR          = DATA_DIR / "normalized"
NORM_CEX_BYBIT    = NORM_DIR / "cex" / "bybit"
NORM_CEX_BINANCE  = NORM_DIR / "cex" / "binance"
NORM_BENCHMARKS   = NORM_DIR / "benchmarks" / "coinbase"
NORM_SPOT         = NORM_DIR / "spot"
NORM_DEFI         = NORM_DIR / "defi"

# --- Analysis (single consolidated output) ---
ANALYSIS_DIR      = DATA_DIR / "analysis"
WINDOWS_DIR       = ANALYSIS_DIR / "windows"
DATASETS_DIR      = ANALYSIS_DIR / "datasets"
REPORTS_DIR       = ANALYSIS_DIR / "reports"

# --- Econometric-ready ---
ECON_DIR          = DATA_DIR / "econ"

# --- Other ---
NOTEBOOKS_DIR     = PROJECT_ROOT / "notebooks"
DUNE_QUERIES_DIR  = PROJECT_ROOT / "dune_queries"
PAPER_DIR         = PROJECT_ROOT / "paper"

# ──────────────────────────────────────────────────────────────
# 4. FILE PATHS  (all normalized inputs)
# ──────────────────────────────────────────────────────────────
class FILES:
    # Bybit perp (primary venue)
    bybit_klines      = NORM_CEX_BYBIT / "klines_1h.parquet"
    bybit_funding     = NORM_CEX_BYBIT / "funding_1h.parquet"
    bybit_oi          = NORM_CEX_BYBIT / "open_interest_1h.parquet"

    # Binance futures (secondary / diagnostic)
    binance_futures   = NORM_CEX_BINANCE / "binance_futures_ethusdt_1h_normalized.parquet"

    # Spot benchmarks (CCData CCCAGG)
    btc_spot          = NORM_SPOT / "btc_ccdata_1h.parquet"
    eth_spot          = NORM_SPOT / "eth_ccdata_1h.parquet"

    # Coinbase benchmark
    coinbase_candles  = NORM_BENCHMARKS / "candles_repaired.parquet"

    # DeFi liquidations (to be produced)
    defi_liq          = NORM_DEFI / "liquidations_1h.parquet"

    # --- Analysis outputs ---
    master_calendar   = WINDOWS_DIR / "master_calendar_1h.parquet"
    window_metadata   = WINDOWS_DIR / "window_metadata.json"
    cex_diagnostics   = DATASETS_DIR / "cex_diagnostics_1h.parquet"

    # --- Econometric datasets ---
    econ_core_predefi = ECON_DIR / "econ_core_predefi_1h.parquet"
    econ_core_full    = ECON_DIR / "econ_core_full_1h.parquet"

# ──────────────────────────────────────────────────────────────
# 5. ECONOMETRIC PARAMETERS
# ──────────────────────────────────────────────────────────────
class ECON:
    # VAR (linear benchmark)
    var_max_lags       = 48
    var_ordering       = ["ret", "log_liq"]

    # Quantile local projections (main specification)
    quantiles          = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    lp_horizons        = list(range(0, 25))   # h = 0..24 hours
    lp_n_boot          = 1000

    # Regime thresholds
    stress_pctile      = 95       # liquidation volume
    high_oi_pctile     = 80       # OI crowding regime

    # Transformations
    log_offset         = 1.0      # ln(1 + L_t)
    vol_window         = 168      # 7-day rolling vol (hours)

    # Inference
    nw_lags            = 12       # Newey-West HAC lags
    block_boot_size    = 24       # block bootstrap block length

# ──────────────────────────────────────────────────────────────
# 6. SYMBOLS
# ──────────────────────────────────────────────────────────────
PRIMARY_SYMBOL     = "ETHUSDT"
PRIMARY_VENUE      = "bybit"
BENCHMARK_ASSET    = "BTC"

# ──────────────────────────────────────────────────────────────
# 7. HELPERS
# ──────────────────────────────────────────────────────────────
def ensure_dirs():
    """Create all output directories. Safe to call repeatedly."""
    for d in [
        RAW_CEX_BYBIT, RAW_CEX_BINANCE, RAW_BENCHMARKS, RAW_DEFI,
        NORM_CEX_BYBIT, NORM_CEX_BINANCE, NORM_BENCHMARKS, NORM_SPOT, NORM_DEFI,
        WINDOWS_DIR, DATASETS_DIR, REPORTS_DIR, ECON_DIR,
        DUNE_QUERIES_DIR, PAPER_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 8. CONVENIENCE OBJECT
# ──────────────────────────────────────────────────────────────
class _Cfg:
    ROOT        = PROJECT_ROOT
    START       = START_UTC
    END_EXCL    = END_UTC_EXCL
    FREQ        = FREQ
    FILES       = FILES
    ECON        = ECON
    ensure_dirs = staticmethod(ensure_dirs)

    def __repr__(self):
        return f"Config(root={self.ROOT}, window=[{self.START}, {self.END_EXCL}))"

CFG = _Cfg()

if __name__ == "__main__":
    ensure_dirs()
    print(CFG)
    print("✅ All directories created.")
