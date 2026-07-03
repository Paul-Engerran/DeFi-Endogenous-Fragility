#!/usr/bin/env python3
"""
run_size_ratio.py  —  [AUXILIARY SIZE EVIDENCE — does NOT change the main spec]

The SIZE-RATIO evidence backing the claim:

    "DeFi ETH liquidation flow is << 1% of ETH market turnover, hence too
     small to drive the aggregate downside."

This script puts the two sides of that inequality on the same scale.

NUMERATOR  (OUR DATA, hard fact)
────────────────────────────────
From data/econ/econ_core_full_1h.parquet, column `liq_usd_total` (USD value of
DeFi liquidations per hour, the basis of the `log_liq` shock). We resample the
hourly series to calendar days and report:
  - total cumulative liquidation $ over the sample,
  - mean and median DAILY $,
  - max single-day $,
  - worst rolling-7d $ (calendar, 7-day trailing sum of daily totals),
  - % of nonzero hours.
This is the on-chain DeFi-ETH liquidation flow the paper studies. It is OUR
measured quantity — not an estimate.

DENOMINATOR  (LITERATURE / INDUSTRY ESTIMATES, flagged)
───────────────────────────────────────────────────────
ETH spot + perpetual-swap turnover, and CEX-perp single-day cascade
magnitudes, taken from verified literature and industry sources.

  *** WARNING ***  These denominators are INDUSTRY / CEX-REPORTED ESTIMATES.
  They are partly wash-trade-inflated (Gan et al. 2022, ACM AFT, on flash-loan
  wash trading) and are NOT peer-reviewed point estimates. They are reported
  here only to fix the ORDER OF MAGNITUDE of the inequality. The asymmetry we
  rely on (numerator << denominator) is so large — three to four orders of
  magnitude — that it is robust to any plausible re-scaling of the denominator,
  including deflating reported volume for wash trading. Treat every
  denominator below as an order-of-magnitude bound, not a precise figure. The
  numerator is the only hard number here.

Turnover anchors (industry estimates):
  - CEX SPOT ETH turnover ~ $3-6 T / yr            -> midpoint $4.5 T/yr.
  - PRIMARY perp anchor = OI-weighted: top-10 perp notional $58.5 T in 2024 x
    ETH OI-share 21% = $12.285 T/yr (implied perp:spot ~2.7x, consistent with
    the measured market ratio). This is the conservative (smaller) denominator,
    i.e. the choice that makes the DeFi share look LARGEST.
  - Spot+perp ETH (PRIMARY) = $4.5 T + $12.285 T = $16.785 T/yr.
  - Daily turnover = annual / 365.
  - ALTERNATIVE perp anchor (cross-check row, coarser basis): perp = 7 x spot
    midpoint = $31.5 T/yr -> spot+perp = $36 T/yr. Retained as a robustness row.

Cascade magnitudes (single-day CEX-perp liquidation events, industry estimates):
  - $1.4 B  Black Thursday (12 Mar 2020)
  - $8.6 B  May 2021 cascade
  - $19  B  10 Oct 2025 (largest in history)

COMPUTE
───────
  ratio_pct = numerator_defi_usd / denominator_eth_usd * 100
For turnover rows the denominator is the DAILY turnover (annual / 365), matched
to the corresponding daily numerator (mean-day, max-day). For the worst-7d
numerator the denominator is 7 x daily turnover (a 7-day turnover window). For
the cascade rows the denominator is the single-day cascade magnitude itself and
the "ratio" is our DeFi flow as a fraction of that one CEX-perp event.

ASSUMPTIONS (stated explicitly; see meta JSON)
  A. Spot midpoint = $4.5 T/yr (mid of the $3-6 T/yr range).
  B. Perp = OI-weighted $12.285 T/yr (PRIMARY, implied ~2.7x spot); 7x spot is
     the ALT cross-check. 365-day year for annual->daily.
  C. Denominators are total-ETH-market figures (spot+perp), NOT DeFi-only; the
     comparison is deliberately "DeFi liq vs the WHOLE ETH market".

CLI
───
    .venv/bin/python scripts/aux/run_size_ratio.py
    # options: --spot_lo/--spot_hi/--spot_mid, --perp_mult, --out_dir

OUTPUT
──────
  data/econ/size_ratio.csv
      columns: metric, numerator_defi_usd, denominator_eth_usd, ratio_pct,
               source
  data/econ/size_ratio_meta.json
      run provenance + every assumption + the WARNING on the denominators.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/aux/ -> ROOT is parents[2]
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import ECON_DIR, PROJECT_ROOT  # noqa: E402

PARQUET = Path(ECON_DIR) / "econ_core_full_1h.parquet"
LIQ_COL = "liq_usd_total"
DATE_COL = "date"

# ── Denominator constants (INDUSTRY ESTIMATES — see module docstring) ──
TRILLION = 1_000_000_000_000.0
BILLION = 1_000_000_000.0
DAYS_PER_YEAR = 365.0

# Source tags (kept short; full provenance in meta JSON)
SRC_OURS = "OUR DATA: econ_core_full_1h.parquet[liq_usd_total]"
SRC_TURN = ("INDUSTRY EST (PRIMARY): spot $3-6T/yr (mid $4.5T), perp OI-weighted "
            "(top-10 $58.5T x ETH OI-share 21% = $12.285T/yr); order-of-magnitude")
SRC_TURN_X = ("INDUSTRY EST ALT: perp = 7x spot ($31.5T/yr), coarser "
              "spot-multiple convention; wash-inflated, not peer-reviewed")
SRC_BT = "INDUSTRY EST: Black Thursday 12-Mar-2020 ~$1.4B CEX-perp cascade"
SRC_MAY = "INDUSTRY EST: May-2021 ~$8.6B CEX-perp cascade"
SRC_OCT = "INDUSTRY EST: 10-Oct-2025 ~$19B CEX-perp cascade (largest)"

OUT_COLS = ["metric", "numerator_defi_usd", "denominator_eth_usd",
            "ratio_pct", "source"]


def compute_numerator(parquet: Path) -> dict:
    """Read OUR hourly liq_usd_total; return the DeFi-side numerator stats."""
    df = pd.read_parquet(parquet, columns=[DATE_COL, LIQ_COL])
    s = df.set_index(DATE_COL).sort_index()[LIQ_COL].astype(float)

    if s.isna().any():
        raise ValueError(f"{LIQ_COL} contains NaNs; expected none.")
    if (s < 0).any():
        raise ValueError(f"{LIQ_COL} contains negatives; expected none.")

    daily = s.resample("1D").sum()
    roll7 = daily.rolling(7).sum()  # trailing 7-calendar-day sum

    worst7_end = roll7.idxmax()
    max_day = daily.idxmax()

    return {
        "total_usd": float(s.sum()),
        "mean_daily_usd": float(daily.mean()),
        "median_daily_usd": float(daily.median()),
        "max_day_usd": float(daily.max()),
        "max_day_date": str(max_day.date()),
        "worst_7d_usd": float(roll7.max()),
        "worst_7d_end_date": str(worst7_end.date()),
        "pct_nonzero_hours": float(100.0 * (s > 0).mean()),
        "n_hours": int(len(s)),
        "n_days": int(len(daily)),
        "n_nonzero_days": int((daily > 0).sum()),
        "date_min": str(s.index.min()),
        "date_max": str(s.index.max()),
        "span_days": float((s.index.max() - s.index.min()).total_seconds()
                           / 86400.0),
    }


def compute_denominator(spot_mid: float, perp_mult: float) -> dict:
    """ETH turnover denominators (INDUSTRY ESTIMATES). All in USD.

    PRIMARY (headline): OI-weighted perp = top-10 perp $58.5T/yr x ETH
    open-interest share 21% = $12.285T/yr. Implied perp:spot ~ 2.7x, consistent
    with the measured market ratio (He, Manela, Ross & von Wachter 2023;
    CoinGecko perpetuals report 2024). This is the CONSERVATIVE (smaller)
    denominator, i.e. the choice that makes the DeFi share look LARGEST.
    ALTERNATIVE (cross-check row): perp = perp_mult x spot (default 7x), the
    coarser spot-multiple convention, retained as a robustness row.
    """
    spot_yr = spot_mid * TRILLION
    # PRIMARY: OI-weighted perp (top-10 perp $58.5T x ETH OI share 21%)
    perp_yr_primary = 58.5 * TRILLION * 0.21
    spotperp_yr = spot_yr + perp_yr_primary
    spotperp_day = spotperp_yr / DAYS_PER_YEAR

    # ALTERNATIVE: perp = perp_mult x spot (coarser spot-multiple convention)
    perp_yr_alt = perp_mult * spot_yr
    spotperp_yr_alt = spot_yr + perp_yr_alt
    spotperp_day_alt = spotperp_yr_alt / DAYS_PER_YEAR

    return {
        "spot_yr": spot_yr,
        "perp_yr_primary": perp_yr_primary,
        "spotperp_yr_primary": spotperp_yr,
        "spotperp_day_primary": spotperp_day,
        "perp_yr_alt": perp_yr_alt,
        "spotperp_yr_alt": spotperp_yr_alt,
        "spotperp_day_alt": spotperp_day_alt,
        "cascade_black_thursday": 1.4 * BILLION,
        "cascade_may_2021": 8.6 * BILLION,
        "cascade_oct_2025": 19.0 * BILLION,
    }


def build_rows(num: dict, den: dict) -> pd.DataFrame:
    """Assemble the [metric, numerator, denominator, ratio_pct, source] table."""
    day = den["spotperp_day_primary"]
    week = 7.0 * day
    day_alt = den["spotperp_day_alt"]

    def pct(n: float, d: float) -> float:
        return float(n / d * 100.0)

    rows = [
        # ── turnover ratios (PRIMARY denominator: spot+perp, 7x spot) ──
        {
            "metric": "mean_daily_defi_liq_vs_daily_eth_spot_perp_turnover",
            "numerator_defi_usd": num["mean_daily_usd"],
            "denominator_eth_usd": day,
            "ratio_pct": pct(num["mean_daily_usd"], day),
            "source": SRC_TURN,
        },
        {
            "metric": "median_daily_defi_liq_vs_daily_eth_spot_perp_turnover",
            "numerator_defi_usd": num["median_daily_usd"],
            "denominator_eth_usd": day,
            "ratio_pct": pct(num["median_daily_usd"], day),
            "source": SRC_TURN,
        },
        {
            "metric": "max_day_defi_liq_vs_daily_eth_spot_perp_turnover",
            "numerator_defi_usd": num["max_day_usd"],
            "denominator_eth_usd": day,
            "ratio_pct": pct(num["max_day_usd"], day),
            "source": SRC_TURN,
        },
        {
            "metric": "worst_7d_defi_liq_vs_7d_eth_spot_perp_turnover",
            "numerator_defi_usd": num["worst_7d_usd"],
            "denominator_eth_usd": week,
            "ratio_pct": pct(num["worst_7d_usd"], week),
            "source": SRC_TURN,
        },
        # ── ALTERNATIVE (cross-check) turnover ratio with the 7x-spot perp
        #    denominator (coarser convention, larger denominator, smaller share) ──
        {
            "metric": "max_day_defi_liq_vs_daily_eth_spot_perp_turnover_ALT_7XSPOT",
            "numerator_defi_usd": num["max_day_usd"],
            "denominator_eth_usd": day_alt,
            "ratio_pct": pct(num["max_day_usd"], day_alt),
            "source": SRC_TURN_X,
        },
        # ── TYPICAL DeFi daily flow vs largest CEX-perp cascade ──
        #    (this is the row that substantiates the "3-4 orders of
        #     magnitude smaller" statement: typical daily DeFi flow, not the
        #     one outlier day, against a single CEX-perp cascade event)
        {
            "metric": "mean_daily_defi_liq_vs_oct2025_cascade",
            "numerator_defi_usd": num["mean_daily_usd"],
            "denominator_eth_usd": den["cascade_oct_2025"],
            "ratio_pct": pct(num["mean_daily_usd"], den["cascade_oct_2025"]),
            "source": SRC_OCT,
        },
        {
            "metric": "median_daily_defi_liq_vs_oct2025_cascade",
            "numerator_defi_usd": num["median_daily_usd"],
            "denominator_eth_usd": den["cascade_oct_2025"],
            "ratio_pct": pct(num["median_daily_usd"], den["cascade_oct_2025"]),
            "source": SRC_OCT,
        },
        # ── DeFi flow vs single-day CEX-perp cascades (worst day) ──
        {
            "metric": "max_day_defi_liq_vs_black_thursday_cascade",
            "numerator_defi_usd": num["max_day_usd"],
            "denominator_eth_usd": den["cascade_black_thursday"],
            "ratio_pct": pct(num["max_day_usd"], den["cascade_black_thursday"]),
            "source": SRC_BT,
        },
        {
            "metric": "max_day_defi_liq_vs_may2021_cascade",
            "numerator_defi_usd": num["max_day_usd"],
            "denominator_eth_usd": den["cascade_may_2021"],
            "ratio_pct": pct(num["max_day_usd"], den["cascade_may_2021"]),
            "source": SRC_MAY,
        },
        {
            "metric": "max_day_defi_liq_vs_oct2025_cascade",
            "numerator_defi_usd": num["max_day_usd"],
            "denominator_eth_usd": den["cascade_oct_2025"],
            "ratio_pct": pct(num["max_day_usd"], den["cascade_oct_2025"]),
            "source": SRC_OCT,
        },
        # ── total cumulative DeFi flow vs single largest CEX-perp event ──
        {
            "metric": "total_sample_defi_liq_vs_oct2025_single_day_cascade",
            "numerator_defi_usd": num["total_usd"],
            "denominator_eth_usd": den["cascade_oct_2025"],
            "ratio_pct": pct(num["total_usd"], den["cascade_oct_2025"]),
            "source": SRC_OCT,
        },
    ]
    df = pd.DataFrame(rows)[OUT_COLS]
    return df


def build_meta(num: dict, den: dict, args: argparse.Namespace) -> dict:
    return {
        "purpose": ("Size-ratio evidence backing A1/A6: DeFi ETH liquidation "
                    "flow is <<1% of ETH market turnover."),
        "numerator": {
            "source": SRC_OURS,
            "column": LIQ_COL,
            "note": "OUR measured data, hard fact (not an estimate).",
            **num,
        },
        "denominator": {
            "WARNING": (
                "INDUSTRY / CEX-REPORTED ESTIMATES, partly wash-trade-inflated "
                "(Gan et al. 2022, ACM AFT), NOT peer-reviewed. Order-of-"
                "magnitude bounds only. The numerator<<denominator inequality "
                "(3-4 orders of magnitude) is robust to any plausible "
                "re-scaling, including deflating for wash trading."
            ),
            "sources": "Industry and practitioner estimates (CEX-perp cascade magnitudes; ETH spot/perp turnover); order-of-magnitude bounds only.",
            "assumptions": {
                "spot_range_usd_per_yr": [args.spot_lo * TRILLION,
                                          args.spot_hi * TRILLION],
                "spot_midpoint_usd_per_yr": args.spot_mid * TRILLION,
                "perp_basis_PRIMARY": ("OI-weighted: top-10 perp $58.5T 2024 x "
                                       "ETH OI-share 21% = $12.285T/yr "
                                       "(implied perp:spot ~2.7x)"),
                "perp_multiple_of_spot_ALT": args.perp_mult,
                "days_per_year": DAYS_PER_YEAR,
                "denominators_are_total_eth_market_spot_plus_perp": True,
                "note": ("PRIMARY is the OI-weighted (smaller) denominator, the "
                         "conservative choice that maximises the DeFi share; the "
                         "7x-spot convention is retained as the ALT cross-check row."),
            },
            **den,
        },
        "run": {
            "parquet": str(PARQUET),
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    ap.add_argument("--spot_lo", type=float, default=3.0,
                    help="Low end of CEX spot ETH turnover, $T/yr. Default 3.")
    ap.add_argument("--spot_hi", type=float, default=6.0,
                    help="High end of CEX spot ETH turnover, $T/yr. Default 6.")
    ap.add_argument("--spot_mid", type=float, default=4.5,
                    help="Spot midpoint, $T/yr. Default 4.5 (mid of 3-6).")
    ap.add_argument("--perp_mult", type=float, default=7.0,
                    help="Perp turnover as a multiple of spot. Default 7.")
    ap.add_argument("--out_dir", type=Path, default=Path(ECON_DIR))
    args = ap.parse_args()

    print("run_size_ratio: SIZE-RATIO evidence (A1/A6)", flush=True)
    print(f"  numerator parquet: {PARQUET}", flush=True)
    print(f"  spot mid=${args.spot_mid}T/yr  perp={args.perp_mult}x spot "
          f"(PRIMARY)  365d/yr", flush=True)

    if not PARQUET.exists():
        ap.error(f"numerator parquet not found: {PARQUET}")

    num = compute_numerator(PARQUET)
    den = compute_denominator(args.spot_mid, args.perp_mult)
    df = build_rows(num, den)
    meta = build_meta(num, den, args)

    # ── console summary ──
    print("\nNUMERATOR (OUR DATA):", flush=True)
    print(f"  sample            {num['date_min']} -> {num['date_max']}  "
          f"({num['span_days']:.0f}d, {num['n_days']} cal-days)", flush=True)
    print(f"  total cumulative  ${num['total_usd']:,.0f}", flush=True)
    print(f"  mean daily        ${num['mean_daily_usd']:,.0f}", flush=True)
    print(f"  median daily      ${num['median_daily_usd']:,.0f}", flush=True)
    print(f"  max single-day    ${num['max_day_usd']:,.0f}  "
          f"({num['max_day_date']})", flush=True)
    print(f"  worst rolling-7d  ${num['worst_7d_usd']:,.0f}  "
          f"(end {num['worst_7d_end_date']})", flush=True)
    print(f"  nonzero hours     {num['pct_nonzero_hours']:.2f}%", flush=True)

    print("\nDENOMINATOR (INDUSTRY ESTIMATES — order of magnitude only):",
          flush=True)
    print(f"  ETH spot+perp     ${den['spotperp_yr_primary']/TRILLION:.1f}T/yr "
          f"-> ${den['spotperp_day_primary']/BILLION:.1f}B/day (PRIMARY)",
          flush=True)
    print(f"  ETH spot+perp ALT ${den['spotperp_yr_alt']/TRILLION:.1f}T/yr "
          f"-> ${den['spotperp_day_alt']/BILLION:.1f}B/day", flush=True)

    print("\nRATIO TABLE:", flush=True)
    with pd.option_context("display.max_colwidth", 38,
                           "display.width", 160):
        show = df.copy()
        show["numerator_defi_usd"] = show["numerator_defi_usd"].map(
            lambda x: f"${x:,.0f}")
        show["denominator_eth_usd"] = show["denominator_eth_usd"].map(
            lambda x: f"${x:,.0f}")
        show["ratio_pct"] = show["ratio_pct"].map(lambda x: f"{x:.6f}%")
        print(show.to_string(index=False), flush=True)

    # ── write outputs ──
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "size_ratio.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv}", flush=True)

    meta_path = out_dir / "size_ratio_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)

    # ── verification: head/tail/shape of the CSV (per project convention) ──
    chk = pd.read_csv(out_csv)
    print(f"\nVERIFY size_ratio.csv  shape={chk.shape}", flush=True)
    print("HEAD:", flush=True)
    print(chk.head().to_string(index=False), flush=True)
    print("TAIL:", flush=True)
    print(chk.tail().to_string(index=False), flush=True)

    print("\nHEADLINE: mean-daily DeFi ETH liq = "
          f"{df.iloc[0]['ratio_pct']:.5f}% of daily ETH spot+perp turnover; "
          f"worst single day = "
          f"{df[df.metric.str.startswith('max_day_defi_liq_vs_daily')].iloc[0]['ratio_pct']:.4f}%.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
