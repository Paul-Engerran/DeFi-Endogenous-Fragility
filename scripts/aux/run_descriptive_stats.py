#!/usr/bin/env python3
"""
run_descriptive_stats.py — [DESCRIPTIVE / Table 1 provenance — does NOT change the spec]

Re-executable computation of EVERY descriptive number cited in the paper's
Table 1 / stylised-facts section (§4), replacing the notebook-only provenance
of notebooks/09_descriptive_stats.ipynb.

This script is a strict SUPERSET of the notebook's descriptive_stats.csv/json:
it reproduces the existing blocks (same definitions, same long format
[variable, statistic, value]) and ADDS the previously informal numbers
with EXPLICIT, documented definitions:

NEW BLOCKS (the Table-1 numbers that previously lived only in informal notes)
----------------------------------------------------------------------------
1. Shape moments (per return column):
     skew         — sample skewness (pandas .skew(), bias-adjusted Fisher-Pearson)
     excess_kurt  — sample EXCESS kurtosis (pandas .kurt(); normal = 0)
   Expected: ret_eth_perp skew ≈ −0.63, excess_kurt ≈ 17.6.

2. tail_asymmetry — the "left ≈ 2× right" numbers with their EXACT definition.
   These are FAR-EXTREME COUNTS at ABSOLUTE thresholds (returns are in
   percent, so the thresholds are ±5% and ±7% one-hour returns), NOT the
   1%/5% quantile values (which are nearly symmetric, ratio ≈ 1.05):
     n_below_minus5 / n_above_plus5   — count of hours with ret ≤ −5 / ret ≥ +5
                                        (expected 50 / 25 → ratio 2.0)
     n_below_minus7 / n_above_plus7   — count of hours with ret ≤ −7 / ret ≥ +7
                                        (expected 13 / 6  → ratio ≈ 2.2)
     ratio_far5, ratio_far7           — left/right count ratios at ±5, ±7
     q_ratio_p1_p99 = |p1| / p99      — quantile-VALUE ratio (expected ≈ 1.05)
     q_ratio_p5_p95 = |p5| / p95      — quantile-VALUE ratio (expected ≈ 1.01)
     min / max                        — expected ≈ −15.5 / +9.2
   The far-extreme counts use ≤ / ≥ (inclusive) on the after-warmup sample.

3. z3 — the third-moment object net of the volatility scale. z is the
   vol-standardised return z_t = ret_eth_perp_t / vol_eth_7d_t, where
   vol_eth_7d is the 168h rolling std (PRE-determined at t: it enters the
   locked spec as a control). Reported:
     z_mean, z_std, z_skew            — moments of z
     z3_mean                          — mean of z³ (raw)
     z3_mean_winsor                   — mean of z³ after two-sided 1% winsorising
                                        of z (limits [p1, p99] of z), the same
                                        de-fattening convention as run_skew_test
   (Descriptive companions of the skew-test regression; OUT of the TOST scope.)

4. liq_crosstab — the contemporaneous reverse-causality cross-tab:
     p_liq_worst1   = P(log_liq_t > 0 | ret_t ≤ p1(ret))   — expected ≈ 85%
     p_liq_best1    = P(log_liq_t > 0 | ret_t ≥ p99(ret))  — expected ≈ 42%
     p_liq_all      = P(log_liq_t > 0)                      — expected ≈ 27%
   plus the same at the 5% tails (p_liq_worst5 / p_liq_best5), and the n of
   each conditioning set. "Contemporaneous" = log_liq in the SAME hour t as
   the return (NOT the lagged shock) — this is the crash→liquidation
   reverse-causality stylised fact, and is documented as such.

SAMPLE — identical to the estimation sample
-------------------------------------------
All statistics are computed on the after-warmup sample produced by
src.estimation.build_df_est_raw (the EXACT estimation panel of the main QLP:
warmup = max(vol_window, 720) + max(lp_horizons) + 2 = 746 rows dropped,
N = 40,582). The 'sample' block reports full-panel first/last obs and the
warmup row count, byte-matching the notebook convention.

OUTPUT
------
  data/econ/descriptive_stats.csv        (long format: variable, statistic, value)
  data/econ/descriptive_stats.json       (same content, nested)
  data/econ/descriptive_stats_meta.json  (run provenance + explicit definitions)

Run
---
    .venv/bin/python scripts/aux/run_descriptive_stats.py
    .venv/bin/python scripts/aux/run_descriptive_stats.py --out_dir /tmp/desc_smoke
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from config import CFG, ECON_DIR  # noqa: E402
from src.estimation import build_df_est_raw, WARMUP_OI_WINDOW  # noqa: E402
from src.io import load_econ_panel  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Definitions (single source of truth, dumped into the meta)
# ──────────────────────────────────────────────────────────────
FAR_THRESHOLDS: list[float] = [5.0, 7.0]   # absolute one-hour return thresholds, in %
PCTL_COLS: list[str] = [
    "ret_eth_perp", "ret_btc_spot", "vol_eth_7d",
    "funding_rate", "basis_bps", "oi",
]
SHAPE_COLS: list[str] = ["ret_eth_perp", "ret_btc_spot"]
RET_COL: str = "ret_eth_perp"
VOL_COL: str = "vol_eth_7d"
LIQ_COL: str = "log_liq"

# Sanity anchors reproduced from notebooks/09 (cross-check vs the paper).
ANCHOR_STD_RET = 0.8
ANCHOR_CUM_LIQ_BN = 2.5


def _pctl_block(s: pd.Series) -> dict:
    """The notebook's per-variable block: N, mean, std, min, p1..p99, max."""
    s = s.dropna()
    return {
        "N":    int(len(s)),
        "mean": float(s.mean()),
        "std":  float(s.std()),
        "min":  float(s.min()),
        "p1":   float(s.quantile(0.01)),
        "p5":   float(s.quantile(0.05)),
        "p25":  float(s.quantile(0.25)),
        "p50":  float(s.quantile(0.50)),
        "p75":  float(s.quantile(0.75)),
        "p95":  float(s.quantile(0.95)),
        "p99":  float(s.quantile(0.99)),
        "max":  float(s.max()),
    }


def compute_stats(df_est: pd.DataFrame, df_full: pd.DataFrame) -> dict:
    """All descriptive blocks (existing notebook blocks + the new Table-1 blocks)."""
    out: dict = {}

    # ── 1. Per-variable percentile blocks (notebook-identical) ──
    for col in PCTL_COLS:
        out[col] = _pctl_block(df_est[col])

    # NEW: shape moments appended inside the return blocks.
    for col in SHAPE_COLS:
        s = df_est[col].dropna()
        out[col]["skew"] = float(s.skew())
        out[col]["excess_kurt"] = float(s.kurt())

    # ── 2. log_liq block (notebook-identical) ──
    liq = df_est[LIQ_COL].dropna()
    out[LIQ_COL] = _pctl_block(liq)
    out[LIQ_COL]["n_nonzero"] = int((liq > 0).sum())
    out[LIQ_COL]["pct_nonzero"] = float(round(100.0 * (liq > 0).mean(), 2))

    # ── 3. liq_usd_total block (notebook-identical) ──
    usd = df_est[["date", "liq_usd_total"]].dropna(subset=["liq_usd_total"])
    nz = usd.loc[usd["liq_usd_total"] > 0, "liq_usd_total"]
    out["liq_usd_total"] = {
        "sum_total_usd":    float(usd["liq_usd_total"].sum()),
        "mean_nonzero_usd": float(nz.mean()),
    }
    for yr, grp in usd.groupby(usd["date"].dt.year):
        out["liq_usd_total"][f"sum_{yr}_usd"] = float(grp["liq_usd_total"].sum())

    # ── 4. sample block (notebook-identical) ──
    out["sample"] = {
        "first_obs":      str(df_full["date"].iloc[0]),
        "last_obs":       str(df_full["date"].iloc[-1]),
        "N_total":        int(len(df_full)),
        "N_after_warmup": int(len(df_est)),
        "warmup_rows":    int(len(df_full) - len(df_est)),
    }

    # ── 5. oi_high block (notebook-identical) ──
    out["oi_high"] = {
        "pct_high": float(round(100.0 * df_est["oi_high"].mean(), 2)),
    }

    # ── 6. NEW: tail_asymmetry — far-extreme COUNTS at absolute thresholds ──
    r = df_est[RET_COL].dropna()
    ta: dict = {}
    for thr in FAR_THRESHOLDS:
        n_lo = int((r <= -thr).sum())
        n_hi = int((r >= thr).sum())
        key = str(int(thr)) if float(thr).is_integer() else str(thr)
        ta[f"n_below_minus{key}"] = n_lo
        ta[f"n_above_plus{key}"] = n_hi
        ta[f"ratio_far{key}"] = float(n_lo / n_hi) if n_hi else np.nan
    ta["q_ratio_p1_p99"] = float(abs(r.quantile(0.01)) / r.quantile(0.99))
    ta["q_ratio_p5_p95"] = float(abs(r.quantile(0.05)) / r.quantile(0.95))
    ta["min"] = float(r.min())
    ta["max"] = float(r.max())
    out["tail_asymmetry"] = ta

    # ── 7. NEW: z3 — vol-standardised third-moment objects ──
    z = (df_est[RET_COL] / df_est[VOL_COL].replace(0, np.nan)).dropna()
    z_lo, z_hi = z.quantile(0.01), z.quantile(0.99)
    z_w = z.clip(lower=z_lo, upper=z_hi)
    out["z3"] = {
        "N":               int(len(z)),
        "z_mean":          float(z.mean()),
        "z_std":           float(z.std()),
        "z_skew":          float(z.skew()),
        "z3_mean":         float((z ** 3).mean()),
        "z3_mean_winsor":  float((z_w ** 3).mean()),
        "winsor_lo_p1":    float(z_lo),
        "winsor_hi_p99":   float(z_hi),
    }

    # ── 8. NEW: liq_crosstab — contemporaneous reverse-causality fact ──
    sub = df_est[[RET_COL, LIQ_COL]].dropna()
    rr, ll = sub[RET_COL], sub[LIQ_COL] > 0
    ct: dict = {"p_liq_all": float(100.0 * ll.mean()), "n_all": int(len(sub))}
    for alpha, tag in ((0.01, "1"), (0.05, "5")):
        lo_thr = rr.quantile(alpha)
        hi_thr = rr.quantile(1.0 - alpha)
        worst = ll[rr <= lo_thr]
        best = ll[rr >= hi_thr]
        ct[f"p_liq_worst{tag}"] = float(100.0 * worst.mean())
        ct[f"p_liq_best{tag}"] = float(100.0 * best.mean())
        ct[f"n_worst{tag}"] = int(len(worst))
        ct[f"n_best{tag}"] = int(len(best))
    out["liq_crosstab"] = ct

    # ── 9. Handover sanity blocks (notebook-identical convention) ──
    std_ret = out[RET_COL]["std"]
    out["anchor_std_ret_eth_perp"] = {
        "value":                 float(round(std_ret, 4)),
        "anchor_estimate":     ANCHOR_STD_RET,
        "deviation_pct":         float(round(100.0 * abs(std_ret - ANCHOR_STD_RET)
                                             / ANCHOR_STD_RET, 1)),
        "anchor_discrepancy":  bool(abs(std_ret - ANCHOR_STD_RET)
                                      / ANCHOR_STD_RET > 0.10),
    }
    cum_bn = out["liq_usd_total"]["sum_total_usd"] / 1e9
    out["anchor_cumulative_liq_usd"] = {
        "value_bn":              float(round(cum_bn, 3)),
        "anchor_estimate_bn":  ANCHOR_CUM_LIQ_BN,
        "deviation_pct":         float(round(100.0 * abs(cum_bn - ANCHOR_CUM_LIQ_BN)
                                             / ANCHOR_CUM_LIQ_BN, 1)),
        "anchor_discrepancy":  bool(abs(cum_bn - ANCHOR_CUM_LIQ_BN)
                                      / ANCHOR_CUM_LIQ_BN > 0.10),
    }
    return out


def to_long_df(stats: dict) -> pd.DataFrame:
    rows = [
        {"variable": var, "statistic": stat, "value": val}
        for var, block in stats.items()
        for stat, val in block.items()
    ]
    return pd.DataFrame(rows, columns=["variable", "statistic", "value"])


def build_meta(stats: dict) -> dict:
    return {
        "script": "scripts/aux/run_descriptive_stats.py",
        "purpose": ("Re-executable provenance for every Table-1 / §4 descriptive "
                    "number (supersedes the notebook-only computation of "
                    "notebooks/09_descriptive_stats.ipynb)."),
        "sample": ("after-warmup estimation panel from src.estimation."
                   "build_df_est_raw (warmup = max(vol_window, 720) + "
                   "max(lp_horizons) + 2)"),
        "definitions": {
            "skew": "pandas Series.skew() — bias-adjusted sample skewness",
            "excess_kurt": "pandas Series.kurt() — sample EXCESS kurtosis (normal=0)",
            "tail_asymmetry": (
                "FAR-EXTREME COUNTS at ABSOLUTE thresholds on one-hour returns in "
                "percent: n_below_minus5 = #{ret <= -5%}, n_above_plus5 = "
                "#{ret >= +5%} (idem ±7%); ratio_farX = left/right count ratio. "
                "NOT the 1%/5% quantile values — those are reported as "
                "q_ratio_p1_p99 = |p1|/p99 and q_ratio_p5_p95 = |p5|/p95 and are "
                "nearly symmetric (~1.05/~1.01). The paper's 'left ≈ 2× right' is "
                "the COUNT ratio at ±5%/±7%, and must be labelled as such."
            ),
            "z3": (
                "z_t = ret_eth_perp_t / vol_eth_7d_t (vol_eth_7d = 168h rolling "
                "std, predetermined control of the locked spec). z3_mean_winsor "
                "winsorises z two-sided at its 1%/99% quantiles before cubing "
                "(same de-fattening convention as run_skew_test)."
            ),
            "liq_crosstab": (
                "P(log_liq_t > 0 | ret_t in tail), CONTEMPORANEOUS hour t (not "
                "the lagged shock): worst1 = {ret <= p1(ret)}, best1 = "
                "{ret >= p99(ret)} (idem 5%). This is the crash→liquidation "
                "reverse-causality stylised fact."
            ),
        },
        "expected_anchors": {
            "skew_ret_eth": -0.63, "excess_kurt_ret_eth": 17.6,
            "far5_counts": [50, 25], "far7_counts": [13, 6],
            "q_ratio_p1_p99": 1.05, "min_max": [-15.5, 9.2],
            "p_liq_worst1_best1_all": [85, 42, 27],
        },
        "panel": str(CFG.FILES.econ_core_full),
        "n_after_warmup": stats["sample"]["N_after_warmup"],
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _unlink_readonly(path: Path) -> bool:
    """Remove a pre-existing output file, returning True if it was read-only.

    The canonical artefacts in data/econ/ are chmod 400 on purpose (protection
    against accidental overwrite). Unlinking needs only directory write
    permission; the caller re-applies the read-only bit after writing so the
    protection convention survives the refresh.
    """
    if not path.exists():
        return False
    was_readonly = not (path.stat().st_mode & 0o200)
    path.unlink()
    return was_readonly


def _restore_mode(path: Path, was_readonly: bool) -> None:
    if was_readonly:
        path.chmod(0o400)


def save_outputs(stats: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_long = to_long_df(stats)

    csv_path = out_dir / "descriptive_stats.csv"
    ro = _unlink_readonly(csv_path)
    df_long.to_csv(csv_path, index=False)
    _restore_mode(csv_path, ro)
    print(f"  wrote {csv_path}", flush=True)

    json_path = out_dir / "descriptive_stats.json"
    ro = _unlink_readonly(json_path)
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    _restore_mode(json_path, ro)
    print(f"  wrote {json_path}", flush=True)

    meta_path = out_dir / "descriptive_stats_meta.json"
    ro = _unlink_readonly(meta_path)
    with open(meta_path, "w") as f:
        json.dump(build_meta(stats), f, indent=2)
    _restore_mode(meta_path, ro)
    print(f"  wrote {meta_path}", flush=True)

    # Convention: after modifying a CSV, print head/tail/shape.
    print(f"\n--- descriptive_stats.csv ---", flush=True)
    print(f"shape: {df_long.shape}", flush=True)
    print("HEAD:", flush=True)
    print(df_long.head(8).to_string(index=False), flush=True)
    print("TAIL:", flush=True)
    print(df_long.tail(8).to_string(index=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    print("run_descriptive_stats: Table-1 provenance recompute", flush=True)
    df_full = load_econ_panel()
    df_est = build_df_est_raw(horizons=list(CFG.ECON.lp_horizons))
    print(f"  full panel rows={len(df_full):,}  after-warmup rows={len(df_est):,}",
          flush=True)

    stats = compute_stats(df_est, df_full)

    # Console echo of the Table-1 anchors (vs expected values cross-checked
    # against the paper).
    r = stats["ret_eth_perp"]
    ta = stats["tail_asymmetry"]
    ct = stats["liq_crosstab"]
    print(f"\n  skew={r['skew']:+.3f} (exp −0.63)   "
          f"excess_kurt={r['excess_kurt']:.1f} (exp 17.6)", flush=True)
    print(f"  far ±5%: {ta['n_below_minus5']}/{ta['n_above_plus5']} (exp 50/25)   "
          f"far ±7%: {ta['n_below_minus7']}/{ta['n_above_plus7']} (exp 13/6)",
          flush=True)
    print(f"  q-ratios p1/p99={ta['q_ratio_p1_p99']:.3f} (exp ~1.05)   "
          f"p5/p95={ta['q_ratio_p5_p95']:.3f}", flush=True)
    print(f"  crosstab worst1/best1/all = {ct['p_liq_worst1']:.1f}/"
          f"{ct['p_liq_best1']:.1f}/{ct['p_liq_all']:.1f} (exp 85/42/27)", flush=True)

    save_outputs(stats, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
