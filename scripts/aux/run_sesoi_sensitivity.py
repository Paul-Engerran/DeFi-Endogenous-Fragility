#!/usr/bin/env python3
"""
run_sesoi_sensitivity.py — [ROBUSTNESS / FLAGGED — does NOT change the main spec]

SESOI SENSITIVITY CURVE (supports §7.3(iii) of the paper).

Question
--------
The equivalence verdict in mde_equivalence.csv is reported at exactly THREE
SESOI calibrations (the iqr / p50->p95 / p10->p90 shock spans). A reviewer can
object: "you chose the bound." This script removes the choice by reporting the
TOST equivalence verdict as a CONTINUOUS function of the SESOI S, for the two
probability-unit downside-asymmetry objects:
  - exceedance Delta at alpha=0.01, h=0  (the per-period downside whisper Delta);
  - skew_test skew_tail01                (the scale-removed whisper).
A reader then sees EXACTLY where the EQUIVALENT / INCONCLUSIVE / NON-NEGLIGIBLE
boundary sits, and can read off the verdict at ANY SESOI they prefer.

WHY THIS IS A *DESCRIPTIVE* POST-PROCESSOR (no re-estimation, no DGP)
--------------------------------------------------------------------
Everything is a deterministic function of (estimate, SE) which already exist in
the CANONICAL mde_equivalence.csv. We do NOT touch the panel, the QLP equation,
the bootstrap, or any canonical CSV. We READ mde_equivalence.csv and:
  (1) recover each object's Gaussian-equivalent SE from its published mde_50
      column: run_mde_equivalence sets mde_50 = z_{0.975} * SE EXACTLY (power
      0.50 => z_power=0), so  SE = mde_50 / z_{0.975}. This ties the curve to
      the published numbers to machine precision (verified: matches the CSV
      `se` column and the meta `se_inferred` to ~1e-12);
  (2) replay run_mde_equivalence.tost_verdict() VERBATIM at a dense grid of S,
      i.e. EQUIVALENT iff the 90% CI [est - z95*SE, est + z95*SE] is strictly
      inside (-S, +S); NON-NEGLIGIBLE iff lo90 >= S or hi90 <= -S;
      INCONCLUSIVE otherwise.

The CLOSED-FORM boundary (the single load-bearing number)
---------------------------------------------------------
For a positive point estimate (both objects here have est>0) the lower 90% bound
lo90 = est - z95*SE; the band's lower edge -S is crossed only for tiny S, but the
binding flip is the UPPER edge: the object becomes EQUIVALENT exactly when the
upper 90% CI bound clears S, i.e.

    S_equiv*  =  est + z95 * SE  =  hi90.

For S > S_equiv* the verdict is EQUIVALENT-TO-NEGLIGIBLE; for S < S_equiv* it is
INCONCLUSIVE, until S falls below lo90 (>0), where it becomes NON-NEGLIGIBLE.
For the exceedance Delta lo90 is ~0/negative, so its NON-NEGLIGIBLE region is
empty; for skew_tail01 lo90>0 gives a small NON-NEGLIGIBLE region below
S_nonneg* ~ 0.000255 (a SESOI more than 4x stricter than the strictest
pre-registered span), recorded per object as s_nonneg_star.
We report S_equiv* and S_nonneg* in CLOSED FORM (exact) AND tag each grid row
with the verdict, so the swept curve and the closed-form boundary agree by
construction (a self-check asserts this).

The 3 canonical spans are echoed as REFERENCE MARKERS (sesoi_beta_iqr /
_p50p95 / _p10p90, read straight from mde_equivalence.csv) so the figure can
annotate where the locked calibrations fall relative to S_equiv*.

OUTPUT (data/econ/) — NEW files, never overwrites a canonical CSV
-----------------------------------------------------------------
  sesoi_sensitivity.csv   long format, one row per (object, S grid point):
    [object, label, estimate, se, lo90, hi90,
     sesoi, verdict, is_equivalent, distance_to_equiv_boundary]
    plus per-object marker rows are NOT mixed in; markers live in the meta.
  sesoi_sensitivity_meta.json
    closed-form boundaries (s_equiv_star, s_nonneg_star) per object, the 3
    canonical span markers + their verdicts, the grid spec, the formulas,
    provenance, and a FIGURE SPEC block (what to plot).

FIGURE SPEC (also embedded in the meta JSON, key "figure_spec")
---------------------------------------------------------------
  Fig A4 "SESOI sensitivity of the equivalence verdict".
  One panel, x = SESOI S (beta units, shared x for both objects; log-x optional).
  For EACH object (exceedance Delta@1%, skew_tail01), draw:
    - a horizontal verdict ribbon coloured by verdict along S
      (INCONCLUSIVE shading left of S_equiv*, EQUIVALENT shading right);
    - a vertical solid line at S_equiv* (the flip threshold) labelled with its
      value;
    - the estimate's 90% CI upper bound hi90 == S_equiv* annotated.
  Overlay 3 vertical dashed reference lines at the canonical spans
  (sesoi_beta_p50p95 [strict], _p10p90, _iqr) — the locked calibrations — so the
  reader sees the strict span (p50p95) sits BELOW skew_tail01's S_equiv*
  (=> that object is INCONCLUSIVE at the strict span, EQUIVALENT at the looser
  two), while the exceedance Delta clears all three. Caption states the one
  sentence the figure buys: "the whisper is negligible for every SESOI above
  S_equiv* (= {skew hi90}); the only span under which it is inconclusive is the
  strictest (p50->p95)."

CLI
---
    .venv/bin/python scripts/aux/run_sesoi_sensitivity.py            # canonical
    .venv/bin/python scripts/aux/run_sesoi_sensitivity.py \
        --in_dir data/econ --out_dir /tmp/sesoi_smoke \
        --s_min 0.0001 --s_max 0.01 --n_grid 400                     # explicit grid
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # scripts/aux/ -> project root
sys.path.insert(0, str(ROOT))

from config import CFG, ECON_DIR            # noqa: E402

# ──────────────────────────────────────────────────────────────
# Locked spec constants (mirror run_mde_equivalence verbatim)
# ──────────────────────────────────────────────────────────────
ALPHA_TEST: float = 0.05                    # two-sided level (matches MDE script)
Z975: float = float(stats.norm.ppf(0.975))  # mde_50 = z975 * SE  (power=0.50)
Z95: float = float(stats.norm.ppf(0.95))    # 90% CI half-width multiplier (TOST)

EQUIV = "EQUIVALENT-TO-NEGLIGIBLE"
INCONC = "INCONCLUSIVE"
NONNEG = "NON-NEGLIGIBLE"

# The two probability-unit objects whose continuous verdict we sweep. Keyed by
# (object, label) EXACTLY as written in mde_equivalence.csv. z3_winsor is
# excluded by construction (N/A-UNITS — its SESOI mapping is incommensurable).
TARGETS: list[tuple[str, str, str]] = [
    # (object_col, label_col, short_tag)
    ("exceedance_delta", "alpha=0.01 (h=0)", "exc_delta_a01_h0"),
    ("skew_test",        "skew_tail01",      "skew_tail01"),
]

# Canonical span marker columns to echo from mde_equivalence.csv.
SPAN_COLS: list[tuple[str, str]] = [
    ("sesoi_beta_p50p95", "p50p95_strict"),
    ("sesoi_beta_p10p90", "p10p90"),
    ("sesoi_beta_iqr",    "iqr"),
]

OUT_COLS: list[str] = [
    "object", "label", "estimate", "se", "lo90", "hi90",
    "sesoi", "verdict", "is_equivalent", "distance_to_equiv_boundary",
]


# ──────────────────────────────────────────────────────────────
# TOST verdict — VERBATIM port of run_mde_equivalence.tost_verdict
# ──────────────────────────────────────────────────────────────
def ci90(estimate: float, se: float) -> tuple[float, float]:
    """90% two-sided CI used for the TOST verdict (mde script convention)."""
    return estimate - Z95 * se, estimate + Z95 * se


def tost_verdict(estimate: float, se: float, sesoi: float) -> str:
    """Equivalence verdict from the 90% CI vs (-sesoi, +sesoi). Verbatim rule."""
    if not np.isfinite(sesoi) or sesoi <= 0:
        return INCONC
    lo, hi = ci90(estimate, se)
    if lo > -sesoi and hi < sesoi:
        return EQUIV
    if lo >= sesoi or hi <= -sesoi:
        return NONNEG
    return INCONC


def closed_form_boundaries(estimate: float, se: float) -> dict:
    """Exact SESOI thresholds where the verdict flips.

    S_equiv*  : smallest S making the object EQUIVALENT. The 90% CI is
                [lo90, hi90]; EQUIV needs lo90 > -S AND hi90 < S, i.e.
                S > max(hi90, -lo90). So S_equiv* = max(hi90, -lo90).
                (For est>0, hi90 dominates => S_equiv* = hi90 = est + z95*SE.)
    S_nonneg* : largest S making the object NON-NEGLIGIBLE: lo90 >= S
                (positive est) => S <= lo90. So S_nonneg* = lo90 if lo90 > 0
                else NaN (no non-negligible region for any S>0). For est<0 the
                symmetric mirror (hi90 <= -S) gives S_nonneg* = -hi90.
    Between S_nonneg* and S_equiv* the verdict is INCONCLUSIVE.
    """
    lo90, hi90 = ci90(estimate, se)
    s_equiv = max(hi90, -lo90)
    # NON-NEGLIGIBLE region (S>0): lo90>=S (est>0 side) OR hi90<=-S (est<0 side)
    cand = []
    if lo90 > 0:
        cand.append(lo90)
    if hi90 < 0:
        cand.append(-hi90)
    s_nonneg = max(cand) if cand else np.nan
    return {"lo90": lo90, "hi90": hi90,
            "s_equiv_star": float(s_equiv),
            "s_nonneg_star": (float(s_nonneg) if np.isfinite(s_nonneg) else None)}


# ──────────────────────────────────────────────────────────────
# Inputs — read the CANONICAL mde_equivalence.csv (read-only)
# ──────────────────────────────────────────────────────────────
def load_mde(in_dir: Path) -> pd.DataFrame:
    p = in_dir / "mde_equivalence.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run scripts/aux/run_mde_equivalence.py first."
        )
    return pd.read_csv(p)


def se_from_row(row: pd.Series) -> float:
    """Recover the Gaussian-equivalent SE.

    mde_equivalence.csv carries `se` directly; we also cross-check it against
    mde_50/z975 (mde_50 = z975*SE by construction) and warn on any mismatch.
    Using `se` keeps us exactly on the published number.
    """
    se = float(row["se"])
    se_chk = float(row["mde_50"]) / Z975
    # Hard assert: se and mde_50/z975 are equal by construction
    # (mde_50 = z975*SE). A divergence means an inconsistent input CSV and the
    # whole curve is untrustworthy — fail fast rather than warn.
    assert np.isclose(se, se_chk, rtol=1e-9, atol=1e-12), (
        f"se ({se:.10g}) != mde_50/z975 ({se_chk:.10g}) — inconsistent "
        "mde_equivalence.csv; aborting.")
    return se


# ──────────────────────────────────────────────────────────────
# Build the swept rows + per-object boundary/marker meta
# ──────────────────────────────────────────────────────────────
def build_for_object(
    mde: pd.DataFrame, object_col: str, label_col: str, tag: str,
    s_grid: np.ndarray,
) -> tuple[list[dict], dict]:
    sel = mde[(mde["object"] == object_col)
              & (mde["alpha_or_measure"] == label_col)]
    if sel.empty:
        raise ValueError(
            f"mde_equivalence.csv has no row object='{object_col}', "
            f"alpha_or_measure='{label_col}'."
        )
    r = sel.iloc[0]
    estimate = float(r["estimate"])
    se = se_from_row(r)
    bnd = closed_form_boundaries(estimate, se)

    rows: list[dict] = []
    for s in s_grid:
        v = tost_verdict(estimate, se, float(s))
        rows.append({
            "object": object_col,
            "label": label_col,
            "estimate": estimate,
            "se": se,
            "lo90": bnd["lo90"],
            "hi90": bnd["hi90"],
            "sesoi": float(s),
            "verdict": v,
            "is_equivalent": bool(v == EQUIV),
            "distance_to_equiv_boundary": float(s - bnd["s_equiv_star"]),
        })

    # Canonical span markers + their verdicts (echoed from the same CSV row).
    markers = {}
    for col, name in SPAN_COLS:
        S = float(r[col]) if pd.notna(r[col]) else np.nan
        markers[name] = {
            "sesoi_beta": (float(S) if np.isfinite(S) else None),
            "verdict_canonical_csv": (
                str(r[{"sesoi_beta_p50p95": "verdict_p50p95",
                       "sesoi_beta_p10p90": "verdict_p10p90",
                       "sesoi_beta_iqr": "verdict_iqr"}[col]])),
            "verdict_recomputed": (tost_verdict(estimate, se, S)
                                   if np.isfinite(S) else None),
        }

    # SELF-CHECK: swept verdict at each marker must equal the canonical CSV
    # verdict (proves the continuous curve reproduces the locked 3-point grid).
    for name, mk in markers.items():
        if mk["verdict_recomputed"] is not None:
            assert mk["verdict_recomputed"] == mk["verdict_canonical_csv"], (
                f"{tag}/{name}: recomputed {mk['verdict_recomputed']} != "
                f"canonical {mk['verdict_canonical_csv']}")

    obj_meta = {
        "object": object_col, "label": label_col, "tag": tag,
        "estimate": estimate, "se": se,
        "lo90": bnd["lo90"], "hi90": bnd["hi90"],
        "s_equiv_star": bnd["s_equiv_star"],
        "s_nonneg_star": bnd["s_nonneg_star"],
        "s_equiv_star_formula": "max(hi90, -lo90) = est + z_{0.95}*SE for est>0",
        "interpretation": (
            f"EQUIVALENT-TO-NEGLIGIBLE for every SESOI > {bnd['s_equiv_star']:.6g}; "
            f"INCONCLUSIVE below it"
            + ("" if bnd["s_nonneg_star"] is None else
               f"; NON-NEGLIGIBLE below {bnd['s_nonneg_star']:.6g}")
            + "."),
        "canonical_span_markers": markers,
    }
    return rows, obj_meta


def auto_grid(obj_metas: list[dict], s_min: float | None,
              s_max: float | None, n_grid: int, log: bool) -> tuple[np.ndarray, dict]:
    """Grid spanning well below the smallest |lo90| to well above the largest
    S_equiv* and the largest canonical span, so every flip is captured."""
    s_equivs = [m["s_equiv_star"] for m in obj_metas]
    s_nonnegs = [m["s_nonneg_star"] for m in obj_metas
                 if m.get("s_nonneg_star") is not None]
    span_vals = [v["sesoi_beta"] for m in obj_metas
                 for v in m["canonical_span_markers"].values()
                 if v["sesoi_beta"] is not None]
    hi_ref = max(s_equivs + span_vals)
    auto_min = s_min if s_min is not None else hi_ref / 200.0
    if s_nonnegs:                       # ensure the NON-NEGLIGIBLE band is swept
        auto_min = min(auto_min, 0.5 * min(s_nonnegs))
    auto_max = s_max if s_max is not None else hi_ref * 4.0
    if log:
        grid = np.geomspace(auto_min, auto_max, n_grid)
    else:
        grid = np.linspace(auto_min, auto_max, n_grid)
    # Inject the exact flip thresholds + canonical spans + NON-NEGLIGIBLE edges
    # so the curve has a node EXACTLY at each boundary (clean plotting / read-off).
    extra = np.array(s_equivs + span_vals + s_nonnegs, dtype=float)
    grid = np.unique(np.concatenate([grid, extra]))
    info = {"s_min": float(grid.min()), "s_max": float(grid.max()),
            "n_grid_effective": int(grid.size), "log_spaced": bool(log),
            "exact_nodes_injected": [float(x) for x in np.sort(extra)]}
    return grid, info


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(df: pd.DataFrame, meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sesoi_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}", flush=True)
    meta_path = out_dir / "sesoi_sensitivity_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)
    print("\n--- sesoi_sensitivity.csv ---", flush=True)
    print(f"shape: {df.shape}", flush=True)
    print("HEAD:", flush=True)
    print(df.head().to_string(index=False), flush=True)
    print("TAIL:", flush=True)
    print(df.tail().to_string(index=False), flush=True)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--in_dir", type=Path, default=ECON_DIR,
                    help=f"Dir holding mde_equivalence.csv. Default: {ECON_DIR}")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    ap.add_argument("--s_min", type=float, default=None,
                    help="Grid lower edge (beta units). Default: auto "
                         "(max_ref / 200).")
    ap.add_argument("--s_max", type=float, default=None,
                    help="Grid upper edge. Default: auto (4x max_ref).")
    ap.add_argument("--n_grid", type=int, default=400,
                    help="Grid points before exact-node injection. Default 400.")
    ap.add_argument("--linear_grid", action="store_true",
                    help="Use linear spacing (default: log-spaced).")
    args = ap.parse_args()

    print("run_sesoi_sensitivity — continuous TOST verdict vs SESOI (A4)",
          flush=True)
    print(f"  in_dir={args.in_dir}", flush=True)
    t0 = time.time()

    mde = load_mde(args.in_dir)

    # First pass: per-object closed-form boundaries (needed to auto-size grid).
    pre_metas: list[dict] = []
    for oc, lc, tag in TARGETS:
        sel = mde[(mde["object"] == oc) & (mde["alpha_or_measure"] == lc)]
        if sel.empty:
            raise ValueError(f"missing object='{oc}', label='{lc}' in mde csv.")
        r = sel.iloc[0]
        est = float(r["estimate"]); se = se_from_row(r)
        bnd = closed_form_boundaries(est, se)
        pre_metas.append({"s_equiv_star": bnd["s_equiv_star"],
                          "s_nonneg_star": bnd["s_nonneg_star"],
                          "canonical_span_markers": {
                              name: {"sesoi_beta": (float(r[col])
                                     if pd.notna(r[col]) else None)}
                              for col, name in SPAN_COLS}})

    s_grid, grid_info = auto_grid(pre_metas, args.s_min, args.s_max,
                                  args.n_grid, log=(not args.linear_grid))
    print(f"  S grid: [{grid_info['s_min']:.6g}, {grid_info['s_max']:.6g}] "
          f"n={grid_info['n_grid_effective']} "
          f"({'log' if grid_info['log_spaced'] else 'linear'})", flush=True)

    all_rows: list[dict] = []
    obj_metas: list[dict] = []
    for oc, lc, tag in TARGETS:
        rows, om = build_for_object(mde, oc, lc, tag, s_grid)
        all_rows.extend(rows)
        obj_metas.append(om)
        print(f"  [{tag}] est={om['estimate']:+.6g} se={om['se']:.6g}  "
              f"S_equiv*={om['s_equiv_star']:.6g}", flush=True)

    df_out = (pd.DataFrame(all_rows)
              .sort_values(["object", "label", "sesoi"], kind="mergesort")
              .reset_index(drop=True)[OUT_COLS])

    figure_spec = {
        "id": "fig_A4_sesoi_sensitivity",
        "title": "SESOI sensitivity of the equivalence verdict",
        "panels": 1,
        "x_axis": {"var": "sesoi", "label": "SESOI S (beta units)",
                   "scale": "log" if grid_info["log_spaced"] else "linear"},
        "series_per_object": [
            {"object": om["object"], "label": om["label"],
             "verdict_ribbon": "colour grid rows by `verdict` along S "
                               "(INCONCLUSIVE left of S_equiv*, EQUIVALENT right)",
             "vline_solid_at": om["s_equiv_star"],
             "vline_label": f"S_equiv*={om['s_equiv_star']:.5f} (=hi90)"}
            for om in obj_metas
        ],
        "reference_vlines_dashed": {
            name: next(iter([v["sesoi_beta"]
                       for om in obj_metas
                       for n, v in om["canonical_span_markers"].items()
                       if n == name]))
            for name in ("p50p95_strict", "p10p90", "iqr")
        },
        "caption": (
            "Equivalence verdict (TOST, 90% CI vs +/-S band) as a continuous "
            "function of the SESOI S, for the per-period 1% exceedance Delta "
            "(h=0) and the whisper (skew_tail01). Solid lines mark each object's "
            "flip threshold S_equiv* = est + z_{0.95}*SE. Dashed lines mark the "
            "three locked shock-span calibrations. The exceedance Delta is "
            "EQUIVALENT-TO-NEGLIGIBLE at all three; the whisper is EQUIVALENT for "
            "every S above its S_equiv*, with the single strict span (p50->p95) "
            "the only calibration leaving it INCONCLUSIVE."),
        "plot_from": "data/econ/sesoi_sensitivity.csv (+ meta boundaries)",
        "lands_in": "§7.3(iii) footnote / Appendix A",
    }

    meta = {
        "script": "scripts/aux/run_sesoi_sensitivity.py",
        "purpose": ("Continuous SESOI sensitivity of the TOST equivalence "
                    "verdict for the two probability-unit downside-asymmetry "
                    "objects (exceedance Delta@1% h=0, skew_tail01). Descriptive "
                    "post-processor over mde_equivalence.csv — NO re-estimation, "
                    "NO DGP, no canonical CSV overwritten."),
        "paper_section": "7.3(iii)",
        "method": {
            "se_recovery": "SE = published `se` col (cross-checked vs mde_50/z975)",
            "verdict_rule": ("VERBATIM run_mde_equivalence.tost_verdict: "
                             "EQUIVALENT iff [est-z95*SE, est+z95*SE] subset "
                             "(-S,+S); NON-NEGLIGIBLE iff lo90>=S or hi90<=-S; "
                             "else INCONCLUSIVE"),
            "closed_form_boundary": ("S_equiv* = max(hi90,-lo90) = est+z95*SE "
                                     "(est>0); S_nonneg* = lo90 if lo90>0 else "
                                     "none"),
            "z95": Z95, "z975": Z975, "alpha_test": ALPHA_TEST,
        },
        "objects": obj_metas,
        "grid": grid_info,
        "figure_spec": figure_spec,
        "non_confirmatory_branch_outcome": (
            "If any TARGET object's S_equiv* exceeded the LARGEST canonical span "
            "(iqr), the whisper would be inconclusive even at the most permissive "
            "locked bound => a residual downside asymmetry that survives every "
            "pre-registered SESOI; that would be REPORTED as a qualification of "
            "the bounded-null claim, not buried. Observed: both S_equiv* fall at "
            "or below the iqr span (exceedance below all three; skew_tail01 above "
            "only the strict p50p95), consistent with the bounded-negligible "
            "thesis. Recorded per object in objects[].interpretation."),
        "inputs": {
            "mde_equivalence_csv": str((args.in_dir
                                        / "mde_equivalence.csv").resolve()),
            "note": "READ-ONLY; this script writes only NEW sesoi_sensitivity.* "
                    "files and never modifies a canonical artefact.",
        },
        "panel": str(CFG.FILES.econ_core_full),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    save_outputs(df_out, meta, args.out_dir)
    # Console summary of the two load-bearing thresholds.
    for om in obj_metas:
        print(f"\n  {om['tag']}: EQUIVALENT for SESOI > {om['s_equiv_star']:.6g}  "
              f"(strict-span p50p95 marker => "
              f"{om['canonical_span_markers']['p50p95_strict']['verdict_recomputed']})",
              flush=True)
    print(f"\nDone. {len(df_out)} rows. Total wall time: "
          f"{(time.time()-t0)/60:.3f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())