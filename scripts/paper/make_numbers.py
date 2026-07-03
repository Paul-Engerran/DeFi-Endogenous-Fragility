#!/usr/bin/env python3
"""
make_numbers.py — generate paper/numbers.tex (LaTeX macros for EVERY number
cited in the prose).

Rationale: every number printed in the manuscript MUST exist as a
macro generated from a canonical CSV — the writing layer types
\\BetaQoneHtwelve, never "-0.241". This kills the stale-value class of bugs
by construction. Rounding conventions are defined HERE,
once.

Macro inventory follows the manuscript's macro conventions.
All macros wrap values in \\ensuremath{}; percent signs and units stay in prose.

Run
---
    .venv/bin/python scripts/paper/make_numbers.py
Output: paper/numbers.tex  (+ a generation manifest in the file header)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from config import ECON_DIR  # noqa: E402

OUT = ROOT / "paper" / "numbers.tex"


# ── formatting helpers (rounding conventions live HERE, once) ──
def num(x: float, nd: int = 3) -> str:
    """Signed decimal at nd places, LaTeX math minus."""
    return f"{x:.{nd}f}"


def pval(x: float) -> str:
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.2f}"


def pct1(x: float) -> str:
    return f"{x:.1f}"


def intc(x: float) -> str:
    """Integer with thin-space thousands separator."""
    return f"{int(round(x)):,}".replace(",", r"\,")


# ── non-destructive regression check (--check) ──
def _parse_existing(path: Path) -> dict[str, str] | None:
    """Parse the committed numbers.tex into a {name: value} map.

    Compares on the PARSED macro value, not raw bytes, so the generation-date
    header and comments never cause false positives.
    """
    import re
    if not path.exists():
        return None
    rx = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{\\ensuremath\{(.*)\}\}")
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        m = rx.search(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _check(new: dict[str, str]) -> int:
    """Compare recomputed macros to paper/numbers.tex; write nothing.

    Exit 0 if every macro name and value matches; exit 1 (and list the
    differences) otherwise. This is the §0 non-regression guard: after an
    intended data edit, ONLY the deliberately targeted macros may differ, and
    every beta/Delta/MDE/OOS macro must stay identical.
    """
    old = _parse_existing(OUT)
    if old is None:
        print("CHECK: paper/numbers.tex missing — run without --check first", file=sys.stderr)
        return 1
    old_names, new_names = set(old), set(new)
    added = sorted(new_names - old_names)
    removed = sorted(old_names - new_names)
    changed = sorted(n for n in (old_names & new_names) if old[n] != new[n])
    if not (added or removed or changed):
        print(f"CHECK OK: all {len(new)} macros match paper/numbers.tex")
        return 0
    print(f"CHECK FAILED: {len(changed)} changed, {len(added)} added, "
          f"{len(removed)} removed vs paper/numbers.tex", file=sys.stderr)
    for n in changed:
        print(f"  CHANGED {n}: {old[n]!r} -> {new[n]!r}", file=sys.stderr)
    for n in added:
        print(f"  ADDED   {n}: {new[n]!r}", file=sys.stderr)
    for n in removed:
        print(f"  REMOVED {n}: was {old[n]!r}", file=sys.stderr)
    return 1


def main() -> int:
    E = ECON_DIR
    macros: list[tuple[str, str, str]] = []  # (name, value, comment)

    def add(name: str, value: str, comment: str) -> None:
        macros.append((name, value, comment))

    # ── main QLP table (the bait) — C1-C9 ──
    qlp = pd.read_csv(E / "quantile_lp_results.csv")

    def beta(tau, h):
        r = qlp[(np.isclose(qlp.tau, tau)) & (qlp.h == h)]
        return float(r.beta_shock.iloc[0])

    add("BetaQoneHzero", num(beta(0.01, 0), 3), "C1 beta(0.01,h0)")
    add("BetaQoneHone", num(beta(0.01, 1), 3), "C3")
    add("BetaQoneHtwelve", num(beta(0.01, 12), 3), "C2 beta(0.01,h12)")
    add("BetaQoneHtwentyfour", num(beta(0.01, 24), 3), "trajectory end")
    add("BetaMedHzero", num(beta(0.50, 0), 4), "C8")
    add("BetaMedHtwelve", num(beta(0.50, 12), 3), "C9")
    add("RatioTailMedImpact", f"{abs(beta(0.01, 0) / beta(0.50, 0)):.1f}", "C4 ~19x at impact")
    add("RatioTailMedHtwelve", f"{abs(beta(0.01, 12) / beta(0.50, 12)):.1f}", "C5 ~7.5x at h12")
    add("GapLrImpact", f"{abs(beta(0.01, 0)) / abs(beta(0.95, 0)):.2f}", "C6 |b.01|/|b.95| h0")

    # ── descriptive anchors (Table 1) — C10-C15, C29-C31 ──
    de = pd.read_csv(E / "descriptive_stats.csv")

    def stat(var, st):
        return float(de[(de.variable == var) & (de.statistic == st)].value.iloc[0])

    add("SkewEth", num(stat("ret_eth_perp", "skew"), 2), "C10 skew -0.63")
    add("KurtEth", f"{stat('ret_eth_perp', 'excess_kurt'):.1f}", "C11 excess kurt 17.6")
    add("MinRet", f"{stat('ret_eth_perp', 'min'):.1f}", "C12 min -15.5")
    add("MaxRet", f"{stat('ret_eth_perp', 'max'):.1f}", "C12 max +9.2")
    add("FarFiveDown", intc(stat("tail_asymmetry", "n_below_minus5")), "C14 far-extreme 50")
    add("FarFiveUp", intc(stat("tail_asymmetry", "n_above_plus5")), "C14 25")
    add("FarSevenDown", intc(stat("tail_asymmetry", "n_below_minus7")), "C14 13")
    add("FarSevenUp", intc(stat("tail_asymmetry", "n_above_plus7")), "C14 6")
    add("RatioFarFive", f"{stat('tail_asymmetry', 'ratio_far5'):.1f}", "C14 2.0x")
    add("QratioOnePct", f"{stat('tail_asymmetry', 'q_ratio_p1_p99'):.2f}", "C14 1.05")
    add("PLiqWorstOne", pct1(stat("liq_crosstab", "p_liq_worst1")), "C15 85.5")
    add("PLiqBestOne", pct1(stat("liq_crosstab", "p_liq_best1")), "C15 41.6")
    add("PLiqAll", pct1(stat("liq_crosstab", "p_liq_all")), "C15 27.0")
    add("Nobs", intc(stat("sample", "N_after_warmup")), "C29 40,582")
    add("NTotal", intc(stat("sample", "N_total")), "C30 41,328")
    add("WarmupRows", intc(stat("sample", "warmup_rows")), "C31 746")
    add("LiqTotalBn", f"{stat('liq_usd_total', 'sum_total_usd') / 1e9:.2f}", "C20 $2.52bn (panel)")

    # ── apparent-result section (§5): tail betas, Test-M band, interaction ──
    add("BetaFiveHtwelve", num(beta(0.05, 12), 3), "C7 -0.102")
    add("BetaNinetyFiveHtwelve", num(beta(0.95, 12), 3), "C7 +0.077")
    tm = pd.read_csv(E / "robustness_bootstrap_nb07_spec_fast.csv").set_index("h")
    for h, tag in ((0, "Hzero"), (12, "Htwelve"), (24, "Htwentyfour")):
        add(f"TestMLo{tag}", num(tm.loc[h, "ci_lo"], 3), f"Test M band lo h{h}")
        add(f"TestMHi{tag}", num(tm.loc[h, "ci_hi"], 3), f"Test M band hi h{h}")
    qi = pd.read_csv(E / "quantile_interaction_bootstrap_fast.csv").set_index("h")
    for h, tag in ((0, "Hzero"), (12, "Htwelve")):
        add(f"IntCILo{tag}", num(qi.loc[h, "ci_lo"], 3), f"interaction CI lo h{h}")
        add(f"IntCIHi{tag}", num(qi.loc[h, "ci_hi"], 3), f"interaction CI hi h{h}")

    # audit fix: the leverage interaction is null on the TAIL grid but
    # significantly negative at the centre. Report the median interaction so the
    # prose can be scoped to the tail grid instead of "null everywhere".
    def _ixn(tau, h):
        r = qlp[(np.isclose(qlp.tau, tau)) & (qlp.h == h)]
        return float(r.beta_interaction.iloc[0]), float(r.pval_interaction.iloc[0])

    _dixn, _dixnp = _ixn(0.50, 12)
    add("DeltaIxnMedHtwelve", num(_dixn, 3), "M1 median leverage interaction delta(0.50,h12)")
    add("DeltaIxnMedHtwelveP", pval(_dixnp), "M1 median interaction p (~9e-5)")

    # ── deconstruction section (§6): placebo gaps, signed pure-null, cum Δ ──
    import json as _json
    pm = _json.load(open(E / "placebo_symmetric_meta.json"))
    gaps = pm["dgp"]["gaps"]["empirical"]
    for h, tag in (("0", "Hzero"), ("12", "Htwelve")):
        g = gaps[h]
        add(f"GapReal{tag}", num(g["gap_real"], 3), f"placebo gap_real h{h}")
        add(f"GapBandLo{tag}", num(g["gap_placebo_ci_lo"], 3), "sym band lo")
        add(f"GapBandHi{tag}", num(g["gap_placebo_ci_hi"], 3), "sym band hi")
    pnc2 = pd.read_csv(E / "pure_null_circular_shift_by_horizon.csv").set_index("h")
    add("PnCircSignedHtwelve", num(pnc2.loc[12, "artifact_beta_signed_mean"], 3),
        "signed null mean h12 (~0)")
    add("PnCircQloHtwelve", num(pnc2.loc[12, "null_q025"], 2), "null band lo h12")
    add("PnCircQhiHtwelve", num(pnc2.loc[12, "null_q975"], 2), "null band hi h12")
    pec = pd.read_csv(E / "exceedance_paired_cumulative.csv")
    rc = pec[(np.isclose(pec.alpha, 0.01)) & (pec.h == 12)].iloc[0]
    add("DeltaExcCumHtwelve", num(rc.delta, 4), "N8 cumulative Δ h12")
    add("DeltaExcCumHtwelveP", pval(rc.pval), "N8 p=0.023")
    pe_ = pd.read_csv(E / "exceedance_paired.csv")
    rp = pe_[(np.isclose(pe_.alpha, 0.01)) & (pe_.h == 12)].iloc[0]
    add("DeltaExcHtwelveP", pval(rp.pval), "per-period Δ h12 p")

    # ── exceedance / the bounded null — C37, N9 ──
    pe = pd.read_csv(E / "exceedance_paired.csv")
    r = pe[(np.isclose(pe.alpha, 0.01)) & (pe.h == 0)].iloc[0]
    add("DeltaExcImpact", num(r.delta, 5), "C37 +0.00053")
    add("DeltaExcImpactP", pval(r.pval), "canonical p=0.11")
    er = pd.read_csv(E / "exceedance_results.csv")
    er = er[er.method == "lpm"].set_index(["alpha", "h", "side"])
    for a, atag in ((0.01, "One"), (0.05, "Five")):
        for s, stag in (("down", "Down"), ("up", "Up")):
            add(f"Exc{stag}{atag}Hzero", num(er.loc[(a, 0, s), "beta"], 5),
                f"exceedance LPM beta {s} alpha={a} h0")

    # ── MDE / SESOI — C43-C45 ──
    mde = pd.read_csv(E / "mde_equivalence.csv")
    # alpha=0.01 row EXPLICITLY (the first exceedance_delta row is alpha=0.10 —
    # taking .iloc[0] printed the wrong bound in the intro; caught at compile).
    exc = mde[(mde.object == "exceedance_delta")
              & (mde.alpha_or_measure.str.startswith("alpha=0.01"))].iloc[0]
    skw = mde[mde.alpha_or_measure == "skew_tail01"].iloc[0]
    add("MdeEightyExc", num(exc.mde_80, 5), "C43 0.00094")
    add("MdeEightySkew", num(skw.mde_80, 5), "C43")
    add("SesoiStrict", num(exc.sesoi_beta_p50p95, 4), "C44 0.0011")
    add("SesoiIqr", num(exc.sesoi_beta_iqr, 4), "C44 0.0038")
    add("SesoiPtenNinety", num(exc.sesoi_beta_p10p90, 4), "C44 0.0014")

    # ── skew test — C39-C42 ──
    sk = pd.read_csv(E / "skew_test.csv").set_index("measure")
    add("SkewTailOne", num(sk.loc["skew_tail01", "beta"], 5), "C39 +0.00071")
    add("SkewTailOneP", pval(sk.loc["skew_tail01", "pval"]), "p=0.012")
    add("SkewTailFiveP", pval(sk.loc["skew_tail05", "pval"]), "C41 null")
    add("ZthreeWinsor", num(sk.loc["z3_winsor", "beta"], 3), "C42 -0.029")

    # ── dual pure-null — N3 ──
    for mode, tag in (("circular_shift", "Circ"), ("innov_shuffle", "Innov")):
        pn = pd.read_csv(E / f"pure_null_{mode}_by_horizon.csv").set_index("h")
        add(f"Pn{tag}Htwelve", f"{pn.loc[12, 'ratio']:.2f}", f"N3 {mode} ratio h12")
        add(f"Pn{tag}Htwentyfour", f"{pn.loc[24, 'ratio']:.2f}", f"N3 {mode} h24")
    pnc = pd.read_csv(E / "pure_null_circular_shift_by_horizon.csv").set_index("h")
    add("PnCircPermPHtwentyfour", pval(pnc.loc[24, "perm_pval_abs"]), "perm p h24 0.038")

    # ── BTC placebo — N1 ──
    btc = pd.read_csv(E / "btc_vs_eth_profile.csv")
    btc = btc[btc.h.astype(str) != "h"].astype(float).set_index("h")
    for h, tag in ((0, "Hzero"), (12, "Htwelve"), (24, "Htwentyfour")):
        add(f"BtcOverEth{tag}", f"{btc.loc[h, 'beta_001_btc_over_eth']:.2f}",
            f"N1 BTC/ETH beta01 h{h}")
    add("BtcBetaQoneHzero", num(btc.loc[0, "btc_beta_001"], 3), "N1")
    add("BtcBetaQoneHtwelve", num(btc.loc[12, "btc_beta_001"], 3), "N1")
    add("BtcRatioTailMedImpact", f"{btc.loc[0, 'btc_ratio_tail_med']:.1f}", "N1 46.8x")

    # ── OOS — N2 (headline cells: moderate tails, h=1, vs nested) ──
    oos = pd.read_csv(E / "oos_predictive.csv")
    qc = oos[oos.benchmark == "qr_controls"].set_index(["tau", "h"])
    for tau, tag in ((0.05, "Five"), (0.95, "NinetyFive")):
        c = qc.loc[(tau, 1)]
        add(f"OosSkill{tag}Hone", pct1(100 * c.skill), f"N2 skill% tau={tau} h1")
        add(f"OosP{tag}Hone", pval(c.dm_pval), "DM p")
    # OOS multiple-testing: Benjamini-Hochberg FDR across the 20-cell nested grid
    # (referee fix — the surviving positive finding must face the same discipline
    #  as the demolished prior). Deterministic post-processing of the DM p-values.
    qa = oos[oos.benchmark == "qr_controls"].copy()
    pv = qa["dm_pval"].to_numpy(dtype=float)
    mm = len(pv); od = np.argsort(pv); bh = np.empty(mm); prev = 1.0
    for i in range(mm - 1, -1, -1):
        prev = min(prev, pv[od[i]] * mm / (i + 1)); bh[od[i]] = prev
    qa["p_bh"] = bh
    qa = qa.set_index(["tau", "h"])
    add("OosPFiveHoneFDR", pval(float(qa.loc[(0.05, 1), "p_bh"])), "OOS BH-FDR p tau=.05 h1")
    add("OosPNinetyFiveHoneFDR", pval(float(qa.loc[(0.95, 1), "p_bh"])), "OOS BH-FDR p tau=.95 h1")
    add("OosNsurviveFDR", intc(int((bh < 0.05).sum())), "OOS cells surviving BH-FDR<.05")

    # ── OLS-LP mean benchmark (C49): the mean is blind ──
    ols = pd.read_csv(E / "ols_lp_hac_benchmark.csv").set_index("h")
    add("OlsLpBetaHzero", num(ols.loc[0, "beta_shock_ols"], 4), "C49 ~-0.0001")
    add("OlsLpPHzero", f"{ols.loc[0, 'pval_hac']:.2f}", "C49 HAC p")

    # ── vol response — C46 ──
    vr = pd.read_csv(E / "vol_response.csv")
    rv = vr[vr.measure == "rv"].set_index("h")
    add("VolRespHzero", num(rv.loc[0, "beta"], 3), "C46 +0.021")
    add("VolRespHtwelve", num(rv.loc[12, "beta"], 3), "C46 +0.065")

    # ── size ratio — C16-C28 (key cells) ──
    sz = pd.read_csv(E / "size_ratio.csv").set_index("metric")
    add("SizeShareMean", f"{sz.loc['mean_daily_defi_liq_vs_daily_eth_spot_perp_turnover', 'ratio_pct']:.4f}",
        "C16 mean daily share %")
    add("MeanDailyLiqM", f"{sz.loc['mean_daily_defi_liq_vs_daily_eth_spot_perp_turnover', 'numerator_defi_usd'] / 1e6:.2f}",
        "C24 $1.46M/day")
    add("MedianDailyLiqK", f"{sz.loc['median_daily_defi_liq_vs_daily_eth_spot_perp_turnover', 'numerator_defi_usd'] / 1e3:.1f}",
        "C23 $9.3k/day")
    add("SizeShareWorstDay", f"{sz.loc['max_day_defi_liq_vs_daily_eth_spot_perp_turnover', 'ratio_pct']:.2f}",
        "C18 worst-day share %")
    add("MaxDayM", intc(sz.loc["max_day_defi_liq_vs_daily_eth_spot_perp_turnover", "numerator_defi_usd"] / 1e6),
        "C25 $307M")
    _ms_m, _ms_e = f"{sz.loc['median_daily_defi_liq_vs_daily_eth_spot_perp_turnover', 'ratio_pct']:.0e}".split("e")
    add("SizeMedianShare", f"{_ms_m}\\times10^{{{int(_ms_e)}}}",
        "median daily share, % (LaTeX sci notation)")
    add("SizeTotalVsCascade",
        f"{sz.loc['total_sample_defi_liq_vs_oct2025_single_day_cascade', 'ratio_pct']:.0f}",
        "C28 whole-sample total = 13% of one cascade day")

    # ── same-spec SE ratio — N7 ──
    se = pd.read_csv(E / "se_ratio_nb07.csv")
    add("SeRatioMin", f"{se.ratio.min():.1f}", "N7 1.6")
    add("SeRatioMax", f"{se.ratio.max():.1f}", "N7 3.3")

    # ── subsample stability — N5 ──
    sub = pd.read_csv(E / "subsample_stability.csv")
    loo = sub[(sub.subsample == "loo_aug2024") & (sub.object == "beta_q01")
              & (sub.h == 12)].iloc[0]
    add("BetaQoneHtwelveLoo", num(loo.estimate, 3), "N5 -0.171 leave-out Aug-2024")

    # ── additional robustness battery — added 2026-06-18 ──
    # A3 positive control: the battery's demonstrated detection floor (graduates
    # EQUIVALENT@0.5x -> INCONCLUSIVE@1x -> NON-NEGLIGIBLE@2x the SESOI).
    pc = pd.read_csv(E / "positive_control.csv")
    pcm = pc[pc.object == "mde_tost"].set_index("mult")
    add("PosCtrlDeltaHalf", num(pcm.loc[0.5, "estimate"], 5), "A3 planted Δ @0.5x SESOI (EQUIV)")
    add("PosCtrlDeltaOne", num(pcm.loc[1.0, "estimate"], 5), "A3 planted Δ @1x SESOI (INCONCLUSIVE)")
    add("PosCtrlDeltaTwo", num(pcm.loc[2.0, "estimate"], 5), "A3 planted Δ @2x SESOI (NON-NEGLIGIBLE)")

    # A4 SESOI sensitivity: equivalence flip thresholds (no cherry-picked bound).
    a4 = _json.load(open(E / "sesoi_sensitivity_meta.json"))
    s_by = {o["tag"]: o["s_equiv_star"] for o in a4["objects"]}
    add("SesoiEquivExc", num(s_by["exc_delta_a01_h0"], 4), "A4 S_equiv* exceedance Δ 0.0011")
    add("SesoiEquivSkew", num(s_by["skew_tail01"], 4), "A4 S_equiv* whisper 0.0012")

    # A5 MakerDAO coverage gap: bounded exceedance asymmetry robust; level sensitive.
    mxe = _json.load(open(E / "maker_exceedance_meta.json"))
    add("MakerExcOverSesoi", f"{mxe['max_abs_delta_over_sesoi']:.2f}",
        "A5 max|Δexc|/SESOI <=0.48 (bound holds under every allocation)")
    mkb = pd.read_csv(E / "maker_bound.csv")
    mkb25 = mkb[mkb.m <= 0.25 + 1e-9]
    add("MakerLevelImpactPct", f"{100 * mkb25[mkb25.h == 0].rel_dbeta.max():.0f}",
        "A5 max level displacement h0 ~24%")
    add("MakerLevelMediumPct", f"{100 * mkb25[mkb25.h == 12].rel_dbeta.max():.0f}",
        "A5 max level displacement h12 ~65% (prop_oi)")

    # B5 leverage interaction δ: MDE detectability (underpowered, not a tight null).
    mdi = pd.read_csv(E / "mde_interaction.csv")
    add("MdeDeltaHzero", num(mdi[mdi.h == 0].mde_80.iloc[0], 3), "B5 MDE_δ@80 h0 0.109")
    add("MdeDeltaRatioMin", f"{mdi.ratio_mde80_to_direct.min():.1f}", "B5 min ratio 2.1x direct")
    add("MdeDeltaRatioMax", f"{mdi.ratio_mde80_to_direct.max():.1f}", "B5 max ratio 3.4x direct")

    # A1 RIF unconditional lens: combined tail-widening (vol channel) at impact.
    # The asymmetry recombination -(g_0.01+g_0.99) equals the conditional Δ to
    # machine precision (stated in prose), so no separate macro is needed for it.
    rg = pd.read_csv(E / "rif_lp_gap.csv").set_index("h")
    add("RifVolWideningImpact", num(rg.loc[0, "gap_g"], 3), "A1 combined tail-widening h0 (vol channel)")

    # ── non-destructive check mode (§0 non-regression guard) ──
    new = {name: value for name, value, _ in macros}
    if "--check" in sys.argv:
        return _check(new)

    # ── write ──
    lines = [
        "% numbers.tex — GENERATED by scripts/paper/make_numbers.py — DO NOT EDIT",
        "% Every number cited in the prose must come from a macro defined here",
        "% (provenance: data/econ/*.csv canonical artefacts, 2026-06-12).",
        "",
    ]
    for name, value, comment in macros:
        lines.append(f"\\newcommand{{\\{name}}}{{\\ensuremath{{{value}}}}}  % {comment}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT}  ({len(macros)} macros)")
    for line in lines[4:12]:
        print("   ", line)
    print("    ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
