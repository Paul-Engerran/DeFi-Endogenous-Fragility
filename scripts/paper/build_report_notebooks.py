#!/usr/bin/env python3
"""
build_report_notebooks.py — generate notebooks/10 + 11 (the report layer
aligned on the paper's narrative).

The notebooks are READ-ONLY reports: they import the scripts/paper functions
(zero duplication of plotting code) and display the canonical CSV slices that
back each section of the paper. They compute nothing.

Run
---
    .venv/bin/python scripts/paper/build_report_notebooks.py
    # then execute them:
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        notebooks/10_deconstruction_report.ipynb notebooks/11_survivors_report.ipynb
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]

SETUP = '''\
# Setup — read-only report over canonical artefacts (data/econ, 2026-06-12)
import sys
from pathlib import Path
ROOT = Path.cwd().parent if (Path.cwd() / "..").resolve().name else Path.cwd()
ROOT = Path.cwd().parent  # notebooks/ -> repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
pd.set_option("display.width", 140)
import style, make_figures as MF
style.apply()
from config import ECON_DIR
print("canonical dir:", ECON_DIR)'''


def nb10() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell(
        "# 10 — Deconstruction report (paper §6-§7)\n\n"
        "**Read-only report** over the canonical artefacts (2026-06-12 VM run).\n"
        "Narrative: the naive quantile-LP *appears* to show downside "
        "amplification (the bait, §6) — the diagnostics show it is a genuine "
        "but **symmetric** volatility effect, generic to crypto stress, "
        "misread as downside-specificity (§7)."))
    c.append(nbf.v4.new_code_cell(SETUP))

    c.append(nbf.v4.new_markdown_cell(
        "## 1. The bait — naive QLP IRF (Fig 3)\n"
        "β(0.01, h=0) = −0.032 → −0.241 at h=12 (≈19× the median at impact); "
        "the median barely moves. Band: block-bootstrap on the exact "
        "main-table spec."))
    c.append(nbf.v4.new_code_cell(
        'qlp = pd.read_csv(ECON_DIR / "quantile_lp_results.csv")\n'
        'qlp[qlp.h.isin([0,1,3,6,12,24])].pivot_table(index="tau", columns="h",'
        ' values="beta_shock").round(3)'))
    c.append(nbf.v4.new_code_cell("MF.fig3_qlp_irf()"))

    c.append(nbf.v4.new_markdown_cell(
        "## 2. The pillar — symmetric sign-flip placebo (Fig 4)\n"
        "A **zero-skew** world (Rademacher sign-flip on real magnitudes; the "
        "empirical volatility path is preserved exactly, no vol model) "
        "**reproduces** the left-right gap at every horizon: h=0 inside the "
        "band (no genuine impact gap), h=12 dead-centre. The vol-model-scaled "
        "variant (rolling + GARCH, genuinely distinct after the "
        "σ-cancellation fix) agrees 14/14 cells."))
    c.append(nbf.v4.new_code_cell(
        'pl = pd.read_csv(ECON_DIR / "placebo_symmetric.csv")\n'
        'pl[pl.tau.isin([0.01, 0.99])].round(4)'))
    c.append(nbf.v4.new_code_cell("MF.fig4_placebo_gap_distribution()"))

    c.append(nbf.v4.new_markdown_cell(
        "## 3. The dual pure-null (Fig 9)\n"
        "Both principled nulls give **small** artifact shares (circular-shift "
        "0.18, innovation-shuffle 0.08 at h=12), **no signed bias**, and the "
        "real β outside the null band at every horizon (permutation p ≤ .038)."
        " The historical “72%” is reproducible under neither — the "
        "long-horizon β is genuine; the *interpretation* was the artifact."))
    c.append(nbf.v4.new_code_cell(
        'circ = pd.read_csv(ECON_DIR / "pure_null_circular_shift_by_horizon.csv")\n'
        'innov = pd.read_csv(ECON_DIR / "pure_null_innov_shuffle_by_horizon.csv")\n'
        'pd.concat([circ.assign(null="circular_shift"), '
        'innov.assign(null="innov_shuffle")])'
        '[["null","h","true_beta","artifact_beta_mean","ratio",'
        '"artifact_beta_signed_mean","perm_pval_abs"]].round(4)'))
    c.append(nbf.v4.new_code_cell("MF.fig9_pure_null_dual()"))

    c.append(nbf.v4.new_markdown_cell(
        "## 4. Overlap re-creates the fake gap (Fig 5, right panel)\n"
        "The clean per-period exceedance asymmetry Δ is null at every h; the "
        "**cumulative (overlapping)** variant manufactures a significant Δ at "
        "h=12 (p=0.023) — the artifact made visible in one picture."))
    c.append(nbf.v4.new_code_cell("MF.fig5_exceedance_symmetry()"))

    c.append(nbf.v4.new_markdown_cell(
        "## 5. Not ETH-specific — the BTC-outcome placebo (Fig 8)\n"
        "The full QLP re-run with **BTC as outcome** (mirror controls) "
        "reproduces the signature at 0.56–0.75× the ETH magnitude with the "
        "same deepening: the pattern is **generic crypto stress**, not a "
        "DeFi-ETH channel. (Kernel sanity: re-fitted ETH cells match the "
        "canonical table at ~1e-17.)"))
    c.append(nbf.v4.new_code_cell(
        'prof = pd.read_csv(ECON_DIR / "btc_vs_eth_profile.csv")\n'
        'prof = prof[prof.h.astype(str) != "h"].astype(float)\n'
        'prof[prof.h.isin([0,6,12,24])].round(3)'))
    c.append(nbf.v4.new_code_cell("MF.fig8_btc_vs_eth()"))
    return nb


def nb11() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell(
        "# 11 — What survives (paper §8)\n\n"
        "**Read-only report.** Net of the deconstruction, three things stand: "
        "(1) liquidations → future volatility (the consensus channel, "
        "confirmed and extended to the tails); (2) a **symmetric**, "
        "out-of-sample-validated short-horizon tail-risk indicator; (3) a "
        "**bounded** null — downside-specificity is excluded beyond the SESOI."))
    c.append(nbf.v4.new_code_cell(SETUP))

    c.append(nbf.v4.new_markdown_cell(
        "## 1. The volatility channel (A4)\n"
        "Realised-volatility response positive and rising with horizon "
        "(rv: +0.021 at h=0 → +0.065 at h=12), aligned with OECD/Lehar-Parlour."))
    c.append(nbf.v4.new_code_cell(
        'vr = pd.read_csv(ECON_DIR / "vol_response.csv")\n'
        'vr[vr.h.isin([0,1,3,6,12,24])].round(4)'))

    c.append(nbf.v4.new_markdown_cell(
        "## 2. The symmetric tail predictor (A8) — in-sample…\n"
        "Both tails load positively (the exceedance LPM), Δ ≈ 0 everywhere "
        "(Fig 5 left)."))
    c.append(nbf.v4.new_code_cell(
        'res = pd.read_csv(ECON_DIR / "exceedance_results.csv")\n'
        'res[(res.method=="lpm") & (res.h.isin([0,1,12]))]'
        '[["alpha","h","side","beta","ci_lo","ci_hi"]].round(5)'))

    c.append(nbf.v4.new_markdown_cell(
        "## 3. …and out-of-sample (Table 6)\n"
        "Pinball skill vs the nested no-liquidation benchmark: positive in "
        "20/20 cells, significant at both moderate tails at short horizons "
        "(τ=.05 h1 p=.004; τ=.95 h1 p<.001). Honest caveat: does **not** beat "
        "a dedicated GARCH(1,1) — the claim is *incremental information*, "
        "not dominance."))
    c.append(nbf.v4.new_code_cell(
        'oos = pd.read_csv(ECON_DIR / "oos_predictive.csv")\n'
        'oos.assign(skill_pct=(100*oos.skill).round(2))'
        '[["benchmark","tau","h","skill_pct","dm_t","dm_pval"]]'
        '.round({"dm_t":2,"dm_pval":3})'))

    c.append(nbf.v4.new_markdown_cell(
        "## 4. The bounded null (A5/A9) — Fig 7\n"
        "Exceedance Δ@1% is equivalent-to-negligible under **all three** "
        "SESOI spans (strict included); the 1% skew whisper is statistically "
        "real (p=.012) but economically negligible under two spans and "
        "borderline under the strict one."))
    c.append(nbf.v4.new_code_cell(
        'mde = pd.read_csv(ECON_DIR / "mde_equivalence.csv")\nmde'))
    c.append(nbf.v4.new_code_cell("MF.fig7_mde_equivalence()"))

    c.append(nbf.v4.new_markdown_cell(
        "## 5. Stability — splits, leave-out-Aug-2024, block length\n"
        "The profile survives every split (~70% of the h=12 deepening without "
        "the Aug-2024 episode); Δ stays null per subsample; the MDE bound is "
        "insensitive to the MBB block length (12/24/36/48h; block=24 "
        "reproduces the headline exceedance cells exactly — built-in "
        "cross-check)."))
    c.append(nbf.v4.new_code_cell(
        'sub = pd.read_csv(ECON_DIR / "subsample_stability.csv")\n'
        'sub[sub.h.isin([0,12])].round(5)'))
    c.append(nbf.v4.new_code_cell(
        'bl = pd.read_csv(ECON_DIR / "block_sensitivity.csv")\n'
        'bl[bl.object=="delta_paired"][["block_size","h","estimate",'
        '"ci_lo","ci_hi","mde80"]].round(5)'))
    return nb


def main() -> int:
    out = ROOT / "notebooks"
    for name, builder in (("10_deconstruction_report.ipynb", nb10),
                          ("11_survivors_report.ipynb", nb11)):
        nb = builder()
        nb.metadata["kernelspec"] = {"name": "python3",
                                     "display_name": "Python 3",
                                     "language": "python"}
        nbf.write(nb, out / name)
        print(f"  wrote {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
