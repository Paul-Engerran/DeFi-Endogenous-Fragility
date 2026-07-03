#!/usr/bin/env python3
"""
make_figures.py — generate the paper's figures (vector PDF) from canonical CSVs.

One function per figure; file names are stable artifact identifiers (the set
skips fig6) — the compiled paper numbers figures by order of appearance, so
the printed numbers differ for the last four (printed 5=fig9, 6=fig8, 7=fig5, 8=fig7). The paper layer
computes NO statistics — only reads canonical artefacts and (where the plotted object is
itself defined as a transformation, e.g. the per-sim placebo gap) applies the
documented arithmetic.

  fig1_liquidations_timeseries   panel: liq USD + ETH price
  fig2_return_distribution       ETH hourly returns, log-y, normal overlay
  fig3_qlp_irf                   the bait: beta_h(tau) fanning + bootstrap band on tau=.01
  fig4_placebo_gap_distribution  per-sim placebo gaps vs real gap (sign_flip)
  fig5_exceedance_symmetry       beta_down/up + Delta per-period VS cumulative
  fig7_mde_equivalence           forest plot: estimates vs SESOI spans
  fig8_btc_vs_eth                BTC-outcome placebo profile vs ETH
  fig9_pure_null_dual            both nulls: artifact magnitude + ratio profile

Run
---
    .venv/bin/python scripts/paper/make_figures.py            # all
    .venv/bin/python scripts/paper/make_figures.py fig4 fig8  # subset
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from config import CFG, ECON_DIR  # noqa: E402
import style  # noqa: E402
from style import (C_ALT, C_BAND, C_DOWN, C_MAIN, C_NULL, C_UP, COLW, FULLW,
                   H_STD, H_TALL, save)  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"


# ────────────────────────────────────────────────────────────
def fig1_liquidations_timeseries() -> None:
    df = pd.read_parquet(CFG.FILES.econ_core_full,
                         columns=["date", "liq_usd_total", "close_eth_spot"])
    daily = (df.set_index("date")
               .resample("1D").agg({"liq_usd_total": "sum",
                                    "close_eth_spot": "last"}))
    fig, ax = plt.subplots(figsize=(FULLW, H_STD))
    ax.bar(daily.index, daily.liq_usd_total / 1e6, width=1.0,
           color=C_DOWN, lw=0, alpha=0.85,
           label="DeFi liquidations (daily, \\$M, left)")
    ax.set_ylabel("Liquidations (\\$M / day)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.01)
    ax2 = ax.twinx()
    ax2.plot(daily.index, daily.close_eth_spot, color=C_MAIN, lw=0.9,
             label="ETH price (right)")
    ax2.set_ylabel("ETH price (\\$)")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    save(fig, FIG_DIR, "fig1_liquidations_timeseries")


def fig2_return_distribution() -> None:
    df = pd.read_parquet(CFG.FILES.econ_core_full, columns=["ret_eth_perp"])
    r = df.ret_eth_perp.dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(COLW, H_STD))
    bins = np.linspace(-16, 10, 131)
    ax.hist(r, bins=bins, density=True, color=C_MAIN, alpha=0.75, lw=0)
    x = np.linspace(-16, 10, 400)
    mu, sd = r.mean(), r.std()
    ax.plot(x, np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
            color=C_NULL, lw=1.0, ls="--", label="Normal($\\mu$, $\\sigma$)")
    ax.axvline(-5, color=C_DOWN, lw=0.6, ls=":")
    ax.axvline(5, color=C_UP, lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel("hourly log-return (\\%)")
    ax.set_ylabel("density (log scale)")
    ax.legend()
    save(fig, FIG_DIR, "fig2_return_distribution")


def fig3_qlp_irf() -> None:
    qlp = pd.read_csv(ECON_DIR / "quantile_lp_results.csv")
    m = pd.read_csv(ECON_DIR / "robustness_bootstrap_nb07_spec_fast.csv")
    fig, ax = plt.subplots(figsize=(FULLW, H_TALL))
    taus = [(0.01, C_DOWN, "-", 1.6), (0.05, C_DOWN, "--", 1.0),
            (0.50, C_NULL, "-", 1.0), (0.95, C_UP, "--", 1.0)]
    for tau, c, ls, lw in taus:
        s = qlp[np.isclose(qlp.tau, tau)].sort_values("h")
        ax.plot(s.h, s.beta_shock, color=c, ls=ls, lw=lw,
                label=f"$\\tau={tau:.2f}$")
    ms = m.sort_values("h")
    ax.fill_between(ms.h, ms.ci_lo, ms.ci_hi, color=C_DOWN, alpha=0.15, lw=0,
                    label="95\\% block-bootstrap CI ($\\tau{=}0.01$, Test M)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("horizon $h$ (hours)")
    ax.set_ylabel("$\\beta_h(\\tau)$")
    ax.legend(ncol=2)
    save(fig, FIG_DIR, "fig3_qlp_irf")


def fig4_placebo_gap_distribution() -> None:
    draws = pd.read_csv(ECON_DIR / "placebo_symmetric_draws.csv")
    summ = pd.read_csv(ECON_DIR / "placebo_symmetric.csv")
    hs = [0, 12]
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, H_STD))
    for ax, h in zip(axes, hs):
        d = draws[draws.h == h].pivot_table(index="sim", columns="tau",
                                            values="beta")
        gaps = d[0.01].abs() - d[0.99].abs()   # the documented gap statistic
        gr = float(summ[(summ.h == h)
                        & np.isclose(summ.tau, 0.01)].gap_real.iloc[0])
        ax.hist(gaps, bins=40, color=C_NULL, alpha=0.8, lw=0, density=True,
                label="symmetric placebo\n(500 sims, sign-flip)")
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        ax.axvline(lo, color=C_NULL, lw=0.7, ls=":")
        ax.axvline(hi, color=C_NULL, lw=0.7, ls=":")
        ax.axvline(gr, color=C_DOWN, lw=1.6,
                   label=f"real gap = {gr:+.3f}")
        ax.set_title(f"$h={h}$")
        ax.set_xlabel("$|\\beta(0.01)| - |\\beta(0.99)|$")
        if h == hs[0]:
            ax.set_ylabel("density")
        ax.legend()
    save(fig, FIG_DIR, "fig4_placebo_gap_distribution")


def fig5_exceedance_symmetry() -> None:
    res = pd.read_csv(ECON_DIR / "exceedance_results.csv")
    pp = pd.read_csv(ECON_DIR / "exceedance_paired.csv")
    cum = pd.read_csv(ECON_DIR / "exceedance_paired_cumulative.csv")
    alpha = 0.01
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, H_STD))

    ax = axes[0]
    for side, c, lab in (("down", C_DOWN, "$\\beta^{down}$"),
                         ("up", C_UP, "$\\beta^{up}$")):
        s = res[(np.isclose(res.alpha, alpha)) & (res.side == side)
                & (res.method == "lpm")].sort_values("h")
        ax.plot(s.h, s.beta, color=c, lw=1.2, label=lab)
        ax.fill_between(s.h, s.ci_lo, s.ci_hi, color=c, alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("per-period (clean)")
    ax.set_xlabel("horizon $h$")
    ax.set_ylabel(f"LPM coef., $\\alpha={alpha:.2f}$")
    ax.legend()

    ax = axes[1]
    p = pp[np.isclose(pp.alpha, alpha)].sort_values("h")
    c = cum[np.isclose(cum.alpha, alpha)].sort_values("h")
    ax.plot(p.h, p.delta, color=C_MAIN, lw=1.2, label="$\\Delta$ per-period")
    ax.fill_between(p.h, p.ci_lo, p.ci_hi, color=C_MAIN, alpha=0.15, lw=0)
    ax.plot(c.h, c.delta, color=C_ALT, lw=1.2, ls="--",
            label="$\\Delta$ cumulative (overlap)")
    ax.fill_between(c.h, c.ci_lo, c.ci_hi, color=C_ALT, alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("asymmetry $\\Delta = \\beta^{down} - \\beta^{up}$")
    ax.set_xlabel("horizon $h$")
    ax.legend()
    save(fig, FIG_DIR, "fig5_exceedance_symmetry")


def fig7_mde_equivalence() -> None:
    pe = pd.read_csv(ECON_DIR / "exceedance_paired.csv")
    sk = pd.read_csv(ECON_DIR / "skew_test.csv").set_index("measure")
    mde = pd.read_csv(ECON_DIR / "mde_equivalence.csv")
    exc = mde[mde.alpha_or_measure.str.contains("0.01", na=False)].iloc[0]
    sesoi = {"strict ($p_{50}{\\to}p_{95}$)": exc.sesoi_beta_p50p95,
             "$p_{10}{\\to}p_{90}$": exc.sesoi_beta_p10p90,
             "IQR": exc.sesoi_beta_iqr}

    rows = []
    r = pe[(np.isclose(pe.alpha, 0.01)) & (pe.h == 0)].iloc[0]
    rows.append(("exceedance $\\Delta$, $\\alpha{=}1\\%$, $h{=}0$",
                 r.delta, r.ci_lo, r.ci_hi))
    r = sk.loc["skew_tail01"]
    rows.append(("skew tail indicator, $\\tau{=}1\\%$", r.beta, r.ci_lo, r.ci_hi))

    fig, ax = plt.subplots(figsize=(FULLW, 2.0))
    colors = ["#DDDDDD", "#CCCCCC", "#BBBBBB"]
    for (lab, v), col in zip(sorted(sesoi.items(), key=lambda kv: -kv[1]),
                             colors):
        ax.axvspan(-v, v, color=col, alpha=0.6, lw=0, label=f"SESOI {lab}")
    for i, (lab, est, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.errorbar(est, y, xerr=[[est - lo], [hi - est]], fmt="o",
                    color=C_MAIN, capsize=2.5, ms=4, lw=1.2)
        ax.text(ax.get_xlim()[0], y + 0.22, lab, fontsize=8, va="bottom")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("effect size ($\\beta$ units per unit log-liq)")
    ax.legend(loc="lower right", fontsize=7)
    save(fig, FIG_DIR, "fig7_mde_equivalence")


def fig8_btc_vs_eth() -> None:
    prof = pd.read_csv(ECON_DIR / "btc_vs_eth_profile.csv")
    prof = prof[prof.h.astype(str) != "h"].astype(float).sort_values("h")
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, H_STD))

    ax = axes[0]
    ax.plot(prof.h, prof.eth_beta_001, color=C_MAIN, lw=1.4,
            label="ETH (main spec)")
    ax.plot(prof.h, prof.btc_beta_001, color=C_ALT, lw=1.4, ls="--",
            label="BTC outcome (mirror controls)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("horizon $h$")
    ax.set_ylabel("$\\beta_h(0.01)$")
    ax.legend()

    ax = axes[1]
    ax.plot(prof.h, prof.beta_001_btc_over_eth, color=C_NULL, lw=1.2)
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("horizon $h$")
    ax.set_ylabel("$\\beta^{BTC}_h / \\beta^{ETH}_h$ at $\\tau{=}0.01$")
    save(fig, FIG_DIR, "fig8_btc_vs_eth")


def fig9_pure_null_dual() -> None:
    circ = pd.read_csv(ECON_DIR / "pure_null_circular_shift_by_horizon.csv")
    innov = pd.read_csv(ECON_DIR / "pure_null_innov_shuffle_by_horizon.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, H_STD))

    ax = axes[0]
    ax.plot(circ.h, circ.true_abs_beta, color=C_MAIN, lw=1.5,
            label="real $|\\beta(0.01,h)|$", marker="o", ms=2.5)
    for df, c, lab in ((circ, C_ALT, "circular-shift null"),
                       (innov, C_NULL, "innovation-shuffle null")):
        ax.plot(df.h, df.artifact_beta_mean, color=c, lw=1.1, ls="--",
                label=f"{lab}: mean $|\\beta_{{null}}|$")
        ax.fill_between(df.h, df.null_q025.abs() * 0,  # baseline 0
                        np.maximum(df.null_q975.abs(), df.null_q025.abs()),
                        color=c, alpha=0.10, lw=0)
    ax.set_xlabel("horizon $h$")
    ax.set_ylabel("$|\\beta|$")
    ax.legend(fontsize=7)

    ax = axes[1]
    for df, c, lab in ((circ, C_ALT, "circular shift"),
                       (innov, C_NULL, "innovation shuffle")):
        ax.plot(df.h, 100 * df.ratio, color=c, lw=1.3, marker="o", ms=2.5,
                label=lab)
    ax.set_ylim(0, 100)
    ax.set_xlabel("horizon $h$")
    ax.set_ylabel("artifact share: mean$|\\beta_{null}|/|\\beta|$ (\\%)")
    ax.legend()
    save(fig, FIG_DIR, "fig9_pure_null_dual")


ALL = {fn.__name__.split("_")[0]: fn for fn in (
    fig1_liquidations_timeseries, fig2_return_distribution, fig3_qlp_irf,
    fig4_placebo_gap_distribution, fig5_exceedance_symmetry,
    fig7_mde_equivalence, fig8_btc_vs_eth, fig9_pure_null_dual)}


def main() -> int:
    style.apply()
    wanted = sys.argv[1:] or list(ALL)
    for key in wanted:
        ALL[key]()
    print(f"Done — {len(wanted)} figure(s) in paper/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
