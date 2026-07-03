#!/usr/bin/env python3
"""
run_mde_interaction.py  —  [Leverage-interaction MDE — turns the 0/6 interaction null from
"the leverage-cycle amplification slope is indistinguishable from zero" into a
DETECTABILITY statement: the smallest delta-slope-shift the design could have
detected, set against the direct effect it would have to amplify.]

MINIMUM DETECTABLE EFFECT (MDE) for the DELTA (regime-slope-shift) interaction
object at the headline tail quantile tau=0.01. This is the inferential companion
to the manuscript's leverage-interaction paragraph: the `shock x oi_high` interaction is a
non-rejection at every reported horizon, and on its own that is uninformative —
it could mean "leverage-cycle amplification is genuinely absent" or "we had no
power to see it". This script makes the distinction explicit by computing the
smallest interaction slope-shift the bootstrap could have detected (MDE_delta),
and reporting it as a ratio against the *direct* tail effect |beta_direct| the
interaction would have to amplify.

WHY THIS IS A *DERIVED* OBJECT (no re-estimation, no DGP)
--------------------------------------------------------
This is a thin, auditable POST-PROCESSOR — it reuses the EXACT se_from_ci -> MDE
machinery of run_mde_equivalence.py (specified in mde_equivalence_meta.json), but
applies it to the CANONICAL interaction bootstrap CIs already on disk:

    quantile_interaction_bootstrap_fast.csv  (the percentile [2.5, 97.5] CIs of
    the delta = interaction slope-shift, n_boot=1000, keyed by horizon h; the
    file is the tau=0.01 headline-tail object).

    SE          = (ci_hi - ci_lo) / 2 / z_{0.975}
    MDE_delta@80 = (z_{0.975} + z_{0.80}) * SE

Re-deriving SE from the published half-width (rather than re-running the
bootstrap) guarantees MDE_delta is exactly consistent with the canonical
intervals and keeps this a NEW read-only add-on. NOTHING is re-estimated; the
locked QLP equation and the canonical bootstrap are untouched.

IDENTIFICATION GUARDRAIL (do NOT import the asymmetry object's reasoning)
------------------------------------------------------------------------
The asymmetry/MDE-equivalence machine (run_mde_equivalence.py) runs a TOST
against a locked tail-probability SESOI. That logic does NOT transfer here:

  * delta is a REGIME-SLOPE-SHIFT (the change in the shock's tail effect when
    oi_high flips on), NOT a down-minus-up difference. It therefore does NOT
    inherit the sigma-cancellation that narrows the asymmetry object's CI — and
    that absence of cancellation is precisely WHY the delta CI is wide. Importing
    the asymmetry object's bounded-negligible reasoning here would be a category
    error.
  * There is NO pre-registered delta-SESOI in interaction-slope units. We
    therefore report MDE ONLY — no TOST, no equivalence verdict, no SESOI band.
    The honest reading is the MDE/|beta_direct| ratio, not an equivalence call.

DECISION OBJECT (pre-planned branch)
------------------------------------
For each h we report MDE_delta@80 and the ratio MDE_delta / |beta_direct(0.01,h)|
(direct effect = |beta_shock| from quantile_lp_results.csv at tau=0.01).

  * CONFIRMATORY (expected): MDE_delta >> |beta_direct| (ratio > 1, materially)
    => the interaction null is UNDERPOWERED — a DETECTABILITY null. The design
    could only have caught an amplification several times the direct effect, so
    the non-rejection cannot bound amplification to be small. This supports the
    leverage-interaction prose already in the manuscript.
  * NON-CONFIRMATORY: MDE_delta SMALL (comparable to or below |beta_direct|,
    ratio <~ 1) => the non-rejection would be a genuine TIGHT null, and the
    leverage-interaction prose must FLIP to bounded-negligible language. If this fires it is SURFACED
    in the verdict, not buried.

OUTPUT (data/econ/ — NEW files only)
------------------------------------
- mde_interaction.csv   columns:
    tau, h, ci_lo, ci_hi, se, mde_50, mde_80,
    beta_direct, abs_beta_direct, ratio_mde80_to_direct, verdict
- mde_interaction_meta.json   formulas, the per-h decision object, the
    identification guardrail, the pre-planned branch + which branch fired,
    provenance. Mirrors mde_equivalence_meta.json house style.

NEVER touches mde_equivalence.csv or quantile_interaction_bootstrap_fast.csv.

CLI
---
    # smoke (writes to /tmp ONLY)
    .venv/bin/python scripts/aux/run_mde_interaction.py \
        --horizons 0,12 --out_dir /tmp/dmde_smoke

    # canonical horizons (run separately; default out_dir is data/econ)
    .venv/bin/python scripts/aux/run_mde_interaction.py

The arithmetic is identical for smoke vs canonical: the canonical interaction
bootstrap CIs (n_boot=1000) are already on disk, so even a 2-horizon smoke reads
the SAME canonical CIs — only the set of horizons reported differs.
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
# Locked spec constants — IDENTICAL machinery to run_mde_equivalence.py
# (se_from_ci -> MDE), per mde_equivalence_meta.json.
# ──────────────────────────────────────────────────────────────
ALPHA_TEST: float = 0.05            # two-sided test level for MDE
INTERACTION_TAU: float = 0.01       # headline tail quantile of the bootstrap object
DEFAULT_HORIZONS: tuple[int, ...] = (0, 3, 6, 12, 24)

# Verdict labels for the pre-planned MDE-vs-direct branch.
UNDERPOWERED = "UNDERPOWERED-NULL"   # MDE_delta materially > |beta_direct| (CONFIRMATORY)
TIGHT = "TIGHT-NULL"                 # MDE_delta comparable-to/below |beta_direct| (NON-CONFIRMATORY)
# Ratio threshold separating the two branches. > 1 means the design could only
# detect an amplification LARGER than the direct effect itself.
RATIO_TIGHT_CUTOFF: float = 1.0

OUT_COLS: list[str] = [
    "tau", "h", "ci_lo", "ci_hi", "se", "mde_50", "mde_80",
    "beta_direct", "abs_beta_direct", "ratio_mde80_to_direct", "verdict",
]

INTERACTION_CSV = "quantile_interaction_bootstrap_fast.csv"
QLP_CSV = "quantile_lp_results.csv"
OUT_CSV = "mde_interaction.csv"
OUT_META = "mde_interaction_meta.json"


# ──────────────────────────────────────────────────────────────
# Core MDE arithmetic  — reused VERBATIM from run_mde_equivalence.py
# (SE = half-width / z_{0.975};  MDE = (z_{0.975}+z_power)*SE).
# ──────────────────────────────────────────────────────────────
def se_from_ci(ci_lo: float, ci_hi: float, alpha_ci: float = ALPHA_TEST) -> float:
    """Infer the (Gaussian-equivalent) SE from a symmetric percentile CI.

    The bootstrap CI is a [alpha_ci/2, 1-alpha_ci/2] percentile interval, so its
    half-width ~ z_{1-alpha_ci/2} * SE. Invert: SE = halfwidth / z.
    """
    z = stats.norm.ppf(1.0 - alpha_ci / 2.0)
    return (float(ci_hi) - float(ci_lo)) / 2.0 / z


def mde(se: float, power: float, alpha_test: float = ALPHA_TEST) -> float:
    """Standard two-sided MDE: (z_{1-alpha/2} + z_power) * SE, in native units."""
    z_a = stats.norm.ppf(1.0 - alpha_test / 2.0)
    z_p = stats.norm.ppf(power)
    return (z_a + z_p) * se


def branch_verdict(mde_80: float, abs_beta_direct: float) -> str:
    """Pre-planned MDE-vs-direct branch.

    UNDERPOWERED-NULL (CONFIRMATORY): MDE_delta@80 materially exceeds the direct
        effect — the design could only have caught an amplification several times
        the direct tail response, so the non-rejection is a detectability null.
    TIGHT-NULL (NON-CONFIRMATORY): MDE_delta@80 is comparable to or below the
        direct effect — the non-rejection genuinely bounds amplification small;
        SURFACE this (the leverage-interaction prose would have to flip to
        bounded-negligible).
    """
    if not np.isfinite(abs_beta_direct) or abs_beta_direct <= 0:
        return UNDERPOWERED  # no finite direct effect to amplify; treat as detectability-only
    ratio = mde_80 / abs_beta_direct
    return TIGHT if ratio <= RATIO_TIGHT_CUTOFF else UNDERPOWERED


# ──────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────
def load_inputs(in_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the canonical interaction bootstrap CIs + the QLP direct effects."""
    boot_path = in_dir / INTERACTION_CSV
    qlp_path = in_dir / QLP_CSV
    if not boot_path.exists():
        raise FileNotFoundError(
            f"{boot_path} not found — the canonical interaction bootstrap "
            f"({INTERACTION_CSV}) must be on disk."
        )
    if not qlp_path.exists():
        raise FileNotFoundError(
            f"{qlp_path} not found — the canonical QLP results "
            f"({QLP_CSV}) supply the direct tail effects."
        )
    return pd.read_csv(boot_path), pd.read_csv(qlp_path)


def direct_beta(qlp: pd.DataFrame, tau: float, h: int) -> float:
    """|beta_shock| (the direct tail effect) at (tau, h) from quantile_lp_results."""
    sel = qlp[(np.isclose(qlp["tau"], tau)) & (qlp["h"] == h)]
    if sel.empty:
        raise ValueError(f"{QLP_CSV} has no row for tau={tau}, h={h}.")
    return float(sel.iloc[0]["beta_shock"])


def collect_rows(
    df_boot: pd.DataFrame,
    qlp: pd.DataFrame,
    horizons: tuple[int, ...],
    power: float,
) -> tuple[list[dict], list[dict]]:
    """Build CSV rows + per-h diagnostics for the delta-interaction MDE object."""
    rows: list[dict] = []
    diags: list[dict] = []
    for h in horizons:
        sel = df_boot[df_boot["h"] == h]
        if sel.empty:
            raise ValueError(
                f"{INTERACTION_CSV} has no bootstrap row for h={h}."
            )
        r = sel.iloc[0]
        ci_lo, ci_hi = float(r["ci_lo"]), float(r["ci_hi"])
        se = se_from_ci(ci_lo, ci_hi)
        mde_50 = mde(se, 0.50)
        mde_80 = mde(se, power)

        bd = direct_beta(qlp, INTERACTION_TAU, h)
        abs_bd = abs(bd)
        ratio = mde_80 / abs_bd if (np.isfinite(abs_bd) and abs_bd > 0) else np.nan
        verdict = branch_verdict(mde_80, abs_bd)

        rows.append({
            "tau": INTERACTION_TAU,
            "h": int(h),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "se": float(se),
            "mde_50": float(mde_50),
            "mde_80": float(mde_80),
            "beta_direct": float(bd),
            "abs_beta_direct": float(abs_bd),
            "ratio_mde80_to_direct": float(ratio),
            "verdict": verdict,
        })
        diags.append({
            "tau": INTERACTION_TAU,
            "h": int(h),
            "ci_lo_reported": ci_lo,
            "ci_hi_reported": ci_hi,
            "se_inferred": float(se),
            "mde_50": float(mde_50),
            "mde_80_at_power": float(mde_80),
            "beta_direct": float(bd),
            "abs_beta_direct": float(abs_bd),
            "ratio_mde80_to_direct": (float(ratio) if np.isfinite(ratio) else None),
            "verdict": verdict,
            "reading": (
                f"MDE_delta@80={mde_80:.4g} is {ratio:.2f}x the direct tail "
                f"effect |beta_direct|={abs_bd:.4g}: the design could only have "
                f"detected an amplification {('larger than' if ratio > 1 else 'at/below')} "
                f"the direct effect, so the interaction non-rejection is a "
                f"{'DETECTABILITY (underpowered) null' if verdict == UNDERPOWERED else 'genuine TIGHT null'}."
            ) if np.isfinite(ratio) else (
                f"MDE_delta@80={mde_80:.4g}; no finite direct effect to ratio against."
            ),
        })
    return rows, diags


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(
    rows: list[dict],
    diags: list[dict],
    horizons: tuple[int, ...],
    power: float,
    in_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[OUT_COLS]
    csv_path = out_dir / OUT_CSV
    df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}", flush=True)

    # Which pre-planned branch fired (overall)?
    verdicts = {r["verdict"] for r in rows}
    if verdicts == {UNDERPOWERED}:
        branch_fired = "CONFIRMATORY (all horizons UNDERPOWERED-NULL)"
    elif TIGHT in verdicts and UNDERPOWERED in verdicts:
        branch_fired = "MIXED (some horizons TIGHT-NULL — see per-h verdicts; SURFACE)"
    elif verdicts == {TIGHT}:
        branch_fired = ("NON-CONFIRMATORY (all horizons TIGHT-NULL — "
                        "leverage-interaction prose must flip; SURFACE)")
    else:
        branch_fired = "UNDETERMINED"

    z_a = stats.norm.ppf(1.0 - ALPHA_TEST / 2.0)
    meta = {
        "test": "Minimum detectable effect (MDE) for the DELTA interaction "
                "slope-shift at tau=0.01 (leverage-interaction MDE).",
        "what_this_does": (
            "Converts the shock x oi_high interaction non-rejection into a "
            "detectability statement: the smallest leverage-cycle amplification "
            "(delta-slope-shift) the canonical bootstrap could have detected "
            "(MDE_delta@80), set as a ratio against the direct tail effect "
            "|beta_direct| it would have to amplify."
        ),
        "derived_not_reestimated": (
            f"CIs read from canonical {INTERACTION_CSV}; direct effects read from "
            f"canonical {QLP_CSV}. SE inferred from each CI half-width / z_{{0.975}} "
            f"so MDE is exactly consistent with the published intervals. NOTHING "
            f"is re-estimated; the locked QLP equation and the canonical bootstrap "
            f"are untouched."
        ),
        "object": {
            "delta_interaction": (
                f"beta_interaction (the change in the shock's tail effect when "
                f"oi_high flips on) at tau={INTERACTION_TAU}, horizons "
                f"{list(horizons)}; CIs from {INTERACTION_CSV}."
            ),
            "beta_direct": (
                f"|beta_shock| at tau={INTERACTION_TAU} from {QLP_CSV} — the "
                f"direct tail effect the interaction would have to amplify."
            ),
        },
        "formulas": {
            "se_from_ci": "SE = (ci_hi - ci_lo) / 2 / z_{1-alpha/2}",
            "mde": "MDE = (z_{1-alpha/2} + z_power) * SE  (interaction-slope units)",
            "ratio": "ratio = MDE_delta@80 / |beta_direct(tau,h)|",
            "alpha_test": ALPHA_TEST,
            "z_{1-alpha/2}": float(z_a),
            "z_power_default": float(stats.norm.ppf(power)),
            "power_default": power,
            "power_also_reported": 0.50,
            "mde_50_meaning": (
                "power=0.50 => z_power=0 => MDE = z_{1-alpha/2}*SE = CI half-width "
                "(smallest 'just-significant' delta-slope-shift)."
            ),
            "machinery_source": (
                "Identical se_from_ci -> MDE machinery as run_mde_equivalence.py "
                "(mde_equivalence_meta.json)."
            ),
        },
        "identification_guardrail": {
            "delta_is_not_asymmetry": (
                "delta is a REGIME-SLOPE-SHIFT (effect change when oi_high turns "
                "on), NOT a down-minus-up difference. It does NOT inherit the "
                "sigma-cancellation that narrows the asymmetry object's CI — that "
                "absence of cancellation is WHY the delta CI is wide."
            ),
            "no_tost": (
                "There is NO pre-registered delta-SESOI in interaction-slope "
                "units, so NO TOST / equivalence test is run against the "
                "asymmetry SESOI. MDE is reported ONLY. The bounded-negligible "
                "reasoning of the asymmetry/MDE-equivalence object is NOT "
                "imported."
            ),
        },
        "decision_branch": {
            "pre_planned": {
                UNDERPOWERED: (
                    "CONFIRMATORY: MDE_delta@80 materially > |beta_direct| "
                    "(ratio > 1) => detectability (underpowered) null; supports "
                    "the manuscript's leverage-interaction prose."
                ),
                TIGHT: (
                    "NON-CONFIRMATORY: MDE_delta@80 <= |beta_direct| (ratio <= 1) "
                    "=> genuine tight null; the leverage-interaction prose must FLIP to "
                    "bounded-negligible language. SURFACED if it fires."
                ),
                "ratio_cutoff": RATIO_TIGHT_CUTOFF,
            },
            "branch_fired": branch_fired,
        },
        "per_horizon": diags,
        "inputs": {
            "interaction_bootstrap_csv": str((in_dir / INTERACTION_CSV).resolve()),
            "quantile_lp_results_csv": str((in_dir / QLP_CSV).resolve()),
            "note": (
                "Canonical n_boot=1000 interaction CIs are already on disk; smoke "
                "and canonical read the SAME CIs — only the reported horizon set "
                "differs."
            ),
        },
        "panel": str(CFG.FILES.econ_core_full),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    meta_path = out_dir / OUT_META
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)

    # Per-convention verification echo (head/shape of the written CSV).
    print(f"\n  {OUT_CSV}  shape={df.shape}", flush=True)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False), flush=True)
    print(f"\n  Pre-planned branch fired: {branch_fired}", flush=True)
    return df


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def parse_horizons(s: str) -> tuple[int, ...]:
    try:
        hs = tuple(int(x.strip()) for x in s.split(",") if x.strip() != "")
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--horizons must be comma-separated ints: {e}")
    if not hs:
        raise argparse.ArgumentTypeError("--horizons is empty.")
    return hs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--power", type=float, default=0.80,
                    help="Target power for MDE@80 (default 0.80). MDE@50 always "
                         "reported alongside.")
    ap.add_argument("--horizons", type=parse_horizons,
                    default=DEFAULT_HORIZONS,
                    help="Comma-separated horizons to report "
                         f"(default {','.join(map(str, DEFAULT_HORIZONS))}).")
    ap.add_argument("--in_dir", type=Path, default=ECON_DIR,
                    help=f"Directory holding {INTERACTION_CSV} & {QLP_CSV}. "
                         f"Default: {ECON_DIR}")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR,
                    help="Output directory. Smoke runs MUST pass --out_dir /tmp/...")
    args = ap.parse_args()

    if not (0.0 < args.power < 1.0):
        ap.error("--power must be in (0, 1).")

    print("run_mde_interaction — MDE for the delta interaction slope-shift",
          flush=True)
    print(f"  tau={INTERACTION_TAU}  power={args.power}  alpha_test={ALPHA_TEST}",
          flush=True)
    print(f"  horizons={list(args.horizons)}", flush=True)
    print(f"  in_dir={args.in_dir}", flush=True)
    print(f"  out_dir={args.out_dir}", flush=True)

    t0 = time.time()

    # 1. read canonical objects (no re-estimation)
    df_boot, qlp = load_inputs(args.in_dir)

    # 2. build rows + diagnostics
    rows, diags = collect_rows(df_boot, qlp, args.horizons, args.power)

    # 3. save (NEW files only)
    save_outputs(rows, diags, args.horizons, args.power, args.in_dir, args.out_dir)

    print(f"\nDone. Total wall time: {(time.time()-t0):.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
