#!/usr/bin/env python3
"""
run_mde_equivalence.py  —  [ROBUSTNESS — MDE + TOST equivalence; does NOT change the main spec]

MINIMUM DETECTABLE EFFECT (MDE) + EQUIVALENCE (TOST) for the downside-asymmetry
objects. This is the single most critical missing inferential object in the
paper: every "symmetric / no robust downside-specific amplification" claim
(exceedance Delta ~ 0; skew_test ~ 0) is currently a *failure to reject*. On
its own that is uninformative — it could mean "the effect is genuinely tiny" or
"we had no power". This script makes the distinction explicit by (a) computing
the smallest effect each test could have detected (MDE), and (b) running a
two-one-sided-tests (TOST) equivalence test against a pre-registered
smallest-effect-size-of-interest (SESOI). The honest reading is encoded in the
verdict: EQUIVALENT-TO-NEGLIGIBLE only when the data actively bound the effect
inside the SESOI band; otherwise INCONCLUSIVE (with the underpowered caveat
when MDE > SESOI) or NON-NEGLIGIBLE.

WHY THIS IS A *DERIVED* OBJECT (no re-estimation)
-------------------------------------------------
The point estimates and bootstrap CIs already exist on disk (run_exceedance.py
=> exceedance_paired.csv ; run_skew_test.py => skew_test.csv). MDE and TOST are
deterministic functions of (estimate, SE). We therefore READ those CSVs and
infer each object's bootstrap SE from its reported percentile-CI half-width:

    SE = (ci_hi - ci_lo) / 2 / z_{0.975}

(the bootstrap CI is a symmetric-percentile [2.5, 97.5] interval, so its
half-width is ~1.96 SE — the same z used to build it). Re-deriving SE this way,
rather than re-running the bootstrap, guarantees the MDE/TOST numbers are
exactly consistent with the published intervals and keeps this script a thin,
auditable post-processor (mirrors how the project layers diagnostics on top of
the locked estimation pipeline; cf. run_skew_test reusing build_df_est_raw).

OBJECTS (the downside-asymmetry family)
---------------------------------------
- exceedance Delta = beta_down - beta_up  at alpha in {0.10, 0.05, 0.01},
  HORIZON h=0 (the headline per-period downside whisper). Source: exceedance_paired.csv.
  Native units: difference in the shock's effect on the down- vs up-tail
  VIOLATION PROBABILITY, per unit log_liq. (A *probability* LHS.)
- skew_test beta at measures {skew_tail05, skew_tail01, z3_winsor}.
  Source: skew_test.csv.
    * skew_tail05 / skew_tail01 : beta_shock on (1[z<=q_tau] - 1[z>=q_{1-tau}])
      = beta_down - beta_up with SCALE REMOVED. Native units: same as the
      exceedance Delta — a down-minus-up tail-PROBABILITY effect per unit
      log_liq. SESOI mapping APPLIES.
    * z3_winsor : beta_shock on the winsorized standardized return CUBED.
      Native units: a conditional THIRD-MOMENT response, NOT a probability.
      The locked SESOI (a 1pp tail-probability gap) is INCOMMENSURABLE with
      this scale, so its SESOI-based verdicts are reported as N/A-UNITS (we
      still report estimate / SE / MDE, which are unit-free statements about
      what the test could detect in its OWN units). See SESOI rationale below.

1. MDE  (per object)
--------------------
Minimum effect detectable at the conventional two-sided level alpha_test=0.05
and power `power` (default 0.80; we also report 0.50):

    MDE = (z_{1 - alpha_test/2} + z_{power}) * SE

Reported in the coefficient's native units. (power=0.50 => z=0 => MDE reduces
to z_{0.975}*SE = the CI half-width: the smallest effect that would have been
"just significant".)

2. SESOI mapping  (locked: down-up tail-probability gap of 1 percentage point)
------------------------------------------------------------------------------
The economically-negligible threshold is locked at a 1-percentage-point gap
between the down- and up-tail probabilities. The objects' betas are *per unit
of the shock* (log_liq), so to compare we map the 1pp gap into beta-units using
the shock's empirical in-sample span:

    SESOI_beta = 0.01 / Delta_log_liq

i.e. "the per-unit effect that, accumulated across a representative move in the
shock, produces a 1pp down-up probability gap". Because there is no single
canonical 'representative move', Delta_log_liq is computed THREE ways on the
SAME in-sample shock (log_liq.shift(1) over the build_df_est_raw estimation
window — identical sample to the tests) and ALL are reported, so SESOI_beta is
an honest *range*, not a single number:
    - IQR(log_liq)            = p75 - p25
    - p50 -> p95              = q95 - q50
    - p10 -> p90              = q90 - q10
(The shock is zero-inflated, so p10=p25=p50=0 in-sample; the spans therefore
reduce to p75 / p95 / p90 respectively. This is reported transparently in the
meta JSON.) A SMALLER span => a LARGER (more permissive) SESOI_beta; a LARGER
span => a SMALLER (stricter) SESOI_beta. We report verdicts under all three.

3. Equivalence (TOST)  (per object, per SESOI span)
---------------------------------------------------
H0: |effect| >= SESOI_beta   vs   H1: |effect| < SESOI_beta.
Equivalence at the 5% TOST level is established iff the 90% CI of the estimate
lies ENTIRELY inside (-SESOI_beta, +SESOI_beta). We build the 90% CI from the
inferred SE (estimate +/- z_{0.95} * SE) so it is consistent with the same
Gaussian approximation used for the MDE. Verdict per object x span:
    - EQUIVALENT-TO-NEGLIGIBLE : 90% CI subset of (-SESOI, +SESOI).
    - NON-NEGLIGIBLE           : 90% CI lies ENTIRELY beyond the band
                                 (lo90 >= +SESOI or hi90 <= -SESOI), i.e. even
                                 the most favourable plausible value exceeds the
                                 negligibility threshold in magnitude.
    - INCONCLUSIVE             : otherwise (the 90% CI straddles a band edge).

4. Honest MDE-vs-SESOI logic
----------------------------
If MDE(power=0.80) > SESOI_beta for an object/span, the design is UNDERPOWERED
for strict equivalence at that span: a non-significant result there can only
support "we exclude effects larger than MDE", NOT "the effect is proven
negligible". This is recorded per object/span as `underpowered_*` and folded
into the meta logic block. An INCONCLUSIVE verdict with underpowered=True is
the canonical "absence of evidence != evidence of absence" case.

OUTPUT (data/econ/)
-------------------
- mde_equivalence.csv   columns:
    object, alpha_or_measure, estimate, se, mde_50, mde_80,
    sesoi_beta_iqr, sesoi_beta_p50p95, sesoi_beta_p10p90,
    verdict_iqr, verdict_p50p95, verdict_p10p90
- mde_equivalence_meta.json   SESOI rationale, the 3 shock spans + values,
    the MDE/TOST formulas, the underpowered logic, per-object diagnostics,
    provenance.

CLI
---
    # smoke (uses the existing smoke CSVs in data/econ/)
    .venv/bin/python scripts/aux/run_mde_equivalence.py

    # explicit power / alternate input dir
    .venv/bin/python scripts/aux/run_mde_equivalence.py --power 0.8 \
        --in_dir data/econ --out_dir /tmp/mde_smoke

SMOKE NOTE: the SE here is inferred from whatever CIs are on disk. With smoke
CIs (n_boot=150) the SE — and hence MDE/verdicts — are smoke-grade; the
canonical numbers use the n_boot=1000 CIs re-run on the VM. The arithmetic is
identical either way.
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
from src.estimation import build_df_est_raw  # noqa: E402


# ──────────────────────────────────────────────────────────────
# Locked spec constants
# ──────────────────────────────────────────────────────────────
ALPHA_TEST: float = 0.05            # two-sided test level for MDE / TOST
SESOI_PROB_GAP: float = 0.01        # LOCKED: 1 percentage-point down-up tail-prob gap

# Exceedance Delta objects: headline per-period horizon is h=0.
# One object per tail level.
EXCEEDANCE_H: int = 0
EXCEEDANCE_ALPHAS: tuple[float, ...] = (0.10, 0.05, 0.01)

# skew_test measures, in CSV order. The first two are tail-PROBABILITY
# (down-up) effects => SESOI mapping applies. z3_winsor is a third-moment
# response in incommensurable units => SESOI verdicts are N/A-UNITS.
SKEW_MEASURES: tuple[str, ...] = ("skew_tail05", "skew_tail01", "z3_winsor")
PROB_UNIT_MEASURES: frozenset[str] = frozenset({"skew_tail05", "skew_tail01"})

SHOCK_COL: str = "shock"            # build_df_est_raw: log_liq.shift(1)

OUT_COLS: list[str] = [
    "object", "alpha_or_measure", "estimate", "se", "mde_50", "mde_80",
    "sesoi_beta_iqr", "sesoi_beta_p50p95", "sesoi_beta_p10p90",
    "verdict_iqr", "verdict_p50p95", "verdict_p10p90",
]

# Verdict labels
EQUIV = "EQUIVALENT-TO-NEGLIGIBLE"
INCONC = "INCONCLUSIVE"
NONNEG = "NON-NEGLIGIBLE"
NA_UNITS = "N/A-UNITS"


# ──────────────────────────────────────────────────────────────
# Shock spans  (SESOI denominator) — computed on the SAME estimation
# sample the tests use, so SESOI_beta is in-sample-consistent.
# ──────────────────────────────────────────────────────────────
def compute_shock_spans() -> dict:
    """Return the 3 Delta_log_liq spans + supporting quantiles of the in-sample shock.

    The shock is build_df_est_raw's `shock` = log_liq.shift(1) AFTER the warmup
    slice — i.e. the identical regressor (and identical sample size) used by
    run_exceedance / run_skew_test. Computing the spans here, rather than off the
    raw panel, keeps SESOI_beta exactly aligned with the objects' own sample.
    """
    df_est = build_df_est_raw(horizons=[0]).reset_index(drop=True)
    s = df_est[SHOCK_COL].dropna()
    q = {p: float(s.quantile(p)) for p in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)}
    span_iqr = q[0.75] - q[0.25]
    span_p50p95 = q[0.95] - q[0.50]
    span_p10p90 = q[0.90] - q[0.10]
    return {
        "n_shock": int(len(s)),
        "quantiles": q,
        "span_iqr": float(span_iqr),
        "span_p50p95": float(span_p50p95),
        "span_p10p90": float(span_p10p90),
        "zero_inflated_note": (
            "shock = log_liq.shift(1) is zero-inflated; in-sample p10=p25=p50=0, "
            "so IQR==p75, p50->p95==p95, p10->p90==p90. Spans reported as-is."
        ),
    }


def sesoi_from_span(span: float) -> float:
    """Map the locked 1pp tail-prob gap into shock-beta units: 0.01 / span."""
    if not np.isfinite(span) or span <= 0:
        return np.nan
    return SESOI_PROB_GAP / span


# ──────────────────────────────────────────────────────────────
# Core MDE / TOST arithmetic
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


def ci90(estimate: float, se: float) -> tuple[float, float]:
    """90% two-sided CI used for the TOST verdict: estimate +/- z_{0.95} * SE."""
    z90 = stats.norm.ppf(0.95)
    return estimate - z90 * se, estimate + z90 * se


def tost_verdict(estimate: float, se: float, sesoi: float) -> str:
    """Equivalence verdict from the 90% CI vs the (-sesoi, +sesoi) band.

    EQUIVALENT-TO-NEGLIGIBLE : 90% CI strictly inside the band.
    NON-NEGLIGIBLE           : 90% CI entirely beyond the band on one side.
    INCONCLUSIVE             : 90% CI straddles a band edge.
    """
    if not np.isfinite(sesoi):
        return NA_UNITS
    lo, hi = ci90(estimate, se)
    if lo > -sesoi and hi < sesoi:
        return EQUIV
    if lo >= sesoi or hi <= -sesoi:
        return NONNEG
    return INCONC


# ──────────────────────────────────────────────────────────────
# Per-object assembly
# ──────────────────────────────────────────────────────────────
def build_object_row(
    obj: str,
    label: str,
    estimate: float,
    ci_lo: float,
    ci_hi: float,
    sesoi: dict[str, float],
    power: float,
    units_are_probability: bool,
) -> tuple[dict, dict]:
    """Return (csv_row, diag) for one object.

    `sesoi` maps {'iqr','p50p95','p10p90'} -> SESOI_beta value. If the object's
    units are not a tail probability (z3_winsor), SESOI verdicts are N/A-UNITS
    and the SESOI_beta columns are blanked (NaN) — the mapping does not apply —
    but estimate/SE/MDE are still reported (they are unit-free statements about
    detectability in the object's OWN units).
    """
    se = se_from_ci(ci_lo, ci_hi)
    mde_50 = mde(se, 0.50)
    mde_80 = mde(se, power)
    lo90, hi90 = ci90(estimate, se)

    row: dict = {
        "object": obj,
        "alpha_or_measure": label,
        "estimate": float(estimate),
        "se": float(se),
        "mde_50": float(mde_50),
        "mde_80": float(mde_80),
    }
    diag: dict = {
        "object": obj,
        "alpha_or_measure": label,
        "estimate": float(estimate),
        "ci_lo_reported": float(ci_lo),
        "ci_hi_reported": float(ci_hi),
        "se_inferred": float(se),
        "mde_50": float(mde_50),
        "mde_80_at_power": float(mde_80),
        "ci90_lo": float(lo90),
        "ci90_hi": float(hi90),
        "units": "tail_probability_gap_per_log_liq" if units_are_probability
                 else "conditional_third_moment_response_per_log_liq",
        "sesoi_applies": bool(units_are_probability),
        "by_span": {},
    }

    for span_key, csv_key in (
        ("iqr", "sesoi_beta_iqr"),
        ("p50p95", "sesoi_beta_p50p95"),
        ("p10p90", "sesoi_beta_p10p90"),
    ):
        S = sesoi[span_key]
        if units_are_probability:
            row[csv_key] = float(S)
            verdict = tost_verdict(estimate, se, S)
            underpowered = bool(mde_80 > S)
        else:
            row[csv_key] = np.nan       # SESOI mapping incommensurable with units
            verdict = NA_UNITS
            underpowered = None
        row[f"verdict_{span_key}"] = verdict
        diag["by_span"][span_key] = {
            "sesoi_beta": (float(S) if units_are_probability else None),
            "verdict": verdict,
            "underpowered_for_equivalence": underpowered,
            "underpowered_note": (
                None if underpowered is None else (
                    f"MDE@power > SESOI_beta ({mde_80:.6g} > {S:.6g}): underpowered; "
                    f"can only claim 'exclude effects > {mde_80:.6g}', NOT 'proven negligible'."
                    if underpowered else
                    f"MDE@power <= SESOI_beta ({mde_80:.6g} <= {S:.6g}): adequately powered "
                    f"to adjudicate equivalence at this span."
                )
            ),
        }

    return row, diag


# ──────────────────────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────────────────────
def load_inputs(in_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read exceedance_paired.csv and skew_test.csv from in_dir."""
    paired_path = in_dir / "exceedance_paired.csv"
    skew_path = in_dir / "skew_test.csv"
    if not paired_path.exists():
        raise FileNotFoundError(
            f"{paired_path} not found — run scripts/aux/run_exceedance.py first."
        )
    if not skew_path.exists():
        raise FileNotFoundError(
            f"{skew_path} not found — run scripts/aux/run_skew_test.py first."
        )
    return pd.read_csv(paired_path), pd.read_csv(skew_path)


def collect_objects(
    df_paired: pd.DataFrame,
    df_skew: pd.DataFrame,
    sesoi: dict[str, float],
    power: float,
) -> tuple[list[dict], list[dict]]:
    """Build all CSV rows + diagnostics for the downside-asymmetry family."""
    rows: list[dict] = []
    diags: list[dict] = []

    # --- exceedance Delta at h=0, one per tail level ---
    p = df_paired.copy()
    # robust float match on alpha and exact h
    for a in EXCEEDANCE_ALPHAS:
        sel = p[(np.isclose(p["alpha"], a)) & (p["h"] == EXCEEDANCE_H)]
        if sel.empty:
            raise ValueError(
                f"exceedance_paired.csv has no row for alpha={a}, h={EXCEEDANCE_H}."
            )
        r = sel.iloc[0]
        row, diag = build_object_row(
            obj="exceedance_delta",
            label=f"alpha={a:g} (h={EXCEEDANCE_H})",
            estimate=float(r["delta"]),
            ci_lo=float(r["ci_lo"]),
            ci_hi=float(r["ci_hi"]),
            sesoi=sesoi,
            power=power,
            units_are_probability=True,
        )
        rows.append(row)
        diags.append(diag)

    # --- skew_test measures ---
    s = df_skew.set_index("measure")
    for m in SKEW_MEASURES:
        if m not in s.index:
            raise ValueError(f"skew_test.csv missing measure '{m}'.")
        r = s.loc[m]
        row, diag = build_object_row(
            obj="skew_test",
            label=m,
            estimate=float(r["beta"]),
            ci_lo=float(r["ci_lo"]),
            ci_hi=float(r["ci_hi"]),
            sesoi=sesoi,
            power=power,
            units_are_probability=(m in PROB_UNIT_MEASURES),
        )
        rows.append(row)
        diags.append(diag)

    return rows, diags


# ──────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────
def save_outputs(
    rows: list[dict],
    diags: list[dict],
    spans: dict,
    sesoi: dict[str, float],
    power: float,
    in_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)[OUT_COLS]
    csv_path = out_dir / "mde_equivalence.csv"
    df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}", flush=True)

    z_a = stats.norm.ppf(1.0 - ALPHA_TEST / 2.0)
    meta = {
        "test": "MDE + equivalence (TOST) for downside-asymmetry objects (A5/A9)",
        "what_this_does": (
            "Converts the symmetry/skew nulls from 'failed to reject' into "
            "quantified statements: the smallest effect each test could detect "
            "(MDE) and whether the data actively bound the effect below a "
            "pre-registered negligibility threshold (TOST equivalence)."
        ),
        "derived_not_reestimated": (
            "Point estimates and CIs are read from exceedance_paired.csv and "
            "skew_test.csv; SE is inferred from each CI half-width / z_{0.975} "
            "so MDE/TOST are exactly consistent with the published intervals."
        ),
        "objects": {
            "exceedance_delta": (
                f"beta_down - beta_up at alpha in "
                f"{list(EXCEEDANCE_ALPHAS)}, horizon h={EXCEEDANCE_H} "
                f"(per-period downside whisper; from exceedance_paired.csv)."
            ),
            "skew_test": (
                f"beta_shock at measures {list(SKEW_MEASURES)} "
                f"(from skew_test.csv); skew_tail05/01 are tail-probability "
                f"(down-up) effects [SESOI applies], z3_winsor is a "
                f"third-moment response [SESOI N/A-UNITS]."
            ),
        },
        "formulas": {
            "se_from_ci": "SE = (ci_hi - ci_lo) / 2 / z_{1-alpha/2}",
            "mde": "MDE = (z_{1-alpha/2} + z_power) * SE  (native units)",
            "ci90_for_tost": "estimate +/- z_{0.95} * SE",
            "alpha_test": ALPHA_TEST,
            "z_{1-alpha/2}": float(z_a),
            "z_power_default": float(stats.norm.ppf(power)),
            "power_default": power,
            "power_also_reported": 0.50,
            "mde_50_meaning": (
                "power=0.50 => z_power=0 => MDE = z_{1-alpha/2}*SE = CI half-width "
                "(smallest 'just-significant' effect)."
            ),
        },
        "sesoi": {
            "locked_definition": (
                "Smallest-effect-size-of-interest = a down-minus-up tail-"
                "PROBABILITY gap of 1 percentage point (0.01)."
            ),
            "rationale": (
                "The economic claim under test is 'no NEGLIGIBLE-or-larger "
                "downside-specific amplification'. A 1pp gap in tail-violation "
                "probability between the down and up sides is the smallest "
                "down/up asymmetry an investor/risk-manager would plausibly "
                "care about; anything smaller is economically indistinguishable "
                "from symmetry. Object betas are per-unit-shock, so the gap is "
                "mapped to beta-units via the shock's empirical span."
            ),
            "mapping": "SESOI_beta = 0.01 / Delta_log_liq",
            "delta_log_liq_three_ways": {
                "iqr": {"value": spans["span_iqr"], "definition": "p75 - p25 of in-sample shock"},
                "p50p95": {"value": spans["span_p50p95"], "definition": "q95 - q50 of in-sample shock"},
                "p10p90": {"value": spans["span_p10p90"], "definition": "q90 - q10 of in-sample shock"},
            },
            "sesoi_beta_three_ways": {
                "iqr": sesoi["iqr"],
                "p50p95": sesoi["p50p95"],
                "p10p90": sesoi["p10p90"],
            },
            "shock_quantiles_in_sample": spans["quantiles"],
            "n_shock": spans["n_shock"],
            "zero_inflation": spans["zero_inflated_note"],
            "span_direction_note": (
                "Smaller span => larger (more permissive) SESOI_beta; larger "
                "span => smaller (stricter) SESOI_beta. Verdicts reported under "
                "all three so the conclusion's sensitivity to the span choice is "
                "transparent."
            ),
            "shock": "log_liq.shift(1) on the build_df_est_raw estimation sample "
                     "(same regressor & sample as the exceedance / skew tests).",
        },
        "tost_logic": {
            "hypotheses": "H0: |effect| >= SESOI_beta  vs  H1: |effect| < SESOI_beta",
            "rule": "EQUIVALENT iff 90% CI subset of (-SESOI_beta, +SESOI_beta)",
            "verdicts": {
                EQUIV: "90% CI strictly inside the band.",
                NONNEG: "90% CI entirely beyond the band on one side "
                        "(lo90 >= +SESOI or hi90 <= -SESOI).",
                INCONC: "90% CI straddles a band edge.",
                NA_UNITS: "object's units are not a tail probability; SESOI "
                          "mapping does not apply (z3_winsor).",
            },
        },
        "honest_power_caveat": (
            "If MDE@power > SESOI_beta for an object/span, the design is "
            "UNDERPOWERED for strict equivalence there: a non-significant result "
            "supports only 'we exclude effects larger than MDE', NOT 'the effect "
            "is proven negligible'. Recorded per object/span as "
            "underpowered_for_equivalence. INCONCLUSIVE + underpowered=True is "
            "the canonical 'absence of evidence is not evidence of absence' case."
        ),
        "per_object": diags,
        "inputs": {
            "exceedance_paired_csv": str((in_dir / "exceedance_paired.csv").resolve()),
            "skew_test_csv": str((in_dir / "skew_test.csv").resolve()),
            "note": "SE inferred from these CSVs' CIs; smoke CIs => smoke SE/MDE.",
        },
        "panel": str(CFG.FILES.econ_core_full),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    meta_path = out_dir / "mde_equivalence_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  wrote {meta_path}", flush=True)

    # Per-convention verification echo (head/shape of the written CSV).
    print(f"\n  mde_equivalence.csv  shape={df.shape}", flush=True)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False), flush=True)
    return df


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--power", type=float, default=0.80,
                    help="Target power for MDE@80 (default 0.80). MDE@50 always "
                         "reported alongside.")
    ap.add_argument("--in_dir", type=Path, default=ECON_DIR,
                    help=f"Directory holding exceedance_paired.csv & "
                         f"skew_test.csv. Default: {ECON_DIR}")
    ap.add_argument("--out_dir", type=Path, default=ECON_DIR)
    args = ap.parse_args()

    if not (0.0 < args.power < 1.0):
        ap.error("--power must be in (0, 1).")

    print("run_mde_equivalence — MDE + TOST equivalence (A5/A9)", flush=True)
    print(f"  power={args.power}  alpha_test={ALPHA_TEST}  "
          f"SESOI_prob_gap={SESOI_PROB_GAP}", flush=True)
    print(f"  in_dir={args.in_dir}", flush=True)

    t0 = time.time()

    # 1. shock spans -> SESOI_beta (3 ways)
    print("Computing in-sample shock spans (build_df_est_raw) …", flush=True)
    spans = compute_shock_spans()
    sesoi = {
        "iqr": sesoi_from_span(spans["span_iqr"]),
        "p50p95": sesoi_from_span(spans["span_p50p95"]),
        "p10p90": sesoi_from_span(spans["span_p10p90"]),
    }
    print(f"  shock n={spans['n_shock']:,}", flush=True)
    print(f"  Delta_log_liq: IQR={spans['span_iqr']:.4f}  "
          f"p50->p95={spans['span_p50p95']:.4f}  "
          f"p10->p90={spans['span_p10p90']:.4f}", flush=True)
    print(f"  SESOI_beta   : IQR={sesoi['iqr']:.6f}  "
          f"p50p95={sesoi['p50p95']:.6f}  p10p90={sesoi['p10p90']:.6f}",
          flush=True)

    # 2. read objects
    df_paired, df_skew = load_inputs(args.in_dir)

    # 3. build rows + diagnostics
    rows, diags = collect_objects(df_paired, df_skew, sesoi, args.power)

    # 4. save
    save_outputs(rows, diags, spans, sesoi, args.power, args.in_dir, args.out_dir)

    print(f"\nDone. Total wall time: {(time.time()-t0)/60:.2f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
