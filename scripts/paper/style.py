"""
style.py — house style for every paper figure (single source of truth).

Conventions:
- Sizes are declared HERE once and the manuscript never rescales:
  FULLW = 6.5 in (full \\textwidth figure), COLW = 3.25 in (single column).
- Fonts 8-9pt to match the document at 1:1 inclusion.
- All figures are vector PDF, deterministic (no timestamps in metadata).
- make_figures.py functions read ONLY data/econ CSVs — no statistics are
  computed in the paper layer (no statistics are computed here).

Palette: colorblind-safe (Okabe-Ito subset), consistent across figures:
  C_MAIN   - ETH / the headline object (blue)
  C_ALT    - BTC / comparison object (vermillion)
  C_UP     - upper tail / up-violations (teal)
  C_DOWN   - lower tail / down-violations (vermillion)
  C_NULL   - null / placebo bands (grey)
  C_ACCENT - highlights (purple)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Headless/deterministic from the CLI; leave the backend alone inside Jupyter
# (notebooks 10/11 reuse these exact functions and display inline).
_INTERACTIVE = "ipykernel" in sys.modules
if not _INTERACTIVE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ── Geometry (inches) — the manuscript includes at natural size ──
FULLW = 6.5
COLW = 3.25
H_STD = 2.6        # standard panel height
H_TALL = 3.4

# ── Palette (Okabe-Ito) ──
C_MAIN = "#0072B2"     # blue
C_ALT = "#D55E00"      # vermillion
C_UP = "#009E73"       # teal
C_DOWN = "#D55E00"     # vermillion
C_NULL = "#999999"     # grey
C_ACCENT = "#CC79A7"   # purple
C_BAND = "#BBBBBB"

RC = {
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "axes.grid": True,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,            # embed TrueType (journal-safe)
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply() -> None:
    """Apply the house rcParams (call once per process)."""
    plt.rcParams.update(RC)


def save(fig: "plt.Figure", out_dir: Path, name: str) -> Path:
    """Save a figure as deterministic vector PDF under paper/figures/.

    `name` is the final paper name, e.g. 'fig4_placebo_gap_distribution'.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pdf"
    # Project convention: canonical artefacts may be chmod 400 — unlink first
    # (directory write permission suffices), then write fresh.
    if path.exists() and not (path.stat().st_mode & 0o200):
        path.unlink()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None})  # deterministic PDF
    if _INTERACTIVE:
        plt.show()  # inline display in the report notebooks
    plt.close(fig)
    print(f"  wrote {path}", flush=True)
    return path
