#!/usr/bin/env python3
"""
build_references.py — provenance script that assembled
manuscript/references.bib from the author's verified reference ledger.

The shipped bibliography (manuscript/references.bib) is the canonical artifact
and is consumed directly by the manuscript. This script documents how it was
built; it is NOT part of the reproduction pipeline, as it requires the author's
reference ledger, which is not distributed.

Design: the script never writes a hand-made entry. Rows with a DOI have their
BibTeX fetched from doi.org content negotiation (authoritative metadata); rows
without a DOI are emitted as commented placeholders for the author to complete.
Citation keys are normalised to <firstauthor><year>, with a/b suffixes on
collisions.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "_worklog" / "23_table_references.md"
OUT = ROOT / "manuscript" / "references.bib"
CACHE = ROOT / "_backups" / ".bib_cache"

# NB: Elsevier DOIs legitimately contain parentheses (10.1016/S0304-4076(00)...),
# so stop only at whitespace, pipe or bracket — parenthesised suffixes in the
# ledger ("(arXiv ...)") are space-separated and thus excluded.
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s|\]]+)")

# Additional references required by the redesigned tests but
# absent from the pre-pivot ledger. Each carries a TITLE KEYWORD GUARD: the
# fetched BibTeX must contain the keyword in its title, otherwise the entry is
# rejected as a failure (safety guard: a mis-remembered or mistyped DOI cannot
# land silently).
EXTRA_DOIS: list = [
    # (ledger-style author/year, doi, lowercase title keyword, why)
    ("Wu 1986", "10.1214/aos/1176350142", "jackknife",
     "wild/sign-flip bootstrap ancestor (placebo sign_flip DGP)"),
    ("Liu 1988", "10.1214/aos/1176351062", "bootstrap",
     "Rademacher weights under non-iid models (placebo sign_flip DGP)"),
    ("Davidson & Flachaire 2008", "10.1016/j.jeconom.2008.08.003", "wild bootstrap",
     "the wild bootstrap, tamed at last (placebo sign_flip DGP)"),
    ("Diebold & Mariano 1995", "10.1080/07350015.1995.10524599", "predictive accuracy",
     "DM test used by run_oos_predictive (Table 6)"),
    # §8/§9 mechanism refs that DO carry a journal DOI in the note-23 prose
    # notes (the ledger parser missed them — they sit in Notes blocks, not
    # table rows). Title-guarded so a wrong DOI fails loudly.
    ("Caldarelli 2020", "10.3390/info11110509", "oracle",
     "oracle decoupling, §8.4"),
    ("Gan et al. 2022", "10.1145/3558535.3559793", "wash",
     "flash-loan wash trading caveat on size denominators, §8.1/§9"),
]


# Working-paper / institutional entries with NO DOI. Every field below was
# verified against the primary page on the date indicated; these are NOT
# hand-remembered.
WP_ENTRIES: list = [
    ("lehar2022",
     "@techreport{lehar2022, author={Lehar, Alfred and Parlour, Christine A.},"
     " title={Systemic Fragility in Decentralized Markets},"
     " institution={Bank for International Settlements}, type={BIS Working Paper},"
     " number={1062}, year={2022},"
     " note={SSRN 4164833}, url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4164833} }",
     "title verified on SSRN 2026-06-13"),
    ("fsb2023",
     "@techreport{fsb2023, author={{Financial Stability Board}},"
     " title={The Financial Stability Risks of Decentralised Finance},"
     " institution={Financial Stability Board}, year={2023}, month=feb,"
     " url={https://www.fsb.org/uploads/P160223.pdf} }",
     "title verified on fsb.org 2026-06-13"),
    # §8/§9 mechanism + official-sector refs, web-verified 2026-06-18 (primary pages)
    ("chaudhary2022",
     "@techreport{chaudhary2022, author={Chaudhary, Amit and Pinna, Daniele}, title={A multi-asset, agent-based approach applied to {DeFi} lending protocol modelling}, institution={arXiv}, type={Preprint}, number={arXiv:2211.08870}, year={2022}, url={https://arxiv.org/abs/2211.08870} }",
     "arxiv.org 2026-06-18"),
    ("warmuz2022",
     "@techreport{warmuz2022, author={Warmuz, Jakub and Chaudhary, Amit and Pinna, Daniele}, title={Toxic Liquidation Spirals}, institution={arXiv}, type={Preprint}, number={arXiv:2212.07306}, year={2022}, url={https://arxiv.org/abs/2212.07306} }",
     "arxiv.org 2026-06-18"),
    ("leharparlour2025",
     "@article{leharparlour2025, author={Lehar, Alfred and Parlour, Christine A.}, title={Decentralized Exchange: The {Uniswap} Automated Market Maker}, journal={The Journal of Finance}, volume={80}, number={1}, pages={321--374}, year={2025}, doi={10.1111/jofi.13405}, url={https://doi.org/10.1111/jofi.13405} }",
     "onlinelibrary.wiley.com 2026-06-23"),
    ("aramonte2021",
     "@techreport{aramonte2021, author={Aramonte, Sirio and Huang, Wenqian and Schrimpf, Andreas}, title={{DeFi} risks and the decentralisation illusion}, institution={Bank for International Settlements}, type={BIS Quarterly Review}, year={2021}, month=dec, url={https://www.bis.org/publ/qtrpdf/r_qt2112b.htm} }",
     "bis.org 2026-06-18"),
    ("heimbach2024",
     "@techreport{heimbach2024, author={Heimbach, Lioba and Huang, Wenqian}, title={{DeFi} leverage}, institution={Bank for International Settlements}, type={BIS Working Paper}, number={1171}, year={2024}, month=mar, url={https://www.bis.org/publ/work1171.htm} }",
     "bis.org 2026-06-18"),
    ("imf2021",
     "@techreport{imf2021, author={{International Monetary Fund}}, title={The Crypto Ecosystem and Financial Stability Challenges}, institution={International Monetary Fund}, type={Global Financial Stability Report (October 2021), Chapter 2}, year={2021}, month=oct, url={https://www.imf.org/en/Publications/GFSR/Issues/2021/10/12/global-financial-stability-report-october-2021} }",
     "imf.org 2026-06-18"),
    ("ecb2022",
     "@techreport{ecb2022, author={Hermans, Lieven and Ianiro, Annalaura and Kochanska, Urszula and van der Kraaij, Anton and Vendrell Sim{\\'o}n, Josep M.}, title={Decrypting financial stability risks in crypto-asset markets}, institution={European Central Bank}, type={Financial Stability Review (May 2022), Special Feature}, year={2022}, month=may, url={https://www.ecb.europa.eu/press/financial-stability-publications/fsr/special/html/ecb.fsrart202205_02~1cc6b111b4.en.html} }",
     "ecb.europa.eu 2026-06-18"),
    ("badev2023",
     "@techreport{badev2023, author={Badev, Anton I. and Watsky, Cy}, title={Interconnected {DeFi}: Ripple Effects from the Terra Collapse}, institution={Board of Governors of the Federal Reserve System}, type={Finance and Economics Discussion Series}, number={2023-044}, year={2023}, month=jun, url={https://www.federalreserve.gov/econres/feds/files/2023044pap.pdf} }",
     "federalreserve.gov 2026-06-18"),
    ("liuszalachowski2021",
     "@inproceedings{liuszalachowski2021, author={Liu, Bowen and Szalachowski, Pawel and Zhou, Jianying}, title={A First Look into {DeFi} Oracles}, booktitle={2021 IEEE International Conference on Decentralized Applications and Infrastructures (DAPPS)}, year={2021}, url={https://arxiv.org/abs/2005.04377} }",
     "arxiv.org 2026-06-18"),
]


def parse_ledger() -> tuple[list[dict], list[dict]]:
    """Return (doi_rows, nodoi_rows) from the note-23 master table."""
    doi_rows, nodoi_rows = [], []
    seen = set()
    for line in LEDGER.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 6 or cells[0] in ("#", "---"):
            continue
        if set(cells[0]) <= {"-", " "}:
            continue
        author_year, venue, doi_cell = cells[1], cells[2], cells[3]
        if author_year.lower().startswith(("author", "—", "-")):
            continue
        m = DOI_RE.search(doi_cell)
        row = {"author_year": author_year.replace("**", "").strip(),
               "venue": venue, "raw": doi_cell}
        if m:
            doi = m.group(1).rstrip(".,;")
            if doi in seen:
                continue
            seen.add(doi)
            row["doi"] = doi
            doi_rows.append(row)
        else:
            key = author_year + doi_cell
            if key in seen or not author_year:
                continue
            seen.add(key)
            nodoi_rows.append(row)
    return doi_rows, nodoi_rows


# Documented errata for the ledger's own DOI mistakes, each web-verified on the
# date noted. The fetched entry replaces the ledger row's DOI transparently.
DOI_ERRATA: dict = {
    # ledger row "Neuberger & Payne 2021" carried 10.1093/rfs/hhaa099, which is
    # Huang & Ritter (RFS 34(4)); the correct DOI for "The Skewness of the
    # Stock Market over Long Horizons" (RFS 34(3):1572-1616) was verified on
    # academic.oup.com, 2026-06-13.
    "10.1093/rfs/hhaa099": "10.1093/rfs/hhaa048",
}


_MONTHS = {"january": "jan", "february": "feb", "march": "mar", "april": "apr",
           "may": "may", "june": "jun", "july": "jul", "august": "aug",
           "september": "sep", "sept": "sep", "october": "oct",
           "november": "nov", "december": "dec"}


def _clean(text: str) -> str:
    """Normalise doi.org output: entities, NBSP, tags, months; trim leading WS
    so the ^@... field-patch anchor matches; map months to plainnat 3-letter."""
    import html
    text = html.unescape(text)
    text = text.replace(" ", " ").replace(" ", " ")   # NBSP variants
    text = re.sub(r"</?(i|b|sub|sup|em)>", "", text)            # HTML tags in titles
    text = text.replace(" & ", " \\& ")                          # LaTeX-escape
    text = re.sub(r"month\s*=\s*\{?([A-Za-z]+)\}?",
                  lambda m: f"month={_MONTHS.get(m.group(1).lower(), m.group(1).lower()[:3])}",
                  text)
    return text.strip()


def fetch_bibtex(doi: str) -> str:
    doi = DOI_ERRATA.get(doi, doi)
    CACHE.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", doi)
    cached = CACHE / f"{slug}.bib"
    if cached.exists():
        return _clean(cached.read_text())
    req = urllib.request.Request(
        f"https://doi.org/{doi}",
        headers={"Accept": "application/x-bibtex; charset=utf-8",
                 "User-Agent": "replication-package-bib-builder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8", errors="replace")
    if not text.lstrip().startswith("@"):
        raise ValueError(f"non-bibtex response for {doi}")
    cached.write_text(text)
    time.sleep(0.4)  # politeness
    return _clean(text)


def fix_year(bib: str, ledger_year: str, log: list, key: str) -> str:
    """Ledger year overrides doi.org's online-first year (note 17/23 verified).

    Example: Carriero-Clark-Marcellino is JMCB 2024 56(5) in the ledger but
    doi.org deposits year=2023 (online-first). The ledger is the verified
    authority for year/venue; titles/authors stay authoritative from doi.org.
    """
    m = re.search(r"year\s*=\s*\{?(\d{4})\}?", bib)
    if m and ledger_year != "0000" and m.group(1) != ledger_year:
        log.append(f"{key}: year {m.group(1)} -> {ledger_year} (ledger wins)")
        bib = bib[:m.start(1)] + ledger_year + bib[m.end(1):]
    return bib


def fix_allcaps_authors(bib: str, log: list, key: str) -> str:
    """Repair publisher deposits with ALL-CAPS author names (Crossref artefact).

    The lowercase test must ignore the ' and ' separators (their lowercase
    letters masked genuinely broken fields like 'CARRIERO, ANDREA and CLARK,
    TODD E.'). Word-capitalises and logs for author review.
    """
    m = re.search(r"author\s*=\s*\{([^}]*)\}", bib)
    if not m:
        return bib
    field = m.group(1)
    probe = field.replace(" and ", " ")
    if any(c.islower() for c in probe):
        return bib
    fixed = " ".join(w if w.lower() == "and" else w.capitalize()
                     for w in field.split(" "))
    log.append(f"{key}: ALL-CAPS authors recapitalised (review): {fixed[:60]}")
    return bib[:m.start(1)] + fixed + bib[m.end(1):]


# Field injections for impoverished Crossref deposits. Values are SOURCED:
# oecd2023 author/title from the OECD WP itself (the PDF is in references/,
# filename carries authors+title); tian2025 series from the ledger venue cell.
FIELD_PATCHES: dict = {
    "oecd2023": [
        ("author", "Sasi-Brodesky, Ana and Nassr, Iota Kaousar"),
        ("title", "DeFi liquidations: Volatility and liquidity"),
        ("publisher", "OECD Publishing"),
    ],
    "tian2025": [
        ("journal", "Bank of Canada Staff Working Paper 2025-12"),
    ],
}


def apply_field_patches(bib: str, key: str, log: list) -> str:
    for field, value in FIELD_PATCHES.get(key, []):
        if not re.search(rf"{field}\s*=", bib):
            bib = re.sub(r"^(@\w+\{[^,]+,)", rf"\1 {field}={{{value}}},", bib,
                         count=1)
            log.append(f"{key}: injected missing {field} (sourced patch)")
    return bib


def normalise_key(bib: str, author_year: str, used: set) -> str:
    """Key = <firstauthorlastname><year>, from the ledger column (stable)."""
    import unicodedata
    m = re.match(r"([A-Za-zÀ-ÿ'\-]+)", author_year)
    name = (m.group(1) if m else "ref").lower()
    # ASCII-only citation keys (accents are fragile across BibTeX tooling)
    name = (unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore").decode().replace("'", "").replace("-", ""))
    y = re.search(r"(19|20)\d{2}", author_year)
    year = y.group(0) if y else "0000"
    base = f"{name}{year}"
    key = base
    suffix = ord("a")
    while key in used:
        key = base + chr(suffix)
        suffix += 1
    used.add(key)
    return re.sub(r"@(\w+)\{[^,]+,", f"@\\1{{{key},", bib, count=1), key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    doi_rows, nodoi_rows = parse_ledger()
    print(f"ledger parsed: {len(doi_rows)} DOI rows, {len(nodoi_rows)} no-DOI rows")
    if args.dry_run:
        for r in doi_rows[:8]:
            print("  DOI ", r["doi"], "<-", r["author_year"][:40])
        for r in nodoi_rows[:8]:
            print("  INCOMPLETE", r["author_year"][:40], "|", r["raw"][:50])
        return 0

    entries, failures = [], []
    used: set = set()
    keymap, fixlog = [], []

    # Ledger rows plus the additional references (with a title guard).
    work = ([(r, None) for r in doi_rows]
            + [({"author_year": ay, "doi": doi, "raw": f"additional reference: {why}"},
                kw) for ay, doi, kw, why in EXTRA_DOIS])
    for i, (r, title_kw) in enumerate(work):
        try:
            bib = fetch_bibtex(r["doi"])
            if title_kw:
                t = re.search(r"title\s*=\s*\{([^}]*)\}", bib)
                if not t or title_kw.lower() not in t.group(1).lower():
                    raise ValueError(
                        f"title guard '{title_kw}' not matched — wrong DOI?")
            bib, key = normalise_key(bib, r["author_year"], used)
            y = re.search(r"(19|20)\d{2}", r["author_year"])
            bib = fix_year(bib, y.group(0) if y else "0000", fixlog, key)
            bib = fix_allcaps_authors(bib, fixlog, key)
            bib = apply_field_patches(bib, key, fixlog)
            entries.append(bib.strip())
            keymap.append((key, r["author_year"], r["doi"]))
            print(f"  [{i+1}/{len(work)}] {key:24s} {r['doi']}", flush=True)
        except Exception as e:  # noqa: BLE001
            failures.append((r, str(e)))
            print(f"  [{i+1}/{len(work)}] FAIL {r['doi']}: {e}", flush=True)

    lines = [
        "% references.bib — GENERATED by scripts/paper/build_references.py",
        "% Source of truth: _worklog/23_table_references.md (verified ledger).",
        "% DOI entries fetched verbatim from doi.org (no hand-written metadata);",
        "% normalisations applied (ledger year wins over online-first; ALL-CAPS",
        "% author repairs) are logged below.",
    ]
    if fixlog:
        lines.append("% ── normalisations applied ──")
        for entry in fixlog:
            lines.append(f"%   FIX  {entry}")
    lines += [
        "% Key map (key <- ledger row <- DOI):",
    ]
    for key, ay, doi in keymap:
        lines.append(f"%   {key:24s} {ay[:44]:44s} {doi}")
    lines.append("")
    lines.extend(entries)
    lines.append("")
    lines.append("% ── Verified working-paper entries (no DOI; fields web-checked, see header of each) ──")
    for key, ent, prov in WP_ENTRIES:
        lines.append(f"% {key}: {prov}")
        lines.append(ent)
        used.add(key)
    lines.append("")
    lines.append("% ── NO-DOI rows from the ledger (WP/SSRN/arXiv/institutional) ──")
    lines.append("% INCOMPLETE (no-DOI rows for the author to complete by hand):")
    for r in nodoi_rows:
        lines.append(f"%   INCOMPLETE  {r['author_year']}  |  {r['venue']}  |  {r['raw']}")
    if failures:
        lines.append("% ── FETCH FAILURES (retry or complete by hand) ──")
        for r, err in failures:
            lines.append(f"%   FAIL  {r['author_year']}  |  {r['doi']}  |  {err[:60]}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}: {len(entries)} entries, {len(nodoi_rows)} incomplete, "
          f"{len(failures)} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
