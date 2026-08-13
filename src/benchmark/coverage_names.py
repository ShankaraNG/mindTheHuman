"""
Canonical coverage vocabulary.

WHY THIS IS ITS OWN MODULE: three different files name the same cover three
different ways.

    recorder_cover_lines   "Employers' Liability", "Employers Liability"
    deck_benchmark_bands   "Employers Liability"
    deck_current_cover     "EL"

If band_lookup normalises names one way and resolve normalises them another,
a peer statistic gets compared against a band for a different cover and the
mismatch is invisible — the report just quietly benchmarks Cyber against
Professional Indemnity. One vocabulary, imported by both.

NOTHING IS SILENTLY DROPPED. An unrecognised name comes back as
Coverage(canonical=None, raw=<original>, matched=False) so the caller can count
and surface it, rather than having it vanish from a groupby.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Coverage:
    raw: str
    canonical: Optional[str]
    matched: bool


# Ordered most-specific first. First pattern to match wins, so
# "Public and Products Liability" must be tested before bare "Liability",
# and "Medical Malpractice" before "Medical".
_PATTERNS: list[tuple[str, str]] = [
    ("Employers Liability",            r"\bemployer'?s?\b.*\bliab|^el$"),
    ("Public & Products Liability",    r"\bpublic\b.*\bliab|\bproducts?\b.*\bliab|^public\s+products?$|^p\s*and\s*pl$|^ppl$"),
    ("Professional Indemnity",         r"\bprofessional\b.*\bindemnit|^pi\b|\bpi\s+and\b|\berrors?\b.*\bomissions?\b|\be\s*and\s*o\b|\bcivil\s+liab|\bfinancial\s+loss\b|\bbreach\s+of\s+contract\b"),
    ("Directors & Officers",           r"\bdirectors?\b.*\bofficers?\b|^d\s*and\s*o$|\bmanagement\s+liab|\bcorporate\s+legal\s+liab|\bcorporate\s+liab"),
    ("Employment Practices Liability", r"\bemployment\s+practices?\b|^\s*epli?\s*$"),
    ("Cyber",                          r"\bcyber\b|\bdata\s+breach\b|\bnetwork\s+security\b"),
    ("Crime",                          r"\bcrime\b|\bfidelity\b|\bfraud\b|\btheft\b"),
    ("Medical Malpractice",            r"\bmedical\s+malpractice\b|\bmalpractice\b|\bclinical\s+negligence\b|\bmedical\s+regulatory\b"),
    ("Clinical Trials",                r"\bclinical\s+trial|\bno\s+fault\s+compensation\b"),
    ("Intellectual Property",          r"\bintellectual\s+property\b"),
    ("Product Recall",                 r"\brecall\b"),
    ("Business Interruption",          r"\bbusiness\s+interrupt|^bi$|\bloss\s+of\s+(?:profit|revenue)\b"),
    ("Property",                       r"\bproperty\b|\bcommercial\s+combined\b|\bcontents\b|\bbuildings?\b|\bmaterial\s+damage\b|\boffice\s+package\b"),
    ("Stock",                          r"\bstock\b"),
    ("Goods in Transit",               r"\bgoods\s+in\s+transit\b|^git$|\bcargo\b|\bmarine\b"),
    ("Motor / Fleet",                  r"\bmotor\b|\bfleet\b|\bvehicle\b"),
    ("Engineering",                    r"\bengineering\b|\binspection\b|\bplant\b|\bmachinery\b"),
    ("Environmental",                  r"\benvironment|\bpollution\b"),
    ("Terrorism",                      r"\bterror"),
    ("Legal Expenses",                 r"\blegal\s+expense|\btax\s+(?:investigation|protection)\b"),
    ("Personal Accident",              r"\bpersonal\s+accident\b|\bbusiness\s+travel\b|^pa$|\btravel\b"),
    ("Group Life",                     r"\bgroup\s+life\b|\bdeath\s+in\s+service\b|\bkey\s+(?:person|man)\b"),
    ("Private Medical",                r"\bprivate\s+medical\b|^pmi$|\bhealth\s+(?:cash|insurance)\b"),
    ("Income Protection",              r"\bincome\s+protection\b|\bcritical\s+illness\b"),
    ("Mortgage Protection",            r"\bmortgage\s+protection\b"),
    ("Warranty & Indemnity",           r"\bwarrant(?:y|ies)\b.*\bindemnit|^w\s*and\s*i$"),
]

_COMPILED = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _PATTERNS]

CANONICAL_COVERAGES = [name for name, _ in _PATTERNS]

# Covers that are effectively mandatory in the UK for any company with staff.
# Used by the gap rules in Phase 4: a GAP here is always "Immediate action",
# never "Review at renewal", regardless of what peer prevalence says.
STATUTORY = {"Employers Liability"}

# HEADLINE COVERS. client_cover_summary.csv mixes two granularities: some clients
# are recorded at policy level ("Employers' Liability", "Professional Indemnity")
# and others at policy-SECTION level ("Debris Removal", "Trace and Access",
# "Negotiable Money - in safe on premise", "Loss of Limbs"). Benchmarking the
# median of a Debris Removal sub-limit against a Cyber band is meaningless, so
# the resolver restricts itself to these. Everything outside the set is a
# sub-limit and is counted-but-excluded, never silently dropped.
HEADLINE = {
    "Employers Liability", "Public & Products Liability", "Professional Indemnity",
    "Directors & Officers", "Cyber", "Crime", "Business Interruption",
    "Employment Practices Liability", "Medical Malpractice", "Clinical Trials",
    "Product Recall", "Goods in Transit", "Stock",
}

# Property is deliberately NOT headline. Its rows in client_cover_summary are
# contents/section sums insured (median £31k across 45 rows, min £1,124), not a
# headline indemnity limit, so any percentile computed on it is nonsense. Add it
# back only if a real property sum-insured field turns up.
EXCLUDED_FROM_BENCHMARK = {"Property"}


def _tidy(raw: str) -> str:
    """Collapse the cosmetic variants before pattern matching."""
    s = str(raw).strip()
    s = s.replace("&", " and ").replace("'", "'")
    s = re.sub(r"[^\w\s&]", " ", s.replace("&", " and "))
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalise_coverage(raw) -> Coverage:
    """Map any source spelling of a cover onto the canonical vocabulary.

    Returns a Coverage, never a bare string, so an unmatched name is impossible
    to mistake for a matched one at the call site.
    """
    if raw is None:
        return Coverage(raw="", canonical=None, matched=False)
    original = str(raw).strip()
    if not original:
        return Coverage(raw=original, canonical=None, matched=False)

    tidied = _tidy(original)
    for name, pattern in _COMPILED:
        if pattern.search(tidied):
            return Coverage(raw=original, canonical=name, matched=True)
    return Coverage(raw=original, canonical=None, matched=False)


def add_canonical_column(df, source_col: str, target_col: str = "coverage_canonical"):
    """Vectorised helper for the cleaned CSVs. Adds the canonical name plus a
    boolean so an audit of unmatched names is one groupby away."""
    results = df[source_col].map(normalise_coverage)
    df[target_col] = [r.canonical for r in results]
    df[target_col + "_matched"] = [r.matched for r in results]
    return df


def unmatched_report(df, source_col: str):
    """Run this ONCE against recorder_cover_lines and deck_benchmark_bands before
    trusting any benchmark output. Anything listed here is a cover silently
    excluded from every statistic downstream — add a pattern or accept the loss
    deliberately, but do not leave it undiscovered."""
    results = df[source_col].map(normalise_coverage)
    unmatched = [r.raw for r in results if not r.matched]
    from collections import Counter
    return Counter(unmatched).most_common()


if __name__ == "__main__":
    samples = [
        "Employers' Liability", "Employers Liability", "EL",
        "Public and Product Liability", "Public and Products Liability",
        "Cyber", "Cyber Liability", "Directors & Officers", "D&O",
        "Professional Indemnity", "PI", "Clinical Trials",
        "Management Liability", "Widget Insurance", "",
    ]
    for s in samples:
        c = normalise_coverage(s)
        print(f"{'Y' if c.matched else '-'}  {s!r:42} -> {c.canonical}")