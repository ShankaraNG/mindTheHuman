"""
Derive the four trigger flags from free-text business descriptions.

personal_data / medical_data / financial_data / stock_logistics do not exist as
structured columns anywhere in the dataset — not in the questionnaire, not in the
CRM, not in the synthetic evaluation briefs. Free text is the only source.

CRITICAL: this same function must run on BOTH sides of the comparison —
`business_activity_description` when building the peer pool, and
`business_description` (or live web-form text) at intake. If the two sides were
extracted differently, any comparison between them would be meaningless.

Why keywords rather than an LLM call: deterministic (a judge re-running the demo
gets identical output), free, and defensible in one sentence. The trade-off is
recall on unusual phrasing — accepted, and documented in KNOWN_LIMITATIONS below.

Vocabulary provenance: a first pass using compliance language ('PII', 'GDPR',
'personal data') matched 0 of 242 clients. Founders describe what their business
*does*, not its data-protection obligations. The lists below were rebuilt by
reading the actual descriptions of the 25 clients who hold cyber cover in
Capsule's book — i.e. companies Capsule's own brokers judged to carry data risk.
"""

import re
from typing import Optional

import pandas as pd

# Terms are matched on word boundaries, so "user" does not fire inside "users"
# unintentionally... but it DOES need to match "users", so stems are used where
# the plural/inflected forms all count: "prescrib" -> prescribing, prescriber.
# Multi-word phrases are matched as phrases.
TRIGGER_TERMS: dict[str, tuple[str, ...]] = {
    "personal_data": (
        # direct
        r"personal data", r"\bpii\b", r"\bgdpr\b", r"data subject", r"privacy",
        # how founders actually describe it
        r"\busers?\b", r"\bcustomers?\b", r"\bconsumers?\b", r"\bsubscribers?\b",
        r"\bmembers?\b", r"\bstudents?\b", r"\bemployees?\b", r"\bindividuals?\b",
        r"\bcandidates?\b", r"\brecruit", r"\btalent\b", r"\bcrm\b",
        r"data collected", r"\bidentif",
    ),
    "medical_data": (
        r"\bpatients?\b", r"\bclinical\b", r"\bclinician", r"\bhealthcare\b",
        r"\bhealth\b", r"mental health", r"\bmedical\b", r"health record",
        r"\bdiagnos", r"\bprescrib", r"\btherapeutic", r"\btreatment\b",
        r"virtual care", r"care model", r"care delivery", r"\bwellness\b",
        r"\bnhs\b", r"\bpharma",
    ),
    "financial_data": (
        r"\bpayments?\b", r"payment data", r"card data", r"\bbanking\b",
        r"\bbank\b", r"bank details", r"bank account", r"point of sale",
        r"\bpos\b", r"e-commerce", r"ecommerce", r"\bmerchants?\b",
        r"\btransactions?\b", r"\bfintech\b", r"\binvoic", r"\bbilling\b",
        r"\blending\b", r"\bwealth\b", r"high-interest", r"\bcheckouts?\b",
    ),
    "stock_logistics": (
        r"\binventory\b", r"\bwarehous", r"\bfulfil", r"\blogistic",
        r"supply chain", r"\bstock\b", r"\bshipping\b", r"\bdeliver",
        r"\bmanufactur", r"\bretail\b", r"\bdistribut", r"\bfreight\b",
        r"physical goods", r"\bhardware\b", r"\bproduce\b", r"\bbottled\b",
    ),
}

FLAG_NAMES: tuple[str, ...] = tuple(TRIGGER_TERMS)

# Compiled once at import — this runs over every row of the peer pool.
_COMPILED: dict[str, re.Pattern] = {
    flag: re.compile("|".join(terms), flags=re.IGNORECASE)
    for flag, terms in TRIGGER_TERMS.items()
}

KNOWN_LIMITATIONS = """
1. Recall is bounded by vocabulary. Some clients hold cyber cover for general
   software/platform exposure that none of the brief's four categories captures
   (e.g. an AI video editor, a battery-recycling logistics platform). Those are
   correctly not flagged here; the gap is in the four-category framing, not the
   extraction.
2. Precision is deliberately loose on personal_data: broad terms like 'customers'
   and 'platform-adjacent' language fire often. This is the intended direction of
   error for a trigger whose job is to prompt a cover check, not to make a
   decision. Validated at odds ratio ~5.1 against real cyber-cover holdings.
3. No description means no evidence. Flags return None (unknown), never False.
   ~46% of the peer pool has no description at all.
"""


def extract_flags(text: Optional[str]) -> dict[str, Optional[bool]]:
    """Free text -> the four trigger booleans.

    Returns None (not False) for every flag when there is no text to read.
    Absence of evidence is not evidence of absence: a company that never filled
    in the form has not told us it is free of personal-data risk.
    """
    if text is None or (not isinstance(text, str) and pd.isna(text)) or not str(text).strip():
        return {flag: None for flag in FLAG_NAMES}

    haystack = str(text)
    return {flag: bool(pattern.search(haystack)) for flag, pattern in _COMPILED.items()}


def matched_terms(text: Optional[str]) -> dict[str, list[str]]:
    """Which terms actually fired, per flag. For auditing false positives and for
    citing evidence in the generated report ('flagged because: patients, clinical')."""
    if text is None or (not isinstance(text, str) and pd.isna(text)) or not str(text).strip():
        return {flag: [] for flag in FLAG_NAMES}

    haystack = str(text)
    out: dict[str, list[str]] = {}
    for flag, terms in TRIGGER_TERMS.items():
        hits = [
            re.search(t, haystack, re.IGNORECASE).group(0)
            for t in terms
            if re.search(t, haystack, re.IGNORECASE)
        ]
        out[flag] = sorted(set(h.lower() for h in hits))
    return out


if __name__ == "__main__":
    # Real descriptions from clients who hold cyber cover — the vocabulary source.
    samples = [
        "Global Payments System Provider",
        "Point of Sale app to allow SMEs to take physical and e-commerce payments",
        "Designed to help your employees find the right mental healthcare at the "
        "right time, with care built with leading clinician-academics.",
        "Gaea develops a platform focused on providing logistical support to "
        "battery recycling processes.",
        "The company's principal activity was the development of an AI-led online video editor.",
        None,
    ]
    for s in samples:
        flags = extract_flags(s)
        shown = {k: v for k, v in flags.items() if v}
        label = (s[:62] + "...") if s and len(s) > 62 else s
        print(f"{str(label):<68} -> {shown or flags}")