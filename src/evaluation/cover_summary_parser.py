"""
Parses the free-text current_cover_summary field from synthetic_prospect_briefs.csv
into a {cover_name: limit_gbp_or_None} dict, matching the shape orchestrator.run()
expects for current_cover.

This format only exists in the synthetic evaluation briefs — a real prospect
enters cover through the Streamlit checkbox UI, not free text. This parser
exists purely to run the eval harness against the 24 test briefs.
"""

import re

ALIASES = {
    "PI":             "Professional Indemnity",
    "PL":             "Public and Products Liability",
    "EL":             "Employers Liability",
    "D&O":            "Directors and Officers",
    "Cyber":          "Cyber",
    "Crime":          "Crime",
    "Cargo":          "Cargo",
    "BI":             "Business Interruption",
    "ML":             "Directors and Officers",
    "Product Recall": "Product Recall",
    "Legal Expenses": "Legal Expenses",
}

NEGATION_PATTERNS = [r"no\s+([A-Za-z&/\s]+?)(?:,|\.|$)"]


def parse_cover_summary(text: str) -> dict:
    if not text or not isinstance(text, str):
        return {}

    result = {}

    for pattern in NEGATION_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            clause = m.group(1).strip()
            for part in re.split(r"[/,]| and ", clause):
                part = part.strip()
                for abbr, full in ALIASES.items():
                    if re.fullmatch(re.escape(abbr), part, re.IGNORECASE):
                        result[full] = None

    for abbr, full in ALIASES.items():
        pattern = rf"\b{re.escape(abbr)}\b[^£]{{0,15}}£([\d,.]+)\s*([mk]?)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m and full not in result:
            num = float(m.group(1).replace(",", ""))
            mult = m.group(2).lower()
            if mult == "m":
                num *= 1_000_000
            elif mult == "k":
                num *= 1_000
            result[full] = num

    return result


if __name__ == "__main__":
    samples = [
        "Commercial Combined (PL £2m, EL £10m, stock £1.5m), D&O £1m. No cyber, no product recall.",
        "CCI (PL £5m), Cargo £250k, D&O £1m, Cyber £250k.",
        "PI £1m, Cyber £500k, D&O £500k.",
        "Commercial Combined (PL £5m, EL £10m, contents/stock), no D&O, no cyber, fleet of 3 vans.",
        "CCI (PL £5m, EL £10m, machinery, BI 12 months), Fleet, no ML/D&O, Legal Expenses £100k.",
        "PI/media £1m combined, contents, no cyber, no D&O, no event cover.",
    ]
    for s in samples:
        print(f"{s}\n  -> {parse_cover_summary(s)}\n")