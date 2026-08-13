import re
import pandas as pd

CAT_TOKENS = [
    ("no cover", "NO_COVER"),
    ("not covered", "NOT_COVERED"),
    ("not insured", "NOT_INSURED"),
    ("not purchased", "NOT_PURCHASED"),
    ("nil", "NIL"),
    ("various", "VARIOUS"),
    ("full limit", "FULL_LIMIT"),
    ("to be discussed", "TO_BE_DISCUSSED"),
    ("based on own figures/requirements", "BASED_ON_OWN_FIGURES"),
    ("subject to further disclosures", "SUBJECT_TO_DISCLOSURES"),
    ("subject to completion of bi calculation", "SUBJECT_TO_BI_CALC"),
]
CUR = {"£": "GBP", "$": "USD", "€": "EUR", "GBP": "GBP", "USD": "USD", "EUR": "EUR"}
SCALE = {"k": 1_000, "m": 1_000_000}

MONEY_RE = re.compile(
    r"(?P<cur>£|\$|€|\bGBP\b|\bUSD\b|\bEUR\b)?\s?"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)(?P<scale>\s?[mMkK])?(?=\b|[^A-Za-z])"
)
STRUCTURAL_RE = re.compile(
    r"in the aggregate|each and every claim|subject to (further )?disclosures|"
    r"subject to completion of bi calculation|\bminimum\b|\bbut\b|\bto\b|"
    r"\bfull limit\b|\bbased on own figures/requirements\b|\bsubject to disclosures\b",
    re.I,
)
PLUS_RE = re.compile(r"\+")
DASH_RE = re.compile(r"-")
SLASH_RE = re.compile(r"/")


def find_money(text):
    out = []
    for m in MONEY_RE.finditer(text):
        try:
            val = float(m.group("num").replace(",", ""))
        except ValueError:
            continue
        scale = (m.group("scale") or "").strip().lower()
        if scale in SCALE:
            val *= SCALE[scale]
        cur = m.group("cur")
        currency = CUR.get(cur.upper() if cur and cur.isalpha() else cur) if cur else None
        out.append((m.start(), m.end(), val, currency))
    return out


def leftover_text(text, spans):
    covered = [False] * len(text)
    for s, e in spans:
        for i in range(s, min(e, len(text))):
            covered[i] = True
    return re.sub(r"[^A-Za-z]", "", "".join(c for c, k in zip(text, covered) if not k))


def parse_cell(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or not str(raw).strip():
        return {"category": "EMPTY", "low": None, "high": None, "amounts": [],
                "currencies": [], "notes": [], "needs_review": False, "reason": ""}

    text = str(raw).strip()
    compact = re.sub(r"\s+", "", text.lower())

    for phrase, canon in CAT_TOKENS:
        if compact == phrase.replace(" ", "").replace("/", "/"):
            return {"category": canon, "low": None, "high": None, "amounts": [],
                    "currencies": [], "notes": [], "needs_review": False, "reason": ""}

    money = find_money(text)
    notes = []
    if STRUCTURAL_RE.search(text):
        for m in STRUCTURAL_RE.finditer(text):
            notes.append(m.group(0).strip().upper().replace(" ", "_"))
    open_ended = bool(PLUS_RE.search(text))
    is_minimum = bool(re.match(r"\s*minimum\b", text, re.I))
    if open_ended:
        notes.append("AT_LEAST")
    if is_minimum:
        notes.append("MINIMUM_QUALIFIER")

    if not money:
        return {"category": "UNSTRUCTURED_TEXT", "low": None, "high": None, "amounts": [],
                "currencies": [], "notes": notes, "needs_review": True,
                "reason": "no amount found, likely free-text"}

    amounts = [v for _, _, v, _ in money]
    currencies = [c for _, _, _, c in money]
    money_spans = [(s, e) for s, e, _, _ in money]
    structural_spans = [(m.start(), m.end()) for m in STRUCTURAL_RE.finditer(text)]
    structural_spans += [(m.start(), m.end()) for m in PLUS_RE.finditer(text)]

    if len(money) == 2:
        between = text[money[0][1]:money[1][0]]
        if (DASH_RE.search(between) or re.search(r"\bto\b", between, re.I)) and not SLASH_RE.search(between):
            leftover = leftover_text(text, money_spans + structural_spans + [(m.start(), m.end()) for m in DASH_RE.finditer(text)])
            low, high = min(amounts), max(amounts)
            suspicious = low > 0 and high / low > 100
            review = len(leftover) > 12 or suspicious
            reason = f"unexplained text: {leftover!r}" if len(leftover) > 12 else \
                (f"range endpoints inconsistent ({low:,.0f} vs {high:,.0f}), possible missing separator/scale" if suspicious else "")
            return {"category": "RANGE", "low": low, "high": high,
                    "amounts": amounts, "currencies": currencies, "notes": notes,
                    "needs_review": review, "reason": reason}
        if SLASH_RE.search(between):
            leftover = leftover_text(text, money_spans + structural_spans + [(m.start(), m.end()) for m in SLASH_RE.finditer(text)])
            return {"category": "SLASH_PAIR", "low": None, "high": None,
                    "amounts": amounts, "currencies": currencies, "notes": notes,
                    "needs_review": True, "reason": "two amounts joined by '/', meaning unclear"}

    leftover = leftover_text(text, money_spans + structural_spans)
    review = len(leftover) > 12
    category = "OPEN_ENDED" if (open_ended and len(money) == 1) else \
        ("MINIMUM" if (is_minimum and len(money) == 1) else
         ("PARSED_MULTI_AMOUNT" if len(money) > 1 else "PARSED_AMOUNT"))
    return {"category": category, "low": min(amounts), "high": max(amounts),
            "amounts": amounts, "currencies": currencies, "notes": notes,
            "needs_review": review, "reason": f"unexplained text: {leftover!r}" if review else ""}


def parse_dataframe(df, fields=("current_limit", "capsule_recommended_limit")):
    df = df.copy()
    for field in fields:
        parsed = df[field].apply(parse_cell)
        df[f"{field}_category"] = parsed.apply(lambda p: p["category"])
        df[f"{field}_low"] = parsed.apply(lambda p: p["low"])
        df[f"{field}_high"] = parsed.apply(lambda p: p["high"])
        df[f"{field}_amounts"] = parsed.apply(lambda p: ";".join(str(a) for a in p["amounts"]))
        df[f"{field}_currencies"] = parsed.apply(lambda p: ";".join(c or "" for c in p["currencies"]))
        df[f"{field}_notes"] = parsed.apply(lambda p: ";".join(p["notes"]))
        df[f"{field}_needs_review"] = parsed.apply(lambda p: p["needs_review"])
        df[f"{field}_reason"] = parsed.apply(lambda p: p["reason"])
    return df


def load(path):
    if not path.endswith(".csv"):
        path = path + ".csv"
    return pd.read_csv(path, encoding="utf-8-sig")


if __name__ == "__main__":
    df = load("dataset/2_book_of_business/deck_limit_benchmarks")
    out = parse_dataframe(df)
    out.to_csv("data/deck_limit_benchmarks_clean.csv", index=False)
    print(out.filter(like="needs_review").sum())