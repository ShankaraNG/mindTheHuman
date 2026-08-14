import re
import pandas as pd

CAT_TOKENS = [
    ("not disclosed", "NOT_DISCLOSED"),
    ("not included", "NOT_INCLUDED"),
    ("various limits", "VARIOUS"),
    ("various excesses", "VARIOUS"),
    ("unlimited", "UNLIMITED"),
    ("included", "INCLUDED"),
    ("various", "VARIOUS"),
    ("unknown", "UNKNOWN"),
    ("covered", "COVERED"),
    ("nil", "NIL"),
    ("none", "NONE"),
    ("n/a", "NA"),
    ("na", "NA"),
    ("tbc", "TBC"),
    ("inc", "INCLUDED"),
]
CUR = {"£": "GBP", "$": "USD", "€": "EUR", "GBP": "GBP", "USD": "USD", "EUR": "EUR"}
SCALE = {"k": 1_000, "m": 1_000_000}

MONEY_RE = re.compile(
    r"(?P<cur>£|\$|€|\bGBP\b|\bUSD\b|\bEUR\b)?\s?"
    r"(?P<num>\d[\d,]*(?:\.\d+)?)(?P<scale>\s?[mMkK])?(?=\b|[^A-Za-z])"
)
NOTE_RES = {
    "IPT": re.compile(r"\+\s*IPT\b|Insurance Premium Tax", re.I),
    "APPLICABLE_TAXES": re.compile(r"applicable taxes", re.I),
    "ADMIN_FEE": re.compile(r"Admin Fee|Policy Fee", re.I),
    "AGGREGATE": re.compile(r"in the aggregate", re.I),
    "FOOTNOTE": re.compile(r"\*"),
}
SPLIT_RE = re.compile(r"[\n;]+")


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


def find_tokens(text):
    lower = text.lower()
    hits = []
    used = [False] * len(text)
    for phrase, canon in CAT_TOKENS:
        i = 0
        while (idx := lower.find(phrase, i)) != -1:
            end = idx + len(phrase)
            i = idx + 1
            if any(used[idx:end]):
                continue
            if (idx == 0 or not lower[idx - 1].isalpha()) and (end == len(text) or not lower[end].isalpha()):
                hits.append((idx, end, canon))
                for j in range(idx, end):
                    used[j] = True
    return sorted(hits)


def find_notes(text):
    return [name for name, rx in NOTE_RES.items() if rx.search(text)]


def repeated_token(compact):
    for phrase, canon in CAT_TOKENS:
        p = phrase.replace(" ", "")
        if p and len(compact) >= 2 * len(p) and len(compact) % len(p) == 0:
            if compact == p * (len(compact) // len(p)):
                return canon, len(compact) // len(p)
    return None


def parse_cell(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or not str(raw).strip():
        return {"category": "EMPTY", "amounts": [], "currencies": [], "tokens": [],
                "notes": [], "needs_review": False, "reason": ""}

    text = str(raw).strip()
    compact = re.sub(r"\s+", "", text.lower())

    for phrase, canon in CAT_TOKENS:
        if compact == phrase.replace(" ", ""):
            return {"category": canon, "amounts": [], "currencies": [], "tokens": [canon],
                    "notes": [], "needs_review": False, "reason": ""}

    rep = repeated_token(compact)
    if rep:
        canon, n = rep
        return {"category": "REPEATED_CATEGORICAL", "amounts": [], "currencies": [],
                "tokens": [canon] * n, "notes": [], "needs_review": False, "reason": ""}

    parts = [p.strip() for p in SPLIT_RE.split(text) if p.strip()]
    if len(parts) > 1:
        sub = [parse_cell(p) for p in parts]
        amounts = [a for s in sub for a in s["amounts"]]
        currencies = [c for s in sub for c in s["currencies"]]
        tokens = [t for s in sub for t in s["tokens"]]
        review = any(s["needs_review"] or s["category"] == "UNSTRUCTURED_TEXT" for s in sub)
        return {"category": "TIERED_EXPLICIT", "amounts": amounts, "currencies": currencies,
                "tokens": tokens, "notes": [], "needs_review": review,
                "reason": "one or more tiers unclear" if review else ""}

    money = find_money(text)
    tokens = find_tokens(text)
    notes = find_notes(text)

    if tokens and not money:
        recon = "".join(text[s:e] for s, e, _ in tokens)
        if re.sub(r"\s+", "", recon).lower() == compact:
            canon = [c for _, _, c in tokens]
            mixed = len(set(canon)) > 1
            return {"category": "REPEATED_CATEGORICAL" if len(canon) > 1 else canon[0],
                    "amounts": [], "currencies": [], "tokens": canon, "notes": [],
                    "needs_review": mixed, "reason": "mixed categorical tokens" if mixed else ""}

    if money:
        covered = [False] * len(text)
        for s, e, _, _ in money:
            for i in range(s, e):
                covered[i] = True
        for s, e, _ in tokens:
            for i in range(s, e):
                covered[i] = True
        for rx in NOTE_RES.values():
            for m in rx.finditer(text):
                for i in range(m.start(), m.end()):
                    covered[i] = True
        leftover = re.sub(r"[^A-Za-z]", "", "".join(c for c, k in zip(text, covered) if not k))
        review = len(leftover) > 12
        return {"category": "PARSED_MULTI_AMOUNT" if len(money) > 1 else "PARSED_AMOUNT",
                "amounts": [v for _, _, v, _ in money],
                "currencies": [c for _, _, _, c in money],
                "tokens": [c for _, _, c in tokens], "notes": notes,
                "needs_review": review,
                "reason": f"unexplained text: {leftover!r}" if review else ""}

    if tokens:
        return {"category": "CATEGORICAL_WITH_TEXT", "amounts": [], "currencies": [],
                "tokens": [c for _, _, c in tokens], "notes": notes,
                "needs_review": True, "reason": "token mixed with free text"}

    return {"category": "UNSTRUCTURED_TEXT", "amounts": [], "currencies": [],
            "tokens": [], "notes": notes, "needs_review": True,
            "reason": "no amount or token found, likely commentary"}


def parse_dataframe(df, fields=("limit", "excess", "premium")):
    df = df.copy()
    for field in fields:
        parsed = df[field].apply(parse_cell)
        df[f"{field}_category"] = parsed.apply(lambda p: p["category"])
        df[f"{field}_amounts"] = parsed.apply(lambda p: ";".join(str(a) for a in p["amounts"]))
        df[f"{field}_primary_amount"] = parsed.apply(lambda p: p["amounts"][0] if p["amounts"] else None)
        df[f"{field}_currencies"] = parsed.apply(lambda p: ";".join(c or "" for c in p["currencies"]))
        df[f"{field}_tokens"] = parsed.apply(lambda p: ";".join(p["tokens"]))
        df[f"{field}_notes"] = parsed.apply(lambda p: ";".join(p["notes"]))
        df[f"{field}_needs_review"] = parsed.apply(lambda p: p["needs_review"])
        df[f"{field}_reason"] = parsed.apply(lambda p: p["reason"])
    return df


def load(path):
    if not path.endswith(".csv"):
        path = path + ".csv"
    return pd.read_csv(path, encoding="utf-8-sig")


if __name__ == "__main__":
    df = load("dataset/2_book_of_business/deck_current_cover")
    out = parse_dataframe(df)
    out.to_csv("data/deck_current_cover_clean.csv", index=False)
    print(out.filter(like="needs_review").sum())