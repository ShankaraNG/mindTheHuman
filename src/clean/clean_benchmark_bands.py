import os
import re
import pandas as pd

DATA_DIR = "data"
SCALE = {"m": 1_000_000, "k": 1_000}
TOKEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([mMkK])?")


def scan_tokens(text):
    return [(float(n), s.lower() if s else None) for n, s in TOKEN_RE.findall(text)]


def default_scale(text):
    for _, s in scan_tokens(text):
        if s:
            return SCALE[s]
    return 1


def parse_band(band_text, scale):
    text = band_text.strip()
    open_ended = text.endswith("+")
    tokens = scan_tokens(text.rstrip("+"))
    values = [n * (SCALE[s] if s else scale) for n, s in tokens]
    if open_ended:
        return (values[0] if values else None), None, True
    if len(values) == 2:
        return values[0], values[1], False
    if len(values) == 1:
        return 0.0, values[0], False
    return None, None, False


def parse_limit(limit_text, scale):
    text = limit_text.strip()
    open_ended = text.endswith("+")
    tokens = scan_tokens(text.rstrip("+"))
    if not tokens:
        return None, open_ended
    n, s = tokens[0]
    return n * (SCALE[s] if s else scale), open_ended


def unpack(df):
    rows = []
    for i, row in df.iterrows():
        bands = [b.strip() for b in str(row["bands"]).split("|")]
        limits = [l.strip() for l in str(row["recommended_limits"]).split("|")]
        if len(bands) != len(limits):
            rows.append({
                "client_id": row["client_id"], "vertical": row["vertical"], "basis": row["basis"],
                "band_index": None, "band_raw": row["bands"], "band_low": None, "band_high": None,
                "band_open_ended": None, "recommended_limit_raw": row["recommended_limits"],
                "recommended_limit_gbp": None, "recommended_limit_open_ended": None,
                "source_row_index": i, "slide_context": row["slide_context"],
                "needs_review": True, "review_reason": "bands/recommended_limits count mismatch",
            })
            continue

        band_scale = default_scale(row["bands"])
        limit_scale = default_scale(row["recommended_limits"])
        for idx, (b, l) in enumerate(zip(bands, limits)):
            low, high, b_open = parse_band(b, band_scale)
            limit_val, l_open = parse_limit(l, limit_scale)
            rows.append({
                "client_id": row["client_id"], "vertical": row["vertical"], "basis": row["basis"],
                "band_index": idx, "band_raw": b, "band_low": low, "band_high": high,
                "band_open_ended": b_open, "recommended_limit_raw": l,
                "recommended_limit_gbp": limit_val, "recommended_limit_open_ended": l_open,
                "source_row_index": i, "slide_context": row["slide_context"],
                "needs_review": False, "review_reason": "",
            })

    out = pd.DataFrame(rows)
    dup_key = out.groupby(["client_id", "basis"])["source_row_index"].transform("nunique")
    out["duplicate_basis_set"] = dup_key > 1
    out.loc[out["duplicate_basis_set"], "needs_review"] = True
    out.loc[out["duplicate_basis_set"], "review_reason"] = (
        out.loc[out["duplicate_basis_set"], "review_reason"] + ";multiple band-sets for same client_id+basis, no cover field to distinguish"
    ).str.lstrip(";")
    return out


def load(path):
    if not path.endswith(".csv"):
        path += ".csv"
    return pd.read_csv(path, encoding="utf-8-sig")


def run(in_path, out_name="deck_benchmark_bands_clean.csv", out_dir=DATA_DIR):
    df = load(in_path)
    out = unpack(df)
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, out_name), index=False)
    return out


if __name__ == "__main__":
    out = run("dataset/2_book_of_business/deck_benchmark_bands")
    print(f"{len(out)} band rows from source rows")
    print("needs_review:", out["needs_review"].sum())
    print(out.head(12).to_string())