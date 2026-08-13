import os
import pandas as pd

DATA_DIR = "data"

PRIORITY_MAP = {
    "Immediate action": ("IMMEDIATE", 1),
    "Review at renewal": ("REVIEW_AT_RENEWAL", 2),
    "For consideration": ("FOR_CONSIDERATION", 3),
}


def load(path):
    if not path.endswith(".csv"):
        path += ".csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    for c in df.select_dtypes(include=["object", "str"]).columns:
        df[c] = df[c].str.strip()
    return df


def normalize_priority(df):
    df = df.copy()
    mapped = df["priority"].map(PRIORITY_MAP)
    df["priority_code"] = mapped.apply(lambda t: t[0] if isinstance(t, tuple) else pd.NA)
    df["priority_rank"] = mapped.apply(lambda t: t[1] if isinstance(t, tuple) else pd.NA)
    df["priority_unspecified"] = df["priority"].isna()
    return df


def flag_row_types(df):
    df = df.copy()
    df["is_blank_row"] = df["item"].isna() & df["action"].isna() & df["priority"].isna()
    df["is_section_header"] = df["item"].notna() & df["action"].isna() & df["priority"].isna() & ~df["is_blank_row"]
    return df


def run(in_path, out_name="deck_recommendations_clean.csv", out_dir=DATA_DIR):
    df = load(in_path)
    df = normalize_priority(df)
    df = flag_row_types(df)
    cols = ["client_id", "vertical", "item", "action", "priority", "priority_code",
            "priority_rank", "priority_unspecified", "is_section_header", "is_blank_row"]
    df = df[cols]
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, out_name), index=False)
    return df


if __name__ == "__main__":
    df = run("dataset/2_book_of_business/deck_recommendations")
    print(f"{len(df)} rows")
    print("priority_code counts:", df["priority_code"].value_counts(dropna=False).to_dict())
    print("priority_unspecified (has content, no priority label):",
          (df["priority_unspecified"] & ~df["is_section_header"] & ~df["is_blank_row"]).sum())
    print("section headers:", df["is_section_header"].sum())
    print("blank rows:", df["is_blank_row"].sum())