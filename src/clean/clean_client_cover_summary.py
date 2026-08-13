import os
import re
import pandas as pd

DATA_DIR = "data"


def normalize_coverage(s):
    s = re.sub(r"\s+", " ", str(s).strip())
    s = s.replace(" & ", " and ")
    return s


def load(path):
    if not path.endswith(".csv"):
        path += ".csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.drop_duplicates()


def rollup(df):
    df = df.dropna(subset=["coverage"]).copy()
    df["coverage_norm"] = df["coverage"].apply(normalize_coverage)
    df["coverage_key"] = df["coverage_norm"].str.lower()

    display_name = (
        df.groupby(["client_id", "coverage_key"])["coverage_norm"]
        .agg(lambda s: s.value_counts().idxmax())
        .rename("coverage")
    )

    agg = df.groupby(["client_id", "coverage_key"]).agg(
        vertical=("vertical", "first"),
        n_lines=("coverage", "size"),
        n_policies=("policy_id", "nunique"),
        policy_ids=("policy_id", lambda s: ";".join(sorted(s.unique()))),
        total_coverage_premium=("coverage_premium", lambda s: s.sum(min_count=1)),
        max_limit_amount=("limit_amount", "max"),
        min_limit_amount=("limit_amount", "min"),
        limit_basis=("limit_basis", lambda s: ";".join(sorted(s.dropna().unique()))),
        max_excess_amount=("excess_amount", "max"),
        min_excess_amount=("excess_amount", "min"),
        excess_type=("excess_type", lambda s: ";".join(sorted(s.dropna().unique()))),
    )

    out = agg.join(display_name).reset_index(drop=False)
    out = out.drop(columns=["coverage_key"])
    out = out[["client_id", "vertical", "coverage", "n_lines", "n_policies", "policy_ids",
               "total_coverage_premium", "max_limit_amount", "min_limit_amount",
               "limit_basis", "max_excess_amount", "min_excess_amount", "excess_type"]]
    return out.sort_values(["client_id", "coverage"]).reset_index(drop=True)


def run(in_path, out_name="client_cover_summary.csv", out_dir=DATA_DIR):
    df = load(in_path)
    out = rollup(df)
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(os.path.join(out_dir, out_name), index=False)
    assert not out.duplicated(subset=["client_id", "coverage"]).any(), "grain broken: duplicate client_id+coverage"
    return out


if __name__ == "__main__":
    out = run("dataset/2_book_of_business/recorder_cover_lines")
    print(f"{len(out)} client x coverage rows")
    print(out.head(10).to_string())