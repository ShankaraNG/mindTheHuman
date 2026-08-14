import os
import pandas as pd

DATA_DIR = "data"
ANSWER_KEY_COL = "evaluation_notes_expected_themes"


def load(path):
    if not path.endswith(".csv"):
        path += ".csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    for c in df.select_dtypes(include=["object", "str"]).columns:
        df[c] = df[c].str.strip()
    return df


def split(df):
    assert df["brief_id"].is_unique, "grain broken: duplicate brief_id"
    themes = df[["brief_id", ANSWER_KEY_COL]].copy()
    briefs = df.drop(columns=[ANSWER_KEY_COL]).copy()
    return briefs, themes


def run(in_path, out_briefs="synthetic_briefs_clean.csv", out_themes="expected_themes.csv", out_dir=DATA_DIR):
    df = load(in_path)
    briefs, themes = split(df)
    os.makedirs(out_dir, exist_ok=True)
    briefs.to_csv(os.path.join(out_dir, out_briefs), index=False)
    themes.to_csv(os.path.join(out_dir, out_themes), index=False)
    assert ANSWER_KEY_COL not in briefs.columns
    return briefs, themes


if __name__ == "__main__":
    briefs, themes = run("dataset/4_evaluation/synthetic_prospect_briefs")
    print(f"{len(briefs)} briefs -> synthetic_briefs_clean.csv, columns: {list(briefs.columns)}")
    print(f"{len(themes)} rows -> expected_themes.csv, columns: {list(themes.columns)}")