import os
import pandas as pd

DATA_DIR = "data"

STR_COLS_BOB = ["client_id", "vertical", "policy_id", "product", "insurer", "status"]
STR_COLS_CL = ["client_id", "vertical", "policy_id", "product", "section", "coverage",
               "limit_basis", "limit_name", "excess_type"]


def clean_strings(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["nan", "None", ""]), c] = pd.NA
    return df


def load_book_of_business(path):
    if not path.endswith(".csv"):
        path = path + ".csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_strings(df, STR_COLS_BOB)
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
    return df


def load_cover_lines(path):
    if not path.endswith(".csv"):
        path = path + ".csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_strings(df, STR_COLS_CL)
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    return df, n_dropped


def validate(bob, cl, dupes_dropped):
    issues = []
    orphan_cl = cl[~cl["policy_id"].isin(bob["policy_id"])]
    if len(orphan_cl):
        issues.append(("ORPHAN_COVER_LINES", len(orphan_cl), "cover_lines rows with policy_id not in book_of_business"))

    actual_counts = cl.groupby("policy_id").size().rename("actual_n_coverages")
    check = bob.set_index("policy_id")[["n_coverages"]].join(actual_counts).fillna(0)
    mismatch = check[check["n_coverages"] != check["actual_n_coverages"]]
    if len(mismatch):
        issues.append(("N_COVERAGES_MISMATCH", len(mismatch), "policies where n_coverages != actual cover_lines row count"))

    bad_dates = bob[bob["end_date"] < bob["effective_date"]]
    if len(bad_dates):
        issues.append(("END_BEFORE_EFFECTIVE", len(bad_dates), "policies where end_date is before effective_date"))

    neg_prem = bob[bob["total_premium"] < 0]
    if len(neg_prem):
        issues.append(("NEGATIVE_PREMIUM", len(neg_prem), "policies with negative total_premium"))

    zero_prem = bob[bob["total_premium"] == 0]
    if len(zero_prem):
        issues.append(("ZERO_PREMIUM", len(zero_prem), "policies with total_premium == 0"))

    unknown_vertical = bob[bob["vertical"] == "Unknown"]
    if len(unknown_vertical):
        issues.append(("UNKNOWN_VERTICAL", len(unknown_vertical), "policies with client_id/vertical unmapped (UNK-xxx / Unknown)"))

    no_lines = bob[~bob["policy_id"].isin(cl["policy_id"])]
    if len(no_lines):
        issues.append(("POLICY_WITH_NO_COVER_LINES", len(no_lines), "policies present in book_of_business with zero rows in cover_lines"))

    issues.append(("EXACT_DUPLICATE_COVER_LINES_DROPPED", dupes_dropped, "fully duplicate rows removed from cover_lines"))

    return pd.DataFrame(issues, columns=["issue", "count", "description"])


def run(bob_path, cl_path, out_policy="policy_level.csv", out_cover="cover_line_level.csv",
        out_validation="validation_report.csv", status="Active", out_dir=DATA_DIR):
    bob = load_book_of_business(bob_path)
    cl, dupes_dropped = load_cover_lines(cl_path)

    report = validate(bob, cl, dupes_dropped)

    active_bob = bob[bob["status"] == status].copy()
    active_cl = cl[cl["policy_id"].isin(active_bob["policy_id"])].copy()

    os.makedirs(out_dir, exist_ok=True)
    active_bob.to_csv(os.path.join(out_dir, out_policy), index=False)
    active_cl.to_csv(os.path.join(out_dir, out_cover), index=False)
    report.to_csv(os.path.join(out_dir, out_validation), index=False)

    return active_bob, active_cl, report


if __name__ == "__main__":
    active_bob, active_cl, report = run(
        "dataset/2_book_of_business/recorder_book_of_business",
        "dataset/2_book_of_business/recorder_cover_lines",
    )
    print(f"policy_level.csv: {len(active_bob)} active policies")
    print(f"cover_line_level.csv: {len(active_cl)} cover lines")
    print(report.to_string(index=False))