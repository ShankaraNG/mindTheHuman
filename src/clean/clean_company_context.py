import os
import re
import pandas as pd

DATA_DIR = "data"


def parse_amount(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s).strip()
    if not s or s.startswith("{"):
        return None
    s = s.replace("£", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def load_questionnaire(path):
    if not path.endswith(".csv"):
        path += ".csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["turnover_past_fy_gbp"] = df["turnover_past_fy_gbp"].apply(parse_amount)
    df["turnover_current_fy_est_gbp"] = df["turnover_current_fy_est_gbp"].apply(parse_amount)
    for c in ["regulated_business", "manual_labour_roles"]:
        df[c] = df[c].astype("boolean")
    return df


def load_hubspot(path):
    if not path.endswith(".csv"):
        path += ".csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["fca_regulated"] = df["fca_regulated"].astype("boolean")
    return df


def coalesce_with_source(df, cols_in_priority_order, out_col):
    result = pd.Series(pd.NA, index=df.index, dtype=object)
    source = pd.Series(pd.NA, index=df.index, dtype=object)
    for col in cols_in_priority_order:
        mask = result.isna() & df[col].notna()
        result[mask] = df.loc[mask, col]
        source[mask] = col
    df[out_col] = result
    df[f"{out_col}_source"] = source
    return df


def merge_clients(q, h):
    df = q.merge(h, on="client_id", how="outer", suffixes=("_q", "_h"))

    df["vertical"] = df["vertical_q"]
    v_mask = (df["vertical"].isna() | (df["vertical"] == "Unknown")) & df["vertical_h"].notna()
    df.loc[v_mask, "vertical"] = df.loc[v_mask, "vertical_h"]

    df = coalesce_with_source(
        df,
        ["turnover_last12m_gbp", "turnover_current_fy_est_gbp", "turnover_past_fy_gbp",
         "turnover_gbp", "annual_revenue_band"],
        "turnover_final_gbp",
    )
    df["turnover_is_band"] = df["turnover_final_gbp_source"] == "annual_revenue_band"

    df = coalesce_with_source(df, ["employee_count_q", "employee_count_h"], "employee_count_final")

    df = coalesce_with_source(
        df, ["total_funding_raised_gbp", "total_funding_gbp"], "total_funding_final_gbp"
    )

    df["fca_regulated"] = df["fca_regulated"].astype("boolean")
    df["regulated_business"] = df["regulated_business"].astype("boolean")
    both_known = df["fca_regulated"].notna() & df["regulated_business"].notna()
    df["regulated_conflict"] = both_known & (df["fca_regulated"] != df["regulated_business"])

    regulated_final = pd.Series(pd.NA, index=df.index, dtype="boolean")
    regulated_source = pd.Series(pd.NA, index=df.index, dtype=object)
    fca_known = df["fca_regulated"].notna()
    regulated_final[fca_known] = df.loc[fca_known, "fca_regulated"]
    regulated_source[fca_known] = "fca_regulated"
    q_only = (~fca_known) & df["regulated_business"].notna()
    regulated_final[q_only] = df.loc[q_only, "regulated_business"]
    regulated_source[q_only] = "regulated_business"
    df["regulated_final"] = regulated_final
    df["regulated_final_source"] = regulated_source

    return df


def run(q_path, h_path, out_name="client_level.csv", out_dir=DATA_DIR):
    q = load_questionnaire(q_path)
    h = load_hubspot(h_path)
    merged = merge_clients(q, h)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    merged.to_csv(out_path, index=False)

    assert merged["client_id"].is_unique, "client_level.csv grain broken: duplicate client_id"

    return merged


if __name__ == "__main__":
    merged = run(
        "dataset/3_company_context/healthcheck_questionnaire",
        "dataset/3_company_context/hubspot_company_context",
    )
    print(f"{len(merged)} clients")
    print("turnover source:", merged["turnover_final_gbp_source"].value_counts(dropna=False).to_dict())
    print("turnover still missing:", merged["turnover_final_gbp"].isna().sum())
    print("regulated_final source:", merged["regulated_final_source"].value_counts(dropna=False).to_dict())
    print("regulated_final still unknown (NA):", merged["regulated_final"].isna().sum())
    print("regulated_conflict:", merged["regulated_conflict"].sum())