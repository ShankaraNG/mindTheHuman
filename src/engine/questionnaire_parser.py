"""
Questionnaire File Parser

The customer uploads a filled health check questionnaire — same schema as
healthcheck_questionnaire.csv (CSV or Excel). This file has the profile
(turnover, funding, headcount, regulation) but NOT current cover limits.

This module's ONLY job is:
  1. Read the uploaded file (CSV or XLSX), flexible on column naming
  2. Resolve turnover from the questionnaire's 3 competing columns by
     precedence (last12m > current_fy_est > past_fy) — this precedence
     resolution has no home in normalise.py, which expects an ALREADY
     resolved turnover_final_gbp column (that's clean_company_context.py's
     job on the peer side). A raw questionnaire upload is neither peer-shaped
     nor intake-shaped — it needs this one precedence step first.
  3. Map questionnaire columns to CANONICAL field names (not the synthetic
     brief's intake_source names) — regulated_business != fca_regulated,
     they're genuinely different fields (self-declared vs CRM-verified), so
     using intake_source_map() here would silently lose the self-declaration.
  4. Hand off to profile.normalise.normalise_intake(), which explicitly
     supports canonical-keyed payloads per its own docstring.

ALL coercion (bool parsing, number parsing, sentinel handling) happens inside
normalise_intake() — this file does not duplicate that logic.
"""

import sys
from pathlib import Path
import pandas as pd

_SRC = Path(__file__).parent.parent
for _sub in ("", "profile", "benchmark", "engine"):
    _p = str(_SRC / _sub) if _sub else str(_SRC)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profile.normalise import normalise_intake, to_number

COLUMN_ALIASES = {
    "vertical":                      ["vertical", "sector", "capsule_vertical"],
    "employee_count":                ["employee_count", "employees", "headcount", "staff_count"],
    "total_funding_raised_gbp":      ["total_funding_raised_gbp", "funding_raised", "total_raised_gbp"],
    "funding_series":                ["funding_series", "funding_stage", "round"],
    "turnover_last12m_gbp":          ["turnover_last12m_gbp", "turnover_ttm_gbp", "revenue_last12m"],
    "turnover_current_fy_est_gbp":   ["turnover_current_fy_est_gbp", "turnover_current_year", "revenue_current_fy"],
    "turnover_past_fy_gbp":          ["turnover_past_fy_gbp", "turnover_last_fy", "revenue_past_fy"],
    "customer_type":                 ["customer_type", "b2b_b2c", "customer_segment"],
    "regulated_business":            ["regulated_business", "is_regulated", "self_declared_regulated"],
    "software_or_hardware":          ["software_or_hardware", "product_type"],
    "us_revenue_pct":                ["us_revenue_pct", "us_revenue_percent", "pct_us_revenue"],
    "business_activity_description": ["business_activity_description", "activity_description", "description"],
}


def _find_column(df: pd.DataFrame, canonical: str) -> str | None:
    aliases = COLUMN_ALIASES.get(canonical, [canonical])
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in df_cols_lower:
            return df_cols_lower[alias.lower()]
    return None


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Reads a Streamlit UploadedFile (CSV or XLSX) into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, encoding="utf-8-sig")
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {name}. Upload a .csv or .xlsx file.")


def _resolve_turnover(row: pd.Series, df: pd.DataFrame) -> float | None:
    """last12m (most current) > current FY est > past FY."""
    for col_key in ["turnover_last12m_gbp", "turnover_current_fy_est_gbp", "turnover_past_fy_gbp"]:
        col = _find_column(df, col_key)
        if col is None:
            continue
        val = to_number(row.get(col))
        if val is not None and val > 0:
            return val
    return None


def parse_row_to_canonical_payload(row: pd.Series, df: pd.DataFrame) -> dict:
    """
    Maps one questionnaire row to a canonical-keyed payload, ready for
    normalise_intake(). Uses canonical field names as keys throughout so
    normalise_intake's own coercion and sentinel handling apply consistently
    with every other entry point in the pipeline.
    """
    def get(canonical_alias_key, default=None):
        col = _find_column(df, canonical_alias_key)
        return row.get(col, default) if col else default

    payload = {
        "vertical":               get("vertical"),
        "turnover_gbp":           _resolve_turnover(row, df),
        "total_funding_gbp":      get("total_funding_raised_gbp"),
        "employee_count":         get("employee_count"),
        "regulated":              get("regulated_business"),
        "customer_type":          get("customer_type"),
        "software_or_hardware":   get("software_or_hardware"),
        "us_revenue_pct":         get("us_revenue_pct"),
        "business_description":  get("business_activity_description"),
        # funding_stage's canonical name has no peer_source (STUB), but its
        # intake_source IS "funding_series" — normalise_intake looks up this
        # exact raw key, not the canonical "funding_stage" name, for this
        # one field. See schema.BY_NAME["funding_stage"].intake_source.
        "funding_series":         get("funding_series"),
    }
    return {k: v for k, v in payload.items() if v is not None}


def parse_questionnaire_file(uploaded_file) -> dict:
    """
    Full pipeline: read file -> canonical payload per row -> normalise_intake().
    Returns {"raw_df": DataFrame, "profiles": [canonical profile dict, ...]}.
    Caller picks a row if multi-row.
    """
    df = read_uploaded_file(uploaded_file)
    if df.empty:
        raise ValueError("Uploaded file has no rows.")

    profiles = []
    for idx, row in df.iterrows():
        payload = parse_row_to_canonical_payload(row, df)
        profile = normalise_intake(payload)
        profile["_row_index"] = idx
        profile["_client_id"] = row.get("client_id", f"ROW-{idx}")
        profiles.append(profile)

    return {"raw_df": df, "profiles": profiles}