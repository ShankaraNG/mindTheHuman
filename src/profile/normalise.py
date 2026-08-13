"""
Raw data -> canonical company profile.

This is the only module that BUILDS profiles. schema.py declares what a profile
must look like; this module produces them. Two entry points, one output shape:

    normalise_peer(row)     client_level.csv row  -> canonical profile
    normalise_intake(data)  synthetic brief / web form -> canonical profile

Both go through the same coercion and the same extract_text call, because a
trigger flag derived one way on the peer side and another way at intake would
make the two incomparable.

WHAT THIS MODULE DOES NOT DO — deliberately:
  - It does not log-transform or percentile-rank the weight fields. schema.py
    marks those transforms as required, but percentile rank is a property of the
    whole pool, not of one row, so it cannot be computed during per-row
    normalisation. Raw values are preserved here and `add_rank_columns()` is
    provided for the pool-level pass. Stage 1 must use ranked/logged values for
    distance, never the raw columns.
  - It does not drop unusable rows. It records usability via
    schema.validate_profile and leaves the filtering decision to Stage 1, so the
    peer pool keeps an auditable record of what was excluded and why.

CHANGE FROM THE FIRST VERSION: funding_stage is no longer a permanent stub.
It was previously left as None for every row because no single source column
maps to it — that is why client_level_normalised.csv came back 242/242 null and
the engine's peer filter was silently running vertical-only. funding_stage is
now resolved here, once, from the two raw source columns that between them
cover it (questionnaire's funding_series, CRM's stage_of_evolution), via
stage_ladder.resolve_stage(). This is also why the stage resolution that used
to live inside peer_match.py at query time has been removed from there — one
place computes it, everywhere else just reads the field.
"""

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

import extract_text
import schema
import stage_ladder
from schema import Role, Status


# --------------------------------------------------------------------------
# Coercion — source data is messy in known, documented ways
# --------------------------------------------------------------------------

_TRUE = {"true", "t", "yes", "y", "1"}
_FALSE = {"false", "f", "no", "n", "0"}


def to_number(value: Any) -> Optional[float]:
    """Money/count -> float. Handles '£1,250,000', '1250000', 1250000.0, blanks.

    Applied to EVERY numeric field, not just the two the Phase 1 cleaner parsed.
    Some money columns arrive already numeric and some do not; assuming which is
    which is how a string ends up in a distance calculation.
    """
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return None if pd.isna(value) else float(value)
    s = str(value).strip()
    if not s or s.startswith("{"):
        return None
    s = s.replace("£", "").replace("$", "").replace(",", "").replace("%", "")
    if s.upper().startswith("GBP"):
        s = s[3:]
    try:
        return float(s)
    except ValueError:
        return None


def to_boolean(value: Any) -> Optional[bool]:
    """Tri-state boolean. A blank stays None — never False.

    An unanswered question is not a declaration of 'no'. Coercing missing to
    False here would silently turn 73 unknown regulated-status clients into 73
    confirmed-unregulated ones.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        if pd.isna(value):
            return None
        return bool(value)
    s = str(value).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def to_text(value: Any, sentinels: tuple = ()) -> Optional[str]:
    """Clean string, with schema-declared sentinels mapped to None.

    This is where vertical='Unknown' becomes None. Skipping it is the bug that
    made vertical look 100% complete when it is 79.8%.
    """
    if value is None:
        return None
    if not isinstance(value, str) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s in sentinels:
        return None
    return s


_COERCE = {
    "float": to_number,
    "boolean": to_boolean,
    "str": to_text,
    "category": to_text,
}


# --------------------------------------------------------------------------
# funding_stage — resolved, not mapped
# --------------------------------------------------------------------------

# Raw column names on each side. questionnaire/hubspot use these directly;
# synthetic briefs and live intake may use either of these two spellings too.
# If schema.py already declares peer_source/intake_source for a field named
# "funding_series" or "stage_of_evolution", prefer those over this hardcoded
# list — this is a fallback so the resolution doesn't silently no-op if the
# schema hasn't been updated to declare them.
_FUNDING_SERIES_KEYS = ("funding_series", "funding_round")
_STAGE_OF_EVOLUTION_KEYS = ("stage_of_evolution", "stage")


def _first_present(raw: Mapping[str, Any], keys: tuple) -> Optional[Any]:
    for k in keys:
        if k in raw and raw[k] is not None and not (
            isinstance(raw[k], float) and pd.isna(raw[k])
        ):
            return raw[k]
    return None


def _resolve_funding_stage(raw: Mapping[str, Any]) -> tuple:
    """Returns (label_for_profile, resolution) so the caller can both store the
    canonical label and, if useful, inspect .ordinal / .conflict for an audit."""
    funding_series = _first_present(raw, _FUNDING_SERIES_KEYS)
    stage_of_evolution = _first_present(raw, _STAGE_OF_EVOLUTION_KEYS)
    resolution = stage_ladder.resolve_stage(funding_series, stage_of_evolution)
    return resolution.label, resolution


# --------------------------------------------------------------------------
# Core: build one profile
# --------------------------------------------------------------------------

def _build(
    raw: Mapping[str, Any],
    source_map: dict[str, str],
    description_key: Optional[str],
) -> dict[str, Any]:
    """Shared normalisation path for both peer and intake."""
    profile: dict[str, Any] = {}

    for spec in schema.SCHEMA:
        if spec.status is Status.EXCLUDED:
            continue                      # volume: absent from every source
        if spec.role is Role.TRIGGER:
            continue                      # derived below, not mapped from a column
        if spec.name == "funding_stage":
            continue                      # resolved below, not column-mapped

        src = source_map.get(spec.name)
        if src is None or src not in raw:
            profile[spec.name] = None
            continue

        coerce = _COERCE.get(spec.dtype, to_text)
        profile[spec.name] = (
            coerce(raw[src], spec.sentinels) if coerce is to_text else coerce(raw[src])
        )

    # funding_stage: resolved from two incompatible source taxonomies onto one
    # ordinal ladder, not mapped from a single column. See stage_ladder.py for
    # why (Zombie/Dead -> None, funding_series takes precedence, etc).
    label, resolution = _resolve_funding_stage(raw)
    profile["funding_stage"] = label
    profile["_funding_stage_ordinal"] = resolution.ordinal
    profile["_funding_stage_source"] = resolution.source
    profile["_funding_stage_conflict"] = resolution.conflict

    # The four trigger flags: derived from free text, identically on both sides.
    description = raw.get(description_key) if description_key else None
    profile.update(extract_text.extract_flags(description))

    return profile


def normalise_peer(row: Mapping[str, Any]) -> dict[str, Any]:
    """One client_level.csv row -> canonical profile."""
    return _build(
        raw=row,
        source_map=schema.peer_source_map(),
        description_key=schema.BY_NAME["business_description"].peer_source,
    )


def normalise_intake(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One synthetic brief / web-form submission -> canonical profile.

    Accepts canonical field names as well as raw intake names, so an API caller
    can post either shape without knowing the source column conventions.
    """
    source_map = schema.intake_source_map()
    merged = dict(payload)
    for canonical, raw_name in source_map.items():
        if canonical in payload and raw_name not in merged:
            merged[raw_name] = payload[canonical]

    # funding_stage isn't in source_map (resolved, not mapped — see _build), so
    # its canonical-name alias has to be handled explicitly here, the same way
    # business_description is below, or a caller posting {"funding_stage": "..."}
    # or {"funding_series": "..."} would have it silently ignored.
    for canonical_alias in ("funding_series", "stage_of_evolution"):
        if canonical_alias in payload and canonical_alias not in merged:
            merged[canonical_alias] = payload[canonical_alias]

    desc_key = schema.BY_NAME["business_description"].intake_source
    if desc_key not in merged and "business_description" in payload:
        merged[desc_key] = payload["business_description"]

    return _build(raw=merged, source_map=source_map, description_key=desc_key)


# --------------------------------------------------------------------------
# Pool-level: build the peer pool
# --------------------------------------------------------------------------

def normalise_peer_pool(df: pd.DataFrame, add_ranks: bool = True) -> pd.DataFrame:
    """client_level.csv -> canonical peer pool, one row per client.

    Adds provenance columns rather than dropping anything:
      profile_usable        - passes schema validation (locked filters present)
      usable_weight_dims    - how many of the weight dimensions are populated
      missing_filters       - which locked filters are absent
      has_description       - whether trigger flags could be derived at all
      funding_stage_conflict- funding_series and stage_of_evolution disagreed by
                               more than one rung (see stage_ladder.py). Worth a
                               groupby before trusting the stage filter: if this
                               is common, the ladder mapping needs revisiting,
                               not the data.
    """
    profiles = [normalise_peer(row) for row in df.to_dict("records")]
    out = pd.DataFrame(profiles)

    reports = [schema.validate_profile(p) for p in profiles]
    out["profile_usable"] = [r["usable"] for r in reports]
    out["usable_weight_dims"] = [r["usable_weight_dims"] for r in reports]
    out["missing_filters"] = [";".join(r["missing_filters"]) for r in reports]
    out["has_description"] = out["business_description"].notna()
    out["funding_stage_conflict"] = out["_funding_stage_conflict"]

    if add_ranks:
        out = add_rank_columns(out)
    return out


def add_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log and percentile-rank columns for the weight fields.

    Raw turnover spans ~£22k to ~£200m; unscaled Euclidean distance over these
    would be dominated entirely by turnover and would effectively ignore
    headcount and funding. Percentile rank is also consistent with the Spearman
    (rank-based) redundancy check that justified keeping all three dimensions.

    Ranks are computed within the pool, so this is a pool-level operation. A
    single incoming prospect must be ranked AGAINST this pool, not on its own —
    see rank_against_pool().
    """
    out = df.copy()
    for spec in schema.weight_fields():
        col = spec.name
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_log"] = np.log1p(values.clip(lower=0))
        out[f"{col}_pct"] = values.rank(pct=True)     # NaN stays NaN, not 0
    return out


def rank_against_pool(profile: Mapping[str, Any], pool: pd.DataFrame) -> dict[str, Optional[float]]:
    """Where does one prospect sit within the pool, per weight dimension?

    Returns percentile in [0, 1], or None where either side lacks the value.
    This is what lets a prospect be compared to peers on the same scale.
    """
    out: dict[str, Optional[float]] = {}
    for spec in schema.weight_fields():
        col = spec.name
        value = to_number(profile.get(col))
        if value is None or col not in pool.columns:
            out[f"{col}_pct"] = None
            continue
        reference = pd.to_numeric(pool[col], errors="coerce").dropna()
        out[f"{col}_pct"] = (
            float((reference <= value).mean()) if len(reference) else None
        )
    return out


if __name__ == "__main__":
    # Fabricated rows shaped like client_level.csv — real data is gitignored.
    fake_peers = pd.DataFrame([
        {"client_id": "TEC-001", "vertical": "Tech", "regulated_final": True,
         "turnover_final_gbp": "£2,000,000", "employee_count_final": 30,
         "total_funding_final_gbp": 5_000_000, "us_revenue_pct": 10,
         "business_activity": "Computer Software",
         "business_activity_description": "SaaS platform for enterprise customers.",
         "funding_series": "Series A", "stage_of_evolution": "Venture"},
        {"client_id": "UNK-004", "vertical": "Unknown", "regulated_final": None,
         "turnover_final_gbp": None, "employee_count_final": None,
         "total_funding_final_gbp": None, "us_revenue_pct": None,
         "business_activity": None, "business_activity_description": None,
         "funding_series": None, "stage_of_evolution": "Zombie"},
        {"client_id": "MED-002", "vertical": "MedTech", "regulated_final": False,
         "turnover_final_gbp": 800_000, "employee_count_final": 12,
         "total_funding_final_gbp": 3_000_000, "us_revenue_pct": None,
         "business_activity": "Scientific Research",
         "business_activity_description": "Virtual care platform for patients, built with clinicians.",
         "funding_series": "Seed", "stage_of_evolution": "Established"},  # conflict, on purpose
    ])

    pool = normalise_peer_pool(fake_peers)
    cols = ["client_id", "vertical", "turnover_gbp", "employee_count",
            "funding_stage", "funding_stage_conflict",
            "personal_data", "medical_data", "profile_usable",
            "usable_weight_dims", "missing_filters"]
    print(pool[cols].to_string(index=False))

    print("\n-- intake (synthetic brief shape) --")
    brief = {
        "brief_id": "SYN-001", "vertical": "Consumer", "employee_count": 45,
        "turnover_gbp": 12_000_000, "total_funding_raised_gbp": 8_000_000,
        "funding_series": "Series A", "us_revenue_pct": 5, "fca_regulated": "false",
        "business_description": "Direct-to-consumer premium pet food, sold via subscription "
                                "with e-commerce checkout and warehouse fulfilment.",
    }
    prospect = normalise_intake(brief)
    for k, v in prospect.items():
        print(f"   {k:<28} {v!r}")

    print("\n-- validation --")
    print("  ", schema.validate_profile(prospect))
    print("\n-- prospect vs pool percentiles --")
    print("Length of pool", len(pool))
    print("  ", rank_against_pool(prospect, pool))