"""
Canonical company profile schema — the contract every stage codes against.

This module does NOT transform data. It declares:
  - which fields exist in the canonical profile
  - where each comes from, on BOTH sides (peer pool + prospect intake)
  - what ROLE each plays in the engine (filter / weight / trigger)
  - how missing values must be handled
  - the diagnostic evidence behind each decision

`normalise.py` reads this to build canonical profiles.
`engine/stage*.py` reads this to know how to use them.
Nothing should hardcode a field name that isn't declared here.

All completeness figures are measured against data/client_level.csv (n=242),
the outer merge of healthcheck_questionnaire.csv (221) + hubspot_company_context.csv (197).
See notebooks/01_feature_scoping.ipynb for the diagnostic run.
"""

from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Optional

import pandas as pd


# --------------------------------------------------------------------------
# Roles — how a field participates in the engine
# --------------------------------------------------------------------------

class Role(str, Enum):
    FILTER = "filter"      # hard rule: peers MUST match. Non-matching peers excluded.
    WEIGHT = "weight"      # similarity dimension: contributes to peer distance ranking.
    TRIGGER = "trigger"    # fires a cover-check rule. Does NOT affect peer selection.
    METADATA = "metadata"  # carried through for reporting/provenance, not used in logic.
    EXCLUDED = "excluded"  # deliberately not used. Reason recorded in `evidence`.


class Status(str, Enum):
    LOCKED = "locked"            # decided, evidenced, safe to build against.
    PROVISIONAL = "provisional"  # role not yet settled — treat as optional, not load-bearing.
    STUB = "stub"                # blocked on upstream work. Must not be relied on yet.
    EXCLUDED = "excluded"        # confirmed unusable.


class NAPolicy(str, Enum):
    EXCLUDE_PEER = "exclude_peer"        # peer unusable without this. Drop from pool.
    SKIP_DIMENSION = "skip_dimension"    # omit from distance calc; count dims actually used.
    UNKNOWN_NOT_FALSE = "unknown_not_false"  # None != False. Never coerce missing to negative.
    PASS_THROUGH = "pass_through"        # missing is fine, carry as None.


# --------------------------------------------------------------------------
# Field specification
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    name: str                        # canonical field name
    role: Role
    status: Status
    dtype: str
    na_policy: NAPolicy
    peer_source: Optional[str]       # column in client_level.csv
    intake_source: Optional[str]     # column in synthetic_prospect_briefs.csv / web form
    completeness: Optional[float]    # % usable in peer pool (sentinel-aware)
    sentinels: tuple = ()            # non-null values that mean "missing"
    transform: Optional[str] = None  # required preprocessing before use
    evidence: str = ""               # why this role/decision — for the schema doc + judges


# --------------------------------------------------------------------------
# THE SCHEMA
# --------------------------------------------------------------------------

SCHEMA: list[FieldSpec] = [

    # ---- identity -------------------------------------------------------
    FieldSpec(
        name="client_id",
        role=Role.METADATA,
        status=Status.LOCKED,
        dtype="str",
        na_policy=NAPolicy.EXCLUDE_PEER,
        peer_source="client_id",
        intake_source="brief_id",
        completeness=100.0,
        evidence="Universal join key. UNK- prefix denotes dataset-designated unclassified clients; "
                 "SYN- prefix is synthetic and never joins to real data.",
    ),

    # ---- FILTERS --------------------------------------------------------
    FieldSpec(
        name="vertical",
        role=Role.FILTER,
        status=Status.LOCKED,
        dtype="category",
        na_policy=NAPolicy.EXCLUDE_PEER,
        peer_source="vertical",
        intake_source="vertical",
        completeness=79.8,
        sentinels=("Unknown",),
        evidence="Hard filter — a Tech company is never benchmarked against Consumer. "
                 "Naive .notna() reports 100% because 'Unknown' is a real string, not null. "
                 "Sentinel-aware completeness is 79.8% (193/242): all 49 'Unknown' rows are "
                 "UNK-prefixed clients, an exact set match, i.e. the dataset's own unclassified "
                 "bucket (DATA_DICTIONARY.md convention) — not a merge defect. Excluded from the "
                 "peer pool: a hard filter has no valid fallback.",
    ),
    FieldSpec(
        name="funding_stage",
        role=Role.FILTER,
        status=Status.STUB,          # <-- BLOCKED. Do not build against this yet.
        dtype="category",
        na_policy=NAPolicy.SKIP_DIMENSION,
        peer_source=None,            # no resolved column exists in client_level.csv
        intake_source="funding_series",
        completeness=None,
        evidence="BLOCKED: Phase 1 never coalesced this. Two incompatible taxonomies remain — "
                 "questionnaire funding_series (Pre-Seed/Seed/Series A-D+, 127/221) vs HubSpot "
                 "stage_of_evolution (Seed/Venture/Growth/Established/Zombie/Dead, 133/197). "
                 "'Venture'/'Growth' are not funding rounds, so there is no clean mapping yet. "
                 "Intended role is an adjacent-stage filter (Series A also pulls Seed + Series B) "
                 "to avoid tiny cohorts. Until a reconciliation table exists, Stage 1 must filter "
                 "on vertical only and treat this as a no-op.",
    ),
    FieldSpec(
        name="regulated",
        role=Role.FILTER,
        status=Status.PROVISIONAL,   # filter vs weight not yet settled
        dtype="boolean",
        na_policy=NAPolicy.UNKNOWN_NOT_FALSE,
        peer_source="regulated_final",
        intake_source="fca_regulated",
        completeness=69.8,
        evidence="PROVISIONAL — filter vs weight undecided; needs its own cross-tab before locking. "
                 "Resolved in Phase 1 from fca_regulated (67/197) with fallback to self-declared "
                 "regulated_business (139/221); because the CRM field is the sparser of the two, "
                 "most final values originate from self-declaration. Missing must stay None: an "
                 "unanswered form is not a declaration of 'not regulated'.",
    ),

    # ---- WEIGHTS (peer similarity dimensions) ---------------------------
    # Spearman on the three below: 0.23-0.63. Weak-to-moderate, so all three are
    # retained independently — they are not three copies of one "size" signal.
    FieldSpec(
        name="turnover_gbp",
        role=Role.WEIGHT,
        status=Status.LOCKED,
        dtype="float",
        na_policy=NAPolicy.SKIP_DIMENSION,
        peer_source="turnover_final_gbp",
        intake_source="turnover_gbp",
        completeness=89.3,
        transform="log1p_or_percentile_rank",
        evidence="Resolved in Phase 1 by documented source precedence (last12m > current_fy_est > "
                 "past_fy > CRM turnover > CRM band); merge lifts completeness ~33pp over the best "
                 "single source. MUST be log-transformed or percentile-ranked before any distance "
                 "calculation — raw values span ~£22k to ~£200m and would otherwise dominate every "
                 "other dimension. Rank-based treatment is also consistent with the Spearman "
                 "redundancy check. Note: CRM band fallback carries known bad values (£10bn, £83k) "
                 "and is flagged via turnover_is_band.",
    ),
    FieldSpec(
        name="employee_count",
        role=Role.WEIGHT,
        status=Status.LOCKED,
        dtype="float",
        na_policy=NAPolicy.SKIP_DIMENSION,
        peer_source="employee_count_final",
        intake_source="employee_count",
        completeness=90.1,
        transform="log1p_or_percentile_rank",
        evidence="Correlates only 0.63 with turnover (~40% shared variance), so it carries genuine "
                 "independent signal — a capital-light SaaS firm and a labour-intensive one can "
                 "share revenue but not headcount. Overall 90.1% looks marginally below HubSpot's "
                 "own 91.4%, but that is a denominator effect (242 vs 197 rows), not a coalesce "
                 "fault: within the 197 CRM-matched clients, completeness rises to 95.9%.",
    ),
    FieldSpec(
        name="total_funding_gbp",
        role=Role.WEIGHT,
        status=Status.LOCKED,
        dtype="float",
        na_policy=NAPolicy.SKIP_DIMENSION,
        peer_source="total_funding_final_gbp",
        intake_source="total_funding_raised_gbp",
        completeness=88.4,
        transform="log1p_or_percentile_rank",
        evidence="Weakest correlations of the three (0.23 with turnover, 0.38 with employees) — "
                 "funding reflects capital raised, not revenue earned, so a well-funded pre-revenue "
                 "startup and a bootstrapped profitable one diverge sharply. Most independent of "
                 "the size proxies and therefore the strongest argument against collapsing them.",
    ),
    FieldSpec(
        name="us_revenue_pct",
        role=Role.WEIGHT,
        status=Status.PROVISIONAL,   # weight vs trigger not yet settled
        dtype="float",
        na_policy=NAPolicy.SKIP_DIMENSION,
        peer_source="us_revenue_pct",
        intake_source="us_revenue_pct",
        completeness=18.1,
        evidence="PROVISIONAL — weight vs trigger undecided. Only 18.1% complete in the peer pool "
                 "(40/221, later form variant) but 100% in the synthetic briefs, so intake is "
                 "reliable while the comparator side is not. Too sparse to weight peer distance "
                 "credibly; more defensible as a trigger (US exposure raises PI/D&O relevance). "
                 "Decide before locking, or defer and document.",
    ),

    # ---- TRIGGERS (fire cover-check rules; never affect peer selection) --
    FieldSpec(
        name="business_activity",
        role=Role.METADATA,
        status=Status.LOCKED,
        dtype="str",
        na_policy=NAPolicy.PASS_THROUGH,
        peer_source="business_activity",
        intake_source=None,   # synthetic briefs carry no structured activity label
        completeness=62.9,
        evidence="Structured form-taxonomy label. Carried for reporting and rationale text. Not a "
                 "filter: free-text taxonomy with a long tail (35 'Computer Software' vs many "
                 "singletons) would fragment cohorts. No equivalent in the synthetic briefs, so it "
                 "cannot be required at intake.",
    ),
    FieldSpec(
        name="business_description",
        role=Role.METADATA,
        status=Status.LOCKED,
        dtype="str",
        na_policy=NAPolicy.PASS_THROUGH,
        peer_source="business_activity_description",
        intake_source="business_description",
        completeness=54.1,
        evidence="Raw input to extract_text.py — the ONLY source for the four trigger flags below. "
                 "131/242 populated on the peer side (form drop-off: ~37% of respondents answered "
                 "only 8 of ~29 questions, so firmographic and free-text blanks cluster together "
                 "and are a collection artifact, not random noise). 24/24 on the synthetic side, "
                 "but a live web form will not match that rate.",
    ),

    # The four flags below are NOT columns anywhere in the dataset — not in the
    # questionnaire, not in the CRM, and not in the synthetic briefs. They are
    # derived from free text. Because the peer pool has no equivalent value to
    # measure distance against, TRIGGER is the only role structurally available
    # to them: they cannot filter or weight peer selection.
    FieldSpec(
        name="personal_data",
        role=Role.TRIGGER,
        status=Status.LOCKED,
        dtype="boolean",
        na_policy=NAPolicy.UNKNOWN_NOT_FALSE,
        peer_source=None,
        intake_source=None,
        completeness=None,
        transform="extract_text.extract_flags",
        evidence="Validated: clients flagged True hold cyber cover at 30% (12/40) vs 7.7% (7/91) "
                 "when False — odds ratio ~5.1 on n=131. Real association against Capsule's own "
                 "broker decisions, so the flag tracks something genuine rather than noise. "
                 "First keyword attempt using compliance vocabulary ('PII', 'GDPR', 'personal "
                 "data') scored 0/242: founders describe what their business does, not its data "
                 "obligations, so the working word list was rebuilt from the actual descriptions "
                 "of the 25 cyber-covered clients. Imperfect by design — 28 flagged clients hold "
                 "no cyber cover, which is often the gap the tool exists to surface.",
    ),
    FieldSpec(
        name="medical_data",
        role=Role.TRIGGER,
        status=Status.LOCKED,
        dtype="boolean",
        na_policy=NAPolicy.UNKNOWN_NOT_FALSE,
        peer_source=None,
        intake_source=None,
        completeness=None,
        transform="extract_text.extract_flags",
        evidence="No structured column in any source. Derived from free text. Not separately "
                 "cross-tab validated — MedTech cohort (25 clients) is too small for a meaningful "
                 "2x2. Vocabulary evidenced from real descriptions ('clinician', 'virtual care', "
                 "'prescribing', 'treatment of Depression and Anxiety').",
    ),
    FieldSpec(
        name="financial_data",
        role=Role.TRIGGER,
        status=Status.LOCKED,
        dtype="boolean",
        na_policy=NAPolicy.UNKNOWN_NOT_FALSE,
        peer_source=None,
        intake_source=None,
        completeness=None,
        transform="extract_text.extract_flags",
        evidence="No structured column in any source. Derived from free text. Risk is usually "
                 "implicit rather than stated: 'Point of Sale app to allow SMEs to take physical "
                 "and e-commerce payments' never says 'financial data' but plainly handles it.",
    ),
    FieldSpec(
        name="stock_logistics",
        role=Role.TRIGGER,
        status=Status.LOCKED,
        dtype="boolean",
        na_policy=NAPolicy.UNKNOWN_NOT_FALSE,
        peer_source=None,
        intake_source=None,
        completeness=None,
        transform="extract_text.extract_flags",
        evidence="No structured column in any source. Derived from free text. "
                 "software_or_hardware ('Hardware/Physical goods', 29/221) is a partial "
                 "corroborating signal but is not the same construct and is not substituted for it.",
    ),

    # ---- EXCLUDED -------------------------------------------------------
    FieldSpec(
        name="volume",
        role=Role.EXCLUDED,
        status=Status.EXCLUDED,
        dtype="float",
        na_policy=NAPolicy.PASS_THROUGH,
        peer_source=None,
        intake_source=None,
        completeness=0.0,
        evidence="Named as a risk driver in the brief but absent from every source, including the "
                 "synthetic evaluation briefs. Nearest columns (policies_needed, quotes, "
                 "bound_quotes) describe insurance-shopping behaviour, not business or transaction "
                 "throughput. Excluded and documented rather than force-fit onto an unrelated "
                 "field — declaring the gap is more defensible than proxying it badly.",
    ),
]


# --------------------------------------------------------------------------
# Accessors — engine stages should use these, not re-filter SCHEMA by hand
# --------------------------------------------------------------------------

BY_NAME: dict[str, FieldSpec] = {f.name: f for f in SCHEMA}


def fields_by_role(role: Role, usable_only: bool = True) -> list[FieldSpec]:
    """Fields for a given role. usable_only drops STUB/EXCLUDED specs."""
    out = [f for f in SCHEMA if f.role is role]
    if usable_only:
        out = [f for f in out if f.status not in (Status.STUB, Status.EXCLUDED)]
    return out


def filter_fields(usable_only: bool = True) -> list[FieldSpec]:
    return fields_by_role(Role.FILTER, usable_only)


def weight_fields(usable_only: bool = True) -> list[FieldSpec]:
    return fields_by_role(Role.WEIGHT, usable_only)


def trigger_fields(usable_only: bool = True) -> list[FieldSpec]:
    return fields_by_role(Role.TRIGGER, usable_only)


def peer_source_map() -> dict[str, str]:
    """canonical name -> client_level.csv column, for fields that have one."""
    return {f.name: f.peer_source for f in SCHEMA if f.peer_source}


def intake_source_map() -> dict[str, str]:
    """canonical name -> synthetic brief / web form field, for fields that have one."""
    return {f.name: f.intake_source for f in SCHEMA if f.intake_source}


def is_missing(value: Any, spec: FieldSpec) -> bool:
    """Sentinel-aware missingness. Plain pd.isna() is NOT sufficient — e.g.
    vertical='Unknown' is a non-null string that means 'no value'."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return True
    return value in spec.sentinels


def validate_profile(profile: dict) -> dict:
    """Check a canonical profile against the schema before it enters the engine.

    Returns a report rather than raising: an incomplete profile is a normal
    condition here, and downstream stages need to know *what* is missing so they
    can report match confidence honestly.
    """
    unknown_keys = sorted(set(profile) - set(BY_NAME))
    missing_filters, missing_soft_filters = [], []
    missing_weights, unknown_triggers = [], []

    for spec in filter_fields():
        if is_missing(profile.get(spec.name), spec):
            # Only a LOCKED filter is fatal. `regulated` is PROVISIONAL and just
            # 69.8% complete — treating it as hard would silently drop ~30% of
            # the peer pool on a role we have not even settled yet.
            if spec.status is Status.LOCKED:
                missing_filters.append(spec.name)
            else:
                missing_soft_filters.append(spec.name)
    for spec in weight_fields():
        if is_missing(profile.get(spec.name), spec):
            missing_weights.append(spec.name)
    for spec in trigger_fields():
        if is_missing(profile.get(spec.name), spec):
            unknown_triggers.append(spec.name)

    usable_weights = len(weight_fields()) - len(missing_weights)
    return {
        # A missing hard filter is fatal: there is no valid fallback.
        "usable": len(missing_filters) == 0 and usable_weights > 0,
        "missing_filters": missing_filters,             # fatal — locked filters
        "missing_soft_filters": missing_soft_filters,   # tolerated — provisional filters
        "missing_weights": missing_weights,
        "usable_weight_dims": usable_weights,
        "total_weight_dims": len(weight_fields()),
        "unknown_triggers": unknown_triggers,   # unknown != False
        "unknown_keys": unknown_keys,
        "stubbed_fields": [f.name for f in SCHEMA if f.status is Status.STUB],
        "provisional_fields": [f.name for f in SCHEMA if f.status is Status.PROVISIONAL],
    }


def to_dataframe() -> "pd.DataFrame":
    """The schema as a table — for the schema doc and the pitch deck."""
    return pd.DataFrame([
        {
            "field": f.name,
            "role": f.role.value,
            "status": f.status.value,
            "peer_source": f.peer_source or "—",
            "intake_source": f.intake_source or "—",
            "completeness_%": f.completeness,
            "na_policy": f.na_policy.value,
            "transform": f.transform or "—",
        }
        for f in SCHEMA
    ])


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)
    print(to_dataframe().to_string(index=False))
    print()
    print("filters :", [f.name for f in filter_fields()])
    print("weights :", [f.name for f in weight_fields()])
    print("triggers:", [f.name for f in trigger_fields()])
    print("stubbed :", [f.name for f in SCHEMA if f.status is Status.STUB])