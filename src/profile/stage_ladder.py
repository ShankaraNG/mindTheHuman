"""
Funding-stage reconciliation — the blocker left open at the end of Phase 2.

THE PROBLEM: two sources describe maturity with incompatible vocabularies.

    healthcheck_questionnaire.funding_series   Pre-Seed | Seed | Series A | B | C | D+
    hubspot_company_context.stage_of_evolution Seed | Venture | Growth | Established
                                               | Zombie | Dead

There is no reconciliation field in the dataset, which is why Stage 1 was told to
filter on vertical only. That is not survivable for the demo: vertical-only
matching compares a pre-revenue seed startup to a £40m Series D in the same
cohort, and limit benchmarking is precisely the thing that must not do that.

THE RESOLUTION: map both onto one coarse 4-rung ordinal, and only ever filter
with a +/-1 adjacency window. The window is what makes the mapping defensible —
"Venture" arguably spans Series A to Series B, but a one-rung mapping error is
absorbed by an adjacency window of one rung. We are not claiming the two
taxonomies are equivalent; we are claiming they agree to within the tolerance we
actually use them at. Say exactly that if a judge asks.

ZOMBIE AND DEAD ARE NOT STAGES. They describe viability, not maturity — a zombie
could be anywhere on the ladder. They map to None (unusable for stage filtering),
not to the top rung. Coercing them to a rung would put failing companies in the
peer cohort of healthy ones on the basis of a field that says nothing about size.

PRECEDENCE: funding_series wins over stage_of_evolution when both are present.
It is finer-grained and self-reported by the company at health-check time,
whereas stage_of_evolution is a CRM-assigned label.
"""

from dataclasses import dataclass
from typing import Optional

# The shared ladder. Deliberately coarse — four rungs, not six — because the
# peer pool is 242 clients and a six-rung filter with adjacency produces cohorts
# in the low single digits.
LADDER = {
    0: "Pre-seed / Seed",
    1: "Early venture (~Series A)",
    2: "Growth (~Series B-C)",
    3: "Late / Established (~Series D+)",
}

_FUNDING_SERIES = {
    "pre-seed": 0, "pre seed": 0, "preseed": 0, "angel": 0,
    "seed": 0, "seed+": 0, "seed extension": 0,
    "series a": 1, "a": 1,
    "series b": 2, "b": 2,
    "series c": 2, "c": 2,
    "series d": 3, "series d+": 3, "d": 3, "d+": 3,
    "series e": 3, "series e+": 3, "growth": 2, "late stage": 3,
    "bootstrapped": None,   # says nothing about maturity — see Zombie/Dead note
    "n/a": None, "unknown": None, "none": None, "": None,
}

_STAGE_OF_EVOLUTION = {
    "seed": 0,
    "venture": 1,
    "growth": 2,
    "established": 3,
    "zombie": None,        # viability, not maturity
    "dead": None,
    "unknown": None, "n/a": None, "none": None, "": None,
}


@dataclass(frozen=True)
class StageResolution:
    ordinal: Optional[int]
    label: Optional[str]
    source: Optional[str]      # which field decided it — goes in report provenance
    raw: Optional[str]
    conflict: bool = False     # both sources present and disagreed by >1 rung


def _lookup(value, table: dict) -> Optional[int]:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    if key in table:
        return table[key]
    # tolerate "Series A (extension)", "Seed round", etc.
    for known, rung in table.items():
        if known and key.startswith(known):
            return rung
    return None


def resolve_stage(funding_series=None, stage_of_evolution=None) -> StageResolution:
    """Both fields in, one ordinal out, with the deciding source recorded.

    Flags a conflict when the two sources disagree by more than one rung. Those
    rows are still usable (funding_series wins) but the count of them is a number
    worth knowing before anyone claims the mapping is sound — if it's large, the
    mapping above is wrong, not the data.
    """
    a = _lookup(funding_series, _FUNDING_SERIES)
    b = _lookup(stage_of_evolution, _STAGE_OF_EVOLUTION)

    conflict = a is not None and b is not None and abs(a - b) > 1

    if a is not None:
        return StageResolution(a, LADDER[a], "funding_series",
                               str(funding_series), conflict)
    if b is not None:
        return StageResolution(b, LADDER[b], "stage_of_evolution",
                               str(stage_of_evolution), conflict)
    raw = funding_series if funding_series not in (None, "") else stage_of_evolution
    return StageResolution(None, None, None,
                           None if raw is None else str(raw), False)


def adjacent(ordinal: Optional[int], window: int = 1) -> Optional[set]:
    """The rungs a prospect at `ordinal` may be compared against.

    Returns None when the prospect's own stage is unknown, which the caller must
    read as "do not apply a stage filter at all" — NOT as "match nothing" and not
    as "assume rung 0".
    """
    if ordinal is None:
        return None
    return {r for r in LADDER if abs(r - ordinal) <= window}


def audit(df, funding_col="funding_series", stage_col="stage_of_evolution"):
    """Run against client_level.csv before trusting the filter. Prints coverage
    and conflict counts — the two numbers that decide whether stage filtering is
    safe to switch on for the demo."""
    rows = [resolve_stage(r.get(funding_col), r.get(stage_col))
            for r in df.to_dict("records")]
    n = len(rows)
    resolved = sum(1 for r in rows if r.ordinal is not None)
    conflicts = sum(1 for r in rows if r.conflict)
    print(f"stage resolved: {resolved}/{n} ({resolved / n:.1%})")
    print(f"conflicts (>1 rung apart): {conflicts}")
    from collections import Counter
    print("by rung:", Counter(r.label for r in rows).most_common())
    print("by deciding source:", Counter(r.source for r in rows).most_common())
    return rows