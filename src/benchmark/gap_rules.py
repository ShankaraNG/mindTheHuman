"""
Task 4 — Gap rules. Compares a BenchmarkResult (Task 3.3) against what the
prospect actually holds, producing a direction and priority per cover.

Neither resolve.py nor band_lookup.py does this comparison — they produce the
BENCHMARK (what peers/bands say a cover should look like), not the GAP (how
the prospect's own declared cover compares to it). This is that missing step.

THE CENTRAL RULE, from resolve.py's own evidence-tier design: a benchmarking
tool that cannot say "we don't know" will confidently upsell. So:

  - TIER_INSUFFICIENT + not held -> we have no evidence either way. NOT flagged
    as a gap — flagged as NEEDS_REVIEW, "insufficient data, broker judgement
    required". Claiming a gap here would be inventing confidence the evidence
    doesn't support.
  - Any other tier + not held (and evidence says peers/bands expect it) -> GAP.
  - Held below reference_limit (< INCREASE_THRESHOLD of it) -> INCREASE.
  - Held at/above reference_limit -> OK.
  - STATUTORY covers (coverage_names.STATUTORY — currently just Employers
    Liability) missing -> ALWAYS Immediate action, regardless of evidence tier.
    This one is not a peer-prevalence judgement, it is UK law.
"""

from dataclasses import dataclass, field
from typing import Optional

try:
    from . import coverage_names
    from .resolve import (BenchmarkResult, TIER_INSUFFICIENT, TIER_PEER_AND_BAND,
                          TIER_PEER_ONLY, TIER_BAND_ONLY, TIER_PREVALENCE_ONLY)
except ImportError:
    import coverage_names
    from resolve import (BenchmarkResult, TIER_INSUFFICIENT, TIER_PEER_AND_BAND,
                         TIER_PEER_ONLY, TIER_BAND_ONLY, TIER_PREVALENCE_ONLY)

GAP = "GAP"
INCREASE = "INCREASE"
OK = "OK"
NEEDS_REVIEW = "NEEDS_REVIEW"          # insufficient evidence — not a claimed gap

IMMEDIATE = "Immediate action"
CONSIDER = "For consideration"
RENEWAL = "Review at renewal"

# Maps each extract_text.py trigger flag to the covers it makes more urgent
# when present. These do NOT invent evidence or change direction — resolve.py's
# evidence tier already decided GAP/INCREASE/NEEDS_REVIEW/OK. This only
# escalates PRIORITY by one step on findings that already exist, on the
# reasoning a real broker would apply: a company handling this kind of data
# faces higher consequence if the gap goes unaddressed, so it deserves more
# urgency, not more certainty it doesn't have. Never touches OK findings —
# nothing to escalate there. Never fires on None (unknown) — only on a
# confirmed True; absence of evidence is not evidence of presence either.
TRIGGER_RELEVANT_COVERS = {
    "personal_data":   {"Cyber", "Professional Indemnity", "Employment Practices Liability"},
    "medical_data":    {"Medical Malpractice", "Clinical Trials", "Cyber", "Professional Indemnity"},
    "financial_data":  {"Cyber", "Crime", "Professional Indemnity"},
    "stock_logistics": {"Product Recall", "Goods in Transit", "Stock", "Business Interruption"},
}

_ESCALATE_ONE_STEP = {RENEWAL: CONSIDER, CONSIDER: IMMEDIATE, IMMEDIATE: IMMEDIATE}

INCREASE_THRESHOLD = 0.75   # held below this fraction of reference_limit -> INCREASE


@dataclass
class GapFinding:
    cover: str
    direction: str
    priority: str
    evidence_tier: str
    current_limit: Optional[float] = None
    reference_limit: Optional[float] = None
    reference_source: Optional[str] = None
    prevalence: Optional[float] = None
    n_cohort: int = 0
    n_holding: int = 0
    note: Optional[str] = None
    is_statutory: bool = False
    triggers_applied: list = field(default_factory=list)


def _priority(direction: str, evidence_tier: str, is_statutory: bool) -> str:
    if is_statutory and direction in (GAP, NEEDS_REVIEW):
        # UK law, not a peer judgement — always urgent regardless of evidence.
        return IMMEDIATE
    if direction == NEEDS_REVIEW:
        return CONSIDER
    if direction == GAP:
        return IMMEDIATE if evidence_tier in (TIER_PEER_AND_BAND, TIER_PEER_ONLY, TIER_BAND_ONLY) else CONSIDER
    if direction == INCREASE:
        return CONSIDER if evidence_tier in (TIER_PEER_AND_BAND, TIER_PEER_ONLY) else RENEWAL
    return RENEWAL  # OK


def _apply_trigger_escalation(finding: GapFinding, profile: dict) -> GapFinding:
    """
    Escalates priority by exactly one step if a confirmed (True, not None)
    trigger flag names this cover as relevant, and the finding isn't already
    OK. Records which flag(s) fired in triggers_applied and appends a plain
    note — this must be visible in the report, not a silent priority bump.
    """
    if finding.direction == OK or not profile:
        return finding

    fired = [flag for flag, covers in TRIGGER_RELEVANT_COVERS.items()
             if profile.get(flag) is True and finding.cover in covers]
    if not fired:
        return finding

    new_priority = _ESCALATE_ONE_STEP.get(finding.priority, finding.priority)
    if new_priority == finding.priority:
        return finding  # already at IMMEDIATE, nothing to escalate

    flag_labels = ", ".join(f.replace("_", " ") for f in fired)
    escalation_note = f"priority raised — business profile indicates {flag_labels}"
    finding.priority = new_priority
    finding.triggers_applied = fired
    finding.note = f"{finding.note}; {escalation_note}" if finding.note else escalation_note
    return finding


def apply_gap_rules(benchmark_result: "BenchmarkResult",
                    current_cover: dict,
                    profile: dict | None = None) -> list[GapFinding]:
    """
    current_cover: {cover_name: limit_gbp_or_None}. Cover names should already
    be canonical (coverage_names.normalise_coverage) — the caller (orchestrator)
    is responsible for that, since only it knows the raw source of the names.

    profile: canonical prospect profile, used ONLY to read the four
    extract_text.py trigger flags for priority escalation (see
    _apply_trigger_escalation). Optional — pass None to skip escalation
    entirely and get the same behaviour as before this existed.

    Returns findings for every cover resolve.py produced evidence for, plus any
    STATUTORY cover missing from that set entirely (defence in depth — a
    statutory gap must never be silently absent just because no peer or band
    row happened to mention it).
    """
    findings = []
    covers_seen = set()

    for name, bm in benchmark_result.covers.items():
        covers_seen.add(name)
        current_limit = current_cover.get(name)
        is_statutory = name in coverage_names.STATUTORY

        if current_limit is None or current_limit == 0:
            if bm.evidence_tier == TIER_INSUFFICIENT and not is_statutory:
                direction = NEEDS_REVIEW
            else:
                direction = GAP
        else:
            ref = bm.reference_limit
            if ref is None:
                direction = NEEDS_REVIEW
            elif current_limit < ref * INCREASE_THRESHOLD:
                direction = INCREASE
            else:
                direction = OK

        finding = GapFinding(
            cover=name, direction=direction,
            priority=_priority(direction, bm.evidence_tier, is_statutory),
            evidence_tier=bm.evidence_tier, current_limit=current_limit,
            reference_limit=bm.reference_limit, reference_source=bm.reference_source,
            prevalence=bm.prevalence, n_cohort=bm.n_cohort, n_holding=bm.n_holding,
            note=bm.note, is_statutory=is_statutory,
        )
        findings.append(_apply_trigger_escalation(finding, profile))

    # Defence in depth: a statutory cover the prospect doesn't hold must appear
    # even if resolve.py had zero evidence for it (empty cohort, no band row).
    for name in coverage_names.STATUTORY:
        if name in covers_seen:
            continue
        current_limit = current_cover.get(name)
        if current_limit is None or current_limit == 0:
            findings.append(GapFinding(
                cover=name, direction=GAP, priority=IMMEDIATE,
                evidence_tier=TIER_INSUFFICIENT, current_limit=None,
                note="statutory requirement — no peer/band evidence available, "
                     "flagged on legal grounds alone", is_statutory=True,
            ))

    order = {IMMEDIATE: 0, CONSIDER: 1, RENEWAL: 2}
    findings.sort(key=lambda f: (order.get(f.priority, 3), f.cover))
    return findings