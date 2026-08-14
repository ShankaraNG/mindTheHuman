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
    Liability) MISSING -> ALWAYS Immediate action, regardless of evidence tier.
    This one is not a peer-prevalence judgement, it is UK law. "Missing" is the
    operative word: a statutory cover the prospect HOLDS but which we have no
    reference limit for is a NEEDS_REVIEW, and escalating that to Immediate
    tells a compliant company its legally-required cover is an emergency.

PREVALENCE GATES URGENCY. "evidence says peers expect it" above is a real
condition, not a figure of speech: resolve.py computes prevalence (share of the
cohort holding the cover) precisely so this module can use it. A cover held by
5 of 16 peers is not something a comparable company is expected to carry, and
reporting it as "Immediate action" is the confident upsell the tier design
exists to prevent. So an unheld cover below MAJORITY_PREVALENCE is still
reported — the prospect may well want it — but as "For consideration", with the
prevalence stated in the note. Statutory covers are exempt: EL is UK law, and
what fraction of the cohort happens to have disclosed it is irrelevant.

TRIGGERS ESCALATE EVIDENCE, NOT ITS ABSENCE. The free-text trigger flags below
raise priority one step on findings that already rest on evidence. They must
never touch a NEEDS_REVIEW finding: NEEDS_REVIEW is this module saying "we do
not know", and a keyword match in a business description is not a substitute for
the peer or band data we just said we lack. extract_text.py documents its own
precision as deliberately loose — "customers" is enough to set personal_data —
so letting it manufacture an Immediate action out of an INSUFFICIENT tier is
exactly the failure the evidence tiers exist to prevent.
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

_PRIORITY_RANK = {IMMEDIATE: 0, CONSIDER: 1, RENEWAL: 2}

INCREASE_THRESHOLD = 0.75   # held below this fraction of reference_limit -> INCREASE

# Share of the cohort that must hold a cover before "you don't have it" is an
# Immediate action rather than something to consider. A simple majority: below
# it, the honest statement is "some comparable companies carry this", which is a
# conversation, not an urgent gap.
MAJORITY_PREVALENCE = 0.5


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


def _priority(direction: str, evidence_tier: str, is_statutory: bool,
              prevalence: Optional[float] = None) -> str:
    if is_statutory and direction == GAP:
        # UK law, not a peer judgement — always urgent regardless of evidence.
        # GAP only: a statutory cover that IS held but has no reference limit
        # comes through as NEEDS_REVIEW, and that is a data gap on our side, not
        # a compliance failure on theirs.
        return IMMEDIATE
    if direction == NEEDS_REVIEW:
        return CONSIDER
    if direction == GAP:
        if evidence_tier not in (TIER_PEER_AND_BAND, TIER_PEER_ONLY, TIER_BAND_ONLY):
            return CONSIDER
        # Known minority prevalence: peers mostly do NOT carry this, so the
        # evidence does not support calling its absence urgent. prevalence is
        # None when it could not be observed at all (cohort too small, or a
        # cover the book cannot report on) — the tier is then the only signal
        # available and is left to stand on its own.
        if prevalence is not None and prevalence < MAJORITY_PREVALENCE:
            return CONSIDER
        return IMMEDIATE
    if direction == INCREASE:
        return CONSIDER if evidence_tier in (TIER_PEER_AND_BAND, TIER_PEER_ONLY) else RENEWAL
    return RENEWAL  # OK


def _evidence_ceiling(finding: GapFinding) -> str:
    """The highest priority this finding's EVIDENCE supports, before triggers.

    Escalation runs after _priority, so without a ceiling it silently undoes the
    prevalence gate: Crime at 31% peer prevalence is capped to CONSIDER by
    _priority, then a financial_data keyword pushes it straight back to
    IMMEDIATE. The cap is what makes the gate mean anything.

    Only GAPs are capped. An INCREASE has a real benchmark and a measured
    shortfall against it, so "you are under-covered AND you handle payment data"
    is a genuine escalation rather than urgency conjured from a word match.
    """
    if finding.is_statutory:
        return IMMEDIATE
    if finding.direction == GAP:
        if (finding.prevalence is not None
                and finding.prevalence < MAJORITY_PREVALENCE):
            return CONSIDER
        if finding.evidence_tier not in (TIER_PEER_AND_BAND, TIER_PEER_ONLY,
                                         TIER_BAND_ONLY):
            return CONSIDER
    return IMMEDIATE


def _apply_trigger_escalation(finding: GapFinding, profile: dict) -> GapFinding:
    """
    Escalates priority by exactly one step if a confirmed (True, not None)
    trigger flag names this cover as relevant, and the finding isn't already
    OK. Records which flag(s) fired in triggers_applied and appends a plain
    note — this must be visible in the report, not a silent priority bump.
    """
    # NEEDS_REVIEW is excluded for the same reason OK is, and it matters more:
    # it means resolve.py found no usable evidence, so there is no finding here
    # to make more urgent — only an admission to overwrite. See module docstring.
    if finding.direction in (OK, NEEDS_REVIEW) or not profile:
        return finding

    fired = [flag for flag, covers in TRIGGER_RELEVANT_COVERS.items()
             if profile.get(flag) is True and finding.cover in covers]
    if not fired:
        return finding

    new_priority = _ESCALATE_ONE_STEP.get(finding.priority, finding.priority)
    if new_priority == finding.priority:
        return finding  # already at IMMEDIATE, nothing to escalate

    # A trigger may raise urgency within what the evidence supports, never past
    # it. Blocked silently: recording "priority raised" on a finding whose
    # priority did not move would be worse than saying nothing.
    if _PRIORITY_RANK[new_priority] < _PRIORITY_RANK[_evidence_ceiling(finding)]:
        return finding

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

        note = bm.note

        if current_limit is None or current_limit == 0:
            if bm.evidence_tier == TIER_INSUFFICIENT and not is_statutory:
                direction = NEEDS_REVIEW
            else:
                direction = GAP
                # Say why this is a conversation and not an urgent gap, in the
                # report itself — the priority alone doesn't explain itself.
                if (not is_statutory and bm.prevalence is not None
                        and bm.prevalence < MAJORITY_PREVALENCE):
                    minority = (f"only {bm.n_holding} of {bm.n_cohort} comparable "
                                f"clients ({bm.prevalence:.0%}) hold this cover — "
                                f"a minority practice, not an expected baseline")
                    # Appended, not substituted: resolve.py's note carries the
                    # reference-limit provenance and any peer-vs-band divergence
                    # warning, which is what a reader challenging this finding
                    # needs most.
                    note = f"{note}; {minority}" if note else minority
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
            priority=_priority(direction, bm.evidence_tier, is_statutory,
                               bm.prevalence),
            evidence_tier=bm.evidence_tier, current_limit=current_limit,
            reference_limit=bm.reference_limit, reference_source=bm.reference_source,
            prevalence=bm.prevalence, n_cohort=bm.n_cohort, n_holding=bm.n_holding,
            note=note, is_statutory=is_statutory,
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