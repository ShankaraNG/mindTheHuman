"""
Task 3.3 — Benchmark resolver. Milestone C.

    canonical profile  ->  a benchmark for EVERY cover type, each carrying an
                           explicit evidence tier

THE EVIDENCE TIERS, strongest first:

    PEER_AND_BAND   n >= 5 peers hold this cover, and a Capsule band exists.
                    Report both. If they disagree, that disagreement is itself
                    reportable and interesting.
    PEER_ONLY       n >= 5 peers hold it, no band row for this cover.
    BAND_ONLY       band exists but fewer than 5 peers hold it. Percentiles are
                    NOT reported — not even with a caveat. A median of three is
                    not a median.
    PREVALENCE_ONLY enough peers to say how many hold it, but too few limits to
                    quantify. Supports "your peers carry this, you don't" without
                    inventing a target number.
    INSUFFICIENT    no band, no usable peers. Says so. Routes to broker review.

This tier is the single most important field in the whole engine's output. It is
what lets the report say "we don't know" — and a benchmarking tool that cannot
say "we don't know" will confidently upsell.

PREVALENCE IS COMPUTED OVER THE WHOLE COHORT, LIMITS ONLY OVER HOLDERS.
Denominators differ by design: 12 of 30 peers hold Product Recall (prevalence
40%), and the P25/median/P75 describe those 12, not all 30. Conflating the two
either dilutes limits toward zero or overstates how normal a cover is.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import band_lookup
    import coverage_names
    import peer_match
except ImportError:
    from . import band_lookup, coverage_names, peer_match


MIN_PEERS_FOR_PERCENTILES = 5     # holders of the cover, not cohort size
MIN_PEERS_FOR_PREVALENCE = 5      # cohort size

TIER_PEER_AND_BAND = "PEER_AND_BAND"
TIER_PEER_ONLY = "PEER_ONLY"
TIER_BAND_ONLY = "BAND_ONLY"
TIER_PREVALENCE_ONLY = "PREVALENCE_ONLY"
TIER_INSUFFICIENT = "INSUFFICIENT"

COVER_SUMMARY_ALIASES = {
    "client_id": ["client_id", "clientid"],
    "coverage":  ["coverage", "coverage_canonical", "cover", "cover_type"],
    # max_limit_amount is the headline indemnity limit; min_limit_amount is the
    # lowest sub-limit on the same cover and would understate every benchmark.
    "limit":     ["max_limit_amount", "limit_gbp", "limit",
                  "indemnity_limit_gbp", "limit_amount_gbp"],
}


@dataclass
class CoverBenchmark:
    coverage: str
    evidence_tier: str
    # peer evidence
    n_cohort: int = 0
    n_holding: int = 0
    prevalence: Optional[float] = None
    p25: Optional[float] = None
    median: Optional[float] = None
    p75: Optional[float] = None
    # band evidence
    band_limit: Optional[float] = None
    band_label: Optional[str] = None
    band_basis: Optional[str] = None
    band_n_observations: int = 0
    band_contributing_clients: list = field(default_factory=list)
    band_tier: Optional[str] = None
    # narrative
    note: Optional[str] = None

    @property
    def reference_limit(self) -> Optional[float]:
        """The single number the gap rules in Phase 4 compare against.

        Median of holders when the peer evidence is strong enough, otherwise the
        band. Deliberately NOT an average of the two — blending a peer statistic
        with a published band produces a number that cannot be sourced to either.
        """
        if self.evidence_tier in (TIER_PEER_AND_BAND, TIER_PEER_ONLY):
            return self.median
        if self.evidence_tier == TIER_BAND_ONLY:
            return self.band_limit
        return None

    @property
    def reference_source(self) -> Optional[str]:
        if self.evidence_tier in (TIER_PEER_AND_BAND, TIER_PEER_ONLY):
            return f"median of {self.n_holding} comparable Capsule clients"
        if self.evidence_tier == TIER_BAND_ONLY:
            return self.band_label
        return None


@dataclass
class BenchmarkResult:
    covers: dict = field(default_factory=dict)      # canonical name -> CoverBenchmark
    cohort: Optional[peer_match.PeerCohort] = None
    band: Optional[band_lookup.BandResult] = None
    profile_echo: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """API response shape. Provenance is not optional — every cover carries
        its tier and its denominators, so the PDF can show its working without a
        second pass over the data."""
        return {
            "cohort": self.cohort.provenance() if self.cohort else None,
            "band": {
                "available": bool(self.band and self.band.available),
                "basis": self.band.basis_used if self.band else None,
                "band_label": self.band.band_label if self.band else None,
                "reason": self.band.reason if self.band else None,
            },
            "covers": {
                name: {
                    "evidence_tier": c.evidence_tier,
                    "reference_limit": c.reference_limit,
                    "reference_source": c.reference_source,
                    "peers": {"cohort": c.n_cohort, "holding": c.n_holding,
                              "prevalence": c.prevalence,
                              "p25": c.p25, "median": c.median, "p75": c.p75},
                    "band": {"limit": c.band_limit, "label": c.band_label,
                             "basis": c.band_basis, "row": c.band_source_row},
                    "note": c.note,
                }
                for name, c in self.covers.items()
            },
        }

    def summary_lines(self) -> list:
        out = []
        for name, c in sorted(self.covers.items()):
            ref = f"£{c.reference_limit:,.0f}" if c.reference_limit else "—"
            prev = f"{c.prevalence:.0%}" if c.prevalence is not None else "—"
            out.append(f"{name:32} {c.evidence_tier:16} ref={ref:>12} "
                       f"held by {prev:>4} of {c.n_cohort}")
        return out


def _resolve_columns(df: pd.DataFrame, aliases: dict, label: str) -> dict:
    lower = {c.lower().strip(): c for c in df.columns}
    resolved, missing = {}, []
    for canonical, options in aliases.items():
        for alias in options:
            if alias.lower() in lower:
                resolved[canonical] = lower[alias.lower()]
                break
        else:
            missing.append(canonical)
    if missing:
        raise KeyError(f"{label} missing columns for {missing}. "
                       f"Actual: {list(df.columns)}. Update the alias map.")
    return resolved


class BenchmarkResolver:
    """Composed once at API start-up.

        resolver = BenchmarkResolver.from_paths(
            pool_csv="data/client_level_normalised.csv",
            cover_summary_csv="data/client_cover_summary.csv",
            bands_csv="data/deck_benchmark_bands_clean.csv",
        )
        result = resolver.resolve(profile)
    """

    def __init__(self, matcher: peer_match.PeerMatcher,
                 cover_summary: pd.DataFrame,
                 bands: band_lookup.BandTable):
        self.matcher = matcher
        self.bands = bands
        self.cols = _resolve_columns(cover_summary, COVER_SUMMARY_ALIASES,
                                     "client_cover_summary")
        cs = cover_summary.copy()
        c = self.cols
        cs = coverage_names.add_canonical_column(cs, c["coverage"])
        self.unmatched_covers = coverage_names.unmatched_report(cs, c["coverage"])
        # Restrict to headline covers. The book records some clients at policy
        # level and others at policy-SECTION level, so without this the median
        # "Property" limit is polluted by Debris Removal and Fixed Glass.
        before = len(cs)
        cs = cs[cs["coverage_canonical"].isin(coverage_names.HEADLINE)]
        self.sublimit_rows_excluded = int(before - len(cs))
        cs[c["limit"]] = pd.to_numeric(cs[c["limit"]], errors="coerce")
        cs[c["client_id"]] = cs[c["client_id"]].astype(str)
        self.cover_summary = cs
        # The only clients that can contribute evidence.
        self.covered_ids = set(cs[c["client_id"]].astype(str))

    @classmethod
    def from_paths(cls, pool_csv: str, cover_summary_csv: str,
                   deck_limits_csv: str,
                   house_bands_csv: Optional[str] = None) -> "BenchmarkResolver":
        return cls(
            matcher=peer_match.PeerMatcher(pd.read_csv(pool_csv)),
            cover_summary=pd.read_csv(cover_summary_csv),
            bands=band_lookup.BandTable.from_paths(deck_limits_csv, house_bands_csv),
        )

    def resolve(self, profile: dict) -> BenchmarkResult:
        cohort = self.matcher.match(profile, eligible_ids=self.covered_ids)
        band = self.bands.lookup(
            profile.get("vertical"),
            turnover_gbp=profile.get("turnover_gbp"),
            total_funding_gbp=profile.get("total_funding_gbp"),
        )

        result = BenchmarkResult(cohort=cohort, band=band,
                                 profile_echo=dict(profile))

        peer_rows = self.cover_summary[
            self.cover_summary[self.cols["client_id"]].isin(cohort.client_ids)
        ] if cohort.client_ids else self.cover_summary.iloc[0:0]

        n_cohort = cohort.n
        cohort_usable = n_cohort >= MIN_PEERS_FOR_PREVALENCE

        # The union of every cover named by either evidence source. A cover that
        # peers hold but no band mentions must still appear (that's a gap the
        # band table can't see), and vice versa.
        covers = set(band.hits) | set(coverage_names.HEADLINE) | set(
            peer_rows["coverage_canonical"].dropna().unique()
            if not peer_rows.empty else []
        )

        for name in sorted(covers):
            hit = band.hits.get(name)
            holders = peer_rows[peer_rows["coverage_canonical"] == name] \
                if not peer_rows.empty else peer_rows

            n_holding = int(holders[self.cols["client_id"]].nunique()) if len(holders) else 0
            limits = holders[self.cols["limit"]].dropna() if len(holders) else pd.Series(dtype=float)
            n_limits = int(len(limits))

            prevalence = (n_holding / n_cohort) if cohort_usable and n_cohort else None

            bm = CoverBenchmark(
                coverage=name,
                evidence_tier=TIER_INSUFFICIENT,
                n_cohort=n_cohort,
                n_holding=n_holding,
                prevalence=prevalence,
                band_limit=hit.recommended_limit if hit else None,
                band_label=hit.band_label if hit else None,
                band_basis=hit.basis if hit else None,
                band_n_observations=hit.n_observations if hit else 0,
                band_contributing_clients=hit.contributing_clients if hit else [],
                band_tier=hit.tier if hit else None,
            )

            has_band = hit is not None and hit.recommended_limit is not None
            has_percentiles = (cohort.sufficient
                              and n_limits >= MIN_PEERS_FOR_PERCENTILES)

            if has_percentiles:
                bm.p25 = float(np.percentile(limits, 25))
                bm.median = float(np.percentile(limits, 50))
                bm.p75 = float(np.percentile(limits, 75))
                bm.evidence_tier = TIER_PEER_AND_BAND if has_band else TIER_PEER_ONLY
                if has_band and bm.median and hit.recommended_limit:
                    ratio = bm.median / hit.recommended_limit
                    if ratio < 0.5 or ratio > 2.0:
                        bm.note = (f"peer median and Capsule band differ by "
                                   f"{ratio:.1f}x — worth a broker eye")
            elif has_band:
                bm.evidence_tier = TIER_BAND_ONLY
                bm.note = (f"only {n_limits} peer limit"
                           f"{'' if n_limits == 1 else 's'} available "
                           f"(need {MIN_PEERS_FOR_PERCENTILES}) — "
                           f"benchmark is Capsule's published band")
            elif cohort_usable and n_holding:
                bm.evidence_tier = TIER_PREVALENCE_ONLY
                bm.note = (f"{n_holding} of {n_cohort} comparable clients hold "
                           f"this cover, but too few disclosed limits to set a "
                           f"target — broker review")
            else:
                bm.note = (cohort.reason or
                           "no comparable peers and no band row for this cover")

            result.covers[name] = bm

        return result


if __name__ == "__main__":
    import sys
    resolver = BenchmarkResolver.from_paths(sys.argv[1], sys.argv[2], sys.argv[3])
    if resolver.unmatched_covers:
        print("UNMATCHED cover names in book of business:")
        for nm, ct in resolver.unmatched_covers:
            print(f"   {ct:3}x {nm!r}")
    demo = {"vertical": "FinTech & Venture", "funding_series": "Series A",
            "turnover_final_gbp": 2_500_000, "employee_count_final": 28,
            "total_funding_final_gbp": 8_000_000}
    res = resolver.resolve(demo)
    print(f"\ncohort n={res.cohort.n} sufficient={res.cohort.sufficient}")
    print(f"widening: {res.cohort.widening_applied}\n")
    for line in res.summary_lines():
        print(line)