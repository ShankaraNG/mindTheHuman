"""
Task 3.2 — Peer selection.

Input:  one canonical profile (from normalise.py) + the normalised peer pool
Output: matched peer client_ids, their distances, and a full record of how the
        cohort was arrived at

TWO DESIGN DECISIONS THAT MATTER MORE THAN THE MATHS:

1. PERCENTILE DISTANCE, NOT RAW DISTANCE. Turnover in this book spans roughly
   four orders of magnitude. Euclidean distance on raw pounds makes every company
   under £1m identical and every company over £10m an outlier. Each size field is
   converted to its percentile position within the pool first, so distance means
   "how far apart in the population", which is also how a broker thinks about it.

2. THE FILTER WIDENS, AND SAYS WHEN IT DID. A hard vertical + stage filter on a
   242-client pool routinely returns 4 peers. Rather than report a median of 4 as
   if it were a benchmark, the filter widens in recorded steps and the final
   `widening_applied` list goes into the report. A cohort of 6 reached by widening
   the stage window is honest; a cohort of 4 presented as a percentile is not.

MISSING DIMENSIONS ARE DROPPED, NOT IMPUTED. A peer with no headcount is compared
on turnover and funding only, with weights renormalised over what's present, and
`dims_used` recorded per peer. Imputing the median would manufacture similarity
on exactly the field we lack evidence for.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import stage_ladder
except ImportError:
    from . import stage_ladder


# Turnover carries the most weight because it is the exposure base most cover
# limits actually scale with. Headcount and funding are corroborating signals.
# These three were kept as independent dimensions on the Phase 2 finding that
# their pairwise Spearman correlation is only 0.23-0.63 — none is a proxy for
# another, so collapsing them would discard real information.
DEFAULT_WEIGHTS = {
    "turnover_gbp": 0.4,
    "employee_count": 0.3,
    "total_funding_gbp": 0.3,
}

# normalise.py already emits <col>_pct percentile columns for the pool. Reuse them
# rather than recomputing: two independently-computed percentile scales for the
# same field is a silent divergence waiting to happen.
PRECOMPUTED_PCT_SUFFIX = "_pct"

MIN_COHORT = 5        # below this, resolve.py must not report percentiles
TARGET_COHORT = 30    # the demo bar from Capsule's deck is 30-50 comparators
MAX_COHORT = 50


def _resolve_prospect_stage(profile: dict) -> "stage_ladder.StageResolution":
    """A prospect profile that came from normalise.py already carries
    _funding_stage_ordinal; a hand-built dict (tests, stub callers) may not, so
    fall back to resolving it from whatever raw-ish fields are present."""
    if "_funding_stage_ordinal" in profile:
        ordinal = profile["_funding_stage_ordinal"]
        label = profile.get("funding_stage")
        return stage_ladder.StageResolution(
            ordinal=ordinal, label=label,
            source=profile.get("_funding_stage_source"),
            raw=None, conflict=bool(profile.get("_funding_stage_conflict")))
    return stage_ladder.resolve_stage(
        profile.get("funding_stage") or profile.get("funding_series"),
        profile.get("stage_of_evolution"))


@dataclass
class Peer:
    client_id: str
    distance: float
    dims_used: list
    vertical: Optional[str] = None
    stage_label: Optional[str] = None


@dataclass
class PeerCohort:
    peers: list = field(default_factory=list)
    n: int = 0
    filters_applied: dict = field(default_factory=dict)
    widening_applied: list = field(default_factory=list)
    pool_size: int = 0
    excluded: dict = field(default_factory=dict)
    sufficient: bool = False        # n >= MIN_COHORT
    reason: Optional[str] = None

    @property
    def client_ids(self) -> list:
        return [p.client_id for p in self.peers]

    def provenance(self) -> dict:
        """Everything the report needs to show its working."""
        return {
            "cohort_size": self.n,
            "pool_size": self.pool_size,
            "filters": self.filters_applied,
            "widening": self.widening_applied,
            "excluded": self.excluded,
            "sufficient_for_percentiles": self.sufficient,
            "peer_ids": self.client_ids,
        }


def _percentile_position(pool_values: pd.Series, value: Optional[float]) -> Optional[float]:
    """Where `value` sits in the pool distribution, 0-1. The prospect and the
    peers are both scored against the same pool, so the two are comparable."""
    if value is None or pd.isna(value):
        return None
    clean = pool_values.dropna()
    if clean.empty:
        return None
    return float((clean < value).mean())


def _distance(prospect_pos: dict, peer_pos: dict, weights: dict) -> tuple:
    """Weighted RMS distance over dimensions present on BOTH sides."""
    dims = [d for d in weights
            if prospect_pos.get(d) is not None and peer_pos.get(d) is not None]
    if not dims:
        return None, []
    total_w = sum(weights[d] for d in dims)
    sq = sum(weights[d] * (prospect_pos[d] - peer_pos[d]) ** 2 for d in dims)
    return float(np.sqrt(sq / total_w)), dims


class PeerMatcher:
    """Built once from the normalised pool, queried per request."""

    def __init__(self, pool: pd.DataFrame,
                 weights: Optional[dict] = None,
                 id_col: str = "client_id"):
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.id_col = id_col
        # reset_index is load-bearing: the percentile arrays below are positional,
        # so a non-RangeIndex pool (e.g. read_csv with index_col) would silently
        # pair each prospect against the wrong peer's size position.
        self.pool = pool.copy().reset_index(drop=True)

        missing = [c for c in self.weights if c not in self.pool.columns]
        if missing:
            raise KeyError(f"peer pool missing size columns {missing}. "
                           f"Columns present: {list(self.pool.columns)}")

        for col in self.weights:
            self.pool[col] = pd.to_numeric(self.pool[col], errors="coerce")

        # Resolve every peer's stage once, up front.
        # normalise.py now resolves funding_stage once, at profile-build time
        # (previously this was recomputed here as a workaround for a funding_stage
        # column that was always null — see normalise.py's changelog note). Read
        # the precomputed ordinal if present; fall back to resolving from raw
        # columns only for pools built before that fix landed.
        if "_funding_stage_ordinal" in self.pool.columns:
            self.pool["_stage_ordinal"] = self.pool["_funding_stage_ordinal"]
            self.pool["_stage_label"] = self.pool.get("funding_stage")
        else:
            stages = [
                stage_ladder.resolve_stage(
                    r.get("funding_stage") or r.get("funding_series"),
                    r.get("stage_of_evolution"))
                for r in self.pool.to_dict("records")
            ]
            self.pool["_stage_ordinal"] = [s.ordinal for s in stages]
            self.pool["_stage_label"] = [s.label for s in stages]

        # Pre-compute each peer's percentile position in the pool.
        self._peer_pos = {}
        self.pct_source = {}
        for col in self.weights:
            pct_col = col + PRECOMPUTED_PCT_SUFFIX
            if pct_col in self.pool.columns:
                vals = pd.to_numeric(self.pool[pct_col], errors="coerce")
                # normalise.py may emit 0-100 or 0-1; detect and rescale
                scale = 100.0 if vals.max(skipna=True) and vals.max() > 1.5 else 1.0
                self._peer_pos[col] = [None if pd.isna(v) else float(v) / scale
                                       for v in vals]
                self.pct_source[col] = pct_col
            else:
                self._peer_pos[col] = [
                    _percentile_position(self.pool[col], v) for v in self.pool[col]
                ]
                self.pct_source[col] = "computed"

    def match(self, profile: dict,
              target: int = TARGET_COHORT,
              max_cohort: int = MAX_COHORT,
              eligible_ids: Optional[set] = None) -> PeerCohort:
        """`eligible_ids` restricts the cohort to clients that actually have cover
        data. THIS IS NOT OPTIONAL FOR LIMIT BENCHMARKING. The pool is 242
        questionnaire/CRM profiles but only 48 of them appear in the book of
        business, so an unrestricted cohort of 30 contains ~4 clients with any
        policy at all — and every prevalence figure computed against it is wrong
        by a factor of seven. Pass the book's client_ids here.
        """
        vertical = profile.get("vertical")
        if vertical in (None, "", "Unknown"):
            return PeerCohort(
                pool_size=len(self.pool),
                reason="prospect vertical unknown — vertical is a hard filter "
                       "with no valid fallback, so no peer cohort can be formed. "
                       "Band lookup only.",
            )

        widening, excluded = [], {}
        df = self.pool

        # --- Hard filter 1: vertical. Never relaxed. -------------------------
        # NaN vertical (the 49 UNK-* clients) and the literal string "Unknown"
        # are the same thing in different files. Both are excluded — vertical is a
        # hard filter with no valid fallback.
        v = df["vertical"].astype(str).str.strip().str.lower()
        matched = df[v.eq(str(vertical).strip().lower()) & ~v.isin(["nan", "unknown", ""])]
        excluded["different_vertical"] = int(len(df) - len(matched))
        if matched.empty:
            return PeerCohort(pool_size=len(df),
                              filters_applied={"vertical": vertical},
                              reason=f"no clients in pool with vertical {vertical!r}")

        # --- Filter 1b: must have cover evidence -----------------------------
        if eligible_ids is not None:
            before = len(matched)
            matched = matched[matched[self.id_col].astype(str).isin(eligible_ids)]
            excluded["no_cover_data"] = int(before - len(matched))
            if matched.empty:
                return PeerCohort(pool_size=len(df),
                                  filters_applied={"vertical": vertical},
                                  excluded=excluded,
                                  reason=f"no {vertical} clients in the book of "
                                         f"business — cannot form an evidence cohort")

        # --- Filter 2: funding stage, adjacency window, widened if needed ----
        prospect_stage = _resolve_prospect_stage(profile)
        stage_note = None
        staged = matched

        if prospect_stage.ordinal is None:
            stage_note = ("prospect stage unknown — stage filter not applied "
                          "(cohort spans all stages within the vertical)")
            widening.append("stage filter skipped: prospect stage unknown")
        else:
            for window in (1, 2):
                allowed = stage_ladder.adjacent(prospect_stage.ordinal, window)
                candidate = matched[matched["_stage_ordinal"].isin(allowed)]
                if len(candidate) >= MIN_COHORT:
                    staged = candidate
                    stage_note = (f"{prospect_stage.label}, +/-{window} rung"
                                  f"{'s' if window > 1 else ''}")
                    if window > 1:
                        widening.append(f"stage window widened to +/-{window} "
                                        f"(+/-1 gave {len(matched[matched['_stage_ordinal'].isin(stage_ladder.adjacent(prospect_stage.ordinal, 1))])} peers)")
                    break
            else:
                staged = matched
                stage_note = "stage filter dropped"
                widening.append("stage filter dropped entirely: even +/-2 rungs "
                                f"gave fewer than {MIN_COHORT} peers")
            excluded["outside_stage_window"] = int(len(matched) - len(staged))

        # --- Rank by weighted percentile distance ----------------------------
        prospect_pos = {c: _percentile_position(self.pool[c], profile.get(c))
                        for c in self.weights}
        if all(p is None for p in prospect_pos.values()):
            # No size signal at all: return the filtered set unranked rather than
            # a fake ordering. Distance of 1.0 flags "not ranked" downstream.
            peers = [Peer(str(r[self.id_col]), 1.0, [], r.get("vertical"),
                          r.get("_stage_label"))
                     for r in staged.to_dict("records")][:max_cohort]
            return PeerCohort(peers=peers, n=len(peers),
                              filters_applied={"vertical": vertical,
                                               "stage": stage_note},
                              widening_applied=widening + [
                                  "unranked: prospect has no turnover, "
                                  "headcount or funding"],
                              pool_size=len(self.pool), excluded=excluded,
                              sufficient=len(peers) >= MIN_COHORT)

        scored, no_shared_dims = [], 0
        for pos_idx, row in zip(staged.index, staged.to_dict("records")):
            peer_pos = {c: self._peer_pos[c][self.pool.index.get_loc(pos_idx)]
                        for c in self.weights}
            dist, dims = _distance(prospect_pos, peer_pos, self.weights)
            if dist is None:
                no_shared_dims += 1
                continue
            scored.append(Peer(str(row[self.id_col]), dist, dims,
                               row.get("vertical"), row.get("_stage_label")))
        if no_shared_dims:
            excluded["no_comparable_size_field"] = no_shared_dims

        scored.sort(key=lambda p: p.distance)
        peers = scored[:max(target, MIN_COHORT)][:max_cohort]

        return PeerCohort(
            peers=peers, n=len(peers),
            filters_applied={"vertical": vertical, "stage": stage_note,
                             "weights": self.weights},
            widening_applied=widening,
            pool_size=len(self.pool), excluded=excluded,
            sufficient=len(peers) >= MIN_COHORT,
            reason=None if len(peers) >= MIN_COHORT else
            f"only {len(peers)} comparable peers — below the {MIN_COHORT} "
            f"minimum for percentile reporting",
        )


if __name__ == "__main__":
    import sys
    pool = pd.read_csv(sys.argv[1])
    matcher = PeerMatcher(pool)
    demo = {"vertical": "FinTech & Venture", "funding_series": "Series A",
            "turnover_final_gbp": 2_500_000, "employee_count_final": 28,
            "total_funding_final_gbp": 8_000_000}
    cohort = matcher.match(demo)
    print(f"n={cohort.n}  sufficient={cohort.sufficient}")
    print(f"filters: {cohort.filters_applied}")
    print(f"widening: {cohort.widening_applied}")
    print(f"excluded: {cohort.excluded}")
    for p in cohort.peers[:10]:
        print(f"   {p.client_id:14} d={p.distance:.3f}  {p.stage_label}  {p.dims_used}")