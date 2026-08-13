"""
Task 3.1 — Reference limits. REWRITTEN after inspecting the real files.

WHAT CHANGED AND WHY. The original design read deck_benchmark_bands_clean.csv as
"vertical + band -> recommended limit per cover". That file cannot support it:

  - Consumer only. 4 clients, all CON-*. Zero rows for FinTech, MedTech or Tech,
    so a band lookup keyed on vertical returns nothing for three of the four
    verticals, including the demo vertical.
  - No cover column at all. Cover names survive only as an unordered slash-list
    in slide_context, and where a client has two band-sets on one basis, nothing
    links set to cover (the parser flagged this itself).
  - Of its four distinct band-sets, two are byte-identical across all four
    clients — good evidence they are Capsule's HOUSE table, not client advice.
    The other two appear for 2 clients and 1 client respectively and are
    unusable without the missing cover mapping.

So this module now reads TWO files and returns a two-tier answer:

  TIER 1  deck_limit_benchmarks_parsed.csv -> vertical x cover median of
          Capsule's own recommended limits. Per-cover, all four verticals,
          78 unique broker judgments. Cohort size always disclosed.
  TIER 2  deck_benchmark_bands_clean.csv sets 1-2 -> a single COVER-AGNOSTIC
          house floor keyed on revenue or funds raised. Used only where tier 1
          has nothing for that cover, and labelled as generic so no one mistakes
          it for a cover-specific recommendation.

NO REVENUE BANDING ON TIER 1. Tempting, but the counts forbid it: FinTech has
Cyber n=6, D&O n=6, Crime n=5, PI n=3. Splitting six observations across five
revenue bands produces medians of one. Vertical x cover, n disclosed, is the
honest resolution of what this data can carry.

LEAKAGE WARNING — READ BEFORE WIRING THE EVAL. capsule_recommended_limit is also
the only labelled ground truth in the dataset, i.e. the natural basis for the
"80% broker agreement" metric. If the same rows set the benchmark AND score the
engine, that metric measures nothing. Either hold out a client split (see
`holdout_split`) or score agreement against peer-derived direction only.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import coverage_names
except ImportError:
    from . import coverage_names


DEDUPE_KEY = ["client_id", "cover", "current_limit", "capsule_recommended_limit"]
MIN_DECK_OBS = 2          # below this, tier 1 is one broker's single opinion


@dataclass
class BandHit:
    coverage: str
    recommended_limit: Optional[float]
    band_label: Optional[str]        # human-readable provenance
    basis: Optional[str]             # "deck median" | "revenue" | "funds raised"
    vertical: Optional[str]
    n_observations: int = 0
    contributing_clients: list = field(default_factory=list)
    tier: str = "DECK_MEDIAN"


@dataclass
class BandResult:
    hits: dict = field(default_factory=dict)
    basis_used: Optional[str] = None
    band_label: Optional[str] = None
    value_used: Optional[float] = None
    generic_floor: Optional[float] = None
    reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.hits) or self.generic_floor is not None


class BandTable:
    """Name kept so resolve.py's import doesn't churn; contents are the two-tier
    reference table described above."""

    def __init__(self, deck_limits: pd.DataFrame,
                 house_bands: Optional[pd.DataFrame] = None):
        d = deck_limits.drop_duplicates(subset=DEDUPE_KEY).copy()
        d = coverage_names.add_canonical_column(d, "cover")
        self.unmatched = coverage_names.unmatched_report(d, "cover")

        # capsule_recommended_limit_low is NaN for the non-numeric categories
        # (FULL_LIMIT, SUBJECT_TO_DISCLOSURES, TO_BE_DISCUSSED...). Those rows
        # still evidence that Capsule addressed the cover, so they are counted
        # separately rather than dropped.
        d["_rec"] = pd.to_numeric(d["capsule_recommended_limit_low"], errors="coerce")
        self.deck = d
        self.unquantified = int(d["_rec"].isna().sum())

        agg = (d.dropna(subset=["coverage_canonical"])
                .groupby(["vertical", "coverage_canonical"]))
        self._table = {}
        for (vert, cov), g in agg:
            quantified = g.dropna(subset=["_rec"])
            self._table[(str(vert).strip().lower(), cov)] = {
                "median": float(np.median(quantified["_rec"])) if len(quantified) else None,
                "n": int(len(quantified)),
                "n_mentioned": int(len(g)),
                "clients": sorted(g["client_id"].astype(str).unique().tolist()),
            }

        self.house = _HouseBands(house_bands) if house_bands is not None else None

    @classmethod
    def from_paths(cls, deck_limits_csv: str,
                   house_bands_csv: Optional[str] = None) -> "BandTable":
        house = pd.read_csv(house_bands_csv) if house_bands_csv else None
        return cls(pd.read_csv(deck_limits_csv), house)

    def lookup(self, vertical: Optional[str],
               turnover_gbp: Optional[float] = None,
               total_funding_gbp: Optional[float] = None) -> BandResult:
        if vertical in (None, "", "Unknown") or pd.isna(vertical):
            return BandResult(reason="vertical unknown — no reference limits apply")

        key = str(vertical).strip().lower()
        result = BandResult()

        for (vert, cov), stats in self._table.items():
            if vert != key or stats["median"] is None:
                continue
            if stats["n"] < MIN_DECK_OBS:
                label = (f"single Capsule recommendation ({stats['clients'][0]}) "
                         f"— indicative only")
            else:
                label = (f"median of {stats['n']} Capsule recommendations "
                         f"in {vertical}")
            result.hits[cov] = BandHit(
                coverage=cov, recommended_limit=stats["median"],
                band_label=label, basis="deck median", vertical=vertical,
                n_observations=stats["n"], contributing_clients=stats["clients"],
                tier="DECK_MEDIAN" if stats["n"] >= MIN_DECK_OBS else "DECK_SINGLE",
            )

        if self.house is not None:
            floor = self.house.floor(turnover_gbp, total_funding_gbp)
            result.generic_floor = floor.get("limit")
            result.basis_used = floor.get("basis")
            result.band_label = floor.get("label")
            result.value_used = floor.get("value")

        if not result.available:
            result.reason = f"no reference limits for vertical {vertical!r}"
        return result

    def holdout_split(self, test_fraction: float = 0.3, seed: int = 0):
        """Split by CLIENT, not by row, so no client's own recommendation can set
        the benchmark it is later scored against. Returns (train_ids, test_ids)."""
        clients = sorted(self.deck["client_id"].astype(str).unique())
        rng = np.random.default_rng(seed)
        rng.shuffle(clients)
        cut = max(1, int(len(clients) * test_fraction))
        return clients[cut:], clients[:cut]


class _HouseBands:
    """The two byte-identical band-sets from deck_benchmark_bands_clean.csv.

    Cover-agnostic by necessity — that file has no cover column. Returns ONE
    number: a generic limit floor for the company's size band.
    """

    def __init__(self, df: pd.DataFrame):
        d = df.copy()
        for c in ("band_low", "band_high", "recommended_limit_gbp"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        # Keep only band-sets that recur across every client that has them —
        # the evidence that they are house-standard rather than client-specific.
        counts = {}
        for (cid, basis, src), g in d.groupby(["client_id", "basis", "source_row_index"]):
            g = g.sort_values("band_index")
            fp = tuple(zip(g.band_low, g.band_high.fillna(-1), g.recommended_limit_gbp))
            counts.setdefault((basis, fp), set()).add(cid)
        n_clients = d["client_id"].nunique()
        self.sets = {basis: fp for (basis, fp), owners in counts.items()
                     if len(owners) == n_clients}

    def floor(self, turnover_gbp=None, total_funding_gbp=None) -> dict:
        for basis, value, label in (("revenue", turnover_gbp, "revenue"),
                                    ("raised", total_funding_gbp, "funds raised")):
            fp = self.sets.get(basis)
            if fp is None or value is None or pd.isna(value) or value <= 0:
                continue
            for lo, hi, limit in fp:
                if lo <= value and (hi < 0 or value <= hi):
                    band = f"£{lo/1e6:g}m+" if hi < 0 else f"£{lo/1e6:g}-{hi/1e6:g}m"
                    return {"limit": float(limit), "basis": label,
                            "label": f"Capsule house band, {band} {label} (generic, "
                                     f"not cover-specific)", "value": float(value)}
            top = fp[-1]
            return {"limit": float(top[2]), "basis": label,
                    "label": f"Capsule house band, top band (value above all bands)",
                    "value": float(value)}
        return {}


if __name__ == "__main__":
    t = BandTable.from_paths("deck_limit_benchmarks_parsed.csv",
                             "deck_benchmark_bands_clean.csv")
    print(f"unmatched cover names: {t.unmatched or 'none'}")
    print(f"rows with unquantifiable recommended limit: {t.unquantified}")
    print(f"house band-sets recurring across all clients: {list(t.house.sets)}")
    for v in ["FinTech", "Tech", "MedTech", "Consumer"]:
        r = t.lookup(v, turnover_gbp=2_500_000, total_funding_gbp=8_000_000)
        print(f"\n{v}: floor={r.generic_floor} ({r.band_label})")
        for cov, h in sorted(r.hits.items()):
            print(f"   {cov:32} £{h.recommended_limit:>12,.0f}  n={h.n_observations}  {h.tier}")