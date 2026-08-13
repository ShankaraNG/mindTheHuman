"""
Orchestrator — wires the real pipeline together.

    profile (canonical, via normalise_intake) + current_cover
      -> BenchmarkResolver.resolve()      [src/benchmark/resolve.py]
      -> apply_gap_rules()                [src/benchmark/gap_rules.py]
      -> RAGRetriever.retrieve() per flagged cover  [stage4_rag.py]
      -> GeminiExplainer.explain()        [gemini_explainer.py — explain only]

Supersedes the old stage1_profile_match / stage2_benchmark / stage3_gap_detect
/ data_loaders files, which are retired. Those did the same job with a flatter,
less evidence-honest design; peer_match.py + band_lookup.py + resolve.py do it
properly (adjacency-window stage filtering, cover-data-eligible cohorts,
5-tier evidence, prevalence-vs-limit separation).
"""

import sys
from pathlib import Path

# peer_match.py / resolve.py use bare `import stage_ladder` / `import
# coverage_names` with a relative-import fallback. Neither works unless BOTH
# sibling package directories are directly on sys.path (not just src/) — this
# is the single place that bootstraps it, so every other entry point (app.py,
# run_eval.py) just imports this module and inherits the fix.
_SRC = Path(__file__).parent.parent
for _sub in ("profile", "benchmark", "engine"):
    _p = str(_SRC / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import resolve
import gap_rules
import coverage_names
from stage4_rag import RAGRetriever
from gemini_explainer import GeminiExplainer


class Orchestrator:
    def __init__(self, data_dir: str, healthcheck_dir: str = None,
                 gemini_api_key: str = None, gemini_model: str = "gemini-2.5-flash"):
        data_dir = Path(data_dir)
        self.resolver = resolve.BenchmarkResolver.from_paths(
            pool_csv=str(data_dir / "client_level_normalised.csv"),
            cover_summary_csv=str(data_dir / "client_cover_summary.csv"),
            deck_limits_csv=str(data_dir / "deck_limit_benchmarks_parsed.csv"),
            house_bands_csv=str(data_dir / "deck_benchmark_bands_clean.csv"),
        )
        self.rag = RAGRetriever(str(data_dir), healthcheck_dir)
        self.explainer = GeminiExplainer(api_key=gemini_api_key, model_name=gemini_model)

    def run(self, profile: dict, current_cover: dict) -> dict:
        """
        profile:       canonical prospect profile (from normalise_intake)
        current_cover: {cover_name: limit_gbp_or_None} — raw names from the UI,
                       canonicalised here before comparison since only the
                       caller knows the raw source.
        """
        canon_cover = {}
        for raw_name, limit in current_cover.items():
            c = coverage_names.normalise_coverage(raw_name)
            canon_cover[c.canonical or raw_name] = limit

        result = self.resolver.resolve(profile)
        findings = gap_rules.apply_gap_rules(result, canon_cover)

        gap_results = []
        for f in findings:
            rag_passages = []
            if f.direction in (gap_rules.GAP, gap_rules.INCREASE):
                rag_passages = self.rag.retrieve(
                    cover=f.cover, vertical=profile.get("vertical", ""),
                    direction=f.direction)

            gap_results.append({
                "cover": f.cover, "direction": f.direction, "priority": f.priority,
                "confidence": f.evidence_tier, "current_limit": f.current_limit,
                "evidence": {
                    "peer_median": f.reference_limit if f.evidence_tier in
                        (resolve.TIER_PEER_AND_BAND, resolve.TIER_PEER_ONLY) else None,
                    "band_rec": f.reference_limit if f.evidence_tier == resolve.TIER_BAND_ONLY else None,
                    "peer_n": f.n_holding, "n_cohort": f.n_cohort,
                    "prevalence": f.prevalence, "reference_source": f.reference_source,
                    "note": f.note,
                },
                "matched_recs": [], "rag_passages": rag_passages,
            })

        immediate = [g for g in gap_results if g["priority"] == gap_rules.IMMEDIATE]
        consider  = [g for g in gap_results if g["priority"] == gap_rules.CONSIDER]

        summary_stats = {
            "immediate_count": len(immediate), "consideration_count": len(consider),
            "cohort_size": result.cohort.n if result.cohort else 0,
            "cohort_confidence": "sufficient" if (result.cohort and result.cohort.sufficient) else "insufficient",
            "n_with_cover_data": result.cohort.n if result.cohort else 0,
            "cover_data_coverage_pct": 100.0 if result.cohort and result.cohort.sufficient else 0.0,
            "covers_assessed": len(gap_results),
        }

        explanation = self.explainer.explain(profile, gap_results, summary_stats)

        return {
            "profile": profile,
            "cohort": result.cohort.provenance() if result.cohort else {},
            "gap_results": gap_results,
            "summary_stats": summary_stats,
            "explanation": explanation,
        }