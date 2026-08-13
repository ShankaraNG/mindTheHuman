"""
Evaluation harness — runs all 24 synthetic prospect briefs through the real
pipeline and scores against evaluation_notes_expected_themes.

Answer-key isolation: evaluation_notes_expected_themes is read ONLY after
gap_results already exist. It is never passed into normalise_intake(),
the orchestrator, or any pipeline input.

Scoring is against gap_results (the deterministic benchmark + gap-rules
output), not the LLM's explanation text — this tests the reasoning engine,
which is the layer that actually decides. The LLM only explains what's
already been found.

Run: python src/evaluation/run_eval.py [--with-llm]
"""

import sys, re, argparse
from pathlib import Path
import pandas as pd

_SRC = Path(__file__).parent.parent
for _sub in ("", "profile", "benchmark", "engine"):
    _p = str(_SRC / _sub) if _sub else str(_SRC)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profile.normalise import normalise_intake
from engine.orchestrator import Orchestrator
from evaluation.cover_summary_parser import parse_cover_summary

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR  = REPO_ROOT / "data"

STOPWORDS = {"the", "a", "an", "of", "for", "vs", "and", "or", "to", "in", "on", "with", "is"}


def _theme_keywords(theme: str) -> set:
    words = re.findall(r"[a-z']+", theme.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _brief_to_profile(brief: pd.Series) -> dict:
    fca = brief.get("fca_regulated")
    if isinstance(fca, str):
        fca = fca.lower() == "true"
    return normalise_intake({
        "vertical":                 brief["vertical"],
        "funding_series":           brief.get("funding_series", ""),
        "turnover_gbp":             brief.get("turnover_gbp"),
        "total_funding_raised_gbp": brief.get("total_funding_raised_gbp"),
        "employee_count":           brief.get("employee_count"),
        "us_revenue_pct":           brief.get("us_revenue_pct"),
        "fca_regulated":            "true" if fca else "false",
        "customer_type":            brief.get("customer_type", "B2B"),
        "software_or_hardware":     brief.get("software_or_hardware", "Software"),
        "business_description":    brief.get("business_description"),
    })


def _score_brief(gap_results: list, expected_themes: str) -> dict:
    """
    Only GAP and INCREASE count as confident findings for scoring purposes.
    NEEDS_REVIEW explicitly means the engine found insufficient evidence and
    is declining to claim a gap — crediting it as a "hit" would reward the
    engine for being uncertain, exactly backwards from what this eval is for.
    """
    themes = [t.strip() for t in expected_themes.split(";") if t.strip()]
    flagged_covers = [r["cover"].lower() for r in gap_results
                      if r["direction"] in ("GAP", "INCREASE")]
    flagged_text = " ".join(flagged_covers)

    hits, theme_results = 0, []
    for theme in themes:
        kws = _theme_keywords(theme)
        matched = bool(kws) and any(kw in flagged_text for kw in kws)
        if matched:
            hits += 1
        theme_results.append({"theme": theme, "matched": matched})

    return {
        "n_themes": len(themes), "n_hit": hits,
        "score": round(hits / len(themes), 3) if themes else None,
        "theme_detail": theme_results,
    }


def run_eval(with_llm: bool = False):
    briefs_path = REPO_ROOT / "dataset" / "4_evaluation" / "synthetic_prospect_briefs.csv"
    if not briefs_path.exists():
        briefs_path = DATA_DIR / "synthetic_briefs_clean.csv"
    briefs = pd.read_csv(briefs_path, encoding="utf-8-sig")

    orch = Orchestrator(data_dir=str(DATA_DIR),
                        healthcheck_dir=str(REPO_ROOT / "dataset" / "1_recommendation_bank"),
                        gemini_api_key="dummy-key" if not with_llm else None)

    results = []
    for _, brief in briefs.iterrows():
        brief_id = brief["brief_id"]
        expected = brief.get("evaluation_notes_expected_themes", "")

        try:
            profile = _brief_to_profile(brief)
            current_cover = parse_cover_summary(brief.get("current_cover_summary", ""))

            report = orch.run(profile, current_cover)
            gap_results = report["gap_results"]
            cohort_size = report["summary_stats"].get("cohort_size", 0)
            score_info = _score_brief(gap_results, expected)

            results.append({
                "brief_id": brief_id, "vertical": brief["vertical"],
                "status": "OK" if cohort_size > 0 else "NO_REAL_COHORT",
                "score": score_info["score"], "n_hit": score_info["n_hit"],
                "n_themes": score_info["n_themes"], "n_findings": len(gap_results),
                "cohort_size": cohort_size,
            })

        except Exception as e:
            results.append({"brief_id": brief_id, "vertical": brief.get("vertical", "?"),
                            "status": f"ERROR: {e}", "score": None, "n_findings": 0})

    df = pd.DataFrame(results)
    out_path = DATA_DIR / "eval_results.csv"
    df.to_csv(out_path, index=False)

    ok = df[df["status"] == "OK"]
    print(f"\n{'='*60}")
    print(f"Ran {len(df)} briefs — {len(ok)} OK, {len(df)-len(ok)} failed/no-peers")
    if len(ok):
        print(f"Mean theme-recall score: {ok['score'].mean():.1%}")
        print(f"\nPer-vertical breakdown:")
        print(ok.groupby("vertical")["score"].agg(["mean", "count"]).to_string())
    failed = df[df["status"] != "OK"]
    if len(failed):
        print(f"\nFailed/skipped briefs:")
        print(failed[["brief_id", "status"]].to_string(index=False))
    print(f"\nFull results written to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true",
                        help="Use real Gemini explanation (requires GEMINI_API_KEY)")
    args = parser.parse_args()
    run_eval(with_llm=args.with_llm)