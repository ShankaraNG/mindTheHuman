import sys
from pathlib import Path
_SRC = Path("src")
for _sub in ("", "profile", "benchmark", "engine"):
    p = str(_SRC / _sub) if _sub else str(_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd
from engine.orchestrator import Orchestrator
from evaluation.cover_summary_parser import parse_cover_summary
from evaluation.run_eval import _brief_to_profile, _score_brief

briefs = pd.read_csv("dataset/4_evaluation/synthetic_prospect_briefs.csv", encoding="utf-8-sig")
tech = briefs[briefs["vertical"] == "Tech"]
orch = Orchestrator(data_dir="data", healthcheck_dir="dataset/1_recommendation_bank",
                    gemini_api_key="dummy-key")

for _, brief in tech.iterrows():
    profile = _brief_to_profile(brief)
    current_cover = parse_cover_summary(brief.get("current_cover_summary", ""))
    report = orch.run(profile, current_cover)
    score = _score_brief(report["gap_results"], brief["evaluation_notes_expected_themes"])
    print(f"\n{brief['brief_id']} — score {score['score']:.0%} ({score['n_hit']}/{score['n_themes']})")
    for t in score["theme_detail"]:
        status = "HIT " if t["matched"] else "MISS"
        print(f"  {status} [{t['method']}] {t['theme']}")