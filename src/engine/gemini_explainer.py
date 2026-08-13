"""
LLM Explainer — google-genai (current SDK, replaces deprecated google-generativeai).

Same contract as before: the LLM receives frozen evidence (direction,
priority, numbers already decided by Stages 1-3) and writes plain-English
rationale only. It cannot change any number.
"""

import os, json, re
from google import genai
from google.genai import types


def _fmt_gbp(n):
    if n is None:
        return "not provided"
    if n >= 1_000_000:
        return f"£{n/1_000_000:.1f}m"
    if n >= 1_000:
        return f"£{n/1_000:.0f}k"
    return f"£{n:,.0f}"


SYSTEM_INSTRUCTION = """You are a Capsule Cover broker assistant writing a plain-English insurance benchmarking report.

RULES — read carefully:
1. The direction (GAP / INCREASE / OK), priority, and all numbers are FROZEN. You did not compute them. Do NOT change them.
2. Your job: explain WHY in 2-3 clear sentences a founder can understand. No jargon.
3. Use the rag_excerpt and historical evidence to ground your explanation — cite similar clients where available.
4. For INSUFFICIENT confidence: say "insufficient peer data — broker review recommended."
5. Never recommend a specific insurer or premium figure.
6. Frame everything as "guidance, not regulated advice."
7. Return ONLY valid JSON — no markdown fences, no preamble, no trailing text."""


class GeminiExplainer:
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def explain(self, profile: dict, gap_results: list, summary_stats: dict) -> dict:
        vertical = profile.get("vertical", "")
        funding  = profile.get("funding_stage", "")

        items_brief = []
        for r in gap_results:
            ev = r.get("evidence", {})
            rag_text = " | ".join(
                p["passage"][:200] for p in r.get("rag_passages", [])[:2] if p.get("passage")
            )
            pair_text = ""
            for p in r.get("matched_pairs", [])[:2]:
                if p.get("current_gbp") and p.get("recommended_gbp"):
                    pair_text += (f"[{p['client_id']}] held {_fmt_gbp(p['current_gbp'])}, "
                                  f"Capsule recommended {_fmt_gbp(p['recommended_gbp'])}. ")

            items_brief.append({
                "cover":         r["cover"],
                "direction":     r["direction"],
                "priority":      r["priority"],
                "confidence":    r["confidence"],
                "current_limit": _fmt_gbp(r["current_limit"]) if r.get("current_limit") else "Not held",
                "peer_median":   _fmt_gbp(ev.get("peer_median")) if ev.get("peer_median") else None,
                "peer_n":        ev.get("peer_n"),
                "band_rec":      ev.get("band_rec"),
                "historical":    pair_text or None,
                "rag_excerpt":   rag_text or None,
            })

        prompt = f"""Write a benchmarking report explanation for this {vertical} {funding} company.

PROSPECT PROFILE:
- Vertical: {vertical}
- Funding: {funding}
- Turnover: {_fmt_gbp(profile.get('turnover_gbp'))}
- Total funding raised: {_fmt_gbp(profile.get('total_funding_gbp'))}
- Employees: {profile.get('employee_count', 'unknown')}
- FCA regulated: {profile.get('regulated', 'unknown')}
- US revenue: {profile.get('us_revenue_pct', 0)}%

PEER COHORT: {summary_stats.get('cohort_size', 0)} comparable companies

COVER FINDINGS (direction + priority already decided — explain them, do not change them):
{json.dumps(items_brief, indent=2)}

Return JSON with this exact structure:
{{
  "executive_summary": "3-4 sentences overall position, most urgent issue first",
  "cover_findings": [
    {{
      "cover": "<name — copy exactly from input>",
      "direction": "<copy exactly from input>",
      "priority": "<copy exactly from input>",
      "confidence": "<copy exactly from input>",
      "rationale": "<your 2-3 sentence plain-English explanation citing evidence>"
    }}
  ],
  "top_actions": [
    {{"action": "<short imperative>", "priority": "<Immediate action|For consideration|Review at renewal>"}}
  ],
  "disclaimer": "This is guidance, not regulated advice. Speak with a Capsule broker to confirm suitability for your circumstances."
}}"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"executive_summary": raw, "parse_error": True,
                    "cover_findings": [], "top_actions": []}
        except Exception as e:
            return {"executive_summary": f"LLM explanation unavailable: {e}",
                    "parse_error": True, "cover_findings": [], "top_actions": []}