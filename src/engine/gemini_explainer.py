"""
LLM Explainer — routed through the event's LiteLLM proxy (OpenAI-compatible
interface), not Google's native genai SDK. The proxy key is not a real
Google API key; it only works against base_url below.

Available models through this proxy (as issued): gemini-3.6-flash,
gemini-3.5-flash, gemini-3.1-pro. Defaulting to gemini-3.6-flash — Google's
own current stable/default Flash tier, and explicitly whitelisted here.

Same contract as before: the LLM receives frozen evidence (direction,
priority, numbers already decided upstream) and writes plain-English
rationale + alternative approaches only. It cannot change any number.
"""

import os, json, re
from openai import OpenAI

DEFAULT_BASE_URL = "https://litellm.perceptura.com"


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
7. For every finding with direction GAP or INCREASE, propose 2-3 named ALTERNATIVE APPROACHES to closing it —
   e.g. raising straight to the peer benchmark, layering primary + excess cover, phasing an increase across
   renewals, bundling with an adjacent cover. These are OPTIONS for a human broker to weigh, not a decision —
   you are not choosing between them or recommending one. Ground each in the rag_excerpt or historical evidence
   where possible; if there's no evidence for a specific alternative, still offer it but say so plainly rather
   than inventing a source. Findings with direction OK or NEEDS_REVIEW get an empty alternatives list — there
   is nothing to close, or not enough evidence to suggest how.
8. Return ONLY valid JSON — no markdown fences, no preamble, no trailing text."""


class GeminiExplainer:
    def __init__(self, api_key: str = None, model_name: str = "gemini-3.6-flash",
                base_url: str = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

        base_url = base_url or os.environ.get("LITELLM_BASE_URL") or DEFAULT_BASE_URL
        self.client = OpenAI(api_key=api_key, base_url=base_url)
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

        user_prompt = f"""Write a benchmarking report explanation for this {vertical} {funding} company.

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
      "rationale": "<your 2-3 sentence plain-English explanation citing evidence>",
      "alternatives": [
        {{"approach": "<short name, e.g. 'Raise to peer median directly'>",
          "detail": "<1-2 sentences on what this involves and any tradeoff>",
          "grounded": true}}
      ]
    }}
  ],
  "top_actions": [
    {{"action": "<short imperative>", "priority": "<Immediate action|For consideration|Review at renewal>"}}
  ],
  "disclaimer": "This is guidance, not regulated advice. Speak with a Capsule broker to confirm suitability for your circumstances."
}}

"alternatives" is a list of 2-3 items for GAP/INCREASE findings, an empty list [] for OK/NEEDS_REVIEW.
"grounded" is true only if the alternative is backed by rag_excerpt or historical evidence in the input —
set it false rather than fabricate a citation."""

        raw = None
        usage = {"input_tokens": 0, "output_tokens": 0}
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=8000,  # raised from 4000 — Gemini 3.x bills internal
                                  # "thinking" tokens at the output rate, counted
                                  # against this same cap, so visible JSON content
                                  # can get truncated well before hitting a pure
                                  # content-length limit. 8000 still keeps the
                                  # absolute worst case at ~£0.05, real margin
                                  # under the £0.10 budget (see metrics.py).
            )
            raw = response.choices[0].message.content
            if getattr(response, "usage", None):
                usage = {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                }
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            parsed = json.loads(clean)
            for finding in parsed.get("cover_findings", []):
                finding.setdefault("alternatives", [])
                for alt in finding["alternatives"]:
                    alt.setdefault("grounded", False)
            parsed["_token_usage"] = usage
            return parsed
        except json.JSONDecodeError:
            return {"executive_summary": "Explanation could not be generated cleanly for this "
                    "report — the underlying findings below are still correct and unaffected, "
                    "only the plain-English writeup failed to parse.",
                    "parse_error": True, "cover_findings": [], "top_actions": [],
                    "_token_usage": usage, "_raw_response_for_debugging": raw}
        except Exception as e:
            return {"executive_summary": f"LLM explanation unavailable: {e}",
                    "parse_error": True, "cover_findings": [], "top_actions": [],
                    "_token_usage": usage}