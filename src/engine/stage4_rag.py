"""
Stage 4 — RAG Retriever

Source data:
  21 healthcheck markdown files — Capsule's actual broker narratives,
  one per client in the recommendation bank.

What it does:
  For each gap/increase finding, finds the most relevant paragraph
  from a real Capsule health check written for a similar company.
  This text is handed to the LLM so it explains using Capsule's
  own language — not generic insurance knowledge.

Matching strategy:
  1. Filter to same vertical
  2. Score each section by keyword overlap with the cover type
  3. Return top 2 narrative passages per finding

The LLM gets:
  - The frozen gap analysis (direction, priority, numbers)
  - The RAG passages (real broker language)
  Its only job: write plain English combining these two sources.
"""

import re
from pathlib import Path


def _split_sections(text: str) -> list[dict]:
    """Split a healthcheck markdown into sections."""
    sections = []
    current_header = "intro"
    current_body = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_body:
                sections.append({
                    "header": current_header,
                    "body":   "\n".join(current_body).strip(),
                })
            current_header = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append({"header": current_header, "body": "\n".join(current_body).strip()})

    return sections


def _score_section(body: str, cover: str, keywords: list[str]) -> float:
    body_lower = body.lower()
    score = sum(1.0 for kw in keywords if kw.lower() in body_lower)
    # Bonus for sections that look like limit tables or observations
    if "| item" in body_lower or "| action" in body_lower:
        score += 0.5
    if "immediate action" in body_lower or "for consideration" in body_lower:
        score += 0.5
    if "£" in body and any(c.isdigit() for c in body):
        score += 0.3
    return score


def _cover_keywords(cover: str) -> list[str]:
    """Extract search keywords from a cover type name."""
    cover_lower = cover.lower()
    base = [w for w in re.split(r"\W+", cover_lower) if len(w) > 3]

    extras = {
        "professional indemnity": ["PI", "indemnity", "financial loss", "errors"],
        "cyber": ["cyber", "data breach", "ransomware", "cyber crime", "data loss"],
        "directors and officers": ["D&O", "directors", "officers", "management liability"],
        "employers liability": ["employers", "EL", "employee"],
        "public and products liability": ["PL", "public liability", "product liability"],
        "crime": ["crime", "fraud", "fidelity", "theft", "social engineering"],
        "employment practices liability": ["EPL", "employment practices", "wrongful dismissal"],
        "corporate legal liability": ["corporate", "entity cover"],
    }

    for key, words in extras.items():
        if key in cover_lower:
            base.extend(words)

    return list(set(base))


class RAGRetriever:
    def __init__(self, data_dir: str, healthcheck_dir: str = None):
        self.data_dir = Path(data_dir)
        hc_dir = Path(healthcheck_dir) if healthcheck_dir else self.data_dir.parent / "dataset" / "1_recommendation_bank"
        self._load(hc_dir)

    def _load(self, hc_dir: Path):
        self.healthchecks = {}

        if not hc_dir.exists():
            # Try relative to data_dir
            hc_dir = self.data_dir / "healthchecks"

        if not hc_dir.exists():
            print(f"RAGRetriever: healthcheck dir not found at {hc_dir}")
            return

        for f in hc_dir.glob("*_healthcheck.md"):
            parts = f.stem.split("_")
            client_id = parts[0]
            vertical  = parts[1] if len(parts) > 1 else "Unknown"
            text = f.read_text(encoding="utf-8", errors="ignore")
            self.healthchecks[client_id] = {
                "vertical": vertical,
                "sections": _split_sections(text),
            }

        print(f"RAGRetriever loaded: {len(self.healthchecks)} healthchecks")

    def retrieve(self, cover: str, vertical: str, direction: str, n: int = 2) -> list[dict]:
        """
        Find the top n most relevant narrative passages for a given cover finding.
        Returns list of {"client_id", "header", "passage", "relevance_score"}.
        """
        if not self.healthchecks:
            return []

        keywords = _cover_keywords(cover)
        if direction == "GAP":
            keywords.extend(["not insured", "missing", "gap", "no cover", "immediate"])
        elif direction == "INCREASE":
            keywords.extend(["increase", "recommend", "inadequate", "below", "higher limit"])

        candidates = []
        for client_id, hc in self.healthchecks.items():
            if hc["vertical"] != vertical:
                continue
            for section in hc["sections"]:
                if len(section["body"]) < 80:
                    continue
                score = _score_section(section["body"], cover, keywords)
                if score > 0:
                    # Extract the most relevant paragraph (max 400 chars)
                    paragraphs = [p.strip() for p in section["body"].split("\n\n")
                                  if len(p.strip()) > 60]
                    best_para = max(
                        paragraphs,
                        key=lambda p: sum(1 for kw in keywords if kw.lower() in p.lower()),
                        default=""
                    )
                    if best_para:
                        # Clean up markdown table pipes
                        passage = re.sub(r"\|.*?\|", "", best_para).strip()
                        passage = re.sub(r"\s+", " ", passage)[:400]
                        candidates.append({
                            "client_id":       client_id,
                            "header":          section["header"],
                            "passage":         passage,
                            "relevance_score": score,
                        })

        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
        return candidates[:n]