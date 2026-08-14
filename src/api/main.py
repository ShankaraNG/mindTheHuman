"""
FastAPI wrapper around the existing pipeline — one POST endpoint that
does exactly what app.py does after Step 3 (profile confirmed, cover
entered): normalise_intake() -> Orchestrator.run() -> return the report.

No new pipeline logic here. This is purely a thin HTTP layer over the
already-tested Streamlit code path.

Run: uvicorn src.api.main:app --reload --port 8000
     (or: python -m uvicorn src.api.main:app --port 8000, from repo root)
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Same bootstrap as every other entry point in this project — profile.py's
# own `import extract_text` etc. need src/profile/ directly on sys.path,
# not just src/. Getting this right FIRST, not after a failure this time.
_SRC = Path(__file__).parent.parent
for _sub in ("", "profile", "benchmark", "engine"):
    _p = str(_SRC / _sub) if _sub else str(_SRC)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Loads .env from the repo root if present — one place to set the key,
# instead of needing $env:GEMINI_API_KEY set fresh in every PowerShell
# session before running uvicorn. Falls back to a real environment
# variable if no .env exists, so this doesn't break anything already
# working via $env: or system-level env vars.
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

from profile.normalise import normalise_intake
from engine.orchestrator import Orchestrator

REPO_ROOT       = _SRC.parent
DATA_DIR        = REPO_ROOT / "data"
HEALTHCHECK_DIR = REPO_ROOT / "dataset" / "1_recommendation_bank"

orchestrator: Optional[Orchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once at startup — constructing this reads 21 RAG decks and
    # builds the whole benchmark resolver. Doing that per-request would
    # make every call slow for no reason.
    global orchestrator
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    orchestrator = Orchestrator(
        data_dir=str(DATA_DIR),
        healthcheck_dir=str(HEALTHCHECK_DIR) if HEALTHCHECK_DIR.exists() else None,
        gemini_api_key=api_key,
    )
    yield


app = FastAPI(
    title="Capsule Cover — Benchmarking API",
    description="Guidance, not regulated advice.",
    lifespan=lifespan,
)


class BenchmarkRequest(BaseModel):
    """
    Same field names normalise_intake() already accepts — this mirrors
    exactly what app.py passes it after Step 2 of the Streamlit flow.
    fca_regulated as a string ("true"/"false") to match normalise's own
    to_boolean() coercion, same as everywhere else in this project.
    """
    vertical: str = Field(..., examples=["Tech"])
    funding_series: Optional[str] = Field(None, examples=["Series A"])
    turnover_gbp: Optional[float] = Field(None, examples=[3_500_000])
    total_funding_raised_gbp: Optional[float] = Field(None, examples=[8_000_000])
    employee_count: Optional[int] = Field(None, examples=[31])
    us_revenue_pct: Optional[float] = Field(0.0, examples=[12.0])
    fca_regulated: Optional[str] = Field(None, examples=["false"])
    customer_type: Optional[str] = Field(None, examples=["B2B"])
    software_or_hardware: Optional[str] = Field(None, examples=["Software"])
    business_description: Optional[str] = Field(
        None, examples=["B2B analytics dashboard for e-commerce, processes customer purchase data."])
    current_cover: dict[str, Optional[float]] = Field(
        default_factory=dict,
        examples=[{"Professional Indemnity": 1_000_000, "Cyber": None}],
        description="Cover name -> limit in GBP, or null if not held.",
    )


@app.get("/health")
def health():
    return {"status": "ok", "orchestrator_loaded": orchestrator is not None}


@app.post("/api/benchmark")
def benchmark(req: BenchmarkRequest):
    if orchestrator is None:
        raise HTTPException(503, "Orchestrator not ready — check server startup logs.")

    payload = req.model_dump(exclude={"current_cover"})
    try:
        profile = normalise_intake(payload)
    except Exception as e:
        raise HTTPException(400, f"Could not normalise profile: {e}")

    try:
        report = orchestrator.run(profile, req.current_cover)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")

    return report