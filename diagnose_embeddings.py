"""
Reproduces run_eval.py's exact import order, then tries to load
sentence-transformers WITHOUT catching the exception — so the REAL
underlying traceback prints, not the generic wrapper message.
"""
import sys
from pathlib import Path
_SRC = Path("src")
for _sub in ("", "profile", "benchmark", "engine"):
    p = str(_SRC / _sub) if _sub else str(_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)

print("Step 1: importing profile.normalise...")
from profile.normalise import normalise_intake
print("  OK")

print("Step 2: importing engine.orchestrator (pulls in openai, resolve, gap_rules)...")
from engine.orchestrator import Orchestrator
print("  OK")

print("Step 3: constructing Orchestrator (loads RAG decks, creates openai client)...")
orch = Orchestrator(data_dir="data", healthcheck_dir="dataset/1_recommendation_bank",
                    gemini_api_key="dummy-key-diagnostic-only")
print("  OK")

print("Step 4: NOW loading sentence-transformers, same process, no exception handling...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
print("  OK — loaded successfully even after the full pipeline import chain")