"""
Theme matching for the eval harness. Two layers:

  1. Vocabulary reuse (coverage_names.py) — fast, deterministic, zero
     dependency risk, proven to add real recall. Always tried first.

  2. Real sentence embeddings, run in a SEPARATE SUBPROCESS via
     embeddings_worker.py. This is not the normal way to call a model —
     it exists specifically to route around a real, confirmed bug:

     Every script in this project that imports `profile.normalise` (i.e.
     every script) adds src/profile/ onto sys.path so `from profile...`
     resolves. That permanently overwrites sys.modules['profile'] for the
     rest of the process — Python's stdlib also has a module literally
     called profile.py (the built-in profiler). Deep inside transformers
     5.15.0's FP8/dynamo integration code, `import cProfile` triggers an
     internal `import profile` expecting the REAL stdlib module. It gets
     our empty package instead, has no .run attribute, and the whole
     import chain dies several layers up with a generic, misleading
     "Could not import module 'PreTrainedModel'" error that has nothing
     to do with the real cause. Confirmed via full traceback, not
     guessed — see project history if you want the whole diagnosis.

     A subprocess has its own sys.modules, untouched, as long as the
     worker script (embeddings_worker.py) never imports anything from
     this project. It's launched once, kept alive for the whole eval run,
     and talked to over stdin/stdout — not spawned per-call, since model
     loading takes real time.
"""

import sys
import subprocess
import json
from pathlib import Path
import re

_SRC = Path(__file__).parent.parent
for _sub in ("", "profile", "benchmark", "engine"):
    _p = str(_SRC / _sub) if _sub else str(_SRC)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import coverage_names


def _covers_mentioned_in_text(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z&]+", text)
    found = set()
    for n in range(1, 5):
        for i in range(len(words) - n + 1):
            span = " ".join(words[i:i + n])
            c = coverage_names.normalise_coverage(span)
            if c.matched:
                found.add(c.canonical)
    return found


def vocabulary_match(theme: str, candidates: list[str]) -> tuple[str | None, bool]:
    mentioned = _covers_mentioned_in_text(theme)
    if not mentioned:
        return None, False
    candidate_canon = {c: coverage_names.normalise_coverage(c).canonical for c in candidates}
    for cand, canon in candidate_canon.items():
        if canon and canon in mentioned:
            return cand, True
    return None, False


class _EmbeddingsWorker:
    """Manages the long-lived subprocess. Lazy-started on first use."""

    def __init__(self):
        self.proc = None
        self.available = None  # None = not attempted yet

    def _start(self):
        worker_path = Path(__file__).parent / "embeddings_worker.py"
        try:
            self.proc = subprocess.Popen(
                [sys.executable, str(worker_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
            resp_line = self.proc.stdout.readline()
            resp = json.loads(resp_line) if resp_line else {}
            if resp.get("status") == "ready":
                self.available = True
                print("[semantic_match] embeddings worker started (subprocess, isolated sys.modules)")
            else:
                self.available = False
                detail = resp.get("detail", "no response")
                import warnings
                warnings.warn(f"embeddings worker failed to load model ({detail}) — vocabulary-only.")
        except Exception as e:
            self.available = False
            import warnings
            warnings.warn(f"embeddings worker subprocess failed to start ({e}) — vocabulary-only.")

    def query(self, theme: str, candidates: list[str]) -> tuple[str | None, float]:
        if self.available is None:
            self._start()
        if not self.available:
            return None, 0.0
        try:
            self.proc.stdin.write(json.dumps({"theme": theme, "candidates": candidates}) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            resp = json.loads(line) if line else {}
            return resp.get("cover"), float(resp.get("score", 0.0))
        except Exception:
            self.available = False
            return None, 0.0

    def shutdown(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.terminate()


_WORKER = _EmbeddingsWorker()

THRESHOLD_EMBEDDINGS = 0.40  # not empirically tuned against real data yet


def best_match(theme: str, candidates: list[str]) -> dict:
    cand, hit = vocabulary_match(theme, candidates)
    if hit:
        return {"cover": cand, "score": 1.0, "method": "vocabulary"}

    cand, score = _WORKER.query(theme, candidates)
    if cand and score >= THRESHOLD_EMBEDDINGS:
        return {"cover": cand, "score": score, "method": "embeddings"}

    return {"cover": None, "score": score, "method": "none"}


def shutdown():
    """Call at the end of a run to cleanly stop the subprocess."""
    _WORKER.shutdown()


if __name__ == "__main__":
    covers = ["Cyber", "Directors & Officers", "Employers Liability",
             "Professional Indemnity", "Public & Products Liability",
             "Intellectual Property", "Business Interruption"]
    tests = [
        "PI/Cyber limits vs MSA requirements and US exposure",
        "D&O ok",
        "consider XOL",
        "management liability gap",
        "IP defence cover",
        "no product recall cover",
    ]
    for theme in tests:
        r = best_match(theme, covers)
        status = "HIT " if r["cover"] else "MISS"
        print(f"{status} [{r['method']:10}] score={r['score']:.3f}  {theme!r:45} -> {r['cover']}")
    shutdown()