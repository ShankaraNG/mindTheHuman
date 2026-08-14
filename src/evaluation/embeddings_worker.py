"""
Standalone embeddings worker — runs in its own subprocess, deliberately.

CRITICAL: this file must NEVER import anything from this project (no
sys.path manipulation, no `from profile...`, no `from engine...`). The
whole reason it's a separate process is that the parent process's
sys.modules['profile'] gets permanently overwritten by src/profile/ the
moment anything imports profile.normalise — which breaks torch._dynamo's
internal `import cProfile` -> `import profile` deep inside transformers
5.15.0's FP8 integration code. A fresh subprocess has its own sys.modules,
untouched, as long as THIS script stays clean of our own project imports.

Protocol: one JSON object per line on stdin, one JSON object per line on
stdout. Kept deliberately simple — no framework, no message length
prefixing, since line-buffered JSON is enough for this volume of calls.

  {"cmd": "ping"}                                    -> {"status": "ready"}
  {"theme": "...", "candidates": ["...", "..."]}     -> {"cover": "..."|null, "score": 0.0-1.0}
  {"cmd": "shutdown"}                                -> process exits
"""

import sys
import json


def main():
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        # Report failure once, then exit — the parent falls back to
        # vocabulary-only if it can't get a "ready" response.
        print(json.dumps({"status": "error", "detail": str(e)}), flush=True)
        return

    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"cover": None, "score": 0.0, "error": "bad_json"}), flush=True)
            continue

        if req.get("cmd") == "shutdown":
            break
        if req.get("cmd") == "ping":
            print(json.dumps({"status": "ready"}), flush=True)
            continue

        theme = req.get("theme", "")
        candidates = req.get("candidates", [])
        if not candidates:
            print(json.dumps({"cover": None, "score": 0.0}), flush=True)
            continue

        import numpy as np
        theme_emb = model.encode([theme])[0]
        cand_embs = model.encode(candidates)
        sims = cand_embs @ theme_emb / (
            np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(theme_emb) + 1e-9
        )
        best_idx = int(np.argmax(sims))
        print(json.dumps({"cover": candidates[best_idx], "score": float(sims[best_idx])}), flush=True)


if __name__ == "__main__":
    main()