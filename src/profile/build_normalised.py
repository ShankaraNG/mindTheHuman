import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2])) 

from src.profile.normalise import normalise_peer_pool

# Offline build step. Run manually when client_level.csv changes. Not called at runtime.
RAW_PATH = Path("data/client_level.csv")
OUT_PATH = Path("data/client_level_normalised.csv")

def main():
    raw = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(raw)} rows from {RAW_PATH}")

    pool = normalise_peer_pool(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUT_PATH, index=False)

    usable = pool["profile_usable"].sum()
    print(f"Wrote {len(pool)} rows to {OUT_PATH}")
    print(f"profile_usable: {usable} True / {len(pool) - usable} False")
    print(pool["missing_filters"].value_counts(dropna=False))

    pool = pd.read_csv("data/client_level_normalised.csv")
    mismatch = pool[(pool["missing_filters"].isna() | (pool["missing_filters"] == "")) & (pool["profile_usable"] == False)]
    print(mismatch[["client_id", "vertical", "profile_usable", "missing_filters", "usable_weight_dims"]])

if __name__ == "__main__":
    main()