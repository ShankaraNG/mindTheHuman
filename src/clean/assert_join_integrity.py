import glob
import os
import pandas as pd

DATA_DIR = "data"
PREFIX_VERTICAL = {"CON": "Consumer", "FIN": "FinTech", "MED": "MedTech", "TEC": "Tech", "UNK": "Unknown"}


def load_all(data_dir=DATA_DIR):
    return {os.path.basename(p): pd.read_csv(p) for p in glob.glob(os.path.join(data_dir, "*.csv"))}


def check_prefix_vertical(dfs):
    violations = []
    for name, df in dfs.items():
        if "client_id" not in df.columns or "vertical" not in df.columns:
            continue
        prefix = df["client_id"].astype(str).str.split("-").str[0]
        expected = prefix.map(PREFIX_VERTICAL)
        bad = df[(expected.notna()) & (df["vertical"] != expected)]
        if len(bad):
            v = bad[["client_id", "vertical"]].copy()
            v["expected_vertical"] = expected[bad.index]
            v["file"] = name
            violations.append(v)
    return pd.concat(violations, ignore_index=True) if violations else pd.DataFrame()


def check_policy_orphans(dfs):
    if "policy_level.csv" not in dfs or "cover_line_level.csv" not in dfs:
        return pd.DataFrame()
    policies = set(dfs["policy_level.csv"]["policy_id"])
    cl = dfs["cover_line_level.csv"]
    orphans = cl[~cl["policy_id"].isin(policies)]
    return orphans[["client_id", "policy_id"]].drop_duplicates() if len(orphans) else pd.DataFrame()


def check_syn_never_joins_real(dfs):
    violations = []
    for name, df in dfs.items():
        for col in ("client_id", "brief_id"):
            if col not in df.columns:
                continue
            syn_rows = df[df[col].astype(str).str.startswith("SYN-")]
            if len(syn_rows) and name != "synthetic_briefs_clean.csv" and name != "expected_themes.csv":
                v = syn_rows[[col]].copy()
                v["file"] = name
                v["column"] = col
                violations.append(v.rename(columns={col: "id"}))
    return pd.concat(violations, ignore_index=True) if violations else pd.DataFrame()


CHECKS = [
    ("client_id prefix matches vertical", check_prefix_vertical),
    ("cover_line_level.policy_id has no orphans vs policy_level", check_policy_orphans),
    ("SYN- ids never appear outside synthetic_briefs/expected_themes", check_syn_never_joins_real),
]


def run(data_dir=DATA_DIR):
    dfs = load_all(data_dir)
    all_passed = True
    for name, fn in CHECKS:
        violations = fn(dfs)
        if len(violations):
            all_passed = False
            print(f"FAIL  {name}  ({len(violations)} violations)")
            print(violations.to_string(index=False))
            print()
        else:
            print(f"PASS  {name}")
    assert all_passed, "join-integrity checks failed, see violations above"
    return all_passed


if __name__ == "__main__":
    run()