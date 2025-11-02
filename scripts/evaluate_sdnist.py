#!/usr/bin/env python3
# evaluate_sdnist.py
import argparse, json
from pathlib import Path
from typing import List, Sequence, Tuple
import numpy as np, pandas as pd

SDNIST_AVAILABLE = False; SDNIST_VERSION = None
try:
    import sdnist  # type: ignore
    SDNIST_AVAILABLE = True
    try: SDNIST_VERSION = getattr(sdnist, "__version__", "unknown")
    except Exception: SDNIST_VERSION = "unknown"
except Exception:
    SDNIST_AVAILABLE = False

def l1_error(df1: pd.DataFrame, df2: pd.DataFrame, cols: Sequence[str]) -> float:
    cols = list(cols) 
    r = df1[cols].astype(str).groupby(cols).size()
    s = df2[cols].astype(str).groupby(cols).size()
    idx = r.index.union(s.index)
    r = r.reindex(idx, fill_value=0).astype(float)
    s = s.reindex(idx, fill_value=0).astype(float)
    if r.sum() == 0 or s.sum() == 0:
        return float("nan")
    r /= r.sum(); s /= s.sum()
    return float(np.abs(r - s).sum())

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate synthetic data with L1 over specified marginals (SDNist optional).")
    # repo-relative defaults (no ../)
    p.add_argument("--real", default="data/dataset_reduced.csv")
    p.add_argument("--syn",  default="results/synthetic.csv")
    p.add_argument("--out_csv",  default="results/metrics.csv")
    p.add_argument("--out_json", default="results/metrics.json")
    # Example: --marginals ZIP_reduced;ZIP_reduced+Age_bin;ZIP_reduced+Income_bin;ZIP_reduced+Age_bin+Category
    p.add_argument("--marginals", default="", help="Override marginals; ';' between marginals, '+' within.")
    return p.parse_args()

def resolve_geo_column(df: pd.DataFrame) -> str:
    for name in ["ZIP_reduced", "ZIP", "ZCTA_reduced", "ZCTA"]:
        if name in df.columns: return name
    raise KeyError("Expected one of: ZIP_reduced, ZIP, ZCTA_reduced, ZCTA")

def parse_marginals_arg(arg: str) -> List[Tuple[str, ...]]:
    if not arg.strip(): return []
    out: List[Tuple[str, ...]] = []
    for spec in arg.split(";"):
        spec = spec.strip()
        if not spec: continue
        cols = tuple(c.strip() for c in spec.split("+") if c.strip())
        if cols: out.append(cols)
    return out

def ensure_bins(df: pd.DataFrame) -> pd.DataFrame:
    """If Age_bin/Income_bin are missing, recreate them in the same way as training."""
    out = df.copy()
    if "Age" in out.columns and "Age_bin" not in out.columns:
        out["Age_bin"] = pd.cut(out["Age"], bins=[18,30,40,50,60,70], labels=False, include_lowest=True)
    if "Income" in out.columns and "Income_bin" not in out.columns:
        # avoid errors if too few unique values
        try:
            out["Income_bin"] = pd.qcut(out["Income"], q=5, labels=False, duplicates="drop")
        except Exception:
            out["Income_bin"] = pd.cut(out["Income"], bins=5, labels=False, include_lowest=True)
    return out

def main() -> None:
    args = parse_args()

    real = pd.read_csv(args.real)
    syn  = pd.read_csv(args.syn)

    # recreate bins if someone passed raw files
    real = ensure_bins(real)
    syn  = ensure_bins(syn)

    geo = resolve_geo_column(real)
    default_marginals = [(geo,), (geo, "Age_bin"), (geo, "Income_bin"), (geo, "Age_bin", "Category")]
    user_marginals = parse_marginals_arg(args.marginals)
    marginals: List[Tuple[str, ...]] = user_marginals if user_marginals else default_marginals

    missing_cols = set()
    for cols in marginals:
        for c in cols:
            if c not in real.columns or c not in syn.columns: missing_cols.add(c)
    if missing_cols:
        raise KeyError(f"Missing required columns in inputs: {sorted(missing_cols)}")

    rows = []
    for cols in marginals:
        rows.append({"marginal": "+".join(cols), "L1_error": l1_error(real, syn, cols)})

    df_out = pd.DataFrame(rows)
    mean_l1 = float(df_out["L1_error"].mean())
    summary = {
        "sdnist_available": SDNIST_AVAILABLE,
        "sdnist_version": SDNIST_VERSION,
        "n_marginals": len(marginals),
        "mean_L1_error": mean_l1,
    }

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w") as f:
        json.dump({"summary": summary, "per_marginal": rows}, f, indent=2)

    sdnist_line = "yes" if SDNIST_AVAILABLE else "no"
    if SDNIST_AVAILABLE and SDNIST_VERSION: sdnist_line += f" (version={SDNIST_VERSION})"
    print("SDNist:", sdnist_line)
    for r in rows: print(f"{r['marginal']}: L1 = {r['L1_error']:.4f}")
    print(f"Mean L1 Error = {mean_l1:.4f}")
    print(f"Wrote {args.out_csv}"); print(f"Wrote {args.out_json}")

if __name__ == "__main__":
    main()