import argparse, json
from pathlib import Path
from typing import List, Sequence, Tuple, Optional, Callable
import numpy as np, pandas as pd

SDNIST_AVAILABLE = False; SDNIST_VERSION = None
_SD_L2_FN: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
try:
    import sdnist
    SDNIST_AVAILABLE = True
    try:
        SDNIST_VERSION = getattr(sdnist, "__version__", "unknown")
    except Exception:
        SDNIST_VERSION = "unknown"
    # Try to locate an L2/Euclidean distance helper inside SDNist (since API surface can vary by version)
    try:
        # common places/names to probe; we normalize vectors before passing
        candidates = []
        if hasattr(sdnist, "metrics"):
            m = sdnist.metrics
            for name in ["l2", "l2_distance", "euclidean", "L2", "euclidean_distance"]:
                if hasattr(m, name) and callable(getattr(m, name)):
                    candidates.append(getattr(m, name))
            for sub in ["distance", "distances", "metrics"]:
                if hasattr(m, sub):
                    submod = getattr(m, sub)
                    for name in ["l2", "l2_distance", "euclidean", "euclidean_distance"]:
                        if hasattr(submod, name) and callable(getattr(submod, name)):
                            candidates.append(getattr(submod, name))
        _SD_L2_FN = candidates[0] if candidates else None
    except Exception:
        _SD_L2_FN = None
except Exception:
    SDNIST_AVAILABLE = False

def _histogram_pair(df1: pd.DataFrame, df2: pd.DataFrame, cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    cols = list(cols)
    r = df1[cols].astype(str).groupby(cols).size()
    s = df2[cols].astype(str).groupby(cols).size()
    idx = r.index.union(s.index)
    r = r.reindex(idx, fill_value=0).astype(float)
    s = s.reindex(idx, fill_value=0).astype(float)
    r_sum, s_sum = r.sum(), s.sum()
    if r_sum == 0 or s_sum == 0:
        return np.array([]), np.array([])
    r = r / r_sum
    s = s / s_sum
    return r.values, s.values

def l1_error(df1: pd.DataFrame, df2: pd.DataFrame, cols: Sequence[str]) -> float:
    r, s = _histogram_pair(df1, df2, cols)
    if r.size == 0 or s.size == 0:
        return float("nan")
    return float(np.abs(r - s).sum())

def l2_error(df1: pd.DataFrame, df2: pd.DataFrame, cols: Sequence[str]) -> float:
    """Compute L2 (Euclidean) distance between normalized histograms for the given marginal.
    If SDNist provides an L2 function, use it; otherwise fall back to numpy."""
    r, s = _histogram_pair(df1, df2, cols)
    if r.size == 0 or s.size == 0:
        return float("nan")
    # Prefer SDNist if available and callable
    if SDNIST_AVAILABLE and callable(_SD_L2_FN):
        try:
            return float(_SD_L2_FN(r, s))
        except Exception:
            pass
    # Fallback: Euclidean distance
    return float(np.sqrt(((r - s) ** 2).sum()))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate synthetic data with L1/L2 over specified marginals (SDNist optional).")
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
        l1 = l1_error(real, syn, cols)
        l2 = l2_error(real, syn, cols)
        rows.append({"marginal": "+".join(cols), "L1_error": l1, "L2_error": l2})

    df_out = pd.DataFrame(rows)
    mean_l1 = float(df_out["L1_error"].mean())
    mean_l2 = float(df_out["L2_error"].mean())

    summary = {
        "sdnist_available": SDNIST_AVAILABLE,
        "sdnist_version": SDNIST_VERSION,
        "n_marginals": len(marginals),
        "mean_L1_error": mean_l1,
        "mean_L2_error": mean_l2,
        "used_sdnist_l2": bool(SDNIST_AVAILABLE and callable(_SD_L2_FN)),
    }

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w") as f:
        json.dump({"summary": summary, "per_marginal": rows}, f, indent=2)

    sdnist_line = "yes" if SDNIST_AVAILABLE else "no"
    if SDNIST_AVAILABLE and SDNIST_VERSION: sdnist_line += f" (version={SDNIST_VERSION})"
    print("SDNist:", sdnist_line, "| L2 via SDNist fn:", "yes" if summary["used_sdnist_l2"] else "no")
    for r in rows:
        print(f"{r['marginal']}: L1 = {r['L1_error']:.4f} | L2 = {r['L2_error']:.4f}")
    print(f"Mean L1 Error = {mean_l1:.4f}")
    print(f"Mean L2 Error = {mean_l2:.4f}")
    print(f"Wrote {args.out_csv}"); print(f"Wrote {args.out_json}")

if __name__ == "__main__":
    main()
