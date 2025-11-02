#!/usr/bin/env python3
"""
Differentially Private Heavy-Hitters for ZIP codes.

- Input : ../data/synthetic_dataset.csv  (columns: ZIP, Age, Income, Category)
- Output: ../data/dataset_reduced.csv    (adds ZIP_reduced; non-HH -> "OTHER")
         ../results/heavy_hitters.csv    (list of kept ZIPs)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="DP heavy-hitters for ZIP")
    p.add_argument("--input",  default="../data/synthetic_dataset.csv")
    p.add_argument("--out_data", default="../data/dataset_reduced.csv")
    p.add_argument("--out_hh",   default="../results/heavy_hitters.csv")
    # defines 3 file paths that can be overrided (where to read, write reduced, write heavy hitters)
    p.add_argument("--epsilon_hh", type=float, default=0.05,
                   help="Privacy budget ε used for HH step (Laplace).")
    p.add_argument("--topk", type=int, default=500,
                   help="Number of heavy-hitter ZIPs to keep.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()
    # adds tunable params: epislon_hh = budget for this step, topk = how many ZIPs to retain, seed = random number seed for reproducibility

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    # Parses arguments and creates a NumPy random generator seeded for repeatability.

    # --- I/O setup
    Path(args.out_data).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_hh).parent.mkdir(parents=True, exist_ok=True)

    # --- Load dataset
    df = pd.read_csv(args.input)
    if "ZIP" not in df.columns:
        raise ValueError(f"Expected column 'ZIP' in {args.input}; got {df.columns.tolist()}")
    #Sanity-check: aborts if the dataset doesn’t contain a ZIP column.


    # Normalize ZIP format to 5 chars (in case upstream changed)
    df["ZIP"] = df["ZIP"].astype(str).str.zfill(5)

    # --- True counts
    counts = df["ZIP"].value_counts().sort_index()
    # Creates a Series mapping each ZIP → number of rows (true counts), sorted by ZIP so that noise aligns.

    # --- DP noisy counts (Laplace with scale 1/ε)
    eps = float(args.epsilon_hh)
    if eps <= 0:
        raise ValueError("--epsilon_hh must be > 0")
    # Reads ε from arguments and ensures it’s positive.
    noise = rng.laplace(loc=0.0, scale=1.0/eps, size=len(counts))
    noisy = counts.astype(float) + noise
    noisy[noisy < 0] = 0.0  # no negative counts
    # Adds Laplace noise with mean 0 and scale 1/ε to each ZIP’s count, producing noisy counts that satisfy ε-DP. Then clips negatives to zero because counts can’t be negative.

    # --- Select top-K noisy ZIPs (the private heavy hitters)
    k = min(int(args.topk), len(noisy))
    top_zip = noisy.sort_values(ascending=False).head(k).index.tolist()

    # --- Reduce domain
    df["ZIP_reduced"] = df["ZIP"].where(df["ZIP"].isin(top_zip), other="OTHER")
    # Creates a new column: keeps ZIPs if they’re in the top-K list, replaces all others with "OTHER". This collapses the long tail so MWEM+PGM runs on a small finite domain.
    
    # --- Save outputs
    df.to_csv(args.out_data, index=False)
    pd.Series(top_zip, name="ZIP").to_csv(args.out_hh, index=False)

    # --- Report
    kept = df["ZIP_reduced"].nunique()
    orig = df["ZIP"].nunique()
    other_share = (df["ZIP_reduced"] == "OTHER").mean()
    print(f"DP-HH done | ε_hh={eps} | topK={k}")
    print(f"   Unique ZIPs: {orig}  →  kept: {kept} (includes 'OTHER')")
    print(f"   'OTHER' share of rows: {other_share:.2%}")
    print(f"   Wrote: {args.out_data}")
    print(f"   Wrote: {args.out_hh}")

    # Computes: how many unique ZIPs were kept (including "OTHER"), how many existed originally, what fraction of rows got lumped into "OTHER".

if __name__ == "__main__":
    main()
