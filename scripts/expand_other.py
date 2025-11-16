
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Expand OTHER bucket to concrete ZIP values.")
    p.add_argument("--real", default="../data/dataset_reduced.csv", help="Real (post-HH) dataset; has ZIP & ZIP_reduced.")
    p.add_argument("--syn",  default="../results/synthetic.csv", help="Synthetic dataset to expand (has ZIP_reduced).")
    p.add_argument("--out",  default="../results/synthetic_expanded.csv", help="Output CSV with ZIP_imputed/ZIP_final.")
    p.add_argument("--strategy", choices=["uniform"], default="uniform") # only randomly select ZIPs for now
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    real = pd.read_csv(args.real); syn = pd.read_csv(args.syn)

    if "ZIP_reduced" not in syn.columns:
        raise KeyError("Expected 'ZIP_reduced' in synthetic CSV.")
    if "ZIP" not in real.columns or "ZIP_reduced" not in real.columns:
        raise KeyError("Real dataset must include 'ZIP' and 'ZIP_reduced'.")
# ensure required columns are present: ZIP and ZIP_reduced 

    tail_pool = real.loc[real["ZIP_reduced"] == "OTHER", "ZIP"].astype(str).str.zfill(5).unique()
    # filters rows binned in OTHER, sleect original ZIP column form them
    if len(tail_pool) == 0:
        syn["ZIP_imputed"] = np.nan
        syn["ZIP_final"] = syn["ZIP_reduced"]
    # if there are no OTHER rows
    else:
        rng = np.random.default_rng(args.seed)
        # rng
        is_other = syn["ZIP_reduced"] == "OTHER"
        # boolean mask over synthetic OTHER rows
        n_other = int(is_other.sum())
        # how many such rows exist
        choices = rng.choice(tail_pool, size=n_other, replace=True)
        # unfirmonly samples OTHER cols
        zip_imputed = np.full(len(syn), fill_value=np.nan, dtype=object)
        # create object w same length as synthetic dataset
        zip_imputed[is_other.to_numpy()] = choices
        # plug sampled ZIPs into positions where synthetic rows are OTHER
        syn["ZIP_imputed"] = zip_imputed
        # save as new col
        syn["ZIP_final"] = syn["ZIP_reduced"].where(~is_other, syn["ZIP_imputed"])
    # builds final zip: for non other rows, keep orginal reduced value and for OTHER rows, use the imputed concrete ZIP
    syn["ZIP_final"] = syn["ZIP_final"].astype(str)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    syn.to_csv(args.out, index=False)
    print(f"Expanded OTHER rows written to: {args.out}")

if __name__ == "__main__":
    main()

# #   python expand_other.py #       --real ../data/dataset_reduced.csv #       --syn  ../results/synthetic.csv #       --ppmf ../data/ppmf_zcta_population.csv #       --strategy uniform|public-weighted #       --out  ../results/synthetic_expanded.csv #       --seed 42
# import argparse
# import numpy as np
# import pandas as pd
# from pathlib import Path

# def parse_args():
#     p = argparse.ArgumentParser(description="Expand OTHER bucket to concrete ZCTA values.")
#     p.add_argument("--real", default="../data/dataset_reduced.csv", help="Real (post-HH) dataset; used to infer tail ZCTAs.")
#     p.add_argument("--syn",  default="../results/synthetic.csv", help="Synthetic dataset to expand.")
#     p.add_argument("--ppmf", default="../data/ppmf_zcta_population.csv", help="Public ZCTA populations (required for public-weighted).")
#     p.add_argument("--strategy", choices=["uniform","public-weighted"], default="uniform", help="Sampling rule for tail ZCTAs.")
#     p.add_argument("--out", default="../results/synthetic_expanded.csv", help="Output CSV with ZCTA_imputed/ZCTA_final.")
#     p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
#     return p.parse_args()

# def main():
#     args = parse_args()

#     real = pd.read_csv(args.real)
#     syn  = pd.read_csv(args.syn)

#     # Validate expected columns
#     if "ZCTA_reduced" not in syn.columns:
#         raise KeyError("Expected 'ZCTA_reduced' in synthetic CSV. Run heavy_hitters and MWEM steps first.")
#     if "ZCTA" not in real.columns or "ZCTA_reduced" not in real.columns:
#         raise KeyError("Real dataset must include both 'ZCTA' and 'ZCTA_reduced' (as produced by heavy_hitters).")

#     # Identify tail pool: concrete ZCTAs that were mapped to OTHER in the reduced real data
#     tail_pool = real.loc[real["ZCTA_reduced"] == "OTHER", "ZCTA"].astype(str).str.zfill(5).unique()
#     if len(tail_pool) == 0:
#         raise ValueError("Tail pool is empty: no rows mapped to OTHER in real data. Nothing to expand.")

#     # Build weights if needed
#     weights = None
#     if args.strategy == "public-weighted":
#         ppmf = pd.read_csv(args.ppmf)
#         if not {"ZCTA","population"}.issubset(ppmf.columns):
#             raise KeyError("PPMF CSV must have columns: ZCTA, population")
#         ppmf["ZCTA"] = ppmf["ZCTA"].astype(str).str.zfill(5)
#         tail_df = pd.DataFrame({"ZCTA": tail_pool})
#         tail_df = tail_df.merge(ppmf[["ZCTA","population"]], on="ZCTA", how="left").fillna({"population": 1.0})
#         weights = tail_df["population"].to_numpy(dtype=float)
#         weights = weights / weights.sum()
#         tail_pool = tail_df["ZCTA"].to_numpy()

#     # RNG
#     rng = np.random.default_rng(args.seed)

#     syn = syn.copy()
#     is_other = syn["ZCTA_reduced"] == "OTHER"
#     n_other = int(is_other.sum())
#     if n_other == 0:
#         # Nothing to expand; just create passthrough columns
#         syn["ZCTA_imputed"] = np.nan
#         syn["ZCTA_final"] = syn["ZCTA_reduced"]
#     else:
#         # Sample tail ZCTAs for OTHER rows
#         if weights is None:
#             choices = rng.choice(tail_pool, size=n_other, replace=True)
#         else:
#             choices = rng.choice(tail_pool, size=n_other, replace=True, p=weights)

#         zcta_imputed = np.full(len(syn), fill_value=np.nan, dtype=object)
#         zcta_imputed[is_other.to_numpy()] = choices
#         syn["ZCTA_imputed"] = zcta_imputed

#         # Construct ZCTA_final
#         syn["ZCTA_final"] = syn["ZCTA_reduced"].where(~is_other, syn["ZCTA_imputed"])

#     # Ensure string types
#     syn["ZCTA_final"] = syn["ZCTA_final"].astype(str)

#     # Save
#     Path(args.out).parent.mkdir(parents=True, exist_ok=True)
#     syn.to_csv(args.out, index=False)
#     print(f"Expanded OTHER for {n_other} rows using strategy='{args.strategy}'. Wrote: {args.out}")

# if __name__ == "__main__":
#     main()
