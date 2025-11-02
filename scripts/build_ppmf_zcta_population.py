
#!/usr/bin/env python3
# build_ppmf_zcta_population.py
#
# Build a small `ppmf_zcta_population.csv` using the Census 2020 Decennial
# PL (redistricting) API, pulling total population by ZCTA5 (P1_001N).
#
# Usage (API mode):
#   python build_ppmf_zcta_population.py --out ../data/ppmf_zcta_population.csv
#
# Usage (local validation/normalize mode):
#   python build_ppmf_zcta_population.py --from_local path/to/zcta_pop.csv --out ../data/ppmf_zcta_population.csv
#
# Notes:
# - API endpoint: https://api.census.gov/data/2020/dec/pl
# - We request: NAME,P1_001N and geography: for=zip code tabulation area:*  (all ZCTAs)
# - Output columns: ZCTA (5-digit, zero-padded), population (int)
#
import argparse
import time
from typing import List, Tuple
from pathlib import Path

import pandas as pd

API_URL = "https://api.census.gov/data/2020/dec/dhc"
API_PARAMS = {
    "get": "NAME,P1_001N",
    "for": "zip code tabulation area:*",
}

def parse_args():
    p = argparse.ArgumentParser(description="Build ppmf_zcta_population.csv from Census 2020 PL API or a local CSV.")
    p.add_argument("--out", default="../data/ppmf_zcta_population.csv", help="Output CSV path (ZCTA,population).")
    p.add_argument("--from_local", default="", help="Optional: path to a local CSV with ZCTA and population to validate/normalize.")
    p.add_argument("--retries", type=int, default=3, help="Number of retries for the API request.")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between retries.")
    return p.parse_args()

def fetch_from_api(retries: int = 3, sleep_s: float = 1.0) -> pd.DataFrame:
    import requests  # local import to fail gracefully if missing
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(API_URL, params=API_PARAMS, timeout=60)
            r.raise_for_status()
            rows = r.json()
            header = rows[0]
            data = rows[1:]
            NAME_idx = header.index("NAME")
            POP_idx = header.index("P1_001N")
            ZCTA_idx = header.index("zip code tabulation area")
            out: List[Tuple[str, int]] = []
            for row in data:
                pop = int(row[POP_idx])
                zcta = str(row[ZCTA_idx]).zfill(5)
                out.append((zcta, pop))
            df = pd.DataFrame(out, columns=["ZCTA", "population"])
            df = df.sort_values(["ZCTA", "population"], ascending=[True, False]).drop_duplicates("ZCTA")
            return df
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to fetch from Census API after {retries} retries: {last_err}")

def normalize_local(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Accept a variety of common column names
    colmap = {c.lower(): c for c in df.columns}
    z_col = None
    p_col = None
    for cand in ["zcta", "zcta5", "zip", "zip_code", "geoid", "geoid10"]:
        if cand in colmap:
            z_col = colmap[cand]
            break
    for cand in ["population", "pop", "p1_001n", "total"]:
        if cand in colmap:
            p_col = colmap[cand]
            break
    if z_col is None or p_col is None:
        raise ValueError("Local CSV must contain ZCTA and population columns (e.g., ZCTA/ZCTA5 and population/P1_001N).")
    out = pd.DataFrame({
        "ZCTA": df[z_col].astype(str).str.zfill(5),
        "population": df[p_col].astype(int)
    })
    out = out.groupby("ZCTA", as_index=False)["population"].sum()
    return out

def main():
    args = parse_args()
    if args.from_local:
        df = normalize_local(args.from_local)
    else:
        df = fetch_from_api(retries=args.retries, sleep_s=args.sleep)

    if df.empty:
        raise ValueError("No rows found for ZCTA population.")
    if (df["population"] < 0).any():
        raise ValueError("Negative population found, aborting.")
    df = df.sort_values("ZCTA")
    out_path = args.out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df)} ZCTAs.")

if __name__ == "__main__":
    main()
