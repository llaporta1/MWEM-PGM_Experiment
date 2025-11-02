
#!/usr/bin/env python3
# make_dataset.py (ZIP-based version)
import argparse
import random
from datetime import date
import numpy as np
import pandas as pd
from faker import Faker

def parse_args():
    p = argparse.ArgumentParser(description="Build synthetic dataset using ZIP metadata + Faker attrs")
    p.add_argument("--zip_csv", default="../data/uszips.csv", help="CSV with a 'zip' column (5-digit, or convertible)")
    p.add_argument("--out", default="../data/synthetic_dataset.csv", help="Output CSV path")
    p.add_argument("--n", type=int, default=100_000, help="Number of synthetic rows to generate")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--locale", default="en_US", help="Faker locale")
    p.add_argument("--income_min", type=int, default=20_000)
    p.add_argument("--income_max", type=int, default=150_000)
    p.add_argument("--income_mu", type=float, default=10.9, help="Lognormal mu (log space)")
    p.add_argument("--income_sigma", type=float, default=0.45, help="Lognormal sigma (log space)")
    return p.parse_args()

def years_between(d1: date, d2: date) -> int:
    return d2.year - d1.year - ((d2.month, d2.day) < (d1.month, d1.day))

def main():
    args = parse_args()
    np.random.seed(args.seed); random.seed(args.seed); Faker.seed(args.seed)
    fake = Faker(args.locale)

    z = pd.read_csv(args.zip_csv)
    colmap = {c.lower(): c for c in z.columns}
    if "zip" not in colmap:
        raise ValueError(f"Expected a 'zip' column in {args.zip_csv}; found: {z.columns.tolist()}")
    z["ZIP"] = z[colmap["zip"]].astype(str).str.zfill(5)
    z = z.dropna(subset=["ZIP"]).copy()

    rng = np.random.default_rng(args.seed)
    weights = rng.zipf(a=2.0, size=len(z))
    weights = weights / weights.sum()

    N = int(args.n)
    sampled_zip = rng.choice(z["ZIP"].to_numpy(), size=N, p=weights)

    today = date.today()
    ages = []
    for _ in range(N):
        dob = fake.date_of_birth(minimum_age=18, maximum_age=80)
        ages.append(years_between(dob, today))
    ages = np.asarray(ages, dtype=int)

    incomes = np.random.lognormal(mean=args.income_mu, sigma=args.income_sigma, size=N).astype(float)
    incomes += np.random.uniform(0, 1000, size=N)
    incomes = np.clip(incomes, args.income_min, args.income_max).astype(int)

    cat_pool = ["A", "B", "C", "D"]; cat_probs = [0.4, 0.3, 0.2, 0.1]
    categories = np.random.choice(cat_pool, size=N, p=cat_probs)

    df = pd.DataFrame({"ZIP": sampled_zip, "Age": ages, "Income": incomes, "Category": categories})
    df.to_csv(args.out, index=False)
    print(f"Synthetic dataset created: {df.shape} -> {args.out}")

if __name__ == "__main__":
    main()

# import argparse
# import math
# import random
# from datetime import date

# import numpy as np
# import pandas as pd
# from faker import Faker


# def parse_args():
#     p = argparse.ArgumentParser(description="Build synthetic dataset using PPMF ZCTA weights + Faker attrs")
#     p.add_argument("--ppmf_csv", default="../data/ppmf_zcta_population.csv",
#                    help="CSV with columns: ZCTA, population (DP counts)")
#     p.add_argument("--out", default="../data/synthetic_dataset.csv",
#                    help="Output CSV path")
#     p.add_argument("--n", type=int, default=100_000,
#                    help="Number of synthetic rows to generate")
#     p.add_argument("--seed", type=int, default=42,
#                    help="Random seed for reproducibility")
#     p.add_argument("--locale", default="en_US",
#                    help="Faker locale") # fake language/region that affects names, dates, etc
#     # Optional knobs to shape the income distribution
#     p.add_argument("--income_min", type=int, default=20_000)
#     p.add_argument("--income_max", type=int, default=150_000)
#     p.add_argument("--income_mu", type=float, default=10.9,  # ≈ $54k mean if lognormal
#                    help="Lognormal mu (in log space)")
#     p.add_argument("--income_sigma", type=float, default=0.45,
#                    help="Lognormal sigma (in log space)")
#     return p.parse_args()


# def years_between(d1: date, d2: date) -> int: # age helper: computes whole years between 2 dates
#     """Whole years between two dates (for age)."""
#     return d2.year - d1.year - ((d2.month, d2.day) < (d1.month, d1.day))


# def main():
#     args = parse_args() # reading CLI args

#     # Seeding for reproducibility
#     np.random.seed(args.seed)
#     random.seed(args.seed)
#     Faker.seed(args.seed)
#     fake = Faker(args.locale)
#     # seeds all three random num generators so reruns produce identical results and instantiates a Faker w locale (tells Faker what cultural/linguistic style to use)

#     # Loads 2-column CSV - ZCTA populations (DP public stats)
#     z = pd.read_csv(args.ppmf_csv)
#     if not {"ZCTA", "population"}.issubset(z.columns):
#         raise ValueError("ppmf_zcta_population.csv must have columns: ZCTA, population")

#     z["ZCTA"] = z["ZCTA"].astype(str).str.zfill(5)
#     # normalize ZCTAs to 5-digit strings 
#     z = z.loc[z["population"].fillna(0) > 0].copy()
#     # drop nonpositive/NaN rows
#     weights = z["population"].astype(float).to_numpy()
#     weights = weights / weights.sum()
#     # build probability weights proportional to population (sum = 1)
#     zctas = z["ZCTA"].to_numpy()
#     # extracts ZCTA values as NumPy array

#     N = int(args.n)
#     #ensure n is an int

#     sampled_zcta = np.random.choice(zctas, size=N, p=weights)
#     # draws n ZTCAs proportionally to population (PPMF) based on probability weights - pure post processing and no privacy cost
#     # weights ae based off of how many people have each zipcode versus total dataset

#     # Note: Faker in a Python loop is fine for N≈100k, but if you want
#     # max speed, you can swap these for vectorized numpy draws later.

#     # Age: use date_of_birth for realistic distribution, clamp 18–80
#     today = date.today()
#     ages = []
#     for _ in range(N):
#         dob = fake.date_of_birth(minimum_age=18, maximum_age=80)
#         ages.append(years_between(dob, today))
#     ages = np.asarray(ages, dtype=int)
#     # generates realistic age by sampling a date of birth from Faker, then converts to integer years 

#     # Income: lognormal shape, clipped to [income_min, income_max]
#     # We add a tiny Faker-driven jitter so it's "Faker-touched"
#     incomes = np.random.lognormal(mean=args.income_mu, sigma=args.income_sigma, size=N)
#     incomes = incomes.astype(float)
#     incomes = incomes + np.fromiter((fake.pyfloat(left=0, right=1000) for _ in range(N)), dtype=float, count=N)
#     incomes = np.clip(incomes, args.income_min, args.income_max).astype(int)
#     # generates income w lognormal base common from income distributions and then adds Faker jitter (0-1000) 
#     # jitters are not privacy requirements, but for veariety and aesthetic (so not too smooth and repetifive)


#     # Category: use a few generic segment labels; choose with Faker
#     # (You can replace these with your own buckets)
#     cat_pool = ["A", "B", "C", "D"]
#     cat_probs = [0.4, 0.3, 0.2, 0.1]
#     # Faker-backed sampler:
#     categories = [fake.random_element(elements=dict(zip(cat_pool, cat_probs))) for _ in range(N)]
#     # creates simple categorical segement w fixed probs using Faker RNG

#     df = pd.DataFrame(
#         {
#             "ZCTA": sampled_zcta,
#             "Age": ages,
#             "Income": incomes,
#             "Category": categories,
#         }
#     )

#     df.to_csv(args.out, index=False)
#     print(f"Synthetic dataset created: {df.shape} → {args.out}")


# if __name__ == "__main__":
#     main()