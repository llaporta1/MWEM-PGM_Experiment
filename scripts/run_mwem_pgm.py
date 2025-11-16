import os, sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level up from scripts/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
# time → how long the model fitting takes (runtime metric)
import pandas as pd
from private_pgm.mwem import MWEM
from private_pgm.workload import Marginals

df = pd.read_csv("data/dataset_reduced.csv")
# This dataset has: A new column ZIP_reduced (top-K ZIPs + “OTHER”), Continuous variables Age and Income, A categorical variable Category.

# discretize continuous vars
df["Age_bin"] = pd.cut(df["Age"], bins=[18,30,40,50,60,70], labels=False)
df["Income_bin"] = pd.qcut(df["Income"], q=5, labels=False)
df = df.dropna()
# So we “bin” Age and Income:,pd.cut() → splits ages into fixed-width bins (18–30, 30–40, …). pd.qcut() → splits income into 5 quantile bins (each containing ~20% of records).
 #.dropna() → removes any records that fell outside the defined ranges.

# Now each record has: ZIP_reduced | Age_bin | Income_bin | Category

cols = ["ZIP_reduced", "Age_bin", "Income_bin", "Category"]
df = df[cols]
# (Removes raw continuous variables since the binned versions replace them)

# domain: all unique values for each column
domain = {c: sorted(df[c].astype(str).unique().tolist()) for c in df.columns}
# Creates a domain = list of all possible values for each attribute.

# workload: which marginals to preserve
query_sets = [
    ("ZIP_reduced",),
    ("ZIP_reduced","Age_bin"),
    ("ZIP_reduced","Income_bin"),
    ("ZIP_reduced","Age_bin","Category"),
]

workload = Marginals(domain, query_sets)

# remaining budget
epsilon = 0.95  # 95 % after HH step
rounds = len(df.columns)
# rounds controls how many MWEM iterations to run (one per attribute)

mwem = MWEM(domain=domain, epsilon=epsilon, rounds=rounds)

t0 = time.time()
model = mwem.fit(df.to_records(index=False), workload)
t1 = time.time()
df.to_records(index=False) # converts the DataFrame to a NumPy structured array — MWEM expects this format.
#mwem.fit():
# I# teratively selects queries (from workload) where the synthetic model diverges most from the real data.
# Measures each selected query with Gaussian noise (DP).
# Updates the internal model (a Probabilistic Graphical Model) to match noisy measurements.
# Tracks runtime from start to finish.
# At the end, model is a fitted private model that encodes an estimated distribution 

syn = model.synthetic_data(rows=len(df))
syn.to_csv("results/synthetic.csv", index=False)

print(f"MWEM done in {t1 - t0:.1f}s with ε={epsilon}")
