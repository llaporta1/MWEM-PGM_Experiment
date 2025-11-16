import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

METRICS = Path("results/metrics.json")
OUT_PNG = Path("results/l1_l2_error_plot.png")

with METRICS.open() as f:
    M = json.load(f)

label_map = {
    "ZIP_reduced": "ZIP",
    "ZIP_reduced+Age_bin": "ZIP×Age",
    "ZIP_reduced+Income_bin": "ZIP×Income",
    "ZIP_reduced+Age_bin+Category": "ZIP×Age×Cat",
}

order = [
    "ZIP_reduced",
    "ZIP_reduced+Age_bin",
    "ZIP_reduced+Income_bin",
    "ZIP_reduced+Age_bin+Category",
]

# Extract values
vals_l1 = {row["marginal"]: float(row["L1_error"]) for row in M["per_marginal"]}
vals_l2 = {row["marginal"]: float(row["L2_error"]) for row in M["per_marginal"]}

labels = [label_map[k] for k in order]
y_l1 = [vals_l1[k] for k in order]
y_l2 = [vals_l2[k] for k in order]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(6, 4))
plt.bar(x - width / 2, y_l1, width, label="L₁ Error")
plt.bar(x + width / 2, y_l2, width, label="L₂ Error")
plt.title("Workload L₁ and L₂ Errors after DP Heavy-Hitter Reduction")
plt.ylabel("Error")
plt.xticks(x, labels)
plt.ylim(0, max(max(y_l1), max(y_l2)) + 0.05)
plt.legend()
plt.tight_layout()

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=160)
print(f"saved {OUT_PNG}")