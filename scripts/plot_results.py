import pandas as pd, matplotlib.pyplot as plt

data = {
    "Marginal": ["ZIP","ZIP×Age","ZIP×Income","ZIP×Age×Cat"],
    "L1_Error": [0.10,0.18,0.20,0.23]  # REPLACE with printed values
}
df = pd.DataFrame(data)

plt.bar(df["Marginal"], df["L1_Error"], color="#1f77b4")
plt.title("Workload L₁ Error after DP Heavy-Hitter Reduction")
plt.ylabel("L₁ Error")
plt.ylim(0, max(df["L1_Error"]) + 0.05)
plt.tight_layout()
plt.savefig("../results/l1_error_plot.png")
plt.show()
