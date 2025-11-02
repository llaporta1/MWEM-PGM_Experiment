**MWEM + PGM Experiment (Baseline)
**
This repository contains the code and results for **“MWEM + PGM Experiment”**, developed by Lauren LaPorta (University of Virginia, 2025).  
It implements a **differentially private synthetic data generation pipeline** using the MWEM + PGM framework with a *Flat (naïve) + Uniform sampling* baseline.  
The project accompanies the full write-up available on Overleaf:  
👉 [Read the paper here](https://www.overleaf.com/read/XXXXXXXXXXXX) ← *replace with your share link.*

---

**Overview**
The pipeline evaluates how differentially private heavy-hitter reduction enables MWEM + PGM to scale to high-cardinality categorical attributes such as ZIP codes.  
This baseline establishes a control case before extending to **Hierarchical** and **Similarity-based** bucketization methods and **weighted** sampling.

**Pipeline Steps
**
1. `make_dataset.py` – Generate a synthetic population dataset (100 k records).  
2. `heavy_hitters.py` – Apply DP heavy-hitter mechanism (εₕₕ = 0.05) to retain top K ZIP codes + “OTHER”.  
3. `run_mwem_pgm.py` – Train the MWEM + PGM model on the compressed domain (ε = 0.95).  
4. `expand_other.py` – Uniformly sample real ZIPs within the “OTHER” bucket.  
5. `evaluate_sdnist.py` – Compute workload L₁ error using the SDNist framework.  
6. `plot_results.py` – Visualize L₁ error across 1-, 2-, and 3-way marginals.

---

**Quick Start**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

1. Generate dataset
python scripts/make_dataset.py --zip_csv data/uszips.csv --out data/synthetic_dataset.csv --n 100000 --seed 42

2. Heavy hitters
python scripts/heavy_hitters.py --input data/synthetic_dataset.csv --out_data data/dataset_reduced.csv --out_hh results/heavy_hitters.csv --epsilon_hh 0.05 --topk 1000 --seed 42

3. Run MWEM + PGM
python scripts/run_mwem_pgm.py

4. Expand OTHER bucket
python scripts/expand_other.py --real data/dataset_reduced.csv --syn results/synthetic.csv --out results/synthetic_expanded.csv --strategy uniform --seed 42

5. Evaluate and plot
python scripts/evaluate_sdnist.py --real data/dataset_reduced.csv --syn results/synthetic.csv --out_csv results/metrics.csv --out_json results/metrics.json
python scripts/plot_results.py
