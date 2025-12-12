# Machine Unlearning for Single-Cell VAEs

STAT 4243 Final Project - Columbia University, Fall 2025

## Project Overview

This study examines whether specific training samples can be "forgotten" from a VAE such that membership inference attacks cannot distinguish forgotten cells from truly unseen cells. Two approaches are compared:

1. **Adversarial Unlearning**: Train the VAE to fool a membership inference attacker using extra-gradient optimization
2. **Fisher Information Scrubbing**: Perturb model parameters based on their influence on specific samples

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the data

The raw 10x Genomics PBMC data is stored in a compressed format. Unzip it before running:

```bash
cd data/filtered_gene_bc_matrices
unzip hg19.zip
cd ../..
```

## Project Structure

```
.
├── data/                          # Data files
│   ├── filtered_gene_bc_matrices/ # Raw 10x data (unzip hg19.zip first)
│   └── adata_processed.h5ad       # Preprocessed AnnData object
├── src/                           # Source code
│   ├── vae.py                     # VAE model definition
│   ├── attacker.py                # Membership inference attacker
│   ├── train.py                   # VAE training
│   ├── train_unlearn_extragradient.py  # Extra-gradient unlearning
│   └── train_fisher_unlearn.py    # Fisher unlearning
├── notebooks/                     # Jupyter notebooks (run in order)
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_vae.ipynb
│   ├── 03_privacy_audit.ipynb
│   ├── 04_adversarial_unlearning.ipynb
│   ├── 05_fisher_unlearning.ipynb
│   ├── 06_theory_analysis.ipynb
│   ├── 07_utility_evaluation.ipynb
│   ├── 08_mog_simulations.ipynb
│   ├── 09_ablations.ipynb
│   └── 10_final_results.ipynb
├── scripts/                       # Shell scripts used during development
├── outputs/                       # Model checkpoints and results
├── reports/                       # Figures and tables
└── report/                        # LaTeX report
```

## Reproducing Results

Run the notebooks in numerical order:

```bash
jupyter notebook notebooks/
```

1. `01_data_preparation.ipynb` - Load and preprocess PBMC-33k data
2. `02_baseline_vae.ipynb` - Train baseline VAE model
3. `03_privacy_audit.ipynb` - Evaluate baseline membership inference vulnerability
4. `04_adversarial_unlearning.ipynb` - Run adversarial unlearning experiments
5. `05_fisher_unlearning.ipynb` - Run Fisher scrubbing experiments
6. `06_theory_analysis.ipynb` - Theoretical analysis of unlearning
7. `07_utility_evaluation.ipynb` - Evaluate model utility after unlearning
8. `08_mog_simulations.ipynb` - Mixture of Gaussians toy experiments
9. `09_ablations.ipynb` - Ablation studies and hyperparameter analysis
10. `10_final_results.ipynb` - Summary of all results

## Key Results

| Method | Post-hoc AUC | Status |
|--------|--------------|--------|
| Baseline | 0.769 | LEAK |
| Frozen Adversarial | >0.98 | FAIL |
| Extra-gradient λ=10 | 0.482 | SUCCESS |
| Fisher (scattered) | 0.499 | SUCCESS |
| Fisher (structured) | 0.814 | FAIL |
| Retrain (floor) | 0.481 | TARGET |

## Hardware Requirements

- GPU recommended (CUDA-compatible)
- ~16GB RAM
- ~10GB disk space

Training times on a single GPU:
- Baseline VAE: ~10 minutes
- Extra-gradient unlearning: ~42 minutes
- Fisher unlearning: ~2 minutes
