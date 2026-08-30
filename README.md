# Reference code implementation (PyTorch)

## Implementation Overview

- heterogeneous temporal graph construction (cognitive, functional, risk nodes)
- UDS version harmonization (FORMVER 2 vs 3)
- heterogeneous GNN encoder with graph neural ODE
- temporal attention pooling
- multi-task heads (classification, concept supervision, progression risk)
- focal loss with gradient accumulation and early stopping
- 5-fold stratified group cross-validation (grouped by ADRC site)
- baseline comparisons (Logistic Regression, Random Forest, Gradient Boosting, MLP)
- explainability (SHAP, concept supervision metrics, temporal attention)

## Expected data format

Place a CSV file named `synthetic_dataset.csv` in the `data/` directory with standard NACC UDS variables including demographics, cognitive scores, functional assessments, comorbidities, and diagnosis (`NACCUDSD`).

Inclusion criteria applied: NACCUDSD in {1, 3, 4}, age >= 50, >= 2 longitudinal visits.

## Run training

```
python src/scripts/run_pipeline.py \
    --data-dir ./data \
    --output-dir ./results \
    --seed 42 \
    --n-epochs 30 \
    --patience 7
```

## Package requirements

```
pip install torch torch-geometric torchdiffeq numpy pandas scikit-learn scipy shap tqdm
```

## What this code does not do automatically

This repo intentionally focuses on the model and training system. It does not fully automate:

- NACC data access (requires a separate Data Use Agreement with https://naccdata.org)
- figure generation (results are saved as CSV/pickle for downstream plotting)
- hyperparameter search (configuration in `config.py`)
- raw EHR data preprocessing beyond the NACC UDS format

Those are separate stages and are best handled as dedicated scripts.

## Data privacy

This repository contains **no real patient data**. The included synthetic dataset uses only publicly documented NACC UDS variable ranges.
