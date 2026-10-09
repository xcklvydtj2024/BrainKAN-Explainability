# BrainKAN-Explainability: A Null-Calibrated Framework for Evaluating Nonlinear Neural Representations Under Cross-Cohort Distribution Shift

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **🔔 IMPORTANT:** 
> **For rigorous mathematical formulations of the support overlap, null-calibrated standardized separation ($Z$-score), and cross-cohort distribution shift quantification, please refer to the [Methodology Whitepaper (PDF)](./BrainKAN_Mathematical_Methodology.pdf).**

This repository provides an evaluation framework for assessing the reliability and identifiability of nonlinear representations learned by Kolmogorov-Arnold Networks (KAN) in task-fMRI connectomics. 

Rather than directly interpreting learned nonlinear geometry as biological signal, this project investigates how nonlinear representations are affected by reference-domain selection and cross-cohort distribution shifts.

## 📌 Main Research Question
**When a nonlinear neural representation changes across datasets, does this change reflect genuine task-associated structure, or reference-domain/model-induced geometric variation?**

## 🎯 Main Findings (HCP $\rightarrow$ CHCP Transfer)

| Evaluation | Observation | Interpretation |
|----------|--------|----------------|
| **Raw Nonlinear Geometry** | Highly sensitive to reference domain | Naive interpretation is confounded by input shifts |
| **Frozen Reference Transfer** | Increased apparent nonlinearity | Extrapolation artifacts mimic structural changes |
| **Null Model Calibration** | Show synchronous geometric inflation ($G_{Frozen} > G_{Native}$) | Identifies reference shift as the primary driver of apparent change |
| **Standardized Separation ($Z$)** | Highly stable across reference conditions | Successfully isolates task-associated variation from model-induced artifacts |

## 🌟 The Core Scientific Narrative

```text
                  BrainKAN
                     │
          High-capacity nonlinear model
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 Nonlinear geometry ≠ automatically interpretable
                                │
                                ▼
         Distribution shift may alter geometry (HCP → CHCP)
                                │
                                ▼
               Need identifiability evaluation
                                │
                                ▼
             Reference + Null calibration framework
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
         Raw Geometry                 Standardized Separation (Z-score)
        (Highly Sensitive)               (Cross-Cohort Stable)
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                 IDENTIFIABILITY BOUNDARY
```

## 🔬 Framework Components

- **1. Fixed Reference Domain**: A shared reference domain is constructed from training data (HCP) only, preventing changes in operating distributions from being conflated with nonlinear geometry.
- **2. Support Overlap ($O_j$)**: Quantifies the potential extrapolation risk when applying a frozen reference grid to the empirical support of the target (CHCP) data.
- **3. Null Model Calibration**: Label-shuffled models are trained to estimate baseline nonlinear deformation caused by model flexibility and data structure.
- **4. Standardized Separation ($Z$-score)**: Measures task-associated separation relative to null expectation, providing a highly stable metric against cross-cohort shift.

## 🚀 Reproducibility

Full reproduction requires substantial compute. You can run all scripts automatically using the provided Makefile.

```bash
make tests      # Run unit tests (requires pytest)
make baseline   # Run baseline GCN/GAT adequacy check
make results    # Re-run all analyses
```

### 📂 Key Implementation Files
- `experiments/chcp_external_validation.py` - Core cross-cohort calibration logic.
- `experiments/chcp_support_overlap.py` - Extrapolation risk quantification.
- `experiments/print_chcp_2x2.py` - 2x2 Identifiability matrix extraction.

## ⚖️ Limitations & Future Work

- **Single Transfer Setting**: This repository presents an empirical evaluation framework currently demonstrated on a single cross-cohort transfer setting (HCP $\rightarrow$ CHCP).
- **Null Sample Size**: Calibration currently operates on $R_{null}=100$ due to computational constraints.
- **Factorial Benchmark**: Future work should implement a fully factorial mechanism × input-shift benchmark to disentangle interaction effects perfectly across multiple cohorts.

## 📚 Citation

If you use this repository, please cite:

```bibtex
@article{BrainKAN2026,
  title={Assessing Identifiability of Nonlinear Neural Representations Under Cross-Cohort Distribution Shift: A Null-Calibrated Framework for BrainKAN Interpretation},
  author={Xin Qi et al.},
  journal={TBD},
  year={2026}
}
```
