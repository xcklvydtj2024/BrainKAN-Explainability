# BrainKAN-Explainability

## Null-Calibrated Evaluation of Nonlinear Neural Representations Under Cross-Cohort Distribution Shift

### Overview

BrainKAN-Explainability is a framework for evaluating the identifiability of nonlinear neural representations learned by KAN models under cross-cohort distribution shifts.

This repository provides an evaluation framework for assessing the reliability of nonlinear representations learned by Kolmogorov-Arnold Networks (KAN) in task-fMRI connectomics. 

Rather than directly interpreting learned nonlinear geometry as biological signal, this project investigates how nonlinear representations are affected by reference-domain selection and cross-cohort distribution shifts.

### Scientific Motivation

High-capacity neural models can learn complex nonlinear functions from brain connectivity data. 
However, nonlinear geometric properties learned by these models may reflect a mixture of task-associated variation and model/reference-dependent deformation.

This project investigates whether nonlinear representations remain identifiable under cross-cohort distribution shifts.

### Research Question

When a nonlinear neural representation changes across datasets, does this change reflect:

1. genuine task-associated structure?
2. or reference-domain/model-induced geometric variation?

### Method Overview

#### 1. Fixed Reference Domain
A shared reference domain is constructed from training data only.
This prevents changes in operating distributions from being conflated with nonlinear geometry.

#### 2. Null Model Calibration
Label-shuffled models are trained to estimate baseline nonlinear deformation caused by model flexibility and data structure.
The raw geometry is calibrated relative to this empirical null distribution.

#### 3. Standardized Separation
The Z-score measures task-associated separation relative to null expectation.

$$
Z = \frac{G_{real} - \mu(G_{null})}{\sigma(G_{null})}
$$

### Cross-Cohort Case Study
We evaluate the framework using an HCP $\rightarrow$ CHCP transfer scenario.
CHCP serves as an independent cohort with substantial distribution discrepancy relative to HCP.

### Main Findings

| Evaluation                | Observation                             |
| ------------------------- | --------------------------------------- |
| Raw nonlinear geometry    | Highly sensitive to reference domain    |
| Frozen reference transfer | Increased apparent nonlinearity         |
| Null models               | Show similar geometric inflation        |
| Null-calibrated Z-score   | More stable across reference conditions |

### Repository Structure

```
BrainKAN-Explainability/

├── data/
│   └── dataset preparation scripts

├── models/
│   ├── kan.py
│   └── baseline_models.py

├── experiments/
│   ├── 00_baseline_models.py
│   ├── 01_q1_q2_q3a_naive_analysis.py
│   ├── 02_q3b_common_reference_geometry.py
│   ├── 03_synthetic_identifiability_benchmark.py
│   ├── 04_q3c_effective_gain.py
│   ├── 05_q2_null_taxonomy.py
│   ├── chcp_external_validation.py
│   ├── chcp_support_overlap.py
│   └── print_chcp_2x2.py

├── stats_utils/
│   └── Statistical testing modules

├── brainkan_math_derivations.tex
├── README.md
└── Makefile
```

### Limitations

This repository presents an empirical evaluation framework demonstrated on a single cross-cohort transfer setting (HCP $\rightarrow$ CHCP).

The null-calibration approach assumes that label-shuffling preserves nuisance-related geometric variation while removing task-associated structure. Future work will investigate additional synthetic and multi-cohort validations.

### Citation

If you use this repository, please cite:

```bibtex
@article{BrainKAN2026,
  title={Assessing Identifiability of Nonlinear Neural Representations Under Cross-Cohort Distribution Shift: A Null-Calibrated Framework for BrainKAN Interpretation},
  author={Xin Qi et al.},
  journal={TBD},
  year={2026}
}
```
