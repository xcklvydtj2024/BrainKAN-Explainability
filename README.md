# BrainKAN-Explainability
## A Null-Calibrated Framework for Evaluating Nonlinear Neural Representations Under Cross-Cohort Distribution Shift

[![License: MIT](https://img.shields.io/badge/License--MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python--3.10%2B-blue.svg)](https://www.python.org/downloads/)

> **🔔 IMPORTANT:** 
> **For rigorous mathematical formulations of the support overlap, null-calibrated standardized separation ($Z$-score), and cross-cohort distribution shift quantification, please refer to the [Methodology Whitepaper (PDF)](./BrainKAN_Mathematical_Methodology.pdf).**

This repository provides an evaluation framework for assessing the reliability and identifiability of nonlinear representations learned by Kolmogorov-Arnold Networks (KAN) in task-fMRI connectomics. 

Rather than directly interpreting learned nonlinear geometry as biological signal, this project investigates how nonlinear representations are affected by reference-domain selection and cross-cohort distribution shifts.

## 📌 Main Research Question
**When a nonlinear neural representation changes across datasets, does this change reflect genuine task-associated structure, or reference-domain/model-induced geometric variation?**

## 🎯 Main Findings (Cross-cohort Reference Substitution)

*Note: The cross-cohort evaluation in this repository tests **reference domain substitution** (applying a source cohort's reference grid to a target-cohort-trained model) to isolate extrapolation distortion, rather than zero-shot model transfer.*

| Evaluation | Observation | Interpretation |
|----------|--------|----------------|
| **Raw Nonlinear Geometry** | Highly sensitive to reference domain | Raw geometry contains reference-dependent components |
| **Frozen Reference Transfer** | Increased apparent nonlinearity | Indicates potential extrapolation-related distortion |
| **Null Model Calibration** | Similar inflation under shifted reference | Suggests a substantial reference-induced contribution |
| **Standardized Separation ($Z$)** | More stable across reference conditions | Provides a stable task-associated separation relative to null expectation |

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
                 IDENTIFIABILITY ASSESSMENT
```

## 🔬 Framework Components

- **1. Fixed Reference Domain**: A shared reference domain is constructed exclusively from the training distribution to prevent operating-regime differences from being conflated with nonlinear geometry.
- **2. Support Overlap ($O_j$)**: Quantifies the potential extrapolation risk when applying a frozen reference grid to the empirical support of the target (CHCP) data.
- **3. Null Model Calibration**: Label-shuffled models are trained to estimate baseline nonlinear deformation caused by model flexibility and data structure.
- **4. Standardized Separation ($Z$-score)**: Measures task-associated separation relative to null expectation, providing a highly stable metric against cross-cohort shift.

## 💾 Data Availability

Preprocessed examples and scripts are provided where permitted. Raw HCP and CHCP data access follows the respective data-use agreements. 
However, the **CHCP** (Chinese Human Connectome Project) dataset is currently restricted due to privacy and data-sharing agreements. Thus, the cross-cohort validation scripts (`chcp_*.py`) cannot be run directly out-of-the-box without authorization. We provide the scripts for transparency and methodology verification.

## 📊 Predictive Adequacy Check

To ensure BrainKAN successfully captures task-related information, we benchmarked it against standard GNNs (GCN, GAT). 
*Note: The primary goal is **not predictive supremacy**, but establishing predictive adequacy. The capacity gap (BrainKAN ~590k params vs standard GCN ~300 params) means predictive comparisons should be interpreted as adequacy checks rather than claims of universal predictive superiority. (We also include high-capacity GCN/GAT baselines (512+ hidden channels) in our code for fairness).*

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

## ⚖️ Limitations & Statistical Caveats

- **Reference vs. Model Transfer**: The current implementation validates reference domain substitution on target-trained models. Future work must strictly differentiate this from frozen model weight transfer.
- **Tail Approximation Uncertainty**: To mitigate high compute costs, the Generalized Pareto Distribution (GPD) tail fit relies on only $R_{null}=100$ null models. This provides a very sparse tail (approx. 10 data points) for parameter estimation, leading to high uncertainty in the extrapolated extreme p-values.
- **Pseudo-replication in Q3C Inference**: The effective gain difference tests currently operate at the subject level. Because subjects in the same CV fold are evaluated on the same trained model, these observations are not strictly independent, which may artificially inflate degrees of freedom and overstate significance. Future implementations should adopt fold-level inference or mixed-effects models.
- **Factorial Benchmark**: Future work should implement a fully factorial mechanism × input-shift benchmark to disentangle interaction effects perfectly across multiple cohorts.

## 📚 Citation

If you use this repository, please cite:

```bibtex
@misc{BrainKAN2026,
  title={Assessing Identifiability of Nonlinear Neural Representations Under Cross-Cohort Distribution Shift: A Null-Calibrated Framework for BrainKAN Interpretation},
  author={Xin Qi et al.},
  note={Preprint / Independent Research Project},
  year={2026}
}
```
