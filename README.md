# BrainKAN-Interpretability-Benchmark
### *Can neural network nonlinearities reveal biological mechanisms, or are they artifacts of evaluation procedures?*

[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch: 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: HCP](https://img.shields.io/badge/Dataset-HCP%20fMRI-success.svg)](https://www.humanconnectome.org/)
[![Reproducibility: 100%](https://img.shields.io/badge/Reproducibility-100%25%20Verified-brightgreen.svg)](#-reproduction)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🧭 Overview

Kolmogorov–Arnold Networks (KANs) parameterize neural network weights with learnable 1D B-splines, offering unprecedented visual interpretability. In computational neuroscience and connectomics, researchers frequently ask: **"Do the nonlinear spline curves learned on task-fMRI reflect genuine biological computation, or are they evaluation and optimization artifacts?"**

This repository provides an empirical and statistical benchmark testing the reliability of KAN interpretability. Using **Human Connectome Project (HCP)** working memory fMRI data ($N=100$ subjects, 2-back vs. 0-back) alongside synthetic ground-truth controls, we systematically stress-test learned edge splines against **reference-domain shifts** and **optimization stochasticity**.

---

## 🎯 Key Findings

### 📌 Finding 1: Nonlinearity estimates depend on reference domains
* **The Illusion**: Evaluating the exact same mathematical function over slightly shifted evaluation intervals causes the measured Deviation-From-Linearity (DFL) to explode by **$+0.1195$** (Exp 2-B), even when the function has zero ground-truth change.
* **The Principle**: Nonlinearity is **not an invariant property of learned weights**; it is an operator conditioned on the chosen reference domain. Without strict support bounding (Common Support), distribution shifts can create massive false interpretations.

### 📌 Finding 2: Single-seed conclusions fail under seed marginalization
* **The Illusion**: A high-precision permutation test on a single default seed (`seed=42`, 500 permutations) yielded an apparent task-induced excess nonlinearity of $\Delta = +0.0017$ with strong statistical significance ($p = 0.0060$).
* **The Reality**: Across 10 independent initialization seeds (Exp 5), **9 out of 10 seeds yielded negative effects** ($\bar{\Delta} = -0.00058$, 95% CI $[-0.00120, +0.00003]$, $p = 0.9513$).
* **The Principle**: **Single-seed conditional inference does not generalize across optimization randomness.** The $p=0.0060$ was a valid conditional result for Seed 42, but mistaking it for a population-level biological claim is a failure of estimand generalization.

### 📌 Finding 3: Permutation-valid inference requires accounting for optimization uncertainty
* **The Invariant**: Across all seeds, folds, and label permutations, BrainKAN models converge to a remarkably stable **intrinsic curvature budget** ($T \approx 0.179 \pm 0.001$).
* **The Principle**: The network maintains substantial baseline spline curvature regardless of label shuffling. True task-induced excess nonlinearity must be evaluated via **Seed-Marginalized Permutation Testing (SMPT)** to marginalize out loss-basin stochasticity.

---

## 📊 Core Diagnostic Figures

### Figure 1: End-to-End Diagnostic Pipeline
The benchmark pipeline connecting raw HCP fMRI graphs, edge-specific B-spline convolutions, reference-domain evaluation, and seed-marginalized permutation tests.
![Figure 1: Pipeline](figures/fig1_pipeline.png)

---

### Figure 2: The Reference Domain Effect (Exp 2 Synthetic Calibration)
Evaluating the **exact same underlying function** on shifted input domains creates a massive apparent nonlinearity jump ($\Delta = +0.1195$), two orders of magnitude larger than real model differences.
![Figure 2: Reference Domain Effect](figures/fig2_reference_domain_effect.png)

---

### Figure 3: Optimization Variability Across Seeds (Exp 1 vs. Exp 5)
Seed 42 is the sole positive outlier ($\Delta = +0.0017$, $p=0.006$). Marginalizing across 10 random seeds reveals that the expected excess nonlinearity is slightly negative ($\bar{\Delta} = -0.00058$, $p=0.9513$).
![Figure 3: Seed Variability](figures/fig3_seed_variability.png)

---

### Figure 4: Methodological Framework — Pitfalls vs. Recommended Practice
A side-by-side comparison illustrating how conventional practices lead to false interpretability claims, and how our proposed practical protocol establishes reproducible, statistically honest conclusions.
![Figure 4: Methodological Framework](figures/fig4_methodological_framework.png)

---

## 🛠️ Recommended Evaluation Protocol

To prevent false claims of "task-induced nonlinearity" in KAN and interpretable GNN research, we propose a practical 3-step evaluation protocol:

1. **Common Support Domain Bounding**:
   Enforce evaluation exclusively over the overlapping empirical support between classes ($\mathcal{S}_{\mathrm{common}} = \operatorname{supp}(X_0) \cap \operatorname{supp}(X_1)$). Never evaluate on shifted data quantiles or unbounded ranges.
2. **Paired Permutation Null Models**:
   Train null models on permuted labels with identical data topology and architecture to estimate the model's baseline curvature budget.
3. **Seed-Marginalized Permutation Testing (SMPT)**:
   Define the target estimand as the marginal expectation across optimization seeds:
   $$\bar{T} = \frac{1}{S}\sum_{s=1}^S T_s, \quad \Delta_{\mathrm{marginal}} = \mathbb{E}_s[T(Y)] - \mathbb{E}_{s, \pi}[T(\pi(Y))]$$
   Do not base scientific conclusions on a single random seed.

---

## 📑 Mathematical Methodology & Theoretical Whitepapers

For detailed mathematical proofs, derivations, and statistical protocols, see the compiled PDF whitepapers:
- 📄 **[BrainKAN Mathematical Methodology (PDF)](./BrainKAN_Mathematical_Methodology.pdf)** — Core mathematical derivations, edge B-spline parameterization, and codebase mappings.
- 📄 **[Anticipated Theoretical & Methodological Questions (PDF)](./docs/Anticipated%20Theoretical%20Questions.pdf)** — Academic defense addressing estimand mismatch, single-seed conditional vs. marginal inference.
- 📄 **[Mathematical Foundations of Spline Geometry Analysis (PDF)](./docs/Mathematical%20Foundations%20of%20Spline%20Geometry%20Analysis.pdf)** — Reference-induced deformation decomposition and estimand duality.
- 📄 **[Statistical and Theoretical Assumptions (PDF)](./docs/Statistical%20and%20Theoretical%20Assumptions.pdf)** — Detailed statistical conditions: exchangeability, common support bounding, and optimization marginalization.

---

## 🚀 Reproduction & Benchmark Commands

All experiments are fully deterministic and reproducible.

### 1. Installation
```bash
git clone https://github.com/xcklvydtj2024/BrainKAN-Explainability.git BrainKAN-Interpretability-Benchmark
cd BrainKAN-Interpretability-Benchmark
pip install -r requirements.txt
```

### 2. Run Synthetic Calibration (Exp 2)
Tests reference-domain shift and density weighting on ground-truth functions:
```bash
python experiments/exp2_synthetic_calibration.py
```

### 3. Run Common Support Domain Control (Exp 3)
Tests whether the measured nonlinearity persists when restricting evaluation strictly to overlapping support:
```bash
python experiments/exp3_reference_domain_control.py
```

### 4. Run Multi-Seed Stability Scan (Exp 5)
Runs parallel evaluation across 10 independent initialization seeds:
```bash
python experiments/run_exp5_parallel.py
```

### 5. Generate Publication Figures
Generates all 4 core figures at 300 DPI:
```bash
python experiments/plot_four_core_figures.py
```

---

## 📁 Repository Structure

```text
BrainKAN-Interpretability-Benchmark/
├── figures/                            # 4 Core publication-quality diagnostic figures
│   ├── fig1_pipeline.png
│   ├── fig2_reference_domain_effect.png
│   ├── fig3_seed_variability.png
│   └── fig4_methodological_framework.png
├── models/
│   └── kan.py                          # Edge-specific B-spline KAN layer & BrainKAN architecture
├── experiments/
│   ├── 07_primary_statistical_analysis.py # DFL extraction & paired permutation core logic
│   ├── exp1_high_precision_permutation.py # High-precision 500-sample permutation runner
│   ├── exp2_synthetic_calibration.py   # Ground-truth synthetic calibration (Exp 2-A/B/C)
│   ├── exp3_reference_domain_control.py# Common support evaluation (Exp 3)
│   ├── exp4_multi_seed_robustness.py   # Hidden capacity & seed sensitivity scan (Exp 4)
│   ├── exp5_runner.py                  # Multi-seed parallel worker (Exp 5)
│   ├── run_exp5_parallel.py            # Orchestrator for 10-seed SMPT test
│   └── plot_four_core_figures.py       # Standalone figure generation script
├── protocol/                           # Frozen experimental protocol & decision rules
│   ├── analysis_plan_v2.md
│   ├── estimands.md
│   └── decision_rules.md
├── results/                            # Experimental CSV outputs and manifest
│   ├── exp1_high_precision/
│   ├── exp2_synthetic/
│   ├── exp3_ref_domain/
│   ├── exp5_seed_stability/
│   └── manifest.json
└── README.md
```

---

## ⚖️ Scientific & Methodological Disclaimer

This benchmark serves as a **methodological study and stress test** for neural network interpretability. The goal is to provide reliable evaluation protocols for white-box AI models in neuroimaging. Our findings show that edge-specific splines possess an intrinsic structural expressivity budget, but individual spline shapes cannot be causally interpreted as in-vivo biological mechanisms without accounting for reference-domain conditioning and optimization randomness.

---

## 📚 Citation

If you use this benchmark or evaluation protocol in your research, please cite:

```bibtex
@article{BrainKAN_Benchmark_2026,
  title={When Nonlinearity Misleads: Reference-Domain and Optimization Variability in Interpreting Kolmogorov--Arnold Network Representations},
  author={Qi, Xin and Collaborators},
  journal={arXiv preprint},
  year={2026}
}
```
