# BrainKAN-Explainability

A statistical benchmark for testing whether interpretable nonlinear structures learned by Kolmogorov–Arnold Networks reflect task-specific information or evaluation artifacts.

[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch: 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: HCP](https://img.shields.io/badge/Dataset-HCP%20fMRI-success.svg)](https://www.humanconnectome.org/)
[![Reproducibility: Deterministic](https://img.shields.io/badge/Reproducibility-Deterministic%20Pipeline-brightgreen.svg)](#-reproducibility--quickstart)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🧭 Overview

Kolmogorov–Arnold Networks (KANs) parameterize neural network weights with explicit univariate B-splines, offering intuitive geometric inspection of learned transformations. In neuroimaging and graph learning, researchers often interpret learned spline curvatures as evidence of task-specific non-linear biological mechanisms.

**However, interpretability itself requires rigorous uncertainty accounting.** When evaluating learned edge functions, two unmodeled confounding factors routinely distort conclusions:
1. **Reference-Domain Shifts**: Measuring curvature across shifted input supports can create substantial apparent effects caused by evaluation choices rather than model differences.
2. **Optimization Stochasticity**: Single initialization seeds can easily produce conditional significance that fails to generalize across the optimization landscape.

> *Note: This benchmark does not argue that neural nonlinearities are biologically meaningless. Instead, it evaluates when such interpretations are statistically justified.*

---

## 💡 Core Methodological Contribution: Multi-Seed Paired Null Protocol

To address these failure modes, we propose **Multi-Seed Paired Null Analysis** paired with **Common Support Domain Bounding**:

```text
Conventional Practice (Fragile)
Single Seed  ──>  Empirical Grid  ──>  Point Estimate  ──>  Risk of Over-interpretation
     
Proposed Protocol (Robust)
Multiple Seeds (S ≥ 10)  ──>  Common Support Bounding  ──>  Paired Permutations  ──>  Marginal Inference E_s[Δ]
```

### Three Rules for Reliable Non-linear Interpretability:
1. **Common Support Bounding**: Evaluate spline curvature strictly on the overlapping support ($\mathcal{S}_{\mathrm{common}} = \operatorname{supp}(X_0) \cap \operatorname{supp}(X_1)$) to prevent extrapolation artifacts.
2. **Paired Topology-Preserving Null Models**: Construct empirical null distributions by permuting labels while preserving graph topology and hyperparameter budgets.
3. **Optimization Marginalization**: Define the target estimand as the marginal expectation across optimization seeds:
   $$\bar{\Delta} = \mathbb{E}_{s \sim \mathcal{S}} \left[ T(Y; s) - \mathbb{E}_\pi [T(\pi(Y); s)] \right]$$

---

## 🔬 Research Questions & Benchmark Results

### Q1. Are spline nonlinearities invariant to evaluation domains?
**Answer: No.** Reference domains strongly affect Deviation-From-Linearity ($\mathrm{DFL} = \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}}} = 1 - R^2$) measurements.
* In synthetic controls (Exp 2-B), evaluating the **exact same static function** over a shifted interval inflated apparent DFL by **$+0.0816 \pm 0.0084$** (>40× larger than real model differences).
* Auditing empirical node supports on real HCP fMRI data across 240 ROIs (`results/primary_analysis/node_support_overlap_stats.csv`) shows a mean within-subject Jaccard overlap of only **$38.52\% \pm 5.08\%$** between 0-Back and 2-Back conditions (covering only $43.47\%$ of the pooled quantile range).
* Bounding evaluations to the Common Support eliminates these extrapolation artifacts (Exp 3).

### Q2. Are single-seed conclusions reliable?
**Answer: No.** Optimization marginalization changes the conclusion.
* Under a single default initialization (`seed=42`, $N=500$ permutations), the model exhibited statistically significant excess nonlinearity ($\Delta = +0.0017$, $p = 0.0060$).
* However, across 10 independent initialization seeds under a fixed data split (Exp 5), **9 out of 10 seeds produced negative differences** ($\bar{\Delta} = -0.00058$, Student's $t$ 95% CI $[-0.00130, +0.00013]$, $p = 0.9513$).
* *Takeaway*: Single-seed conditional inference does not generalize across optimization randomness. The failure to reject the null under marginalization shows that excess nonlinearity is sensitive to initialization rather than an optimization-invariant mechanism.

### Q3. Does task-associated excess nonlinearity survive rigorous controls?
**Answer: Not under the proposed optimization-marginalized estimator.**
* BrainKAN models consistently allocate an invariant **baseline curvature budget** ($T \approx 0.179 \pm 0.001$) across both real and label-permuted data.
* The network inherently leverages non-linear transformations for feature representation, but task supervision does not drive extra curvature beyond this baseline once optimization uncertainty is accounted for.

---

## 📊 Core Diagnostic Figures

### Figure 1: End-to-End Diagnostic Pipeline
The benchmark pipeline connecting raw HCP fMRI graphs, edge-specific B-spline convolutions, reference-domain evaluation, and multi-seed paired permutation tests.
![Figure 1: Pipeline](figures/fig1_pipeline.png)

---

### Figure 2: The Reference Domain Effect (Exp 2 Synthetic Calibration)
Evaluating the **exact same underlying function** on shifted input domains creates a massive apparent nonlinearity jump ($\Delta = +0.0816$), >40× larger than real model differences.
![Figure 2: Reference Domain Effect](figures/fig2_reference_domain_effect.png)

---

### Figure 3: Optimization Variability Across Seeds (Exp 1 vs. Exp 5)
Seed 42 is the sole positive outlier ($\Delta = +0.0017$, $p=0.006$). Marginalizing across 10 random seeds on fixed data reveals that the expected excess nonlinearity is slightly negative ($\bar{\Delta} = -0.00058$, Student's $t$ $p=0.9513$).
![Figure 3: Seed Variability](figures/fig3_seed_variability.png)

---

### Figure 4: Methodological Framework — Pitfalls vs. Recommended Practice
A side-by-side comparison illustrating how conventional practices lead to false interpretability claims, and how our proposed practical protocol establishes reproducible, statistically honest conclusions.
![Figure 4: Methodological Framework](figures/fig4_methodological_framework.png)

---

## 📌 Project Status

**Research prototype & benchmark release**

- [x] Synthetic calibration with ground-truth functions (Exp 2-A/B/C)
- [x] Common support domain control on fMRI connectomes (Exp 3)
- [x] High-precision single-seed permutation baseline (Exp 1, $N=500$)
- [x] Multi-seed stability & optimization marginalization (Exp 5, $S=10$)
- [x] HCP working-memory benchmark ($N=100$ subjects, 2-back vs. 0-back)
- [x] Pre-committed protocol specifications (`protocol/`)

**Planned & Ongoing Extensions:**
- [ ] Cross-cohort evaluation on independent clinical datasets
- [ ] Ultra-large-scale seed profiling ($S \ge 50$)
- [ ] Benchmarking alternative spline representations (Fourier-KAN, Wavelet-KAN)

---

## 📋 Pre-Committed Benchmark Protocol (`protocol/`)

To guarantee methodological transparency and prevent post-hoc hypothesis adjustments, all experimental analyses followed pre-committed protocol specifications formulated prior to multi-seed data evaluation:

```text
protocol/
├── analysis_plan_v2.md    # Pre-committed hypotheses, CV folds, and probe ranges
├── estimands.md           # Formal definitions of conditional vs. marginal estimands
└── decision_rules.md      # Explicit statistical decision criteria
```
*Clarification: These specifications represent internal pre-analysis commitments designed to lock evaluation rules, rather than a formally time-stamped external third-party registry (e.g. OSF).*

---

## 📑 Mathematical Methodology & Theoretical Whitepapers

For detailed mathematical proofs, derivations, and statistical protocols, see the compiled whitepapers:
- 📄 **[BrainKAN Mathematical Methodology (PDF)](./BrainKAN_Mathematical_Methodology.pdf)** — Core mathematical derivations, edge B-spline parameterization, and codebase mappings.
- 📄 **[Anticipated Theoretical & Methodological Questions (PDF)](./docs/Anticipated%20Theoretical%20Questions.pdf)** — Academic defense addressing estimand mismatch, single-seed conditional vs. marginal inference.
- 📄 **[Mathematical Foundations of Spline Geometry Analysis (PDF)](./docs/Mathematical%20Foundations%20of%20Spline%20Geometry%20Analysis.pdf)** — Anchored operational decomposition and estimand duality.
- 📄 **[Statistical and Theoretical Assumptions (PDF)](./docs/Statistical%20and%20Theoretical%20Assumptions.pdf)** — Detailed statistical conditions: exchangeability, common support bounding, and optimization marginalization.

---

## 🚀 Reproducibility & Quickstart

Deterministic pipelines with fixed seeds, logged configurations, and released experiment scripts.

### 1. Installation
```bash
git clone https://github.com/xcklvydtj2024/BrainKAN-Explainability.git BrainKAN-Explainability
cd BrainKAN-Explainability
pip install -r requirements.txt
```

### 2. Run Synthetic Calibration (Exp 2)
Tests reference-domain shift and density weighting on ground-truth functions:
```bash
python experiments/exp2_synthetic_calibration.py
```

### 3. Run Multi-Seed Stability Scan (Exp 5)
Runs parallel evaluation across 10 independent initialization seeds:
```bash
python experiments/run_exp5_parallel.py
```

**Example Terminal Output:**
```text
=== Final Seed Stability Inference (Student's t Formulation) ===
Sample Size (Seeds): 10
Mean Delta: -0.00058
SD Delta:   0.00100
SE Delta:   0.00032
Student's t Critical (df=9, 95%): 2.262
95% Confidence Interval (Student's t): [-0.00130, +0.00013]
t-statistic: -1.8495
p-value (H1: Delta > 0, one-sided): 0.9513
Sign Consistency: 1/10 positive Deltas (Seed 42 only)
```

### 4. Audit Per-Node Empirical Support Overlap
Calculates Jaccard overlap and pooled coverage across all 240 ROIs:
```bash
python experiments/compute_support_overlap_empirical.py
```

### 5. Generate Core Figures
Generates all 4 core publication figures dynamically from experiment CSVs at 300 DPI:
```bash
python experiments/plot_four_core_figures.py
```

---

## 📁 Repository Structure

```text
BrainKAN-Explainability/
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
│   ├── compute_support_overlap_empirical.py # Per-node empirical support overlap audit
│   └── plot_four_core_figures.py       # Dynamic figure generation script
├── protocol/                           # Pre-committed experimental protocol & decision rules
│   ├── analysis_plan_v2.md
│   ├── estimands.md
│   └── decision_rules.md
├── results/                            # Experimental CSV outputs and manifest
│   ├── exp1_high_precision/
│   ├── exp2_synthetic/
│   ├── exp3_ref_domain/
│   ├── exp5_seed_stability/
│   ├── primary_analysis/
│   └── manifest.json
└── README.md
```

---

## ⚖️ Scientific Disclaimer

This benchmark serves as a **methodological study and stress test** for neural network interpretability. The goal is to provide reliable evaluation protocols for white-box AI models in neuroimaging. Our findings show that edge-specific splines possess an intrinsic structural expressivity budget, but individual spline shapes cannot be causally interpreted as in-vivo biological mechanisms without accounting for reference-domain conditioning and optimization randomness.

---

## 📚 Citation

This repository is currently a research prototype and benchmark release.  
A formal academic citation will be provided upon publication / preprint release.
