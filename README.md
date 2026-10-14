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

## 💡 Core Methodological Contributions: Auditing & Regularizing KAN Interpretability

To eliminate domain-selection artifacts, optimization pseudoreplication, and epiphenomenal curvature, this benchmark introduces a four-pillar methodological framework:

```text
Conventional Practice (Fragile)
Single Seed  ──>  Empirical Grid  ──>  Point Estimate  ──>  Risk of Over-interpretation
     
Proposed Rigorous Framework (Audited & Calibrated)
1. Unbiased Reference Domains  ──>  Label-Blind / Re-enacted Support (Eliminates Selection Bias)
2. Curvature Regularization    ──>  Loss + λ ∫|f''(t)|² dt (Tests Predictive Indispensability)
3. Mechanistic Controls        ──>  Linear Null & Nonlinear Controls (Calibrates Type I Error & Power)
4. Hierarchical Inference      ──>  B ≥ 500 Permutation Quantiles + Subject Bootstrap (Decouples Variance)
```

### Four Pillars for Reliable Non-Linear Interpretability:
1. **Unbiased Reference Domain Bounding**: Audit evaluations under label-blind pooled supports, independent split supports, and full-pipeline permutation re-enactments to guarantee finite-sample error rate control.
2. **Curvature Regularization & Ablation**: Penalize edge roughness $\lambda \int |f''(t)|^2 dt$ during training to test whether straightening learned curves impairs test classification accuracy.
3. **Mechanistic Negative & Positive Controls**: Benchmark synthetic linear nulls ($\gamma = 0$) to calibrate Type I error against architectural over-parameterization ($P \approx 5.8 \times 10^5$).
4. **Hierarchical Statistical Inference**: Replace single-sample seed-level $t_9$ tests with large-scale ($B \ge 500$) permutation distribution quantiles, and apply clustered bootstrap at the subject-pair level to decouple algorithmic optimization noise from biological population variability.

---

## 🔬 Research Questions & Benchmark Results

### Q1. Are spline nonlinearities invariant to evaluation domains?
**Answer: No. Domain selection strongly dictates apparent curvature, and naive common supports introduce selection bias.**
* In synthetic controls (Exp 2-B), evaluating the **exact same learned model** (fitted to $`f^*(x) = x + 0.5\sin(2x)`$) over a shifted interval inflated apparent DFL by **+0.0816 ± 0.0084** (>40× larger than real model differences).
* **Reference Domain Selection Bias Audit (`results/domain_bias_correction/`)**:
  * **Legacy Fixed Common Support:** Fixing the intersection interval derived from real labels yielded apparent positive difference ($`\bar{\Delta} = +0.00154`$).
  * **Independent Split Domain:** When common support was derived from an independent subject split and frozen, the difference collapsed ($`\bar{\Delta} = +0.00053`$, $`p = 0.5951`$).
  * **Full-Pipeline Re-enactment:** When recomputing common support dynamically under each label permutation, null models on wider re-enacted overlap exhibited elevated curvature ($`T_{\mathrm{null}} = 0.1547`$), revealing that real model curvature is significantly *lower* than the permuted null ($`\bar{\Delta} = -0.06934`$, $`p = 0.000085`$). This confirms the legacy positive excess was a domain-selection artifact.

### Q2. Are single-seed conclusions reliable across optimization randomness, and how should inference be calibrated?
**Answer: No. Optimization noise dominates biological variability, and seed-level $t$-tests commit pseudoreplication.**
* **Pooled Grid Failure:** On the pooled range, extrapolation artifacts dominate, yielding negative differences for 9/10 seeds ($`\bar{\Delta}_{\mathrm{pool}} = -0.00058`$, $`p = 0.9513`$).
* **Calibrating Finite-Sample Quantiles ($`B=500`$):** Replacing the flawed 10-seed Student's $t_9$ test ($`t=5.186, p=0.0003`$, which inflated degrees of freedom on a single sample) with the empirical quantile test on $B=500$ permutations yields an honest non-parametric $p$-value ($`p = 0.0020 \sim 0.0339`$).
* **Subject-Pair Clustered Bootstrap & Variance Decoupling:** Clustered bootstrap across $N=100$ subject pairs ($M=1000$ replicates) yields a population 95% CI of $[+0.00022, +0.00105]$ ($`\mathrm{SE}_{\mathrm{pop}} = 0.00022`$).
* **Variance Decomposition:** Algorithmic optimization randomness accounts for **86.0%** of inferential variance, whereas biological population heterogeneity accounts for **14.0%**, demonstrating why single-seed findings fail to generalize.

### Q3. Are learned edge nonlinearities functionally necessary for task prediction?
**Answer: No. Incremental predictive ablation demonstrates that spline curvature is epiphenomenal.**
* **Second-Derivative Curvature Regularization (`experiments/curvature_ablation.py`):** We added an edge roughness penalty $\lambda \int |f''(t)|^2 dt$ to the classification loss across $\lambda \in [0, 10.0]$.
* **Predictive Invariance:** As $\lambda$ increased to $10.0$, edge curvature energy plummeted by **66.4%** ($1.0157 \to 0.3416$) and DFL dropped by **46.5%** ($0.1472 \to 0.0787$); however, 5-fold cross-validation test classification accuracy was completely preserved ($92.0\% \to 91.5\%$, $\Delta\mathrm{Acc} = -0.5\%$, paired $t$-test $p = 0.6213$).
* **Scientific Verdict:** BrainKAN's learned nonlinear curves can be forcibly flattened without compromising predictive fidelity, proving that edge curvatures are non-essential over-parameterization artifacts rather than necessary biological mechanisms.

### Q4. Can over-parameterization generate false-positive claims? (Mechanistic Controls)
**Answer: Yes for raw metrics, but paired permutation tests strictly control Type I error.**
* **Linear Negative Control ($\gamma = 0.0$):** Synthetic brain networks with purely linear mean shifts and Gaussian noise still produced a raw BrainKAN DFL of $\text{Mean Raw } T_{\mathrm{real}} = 0.2249 \gg 0$ due to 589k B-spline parameter capacity, proving raw DFL cannot serve as evidence of nonlinearity.
* **Type I Error Rate:** The paired permutation null effectively subtracted this baseline capacity ($\Delta = -0.000169 \pm 0.000804$), achieving a calibrated empirical False Positive Rate of **0.0% (0 / 10 rejections at $\alpha=0.05$)**.
* **Nonlinear Positive Control ($\gamma \in \{0.2, 0.5, 1.0\}$):** Whole-graph averaged DFL exhibited limited statistical power (Power $\approx 0.0$) against global background capacity noise, showing that future mechanistic discovery requires subnetwork-focused rather than whole-graph-averaged inference.

---

## 📊 Core Diagnostic Figures

### Figure 1: End-to-End Diagnostic Pipeline
The benchmark pipeline connecting raw HCP fMRI graphs, edge-specific B-spline convolutions, reference-domain evaluation, and multi-seed paired permutation tests.
![Figure 1: Pipeline](figures/fig1_pipeline.png)

---

### Figure 2: The Reference Domain Effect (Exp 2 Synthetic Calibration)
Evaluating the **exact same learned model** (fitted to $`f^*(x) = x + 0.5\sin(2x)`$) on shifted input domains creates a massive apparent nonlinearity jump ($`\Delta = +0.0816 \pm 0.0084`$), >40× larger than real model differences.
![Figure 2: Reference Domain Effect](figures/fig2_reference_domain_effect.png)

---

### Figure 3: Dual Estimand Evaluation Across 10 Seeds (Exp 5: Pooled Domain vs. Strict Common Support)
Side-by-side comparison illustrating why reference-domain control is decisive:
* **Panel A (Pooled Domain $`\mathcal{S}_{\mathrm{pooled}}`$)**: Extrapolation artifacts dominate; Seed 42 is the sole positive outlier ($`\Delta = +0.0017`$), while 9/10 seeds produce negative differences ($`\bar{\Delta} = -0.00058`$, $`p = 0.9513`$).
* **Panel B (Strict Common Support $`\mathcal{S}_{\mathrm{common}}`$)**: Extrapolation noise is filtered out; 9/10 seeds exhibit consistent positive excess curvature ($`\bar{\Delta} = +0.00090`$, Student's $`t`$ 95% CI $[+0.00051, +0.00129]$, $`t = 5.186`$, $`p = 0.0003`$).
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
- [x] High-precision single-seed permutation baseline (Exp 1, $`N=500`$)
- [x] Multi-seed stability & optimization marginalization (Exp 5, $`S=10`$)
- [x] HCP working-memory benchmark ($`N=100`$ subjects, 2-back vs. 0-back)
- [x] Pre-committed protocol specifications (`protocol/`)
- [x] Unbiased Reference Domain Protocol Suite (`protocol/reference_domains.py`: Label-blind, Independent split, Full re-enactment)
- [x] Mechanistic Negative & Positive Controls (`experiments/mechanistic_controls.py`: Type I Error = 0.0%)
- [x] Curvature Regularization ($\lambda \int |f''(t)|^2 dt$) & Incremental Predictive Ablation (`experiments/curvature_ablation.py`)
- [x] Hierarchical Statistical Inference Framework (`stats_utils/hierarchical_inference.py`: $B \ge 500$ Permutation Quantiles & Subject Bootstrap)

**Planned & Ongoing Extensions:**
- [ ] Cross-cohort evaluation on independent clinical datasets
- [ ] Ultra-large-scale seed profiling ($`S \ge 50`$)
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
- 📌 **[BrainKAN Two-Page Research Evidence Brief (PDF)](./docs/BrainKAN_Research_Evidence_Brief.pdf)** ([Markdown Version](./docs/BrainKAN_Research_Evidence_Brief.md) | [LaTeX Source](./docs/BrainKAN_Research_Evidence_Brief.tex)) — Concise 2-page technical dossier for academic committees: Page 1 covers the core research problem, B-spline architecture, HCP dataset, and experimental design; Page 2 details empirical results, inferential boundaries, and full codebase paths.
- 🔍 **[BrainKAN Data and Code Audit Report (MD)](./docs/audit_report.md)** — All 5 methodological audit items closed (pairing integrity, reference domain leak prevention, exception fallback audit, model parameter matching).
- 🎓 **[Mathematical Research Evidence: Methodological Analysis (MD)](./docs/mathematical_research_evidence.md)** ([LaTeX Source](./docs/mathematical_research_evidence.tex)) — Candidate research evidence dossier: empirical DFL metric definition & domain sensitivity, paired permutation null hypothesis testing ($`p \approx 0.00599`$), and estimand-dependent sign reversal under empirical common support.
- 🛡️ **[Methodological Assumptions, Boundary Conditions, and Empirical Audit (MD)](./docs/critical_methodological_defense_and_bounds.md)** — Comprehensive methodological audit of estimands, assumptions, and limitations: optimization sensitivity vs. population uncertainty, finite permutation precision ($`B=15`$), empirical fallback audit (0/240), evaluation gauge invariance, representation diagnostics, and $`P \gg N`$ capacity calibration.
- 📋 **[Sample Pairing Manifest (CSV)](./data/subject_pairing_manifest.csv)** — Sequential pairing manifest and cross-validation partition records for 100 HCP subject pairs (`sub-HCP100-001` through `sub-HCP100-100`).
- 📄 **[BrainKAN Mathematical Methodology (PDF)](./docs/BrainKAN_Mathematical_Methodology.pdf)** — Core mathematical derivations, edge B-spline parameterization, and codebase mappings.
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
=== DUAL-SUPPORT MULTI-SEED STABILITY AUDIT (EXP 5: POOLED vs. COMMON SUPPORT) ===
Sample Size: 10 Seeds (df = 9)

ESTIMAND 1: POOLED 1%-99% QUANTILE SUPPORT (S_pooled = Quantile([X_0, X_1]))
  Mean T_real:      0.17939
  Mean T_null:      0.17998
  Mean Delta_pool:  -0.00058 (SD=0.00100, SE=0.00032)
  95% CI (t-dist):  [-0.00130, +0.00013]
  t-statistic:      -1.8493 (p_two_sided=0.0975, p_one_sided=0.9513)
  Sign Consistency: 1/10 seeds positive (Seed 42 only)

ESTIMAND 2: STRICT COMMON SUPPORT INTERSECTION (S_common = supp(X_0) ∩ supp(X_1))
  Mean T_real:      0.08595
  Mean T_null:      0.08505
  Mean Delta_common:+0.00090 (SD=0.00055, SE=0.00017)
  95% CI (t-dist):  [+0.00051, +0.00129]
  t-statistic:      5.1862 (p_two_sided=0.0006, p_one_sided=0.0003)
  Sign Consistency: 9/10 seeds positive
```

### 4. Audit Per-Node Empirical Support Overlap
Calculates Jaccard overlap and pooled coverage across all 48 cortical ROIs (240 functional edges):
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
├── data/                               # Neuroimaging graph features & subject manifest
│   ├── HCP_GNN_features.npz
│   ├── GNN_Edges.pt
│   └── subject_pairing_manifest.csv   # Sample pairing manifest & 5-fold CV splits
├── docs/                               # Methodological audit & theoretical whitepapers
│   ├── BrainKAN_Research_Evidence_Brief.pdf # Concise 2-page candidate technical dossier
│   ├── audit_report.md                 # Formally closed code and data audit report
│   ├── mathematical_research_evidence.md # 3 Primary mathematical case studies dossier
│   ├── critical_methodological_defense_and_bounds.md
│   ├── BrainKAN_Mathematical_Methodology.pdf
│   ├── Anticipated Theoretical Questions.pdf
│   ├── Mathematical Foundations of Spline Geometry Analysis.pdf
│   └── Statistical and Theoretical Assumptions.pdf
├── models/
│   └── kan.py                          # Edge-specific B-spline KAN layer & BrainKAN architecture
├── experiments/
│   ├── domain_bias_correction.py       # Task 1: 4-protocol reference domain benchmark
│   ├── mechanistic_controls.py         # Task 2: Linear negative & nonlinear positive controls
│   ├── curvature_ablation.py           # Task 3: Curvature regularization & predictive ablation
│   ├── hierarchical_inference_runner.py# Task 4: Large-scale permutation & variance decoupling
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
│   ├── reference_domains.py            # Unbiased reference domain protocols (Label-blind, Split, Re-enacted)
│   ├── analysis_plan_v2.md
│   ├── estimands.md
│   └── decision_rules.md
├── stats_utils/                        # Statistical testing & inference utilities
│   ├── hierarchical_inference.py       # B>=500 permutation quantiles & clustered subject bootstrap
│   ├── empirical_null.py
│   ├── gpd_tail.py
│   └── multiple_testing.py
├── results/                            # Experimental CSV outputs and manifest
│   ├── domain_bias_correction/         # Task 1 reference domain bias comparison
│   ├── mechanistic_controls/           # Task 2 linear/nonlinear control records
│   ├── curvature_ablation/             # Task 3 curvature regularization sweep records
│   ├── hierarchical_inference/         # Task 4 hierarchical inference & variance decomposition
│   ├── exp1_high_precision/            # Production Exp 1 permutation baseline (N=500)
│   ├── exp2_synthetic/                 # Production Exp 2 synthetic calibration (N=200)
│   ├── exp3_ref_domain/                # Production Exp 3 reference domain control
│   ├── exp4_multi_seed/                # Production Exp 4 capacity scan
│   ├── exp5_seed_stability/            # Production Exp 5 SMPT dual estimands (S=10)
│   ├── primary_analysis/               # Active empirical node support overlap audit
│   ├── archived_runs/                  # Archived historical pilot diagnostic runs
│   └── manifest.json                   # Full commit SHA, environment & canonical index
└── README.md
```

---

## ⚖️ Scientific Disclaimer

This benchmark serves as a **methodological study and stress test** for neural network interpretability. The goal is to provide reliable evaluation protocols for white-box AI models in neuroimaging. Our findings show that edge-specific splines possess an intrinsic structural expressivity budget, but individual spline shapes cannot be causally interpreted as in-vivo biological mechanisms without accounting for reference-domain conditioning and optimization randomness.

---

## 📚 Citation

This repository is currently a research prototype and benchmark release.  
A formal academic citation will be provided upon publication / preprint release.
