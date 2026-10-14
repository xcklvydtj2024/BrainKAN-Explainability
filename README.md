# BrainKAN-Explainability

A rigorous statistical benchmark for evaluating whether interpretable nonlinear structures learned by Kolmogorov–Arnold Networks (KANs) reflect true task-specific neurobiological mechanisms or evaluation and architectural artifacts.

[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch: 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: HCP](https://img.shields.io/badge/Dataset-HCP%20fMRI-success.svg)](https://www.humanconnectome.org/)
[![Tests: 21 Passed](https://img.shields.io/badge/Tests-21%2F21%20Passed-brightgreen.svg)](#-unit-tests--automated-verification)
[![Reproducibility: Deterministic](https://img.shields.io/badge/Reproducibility-Deterministic%20Pipeline-brightgreen.svg)](#-reproducibility--quickstart)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🧭 Executive Overview

Kolmogorov–Arnold Networks (KANs) parameterize neural network weights with learnable univariate B-splines, allowing direct inspection of learned edge activation functions. In neuroimaging connectomics, learned spline curvature is frequently interpreted as evidence of condition-dependent non-linear communication between brain regions.

**However, interpretability requires rigorous statistical and mechanistic uncertainty accounting.** Through counterfactual synthetic benchmarks, large-scale permutation testing, and reference-domain audits, this repository demonstrates:

1. **Domain Selection Confounding**: Apparent spline curvature is highly sensitive to the evaluation support. Dynamically defining common supports across experimental conditions introduces selection artifacts that reverse sign under full-pipeline label permutation.
2. **Conditional vs. Multi-Seed Generalizability**: On a pre-committed label-blind reference domain, a single initialization seed (Seed 42) exhibits statistically significant excess curvature ($p_{\mathrm{emp}} = 0.00599$ against $B=500$ valid paired permutations). However, across $S=10$ independent initialization seeds, this effect fails to generalize ($\bar{\Delta} = -0.00058 \pm 0.00100$, positive in only 1/10 seeds). Naively comparing a 10-seed average to a 1-seed permutation null creates an invalid mismatched test ($p = 0.00399$) that is formally discarded.
3. **Predictive Necessity Constraints**: Strong curvature regularization substantially reduced first-layer spline roughness, while no statistically detectable accuracy difference was observed in this limited five-fold comparison ($92.0\% \to 91.5\%$, paired $t$-test $p = 0.6213$). These findings suggest that large learned curvature is not sufficient evidence of predictive necessity.
4. **Architectural Identifiability Limits**: Under an identifiable counterfactual Data Generation Mechanism (DGM), an identifiable polynomial regression baseline achieves well-calibrated Type I error ($\mathrm{FPR} = 4.0\%$ under a strictly exchangeable sharp null) and an empirical rejection rate of $7.0\%$ under a linear-shift stress test, with $100\%$ detection power at $\gamma=1.0$ ($\mathrm{AUC} = 1.000$). Conversely, BrainKAN trained end-to-end with whole-graph classification loss does not identify planted non-linear edges (planted DFL = $0.1910$ vs. unplanted DFL = $0.2579$). These observations are consistent with possible effects of global pooling and excess model capacity, but do not isolate either mechanism causally.

---

## 🔬 Core Methodological Pillars

### Pillar 1: Mechanistic Controls & Identifiability Benchmark
**File:** [`experiments/mechanistic_controls.py`](experiments/mechanistic_controls.py)  
**Verification:** [`tests/test_mechanistic_controls.py`](tests/test_mechanistic_controls.py)

To evaluate whether non-linear testing frameworks can identify known ground truth, we constructed a synthetic counterfactual Data Generation Mechanism (DGM) on paired subjects ($i \in \{1,\dots,N\}$, conditions $c \in \{0, 1\}$):
$$x_{v,i,c} = a x_{u,i,c} + \gamma (2c - 1) \tanh(b x_{u,i,c}) + \epsilon_{i,c}$$
- **Counterfactual Control**: Comparisons across $\gamma \in \{0.0, 0.1, 0.2, 0.5, 1.0\}$ use the **exact same baseline topology, node features, and Gaussian noise realizations** $\epsilon_{i,c} \sim \mathcal{N}(0, 0.1^2)$.
- **Planted Graph Topology**: 6 disjoint directed edges carry condition-dependent non-linearities, embedded among 94 background linear edges.
- **Automated Data Injection Check**: Verified via automated unit tests that changing $\gamma$ modifies only planted edge target responses while all other nodes remain identical.

This module explicitly distinguishes two evaluations:
- **Benchmark 1**: Polynomial Nonlinear-Dependence Detection Benchmark (Identifiable Proxy Baseline)
- **Audit 2**: BrainKAN End-to-End Representation Audit (Architectural Evaluation)

#### 1. Null Calibration across 100 Independent Cohorts ($B = 199$ Permutations Each)
Evaluated across **100 independently generated null cohorts** ($\gamma = 0.0$, 80 subject pairs each) with $B=199$ valid paired label permutations per cohort under both sharp null and a linear-shift stress test:

| Null Protocol | Linear Shift | N Cohorts | Permutations $B$ | Rejections ($\alpha=0.05$) | Empirical Rate | Mean $p$-value | Median $p$-value | Calibration / Stress Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strictly Exchangeable Sharp Null** ($H_0: P_0 = P_1$) | $0.0$ | 100 | 199 | 4 / 100 | **$0.0400 \pm 0.0196$** | 0.4773 | 0.4525 | **WELL-CALIBRATED** |
| **Linear-Shift Stress Test** | $0.4$ | 100 | 199 | 7 / 100 | **$0.0700 \pm 0.0255$** | 0.4635 | 0.4300 | **EMPIRICAL STRESS TEST** |

- **Strict Exchangeability**: When $\mu_0 = \mu_1$ (`linear_shift = 0.0`), condition labels are strictly exchangeable, achieving an empirical Type I error rate of $4.0\%$.
- **Linear-Shift Stress Test**: When condition 1 carries a linear distribution shift (`linear_shift = 0.4`), paired label exchangeability no longer strictly holds. The observed empirical rejection rate is $7.0\%$ in this stress test, demonstrating that moderate linear shifts do not produce severe false-positive inflation, though this observation cannot be claimed as formal Type I error control for general composite nulls.

#### 2. Benchmark 1: Polynomial Non-Linear Detection Power & Edge Recovery
Evaluating the identifiable polynomial regression proxy across 120 trials (30 independent cohorts per non-zero $\gamma$, $B=199$ permutations):

| Non-linear Strength $\gamma$ | Trials | Empirical Power ($\alpha=0.05$) | Power SE | Planted Edge AUC | Edge Recall | Edge Precision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\gamma = 0.1$** | 30 | 0.0333 | $\pm 0.0328$ | 0.6132 | 0.0000 | 0.0000 |
| **$\gamma = 0.2$** | 30 | 0.0000 | $\pm 0.0000$ | 0.7070 | 0.0000 | 0.0000 |
| **$\gamma = 0.5$** | 30 | 0.2000 | $\pm 0.0730$ | 0.9879 | 0.5889 | 0.5889 |
| **$\gamma = 1.0$** | 30 | **1.0000** | $\pm 0.0000$ | **1.0000** | **1.0000** | **1.0000** |

This confirms that the statistical testing procedure has sufficient power to reject $H_0$ and isolate planted non-linear edges when evaluated with an identifiable edge-level estimator.

#### 3. Audit 2: BrainKAN End-to-End Representation Audit
When passing the exact same counterfactual cohort into BrainKAN trained end-to-end with graph classification loss:
- $\gamma = 0.0$: Mean Planted DFL = $0.2229$, Unplanted DFL = $0.2563$, Overall = $0.2555$
- $\gamma = 1.0$: Mean Planted DFL = $0.1910$, Unplanted DFL = $0.2579$, Overall = $0.2562$

> **Scientific Conclusion**: The 100% detection power of the polynomial proxy baseline does **not** translate to BrainKAN end-to-end graph classification. BrainKAN exhibits lower DFL on planted edges ($0.1910$) than on unplanted edges ($0.2579$), against an overall background DFL floor of $\sim 0.25$. These observations are consistent with possible effects of global pooling and excess model capacity, but do not isolate either mechanism causally. Consequently, **end-to-end graph classification in deep KANs cannot be treated as an edge-level causal discovery engine without edge-specific supervision or explicit inductive constraints.**

---

### Pillar 2: Hierarchical Permutation Testing & Optimization Sensitivity Audit
**File:** [`experiments/hierarchical_inference_runner.py`](experiments/hierarchical_inference_runner.py)  
**Verification:** [`tests/test_hierarchical_inference.py`](tests/test_hierarchical_inference.py)

#### 1. Single-Seed Conditional Permutation Test (Seed 42, $B = 500$)
On the pre-committed **Label-Blind Training-Fold Domain**, the observed test statistic and null distribution are evaluated on the identical single seed ($s=42$):
- **Observed Test Statistic $T_{\mathrm{obs}}(s=42)$**: $0.178953$
- **Permutation Null Mean ($B=500$)**: $0.177333 \pm 0.000585$ ($95\text{th percentile} = 0.178301$)
- **Excess Curvature $\Delta(s=42)$**: $+0.001620$
- **Exact Empirical $p$-value**:
  $$p_{\mathrm{emp}} = \frac{1 + \sum_{b=1}^{500} \mathbf{1}(T_b^{\mathrm{null}} \ge T_{\mathrm{obs}})}{500 + 1} = \frac{1 + 2}{501} = \mathbf{0.00599}$$
- **Tail GPD $p$-value**: $0.003535$
- **Scope**: Statistically significant conditioned on the Seed 42 initialization.

#### 2. Multi-Seed Optimization Sensitivity Audit ($S = 10$ Seeds)
Evaluating whether this conditional excess curvature replicates across 10 independent random initialization seeds:
- **Observed Curvature**: Mean $\bar{T}_{\mathrm{real}} = 0.179394 \pm 0.001263$ ($\mathrm{CV} = 0.70\%$)
- **Null Baseline Across Seeds**: Mean $\bar{T}_{\mathrm{null}} = 0.179977 \pm 0.001492$
- **Mean Excess Curvature**: $\bar{\Delta} = \mathbf{-0.000583} \pm 0.000996$
- **Replication Count ($\Delta > 0$)**: **1 / 10 seeds** (Seed 42 only; 9 / 10 seeds produce negative $\Delta$)
- **One-Sided Student's $t$-test**: $t = -1.8493, p = 0.9513$
- **Scientific Finding**: The conditional excess curvature observed for Seed 42 fails to generalize across the optimization landscape on the label-blind training-fold domain.

#### 3. Methodological Audit: Formally Discarding Mismatched $p = 0.00399$
Comparing a 10-seed observed mean ($0.179394$) directly against Seed 42's single-seed null distribution ($\mu = 0.177333$) previously produced an apparent $p = 0.00399$. **This comparison is methodologically invalid**:
- An observed test statistic and its permutation null must follow the exact same evaluation procedure.
- Evaluating consistently within Exp 5 reveals that the 10-seed null baseline ($0.179977$) exceeds the 10-seed real mean ($0.179394$).
- The apparent $p = 0.00399$ is formally discarded as an artifact of comparing an averaged statistic against a single-seed null.

---

### Pillar 3: Reference Domain Design & Eliminating Confounded Significance
**File:** [`experiments/domain_bias_correction.py`](experiments/domain_bias_correction.py)  
**Verification:** [`tests/test_reference_domains.py`](tests/test_reference_domains.py)

#### 1. Audited Evaluation Protocols
- **Scheme A (Primary: Label-Blind Training-Fold Domain)**: Evaluates real and permuted models on a fixed reference domain extracted from training fold inputs ignoring labels.
- **Scheme B (Independent Held-Out Reference Set)**: Exactly 20 subject pairs (40 graphs) are permanently held out solely for reference domain extraction and never used in model training or testing ($\mathrm{ref\_subjects} \cap \mathrm{cv\_subjects} = \emptyset$, verified by unit test). The remaining 80 pairs undergo 5-fold CV.
- **Full Pipeline Reenactment**: Dynamically re-determines common support under each permuted label assignment.

#### 2. Full-Pipeline Permutation Testing Across Folds
Replacing flawed 5-fold Student's $t$-tests (which previously claimed $p = 0.000085$ on dependent folds) with exact permutation testing across folds reveals:

| Protocol | Mean Real $T_{\mathrm{real}}$ | Mean Null $T_{\mathrm{null}}$ | Real Excess $\Delta$ | Positive Folds | Permutation $B$ | Permutation $p$-value | Scientific Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Pipeline Reenactment** | 0.09058 | 0.16433 | **-0.07375** | 0 / 5 | 10 | **1.0000** | Non-significant; dynamic support inflates null curvature |
| **Independent Split (Scheme B)** | 0.05074 | 0.04983 | +0.00091 | 4 / 5 | 10 | **0.0909** | Moderate positive trend under sample reduction (80 pairs) |
| **Label-Blind (Scheme A, Primary)** | 0.18853 | 0.18783 | +0.00070 | 2 / 5 | 10 | **0.0909** | Unbiased reference domain; free of label selection bias |
| **Legacy Fixed Support** | 0.09058 | 0.09085 | -0.00027 | 3 / 5 | 10 | **0.8182** | Non-significant under paired permutation |

---

### Pillar 4: Curvature Regularization & Predictive Ablation
**File:** [`experiments/curvature_ablation.py`](experiments/curvature_ablation.py)

To test whether learned edge non-linearities are functionally indispensable for cognitive task prediction, we introduced an edge roughness penalty to the training objective:
$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{CE}} + \lambda \sum_{e \in E} \int |f_e''(t)|^2 \, dt$$

Across a sweep of $\lambda \in [0.0, 10.0]$:
- Curvature energy drops by **$66.4\%$** ($1.0157 \to 0.3416$).
- Empirical DFL drops by **$46.5\%$** ($0.1472 \to 0.0787$).
- **5-Fold Cross-Validation Test Accuracy**: $92.0\% \to 91.5\%$ (paired $t$-test $p = 0.6213$).
- **Calibrated Scientific Conclusion**: Strong curvature regularization substantially reduced first-layer spline roughness, while no statistically detectable accuracy difference was observed in this limited five-fold comparison. These findings suggest that large learned curvature is not sufficient evidence of predictive necessity. Note that $p > 0.05$ does not prove predictive equivalence, and deeper network layers retain non-linear capacity.

---

## 📊 Core Diagnostic Figures

| Figure | Description |
| :---: | :--- |
| **Figure 1**<br>![Figure 1](figures/fig1_pipeline.png) | **End-to-End Diagnostic Pipeline**: Raw HCP fMRI connectomes, edge-specific B-spline convolutions, reference-domain evaluation, and paired permutation framework. |
| **Figure 2**<br>![Figure 2](figures/fig2_reference_domain_effect.png) | **Reference Domain Effect**: Evaluating the *exact same model* over shifted input supports inflates apparent curvature by $>40\times$ ($\Delta = +0.0816$), demonstrating domain confounding. |
| **Figure 3**<br>![Figure 3](figures/fig3_seed_variability.png) | **Dual-Estimand Evaluation**: Comparing pooled quantile grids (extrapolation artifacts) versus strict common support across 10 random initialization seeds. |
| **Figure 4**<br>![Figure 4](figures/fig4_methodological_framework.png) | **Methodological Comparison**: Conventional fragile practices versus our audited, pre-committed evaluation framework. |

---

## 🧪 Unit Tests & Automated Verification

The test suite covers the entire pipeline and runs automatically via pytest:

```bash
pytest tests/ -v
```

**Test Suite Coverage (21 / 21 Passed, 0 Failures):**
- [`tests/test_mechanistic_controls.py`](tests/test_mechanistic_controls.py) (4 tests):
  - `test_data_injection_and_counterfactual_integrity`: Verifies that varying $\gamma$ strictly perturbs planted non-linear edges and leaves background edges identical.
  - `test_linear_null_hypothesis_gamma_zero`: Verifies that at $\gamma=0$, relationship is purely linear in both conditions with Gaussian residuals.
  - `test_planted_edge_recovery_high_gamma`: Verifies high detection power, AUC $\ge 0.85$, and precision/recall $\ge 0.50$ at $\gamma=1.0$.
  - `test_null_calibration_single_run`: Verifies null run completes without spurious excess.
- [`tests/test_hierarchical_inference.py`](tests/test_hierarchical_inference.py) (5 tests):
  - `test_no_synthetic_normal_deltas_in_codebase`: Codebase audit ensuring zero synthetic normal delta sampling.
  - `test_optimization_variance_computation`: Verifies correct SD, variance, and CV calculation across seeds.
  - `test_tail_gpd_fit_monotonicity`: Verifies Generalized Pareto tail CDF monotonicity.
  - `test_empirical_p_value_calculation`: Verifies exact $(1 + k)/(B+1)$ non-parametric formula.
  - `test_multi_seed_permutation_inference_runner`: Verifies Scheme A conditional test ($p \approx 0.00599$) and confirms mismatched comparison is discarded.
- [`tests/test_reference_domains.py`](tests/test_reference_domains.py) (5 tests):
  - `test_independent_split_disjointness`: Asserts $\mathrm{ref\_subjects} \cap \mathrm{cv\_subjects} = \emptyset$.
  - `test_label_blind_domain_permutation_invariance`: Verifies reference domain bounds do not change under label permutation.
  - `test_domain_bias_runner_structure`: Validates all 4 protocols return valid summary records.
  - `test_permutation_p_value_bounded`: Asserts permutation $p$-values are in $(0, 1]$.
  - `test_five_fold_split_completeness`: Verifies 5-fold CV splits cover all 80 CV subjects without leakage.
- [`tests/test_b_spline.py`](tests/test_b_spline.py): Exact Cox-de Boor recursion & partition of unity.
- [`tests/test_curvature_penalty.py`](tests/test_curvature_penalty.py): Analytical second derivative of cubic B-splines.
- [`tests/test_edge_mapping.py`](tests/test_edge_mapping.py): Graph edge index topological mapping.
- [`tests/test_batch_invariance.py`](tests/test_batch_invariance.py): Batch-size invariance of graph forward passes.
- [`tests/test_model_parameters.py`](tests/test_model_parameters.py): Parameter counting and capacity checks.

---

## 🚀 Reproducibility & Quickstart

### 1. Installation
```bash
git clone https://github.com/xcklvydtj2024/BrainKAN-Explainability.git
cd BrainKAN-Explainability
pip install -r requirements.txt
```

### 2. Run All Unit Tests
```bash
pytest tests/
```

### 3. Run Mechanistic Controls (Dual Null Calibration & Power Curve)
```bash
python experiments/mechanistic_controls.py --mode all --n_null 100 --n_perm 199
```

### 4. Run Hierarchical Permutation Inference & Sensitivity Audit
```bash
python experiments/hierarchical_inference_runner.py
```

### 5. Run Reference Domain Bias Correction Audit
```bash
python experiments/domain_bias_correction.py --n_perm 10
```

### 6. Run Curvature Regularization & Ablation
```bash
python experiments/curvature_ablation.py
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
│   └── subject_pairing_manifest.csv   # Subject pairing manifest & 5-fold CV splits
├── docs/                               # Methodological audit & theoretical whitepapers
│   ├── BrainKAN_Research_Evidence_Brief.pdf # Concise 2-page candidate technical brief
│   ├── audit_report.md                 # Formally closed code and data audit report
│   ├── mathematical_research_evidence.md # Mathematical case studies dossier
│   └── critical_methodological_defense_and_bounds.md
├── models/
│   └── kan.py                          # Edge-specific B-spline KAN layer & BrainKAN architecture
├── experiments/
│   ├── mechanistic_controls.py         # Counterfactual DGM, dual null calibration & power curve
│   ├── hierarchical_inference_runner.py# Scheme A conditional test & sensitivity audit
│   ├── domain_bias_correction.py       # Scheme A, Scheme B & Full-Pipeline Reenactment
│   ├── curvature_ablation.py           # Second-derivative curvature roughness penalty
│   ├── exp1_high_precision_permutation.py # High-precision 500-sample permutation baseline
│   ├── exp2_synthetic_calibration.py   # Ground-truth synthetic calibration (Exp 2-A/B/C)
│   ├── exp3_reference_domain_control.py# Common support evaluation (Exp 3)
│   ├── exp4_multi_seed_robustness.py   # Hidden capacity & seed sensitivity scan (Exp 4)
│   ├── exp5_runner.py                  # Multi-seed parallel worker (Exp 5)
│   ├── run_exp5_parallel.py            # Orchestrator for 10-seed SMPT test
│   ├── compute_support_overlap_empirical.py # Empirical node support overlap audit
│   └── plot_four_core_figures.py       # Dynamic figure generation script
├── tests/                              # Comprehensive test suite (21 unit tests)
│   ├── test_mechanistic_controls.py    # Counterfactual DGM & null calibration tests
│   ├── test_hierarchical_inference.py  # Statistical inference & stability tests
│   ├── test_reference_domains.py       # Reference domain disjointness & invariance tests
│   ├── test_b_spline.py                # B-spline mathematical properties
│   ├── test_curvature_penalty.py       # Curvature penalty derivatives
│   ├── test_edge_mapping.py            # Edge index consistency
│   ├── test_batch_invariance.py        # Graph batching invariance
│   └── test_model_parameters.py        # Parameter counts
├── protocol/                           # Pre-committed experimental protocol & decision rules
│   ├── reference_domains.py            # Unbiased reference domain implementations
│   ├── analysis_plan_v2.md
│   ├── estimands.md
│   └── decision_rules.md
├── results/                            # Clean canonical experimental outputs
│   ├── mechanistic_controls/           # Dual null calibration (200 cohorts), power curve, BrainKAN audit
│   ├── hierarchical_inference/         # Scheme A conditional test, sensitivity audit & discarded comparisons
│   ├── domain_bias_correction/         # 4-protocol comparison records and fold outputs
│   ├── curvature_ablation/             # Curvature regularization sweep records
│   ├── exp1_high_precision/            # Production Exp 1 permutation baseline (N=500)
│   ├── exp2_synthetic/                 # Production Exp 2 synthetic calibration (N=200)
│   ├── exp3_ref_domain/                # Production Exp 3 reference domain control
│   ├── exp4_multi_seed/                # Production Exp 4 capacity scan
│   ├── exp5_seed_stability/            # Production Exp 5 SMPT dual estimands (S=10)
│   └── manifest.json                   # Full canonical results index & metadata
└── README.md
```

---

## ⚖️ Scientific Disclaimer

This benchmark serves as a **methodological stress test** for neural network interpretability in neuroimaging. Our findings show that univariate edge B-splines possess an intrinsic capacity to model complex continuous couplings, but individual spline shapes cannot be causally interpreted as biological mechanisms without pre-committed reference domains, counterfactual calibration, and optimization stability checks.

---

## 📚 Citation

```bibtex
@software{brainkan_explainability_2026,
  author = {BrainKAN Methodology Working Group},
  title = {BrainKAN-Explainability: A Statistical Benchmark for Auditing Neural Nonlinearities in Connectomics},
  year = {2026},
  url = {https://github.com/xcklvydtj2024/BrainKAN-Explainability}
}
```
