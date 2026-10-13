# BrainKAN Research Evidence Brief (Two-Page Technical Dossier)

**Project Title:** BrainKAN Explainability Benchmark: Robust Statistical Evaluation of Learned Spline Nonlinearities in Neuroimaging Graph Neural Networks  
**Author:** Xin Qi  
**Repository:** [github.com/xcklvydtj2024/BrainKAN-Explainability](https://github.com/xcklvydtj2024/BrainKAN-Explainability)  
**Production Data Commit SHA:** `4f6c2e8caba09fc7d0ddcda503da87f380424b3f` $\vert$ Release: `8218aeb1fae0b5a084266fbfedbc051c07c7e7b9`  
**Environment:** Python 3.10.20 | PyTorch 2.14.0+cpu | PyTorch Geometric 2.8.0.post1 | Deterministic seeds  

---

<!-- ========================================================================= -->
<!-- PAGE 1: RESEARCH PROBLEM, MODEL ARCHITECTURE, DATASET, AND CORE EXPERIMENTS -->
<!-- ========================================================================= -->

## Page 1: Problem, Model, Dataset, and Core Experimental Design

### 1. Research Motivation & Core Methodological Problem
In graph neural networks for functional neuroimaging, Kolmogorov–Arnold Networks (KANs) replace linear edge weights with learnable univariate B-splines $\Phi_{ij}(x) = \sum_{p} c_{ij,p} B_p(x)$, allowing direct visual inspection of edge-level transformations. In literature, researchers frequently interpret learned spline curvatures as evidence of in-vivo biological computation or task-specific cognitive mechanisms.

**The Foundational Methodological Problem:**  
*Do learned spline nonlinearities reflect task-specific information, or are they mathematical artifacts of operational metric definitions, evaluation domains, and non-convex optimization randomness?*  
Specifically, we identify two unmodeled confounding factors:
1. **Reference-Domain Shifts:** Measuring curvature across shifted input supports creates apparent nonlinearity jumps purely from grid boundary extrapolation rather than model differences.
2. **Optimization Stochasticity:** Single initialization seeds can easily produce statistically significant conditional curvature differences that vanish or reverse across the broader optimization landscape.

```
Conventional Fragile Workflow:
  Single Seed (s=42) ──> Raw Empirical Grid ──> Point Estimate ──> Risk of False Interpretability

Proposed Calibrated Protocol (SMPT):
  Multiple Seeds (S ≥ 10) ──> Common Support Bounding ──> Paired Permutations ──> Marginal Estimand E_s[Δ_s]
```

---

### 2. Model Architecture: BrainKAN Edge-Spline Parameterization
The benchmark evaluates a specialized 2-layer Graph Kolmogorov–Arnold Network tailored for functional connectomes:
- **Layer 1 (Learnable Edge B-Splines):** For each directed functional edge $e = (u, v) \in \mathcal{E}$, message passing computes:
  $$\Phi_e(x_u) = w_{\mathrm{base}, e} \cdot \operatorname{SiLU}(x_u) + \sum_{k=1}^{G+k_o} c_{e, k} B_{k, k_o}(x_u)$$
  where $B_{k, k_o}$ denotes cubic B-spline basis functions ($k_o = 3$) parameterized over $G = 5$ grid intervals ($G + k_o = 8$ basis functions).
- **Exact Parameter Accounting:** With $|E| = 240$ edges, $C_{\mathrm{in}} = 1$, and $C_{\mathrm{out}} = 16$:
  - Spline coefficients: $240 \times 16 \times 1 \times 8 = \mathbf{30,720}$ parameters.
  - Base residual weights: $240 \times 16 \times 1 = \mathbf{3,840}$ parameters.
  - Root self-connections: $16 \times 1 = \mathbf{16}$ parameters.
  - **Layer 1 Total:** $\mathbf{34,576}$ parameters. The complete 2-layer BrainKAN model contains $P = \mathbf{589,779}$ trainable parameters.
- **Operational Metric (Deviation-From-Linearity, DFL):** For any learned function $f$ evaluated on discrete grid $D = \{t_1, \dots, t_K\}$, DFL measures normalized variance unexplained by affine regression:
  $$\operatorname{DFL}(f; D) = \frac{\inf_{a, b} \sum_{k=1}^K |f(t_k) - (at_k + b)|^2}{\sum_{k=1}^K |f(t_k) - \bar{f}|^2} = 1 - R^2$$

---

### 3. Dataset & Data Provenance
- **Primary Cohort (HCP-WM):** 100 healthy young adults from the Human Connectome Project (HCP) 3T Working Memory task-fMRI. 
  - **Task Contrast:** 2-Back (high cognitive load) vs. 0-Back (low cognitive load).
  - **Structure:** 200 total functional graphs (100 strictly paired subject observations).
  - **Graph Topology:** 48 cortical Regions of Interest (ROIs) based on cortical parcellation, connected by 240 directed functional edges ($E = 240$).
  - **Cross-Validation:** 5-fold cross-validation with strict subject-level pairing (`data/subject_pairing_manifest.csv`). 0-Back and 2-Back scans for the same subject never cross fold boundaries. Reference domains are computed dynamically using **only** intra-fold training subjects (zero train-test leakage).
- **External Validation Cohort (CHCP):** Chinese Human Connectome Project dataset used for cross-cohort transportability testing under site and demographic distribution shifts.

---

### 4. Core Experimental Design: Progressive Falsification Suite
The benchmark executes 5 structured, pre-committed experiments crossing input shifts with design-consistent null models:
- **Exp 1 (High-Precision Permutation Test):** Evaluates single-seed (`seed=42`) excess curvature on the **full-graph average DFL** over $N = 500$ sign-flip label permutations on the pooled quantile grid $\mathcal{S}_{\mathrm{pooled}}$. (*Note: This global-graph permutation test is statistically distinct from the edge-level FDR test in exploratory Q3B, where 0/240 individual edges reached significance*).
- **Exp 2 (Synthetic Ground-Truth Calibration):** A 3-condition controlled simulation ($N = 200$) on known function $f^*(x) = x + 0.5\sin(2x)$ assessing DFL sensitivity to uniform, domain-shifted, and density-weighted grids.
- **Exp 3 (Reference Domain Control):** 5-fold CV on real fMRI data contrasting pooled ranges against empirical common support $\mathcal{S}_{\mathrm{common}} = \operatorname{supp}(X_0) \cap \operatorname{supp}(X_1)$.
- **Exp 4 (Capacity Sensitivity Analysis):** Multi-seed capacity sensitivity analysis across BrainKAN hidden dimensions ($d \in \{4, 8, 16\}$ across 5 random initializations), with model parameter counts ($P = 589,779$ vs. standard GNN baselines) reported separately.
- **Exp 5 (Seed-Marginalized Permutation Testing, SMPT):** Multi-seed evaluation across $S = 10$ independent random seeds with $B = 15$ permutations per seed under **dual estimands**:
  1. *Estimand 1 (Pooled Grid $\mathcal{S}_{\mathrm{pooled}}$):* Evaluates $\bar{\Delta}_{\mathrm{pool}}$ over the full $1\%-99\%$ quantile range of combined inputs.
  2. *Estimand 2 (Strict Common Support $\mathcal{S}_{\mathrm{common}}$):* Evaluates $\bar{\Delta}_{\mathrm{common}}$ strictly on the intersection $\operatorname{supp}(X_0) \cap \operatorname{supp}(X_1)$, eliminating boundary extrapolation noise.

---

<!-- ========================================================================= -->
<!-- PAGE 2: PRIMARY RESULTS, INFERENTIAL LIMITATIONS, AND CODE PATHS          -->
<!-- ========================================================================= -->

## Page 2: Primary Results, Inferential Limitations, and Codebase Provenance

### 5. Primary Quantitative Findings

| Experiment | Estimand & Target | Sample Budget | Test Statistic / Effect Size | Uncertainty / p-value | Key Scientific Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1: Permutation Baseline** | Single-seed ($s=42$) excess on $\mathcal{S}_{\mathrm{pooled}}$ (whole-graph average) | $N=500$ permutations | $T_{\mathrm{real}} = 0.17895$<br>$\bar{T}_{\mathrm{null}} = 0.17733$<br>$\Delta_{\mathrm{pool}} = +0.00162$ | $k = 2 / 500$<br>$p = 0.0060$ | Conditional statistical significance for Seed 42 at global level. Fails to generalize across optimization seeds. |
| **Exp 2: Synthetic Calibration** | Gauge sensitivity on $f^*(x)=x+0.5\sin(2x)$ | $N=200$ simulations | **2-A (Uniform):** $\Delta = -0.0022$<br>**2-B (Shifted):** $\Delta = +0.0816$<br>**2-C (Density):** $\Delta = +0.0175$ | $\mathrm{SD} = 0.0084$ on shifted domain | Domain shift creates $>40\times$ apparent inflation on identical model. DFL is gauge-dependent. |
| **Exp 3: Common Support** | HCP $\to$ CHCP cross-cohort transfer | 5-fold CV | HCP domain: $\Delta = +0.0013$<br>Common support: $\Delta = -0.0003$<br>CHCP transfer: $\Delta = -0.0003$ | 5-fold mean | Support restriction eliminates extrapolation; excess curvature is non-distinguishable from zero. |
| **Exp 4: Capacity Analysis** | BrainKAN hidden dimension scan ($d \in \{4, 8, 16\}$) | 5 seeds $\times$ 3 dims | Real $T \approx 0.176 - 0.183$<br>Null $T \approx 0.177 - 0.183$ | High test accuracy ($90\% - 93\%$) | Baseline curvature ($T \approx 0.18$) is allocated by capacity across all widths; parameter counts reported separately. |
| **Exp 5: SMPT (Estimand 1)** | Multi-seed marginal $\bar{\Delta}_{\mathrm{pool}}$ on $\mathcal{S}_{\mathrm{pooled}}$ | $S=10$ seeds<br>$B=15$ per seed | $\bar{T}_{\mathrm{real}} = 0.17939$<br>$\bar{T}_{\mathrm{null}} = 0.17998$<br>$\bar{\Delta}_{\mathrm{pool}} = \mathbf{-0.00058}$ | 95% $t$-CI: $[-0.00130, +0.00013]$<br>$t = -1.8493, p = 0.9513$<br>**Positives: 1/10 seeds** | Under pooled grid, tail extrapolation noise dominates; 9/10 seeds negative. Single-seed inference fails. |
| **Exp 5: SMPT (Estimand 2)** | Multi-seed marginal $\bar{\Delta}_{\mathrm{common}}$ on $\mathcal{S}_{\mathrm{common}}$ | $S=10$ seeds<br>$B=15$ per seed | $\bar{T}_{\mathrm{real}} = 0.08595$<br>$\bar{T}_{\mathrm{null}} = 0.08505$<br>$\bar{\Delta}_{\mathrm{common}} = \mathbf{+0.00090}$ | 95% $t$-CI: $[+0.00051, +0.00129]$<br>$t = 5.1862, p = 0.0003$<br>**Positives: 9/10 seeds** | Boundary extrapolation noise eliminated; 9/10 seeds positive. Evidence of consistent excess curvature. |

---

### 6. Inferential Limitations & Methodological Boundary Conditions
To preserve scientific integrity, our findings are framed within explicit inferential boundaries:
1. **Optimization Sensitivity vs. Population Generalization:**  
   Exp 5 evaluates $S = 10$ optimization initializations on a fixed 100-subject cohort. It demonstrates **initialization stability**, not population-level generalizability across unseen patient cohorts.
2. **Operational Residual Ratio vs. Differential Geometry:**  
   $\mathrm{DFL} = 1 - R^2$ is an operational metric of unexplained variance against affine regression over a discrete grid, **not** coordinate-invariant differential Riemannian curvature.
3. **Finite Permutation Precision & Shared Vectors:**  
   In Exp 5, each seed evaluated $B = 15$ permutations sharing the same flip vectors ($\mathrm{SE}_{\mathrm{MC}} \approx 0.00015$). Exp 1 provides high-precision anchor ($B=500, p=0.0060$).
4. **Representation Inspection vs. Held-Out Generalization:**  
   DFL inspects parameter geometry in internal representation space, not test set classification accuracy. 5-fold CV serves as a subset sensitivity check.
5. **Empirical Fallback Numerical Guard Audit:**  
   The boundary guard `if c_min >= c_max:` was audited across all 5 folds $\times$ 48 ROIs ($240$ instances): **$0 / 240$ triggered ($0.00\%$)**. In 100% of cases, non-empty condition overlaps existed.

---

### 7. Programmatic Provenance & Codebase Path Index

| Component | Repository Path | Functional Responsibility |
| :--- | :--- | :--- |
| **Core Architecture** | [`models/kan.py`](../models/kan.py) | Custom PyTorch Geometric Layer 1 B-spline EdgeConv ($34,576$ params) & BrainKAN ($589,779$ params) |
| **Baseline Architectures** | [`models/baseline.py`](../models/baseline.py) | Canonical GCNBaseline (338 params) and GATBaseline (234 params) architectures |
| **Parameter Verification** | [`compute_baseline_params.py`](../compute_baseline_params.py)<br>[`tests/test_model_parameters.py`](../tests/test_model_parameters.py) | Analytical & runtime parameter counting script and automated test assertions |
| **Data Loader & Verification** | [`utils_data_loading.py`](../utils_data_loading.py) | Subject-paired loader, CV split validator, and manifest integrity assertion |
| **Subject Pairing Manifest** | [`data/subject_pairing_manifest.csv`](../data/subject_pairing_manifest.csv) | Full provenance for 100 paired HCP subjects across 5 folds |
| **Exp 1: Permutation Test** | [`experiments/exp1_high_precision_permutation.py`](../experiments/exp1_high_precision_permutation.py)<br>[`experiments/aggregate_exp1.py`](../experiments/aggregate_exp1.py) | Distributed chunked 500-permutation runner & exact double-precision aggregator |
| **Exp 2: Synthetic Calibration** | [`experiments/exp2_synthetic_calibration.py`](../experiments/exp2_synthetic_calibration.py) | Ground-truth simulation script testing grid gauge dependence ($N=200$) |
| **Exp 3: Common Support** | [`experiments/exp3_reference_domain_control.py`](../experiments/exp3_reference_domain_control.py) | Cross-cohort domain bounding and HCP $\to$ CHCP transfer evaluation |
| **Exp 4: Capacity Scan** | [`experiments/exp4_multi_seed_robustness.py`](../experiments/exp4_multi_seed_robustness.py) | Multi-seed hidden dimension scan ($d \in \{4, 8, 16\}$) across 5 initializations |
| **Exp 5: SMPT Multi-Seed** | [`experiments/exp5_runner.py`](../experiments/exp5_runner.py)<br>[`experiments/run_exp5_parallel.py`](../experiments/run_exp5_parallel.py) | Multi-seed parallel worker and dual-estimand Student's $t$ orchestrator |
| **Support Overlap Audit** | [`experiments/compute_support_overlap_empirical.py`](../experiments/compute_support_overlap_empirical.py) | Evaluates 48 ROI empirical distributions (`results/primary_analysis/node_support_overlap_stats.csv`) |
| **Dynamic Figure Generation** | [`experiments/plot_four_core_figures.py`](../experiments/plot_four_core_figures.py) | Generates Figures 1–4 at 300 DPI directly from result CSV matrices |
| **Primary Results Storage** | [`results/exp1_high_precision/`](../results/exp1_high_precision/)<br>[`results/exp5_seed_stability/`](../results/exp5_seed_stability/) | Canonical experimental outputs and aggregated summary CSVs |
| **Archived Pilot Runs** | [`results/archived_runs/pilot_unpermuted_diagnostic/`](../results/archived_runs/pilot_unpermuted_diagnostic/) | Archived pre-test dry runs ($N_{\mathrm{perms}}=0, \mathrm{effect}=\mathrm{NaN}$) |
| **Execution Manifest** | [`results/manifest.json`](../results/manifest.json) | Full commit SHA `4f6c2e8caba09fc7d0ddcda503da87f380424b3f` & environment metadata |
| **Audit Whitepaper** | [`docs/critical_methodological_defense_and_bounds.md`](../docs/critical_methodological_defense_and_bounds.md) | Exhaustive methodological defense addressing review critiques and bounds |
| **Audit Report (All Closed)** | [`docs/audit_report.md`](../docs/audit_report.md) | Formally closed code and data audit report with resolution commits |
