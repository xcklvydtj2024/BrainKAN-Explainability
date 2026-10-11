# Methodological Assumptions, Boundary Conditions, and Empirical Audit

**Author:** Xin Qi  
**Benchmark:** BrainKAN-Explainability Benchmark  
**Date:** October 2026  
**Document Classification:** Methodological Analysis & Statistical Audit

---

## Executive Summary

This document provides a formal, balanced methodological audit addressing critical questions regarding the inferential validity, statistical estimands, data provenance, and mathematical assumptions of the BrainKAN Explainability Benchmark.

The benchmark's primary objective is **methodological diagnosis**: examining whether post-hoc interpretability metrics in graph neural networks reflect genuine structural invariants versus evaluation artifacts (such as reference domain shifts and optimization stochasticity). Below, we systematically document the exact empirical evidence, mathematical definitions, and inferential boundaries of each experimental component.

---

## 1. Optimization Sensitivity vs. Population Uncertainty (Estimand Demarcation)

### Context & Question
The 10 seeds evaluated in Exp 5 do not represent 10 independent subject cohorts. The observed statistics are derived from fixed data under a fixed 5-fold cross-validation split across optimization initializations.

### Formal Demarcation of Estimands
We distinguish between two distinct sources of variation and their corresponding estimands:

$$\Delta(s, \mathcal{D}) = T(Y; s, \mathcal{D}) - \mathbb{E}_\pi [T(\pi(Y); s, \mathcal{D})]$$

1. **Optimization Sensitivity Estimand (Conditional on Sample Cohort $\mathcal{D}$):**
   $$\bar{\Delta}_{\mathrm{opt}}(\mathcal{D}) = \mathbb{E}_{s \sim \mathcal{S}} \left[ \Delta(s, \mathcal{D}) \mid \mathcal{D} \right]$$
   - Conditioned on the fixed benchmark cohort $\mathcal{D}$, the fixed 5-fold split, a fixed training configuration (15 epochs, Adam optimizer, $\text{lr} = 0.005$, matching the exact runtime specification in `exp5_runner.py`), and a fixed set of 15 permutation flip vectors, this estimand evaluates consistency across random weight initializations ($w_0 \sim \mathcal{S}$) and stochastic batch sequencing.
   - **Empirical Finding in Exp 5:** On the strict common support grid, 9 out of 10 seeds yield a positive DFL difference ($\bar{\Delta}_{\mathrm{common}} = +0.00090 \pm 0.00055$, paired $t = 5.186, p = 0.0003$).
   - **Appropriate Interpretation:** This demonstrates consistency of the positive difference across the tested initializations under this specific optimization regime.
   - **Boundary Conditions:**
     - A fixed budget of 15 training epochs does not constitute mathematical proof of optimization convergence.
     - Conditioning on a shared set of 15 permutation vectors fixes the reference null baseline across seeds; the cross-seed $t$-interval captures initialization variance given this baseline, but does not cover the shared uncertainty of drawing an alternative set of permutation vectors.

2. **Population Generalization Estimand:**
   $$\bar{\Delta}_{\mathrm{pop}} = \mathbb{E}_{\mathcal{D} \sim \mathcal{P}} \left[ \mathbb{E}_{s \sim \mathcal{S}} [\Delta(s, \mathcal{D})] \right]$$
   - This measures expectation across independent draws of subject cohorts from the biological population $\mathcal{P}$.
   - **Inferential Boundary:** Exp 5 **does not** test $H_0^{\mathrm{pop}}$. Subject-level resampling (such as subject-level bootstrap or repeated nested cross-validation) can evaluate sampling uncertainty within the target population, while external clinical cohorts evaluate transfer across acquisition sites. We explicitly restrict our claims to initialization sensitivity on the benchmark dataset.

---

## 2. Permutation Budget, Monte Carlo Precision, and Shared Null Vectors

### Context & Question
Exp 5 evaluates $B = 15$ permutations per seed, and all 10 seeds share the exact same 15 permutation flip vectors. What are the precision limits and baseline uncertainties?

### Technical Audit & Methodological Rationale
- **Computational Budget:** Each seed trains $5 \text{ folds} \times (1 \text{ Real} + 15 \text{ Null}) = 80$ deep neural network models (589,779 parameters each). Running 10 seeds required training **800 models** (over 4 hours of CPU compute).
- **Shared Permutation Vectors:** Seeds share the same 15 flip vectors (`global_flip_vectors`, generated deterministically with seeds $2026 + b$). While this controls the label permutation sequence across initializations (allowing paired comparisons across seeds), it induces covariance between the null estimates across seeds.
- **Monte Carlo Precision Limits & Baseline Uncertainty:**
  - *Exp 1 Baseline (Pooled Domain, Single Seed):* With $B = 500$ independent permutations, the null values yielded sample mean $0.17733$ and sample standard deviation $\sigma_{\mathrm{null}} = 0.000585$, giving $\mathrm{SE}_{\mathrm{MC}} = 0.000585 / \sqrt{500} \approx 0.000026$. With $k = 2$, $p = (1 + 2)/(500 + 1) = 3/501 \approx 0.005988 \approx 0.0060$.
  - *Inferential Distinction:* Exp 1 is a single-seed pooled-domain test, and does not calibrate the multi-seed common-support results of Exp 5. They address distinct questions.
  - *Exp 5 Monte Carlo Uncertainty (示意估算 / Illustrative Proxy):* In Exp 5, experimental logs recorded the seed-level mean null statistic $\hat{\mu}_{\mathrm{null}, s}$ rather than individual permutation runs. If we borrow the within-seed null dispersion scale from Exp 1 ($\sigma_{\mathrm{null}} \approx 0.00058$) as an illustrative order-of-magnitude proxy, the Monte Carlo standard error of a 15-permutation null mean is roughly $\mathrm{SE}_{\mathrm{MC}} \approx 0.00058 / \sqrt{15} \approx 0.00015$. This illustrative proxy highlights that finite permutation error is non-negligible relative to $\bar{\Delta}_{\mathrm{common}} \approx 0.00090$; it should not be construed as a directly measured within-seed standard deviation on the common domain.
- **Methodological Positioning:** Exp 5 is an **algorithmic sensitivity audit** across initializations under a fixed set of permutations, not an asymptotic non-parametric permutation test ($B \to \infty$).

---

## 3. Empirical Audit of the Common Support Fallback Guard

### Context & Question
In `exp5_runner.py`, when the intersection of condition supports is empty (`c_min >= c_max`), the code contains a fallback to the pooled quantile range. How frequently was this triggered in practice?

### Quantitative Audit Across 100% of Evaluated Instances
We audited all 5 cross-validation folds across all 48 cortical ROIs ($5 \times 48 = 240$ instances):

```python
# Verification loop across all 5 folds and 48 ROIs:
c_min = max(c0_min, c1_min)
c_max = min(c0_max, c1_max)
if c_min >= c_max:
    fallback_count += 1
```

- **Total Node-Fold Instances Evaluated:** 240
- **Instances Triggering Fallback (`c_min >= c_max`):** **0 / 240 (0.00%)**
- **Empirical Overlap Distribution:** Across all 48 nodes, the Jaccard overlap between 0-Back and 2-Back empirical distributions ranged from $28.59\%$ to $52.14\%$ (mean $38.52\% \pm 5.08\%$).
- **Conclusion:** The fallback condition `if c_min >= c_max` is purely a **defensive numerical guard** for degenerate distributions. In the actual HCP neuroimaging data, the intersection was non-empty in **100.0% of instances**. Every reported DFL value on $\mathcal{S}_{\mathrm{common}}$ was computed strictly on non-empty condition intersections.

---

## 4. Evaluation Reference Domain: Motivation, Limitations, and Gauge Invariance

### Context & Question
The common support grid is determined using true condition labels and kept fixed during label permutation. Does this prove valid error-rate control under permutation?

### Methodological Distinction: Measurement Comparability vs. Error Rate Control
1. **Measurement Comparability Motivation:**  
   Fixing the evaluation domain $\mathcal{S}_{\mathrm{common}} = \operatorname{supp}(X_0) \cap \operatorname{supp}(X_1)$ ensures that the real model and all null models are evaluated over the exact same spatial interval $[c_{\min}, c_{\max}]$. If the evaluation interval were dynamically recomputed for each shuffled label vector $\pi(Y)$, the shuffled condition distributions would mix, altering the evaluation domain across permutations and conflating representation differences with domain boundary shifts.
2. **Inferential Limitation:**  
   While fixing the evaluation grid achieves measurement comparability, it **does not formally prove valid type I error rate control** under permutation testing, because the true condition labels were used to construct the reference domain.
3. **Appropriate Positioning:**  
   This procedure is an **empirical model comparison and sensitivity analysis under a fixed empirical reference domain**. A fully calibrated, end-to-end permutation test would require determining the evaluation domain on an independent sample split, or validating error rate control via null simulations across the complete pipeline.

---

## 5. Representation Parameter Inspection vs. Held-Out Generalization Error

### Context & Question
Exp 5 inspects fold model parameters rather than computing test set prediction metrics. Can this be described as held-out generalization?

### Explicit Clarification
- **Representation Diagnostics vs. Prediction Metrics:**  
  Deviation-From-Linearity ($\mathrm{DFL}$) inspects the geometric properties of the network's internal edge B-spline functions $\Phi_{ij}(t)$. It is an **internal representation diagnostic** in parameter space, not a held-out classification error or predictive generalization metric.
- **Role of 5-Fold Cross-Validation:**  
  The 5 training subsets overlap by $75\%$ (each pair of folds shares 3 out of 4 training splits). This setup provides a **subset sensitivity check** (verifying that the learned spline shapes are not entirely driven by an arbitrary 20% slice of subjects), but does not represent independent cohort validation.

---

## 6. Sample Pairing Manifest and Loader Verification

### Context & Question
Does the data loader verify biological pairing, or merely class alternation?

### Verification Pipeline (`data/subject_pairing_manifest.csv`)
We released a structured sample pairing manifest and integrated complete validation into the data loader:

- **Manifest File:** [`subject_pairing_manifest.csv`](../data/subject_pairing_manifest.csv)
- **Cohort Structure:** 100 paired subjects (`sub-HCP100-001` through `sub-HCP100-100`).
- **Data Loader Validation (`utils_data_loading.py`):**
  The loader explicitly verifies:
  1. Row count matches $N_{\mathrm{subjects}} = 100$ ($200$ samples total).
  2. Subject IDs are unique across all 100 pairs.
  3. Sample index mapping ($2i$ and $2i+1$).
  4. Condition labels match binary pairs $\{0, 1\}$.
  5. Dimension consistency ($48$ ROIs, $240$ edges).
  6. Cross-validation test fold assignment (verifying each row's `cv_test_fold` against deterministic runtime `KFold(n_splits=5, shuffle=True, random_state=42)`).
- **Provenance Scope:** Upstream raw HCP subject mapping (subject ID, scan session) is maintained in controlled local storage pursuant to HCP Open Access Data Use Terms. This manifest documents the sequential pairing, dimensional integrity, and cross-validation splits used within the benchmark.

---

## 7. Mathematical Definition of DFL: Operational Residual Ratio vs. Differential Curvature

### Context & Question
Is DFL ($1 - R^2$) equivalent to intrinsic differential curvature?

### Mathematical Definition
1. **Operational Definition:**
   $$\mathrm{DFL}(f, \mathcal{S}) = \frac{\inf_{a, b} \sum_{k=1}^K |f(t_k) - (at_k + b)|^2}{\sum_{k=1}^K |f(t_k) - \bar{f}|^2} = 1 - R^2$$
   - DFL measures the proportion of sample variance unexplained by the best-fitting affine linear function on a discrete probe grid $\{t_k\}_{k=1}^K \subset \mathcal{S}$.
2. **Mathematical Precision & Degeneracy Conditions:**
   - **Non-zero variance requirement:** DFL is well-defined when $\mathrm{SS}_{\mathrm{tot}} = \sum_{k=1}^K |f(t_k) - \bar{f}|^2 > 0$. For a constant function ($f(t) \equiv c$), $\mathrm{SS}_{\mathrm{tot}} = 0$, yielding $0/0$. In our codebase, constant functions are assigned $\mathrm{DFL} = 0$ by operational convention.
   - **Discrete Grid Limitation:** $\mathrm{DFL} = 0$ on discrete probe points guarantees colinearity at those specific sample locations, but does not guarantee that the continuous spline is affine everywhere between grid points.
3. **Contrast with Differential Curvature:**
   - Intrinsic Riemannian / differential curvature $\kappa(t) = |f''(t)| / (1 + f'(t)^2)^{3/2}$ is a localized, coordinate-dependent differential operator.
   - In all formal documentation, DFL is designated as an **operational residual variance ratio against affine regression**, not an intrinsic geometric curvature invariant.

---

## 8. Reference Domain Standardization vs. Model Parameter Identifiability

### Context & Question
Does standardizing the evaluation domain resolve neural network parameter identifiability?

### Methodological Concession & Clarification
- **Deep Network Non-Identifiability:**  
  Overparameterized neural networks are non-identifiable in parameter space due to permutation symmetries, scaling invariances, and flat loss basins. Different weight configurations can yield functionally equivalent network predictions.
- **Scope of the Anchored Decomposition:**  
  The anchored decomposition:
  $$G(f, \mathcal{S}) = G(f, \mathcal{S}_0) + \left[ G(f, \mathcal{S}) - G(f, \mathcal{S}_0) \right]$$
  is an **operational accounting identity**, not an identifiability theorem. It isolates empirical measurement variance caused by domain shifting, but does not prove that learned edge splines uniquely identify biophysical mechanisms.

---

## 9. Model Overparameterization ($P \gg N$) and Capacity Calibration

### Context & Question
BrainKAN has ~589k parameters for 100 subjects ($P \gg N$). How should capacity and baselines be interpreted?

### Capacity Accounting & Baseline Qualification
- **Exact Ratio Clarification:**
  - Trainable parameters: $P = 589,779$.
  - Sample size:
    - Computed by subject count: $N_{\mathrm{subjects}} = 100 \implies P / N_{\mathrm{subjects}} \approx 5,898$.
    - Computed by graph count: $N_{\mathrm{graphs}} = 200 \implies P / N_{\mathrm{graphs}} \approx 2,949$.
- **Role of Paired Null Models:**  
  Because $P \gg N$, the network possesses sufficient capacity to absorb label noise. Paired label-permuted null models are trained with identical architecture and hyperparameter budgets on permuted labels. Subtracting the null baseline ($\Delta = T_{\mathrm{real}} - T_{\mathrm{null}}$) evaluates the effect relative to this operational baseline, but cannot guarantee the complete elimination of all overfitting or unobserved confounding.
- **Baseline Alignment:**  
  Existing standard GCN and GAT baselines were not parameter-matched to BrainKAN's 589k parameter budget. We do not claim architectural superiority over tuned graph neural networks; our findings focus on interpretability evaluation methodology.

---

## 10. Summary Matrix of Methodological Commitments

| Topic | Analysis Type | Benchmark Commitment & Scope |
| :--- | :---: | :--- |
| **Optimization Sensitivity ($10$ Seeds)** | Sensitivity Audit | Evaluates initialization variation under fixed cohort and 15 permutations; does not claim population generalization. |
| **Permutation Budget ($B=15$)** | Monte Carlo Approximation | Documented finite-sample error ($\mathrm{SE}_{\mathrm{MC}} \approx 0.00015$); Exp 1 provides high-precision pooled-grid baseline ($B=500, p \approx 0.006$). |
| **Common Support Fallback** | Empirical Audit | Verified to trigger **$0 / 240$ times ($0.00\%$)** across all folds and ROIs on HCP fMRI. Purely a defensive numerical guard. |
| **Fixed Evaluation Grid** | Operational Choice | Provides measurement comparability; formal type I error rate control under label-dependent selection is an open problem. |
| **5-Fold Cross-Validation** | Diagnostic | Tests representation sensitivity to 20% subject hold-out; not an independent test set predictive generalization proof. |
| **Sample Pairing Manifest** | Documentation & Check | Validates sample indices, binary label pairing, and dimensions via [`subject_pairing_manifest.csv`](../data/subject_pairing_manifest.csv). |
| **DFL Definition ($1 - R^2$)** | Operational Metric | Normalized linear variance unexplained on discrete probe grid; not differential Riemannian curvature. |
| **Model Identifiability** | Limitation | Domain control standardizes evaluation coordinates; does not prove parameter identifiability of overparameterized networks. |
| **Overparameterization ($P \gg N$)** | Capacity Baseline | $P = 589,779$ vs. $N_{\mathrm{sub}}=100$ ($P/N \approx 5,898$); paired null models provide an empirical reference baseline. |
