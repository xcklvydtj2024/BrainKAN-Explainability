# Statistical & Theoretical Assumptions
### BrainKAN Interpretability & Diagnostic Benchmark

This document formalizes the statistical, inferential, and mathematical assumptions underlying the BrainKAN Benchmark. It explicitly defines the inferential boundaries, testable null hypotheses, and operational conventions required for rigorous geometric interpretability in neural networks.

---

## 1. Permutation Null Hypothesis & Exchangeability Boundaries

### Definition
We test the **sharp null hypothesis of conditional independence** between subject-level task conditions $Y$ and observed neural activity $X$:
$$H_0: Y \perp X \quad \Longrightarrow \quad (X_i, Y_i)_{i=1}^n \overset{d}{=} (X_i, \pi(Y)_i)_{i=1}^n$$
where $\pi \in \Pi$ represents a paired sign-flip permutation: for paired task blocks within subject $i$ (0-Back vs. 2-Back), $\pi$ either preserves or swaps $(y_{i,0}, y_{i,1})$ with probability $0.5$. This design preserves subject-level graph correlation structures and intra-individual covariates.

### Rigorous Inferential Boundary
- **What this null tests:** Comparing the real model geometric curvature $T(X, Y)$ against the null ensemble $T(X, \pi(Y))$ evaluates whether **training with intact task labels produces excess spline curvature over training with destroyed task-label associations**.
- **What this null does NOT test:** It does **not** construct an isolated test of "non-linear task mechanisms given linear task correlation." If a task feature has a linear association with $Y$, permuting $Y$ destroys *both* linear and non-linear associations. 
- **Scientific Claim:** Consequently, $\Delta_{\mathrm{task}} = T_{\mathrm{real}} - T_{\mathrm{null}}$ represents **task-associated model-level excess curvature**, rather than an isolated biological claim of non-linearity exclusive of linear signal.

---

## 2. Inherent Capacity Baseline ("Nuisance Curvature Budget")

### Statement
Neural networks with flexible parametrizations (such as B-spline activation functions in KANs) possess an intrinsic inductive bias towards non-linear warping, even when fitting pure noise or permuted labels:
$$\mathbb{E}_{\pi}[G(f_{\mathrm{null}}, X)] = T_{\mathrm{baseline}} > 0$$

### Empirical Benchmark Finding
Across all evaluated folds and random initializations, BrainKAN models converge to a remarkably invariant curvature budget:
$$T_{\mathrm{baseline}} \approx 0.179 \pm 0.001$$
Evaluating raw $\mathrm{DFL}(f_{\mathrm{real}})$ without subtracting the paired null baseline leads to the false conclusion that the network discovered biological non-linearities, when in reality it merely consumed its predefined capacity budget.

---

## 3. Reference Domain Anchoring & Support Overlap

### Formulation
Spline evaluation metrics like Deviation-From-Linearity ($\mathrm{DFL}$) are non-local functional metrics that depend explicitly on the evaluation domain $\mathcal{S} \subset \mathbb{R}$:
$$\operatorname{DFL}(f, \mathcal{S}) = \frac{\sum_{t \in \mathcal{S}} (f(t) - \hat{y}_{\mathrm{linear}}(t))^2}{\sum_{t \in \mathcal{S}} (f(t) - \bar{y})^2}$$

### Anchored Operational Accounting
To provide a concrete reference standard against which domain sensitivity can be audited, we define an **anchored operational decomposition** relative to a predefined canonical support $\mathcal{S}_0$:
$$G(f, \mathcal{S}) = G(f, \mathcal{S}_0) + \mathcal{D}(\mathcal{S}; \mathcal{S}_0, f)$$
where:
- $G(f, \mathcal{S}_0)$ is the baseline curvature profile over the canonical anchor $\mathcal{S}_0$.
- $\mathcal{D}(\mathcal{S}; \mathcal{S}_0, f) \equiv G(f, \mathcal{S}) - G(f, \mathcal{S}_0)$ is the domain-shift discrepancy.
*Remark:* This formulation is a definitional accounting identity (gauge fixing) rather than an unconstrained algebraic uniqueness theorem. For non-linear splines ($\Phi''(x) \not\equiv 0$), domain shifts induce non-zero $\mathcal{D}$, whereas for purely affine functions ($\Phi(x) = ax + b$), DFL is identically zero regardless of domain choice.

### Empirical Support Overlap in Real fMRI Data
Auditing the empirical support of 0-Back and 2-Back across 48 cortical ROIs (`results/primary_analysis/node_support_overlap_stats.csv`) reveals:
- Mean cohort-aggregated condition Jaccard overlap: **$38.52\% \pm 5.08\%$** (minimum $28.59\%$, maximum $52.14\%$).
- The intersection covers only **$43.47\%$** of the pooled 1%–99% quantile domain.
- As proven in Exp 2-B, evaluating a fixed non-linear function on shifted domains introduces artificial $\Delta_{\mathrm{DFL}}$ inflation up to **$+0.0816$** (>40x larger than the true task effect), demonstrating that unconstrained empirical support domains introduce major evaluation artifacts.

---

## 4. Optimization Stochasticity & Multi-Seed Inference

### Estimand Distinction
Because deep neural networks navigate non-convex loss landscapes, learned spline parameters are realization-dependent:
1. **Conditional Estimand (Fixed Seed $s$):**
   $$\Delta(s) = T_{\mathrm{real}}(s) - \bar{T}_{\mathrm{null}}(s)$$
   For Seed 42, $\Delta(42) = +0.00172$, achieving conditional significance ($p = 0.0060$ under 500 permutations).
2. **Optimization-Marginalized Estimand:**
   $$\bar{\Delta} = \mathbb{E}_{s \sim \mathcal{S}} [\Delta(s)]$$

### Inferential Qualification
- **Sampling Scope:** The 10 seeds evaluated in Exp 5 represent independent optimization initializations on a **fixed dataset and fixed 5-fold cross-validation split**. They quantify algorithmic training stochasticity, not inter-subject biological cohort variation.
- **Null Distribution Structure:** Each seed computes a paired null mean across 15 label permutations sharing a common pre-generated flip-vector bank.
- **Statistical Test:** The cross-seed summary ($\bar{\Delta} = -0.00058$, $95\%\text{ CI } [-0.00130, +0.00013]$, $p = 0.9513$) is derived from a one-sample Student's $t$-test ($df=9$) across seed-level differences.
- **Conclusion:** Under the predefined one-sided alternative, the marginal effect does not support rejection of the null hypothesis. Conditional significance at a single seed cannot be generalized across optimization randomness.
