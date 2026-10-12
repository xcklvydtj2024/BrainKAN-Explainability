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
- As proven in Exp 2-B, evaluating a fixed model (fitted to $f^*(x) = x + 0.5\sin(2x)$) on shifted domains introduces artificial $\Delta_{\mathrm{DFL}}$ inflation up to **$+0.0816 \pm 0.0084$** (>40x larger than typical task differences), demonstrating that unconstrained empirical support domains introduce major evaluation artifacts.

---

## 4. Optimization Stochasticity & Multi-Seed Inference

### Estimand Distinction
Because deep neural networks navigate non-convex loss landscapes, learned spline parameters are realization-dependent:
1. **Conditional Estimand (Fixed Seed $s$):**
   $$\Delta(s) = T_{\mathrm{real}}(s) - \bar{T}_{\mathrm{null}}(s)$$
   In Exp 1 ($B=500$ permutations, Seed 42), the model achieved conditional significance ($\Delta = +0.00162, p = 0.00599 \approx 0.0060$). In Exp 5's 15-permutation screening, Seed 42 yielded $\Delta(42) = +0.00172$.
2. **Optimization-Marginalized Estimand:**
   $$\bar{\Delta} = \mathbb{E}_{s \sim \mathcal{S}} [\Delta(s)]$$

### Empirical Benchmark Findings (Exp 5: Dual Estimand Contrast)
Evaluating across 10 independent initialization seeds under a fixed data split:
- **Pooled Reference Domain ($\mathcal{S}_{\mathrm{pooled}}$):** $\bar{\Delta}_{\mathrm{pool}} = -0.00058 \pm 0.00100$ (Student's $t$ $df=9$: $t = -1.849$, one-sided $p = 0.9513$, 95% CI: $[-0.00130, +0.00013]$). Only 1 of 10 seeds (Seed 42) exhibited $\Delta > 0$. Boundary extrapolation artifacts dominate.
- **Empirical Common Support ($\mathcal{S}_{\mathrm{common}}$):** $\bar{\Delta}_{\mathrm{common}} = +0.00090 \pm 0.00055$ (Student's $t$ $df=9$: $t = +5.186$, one-sided $p = 0.0003$, 95% CI: $[+0.00051, +0.00129]$). 9 of 10 seeds exhibited $\Delta > 0$.

### Inferential Qualification
- **Sampling Scope:** The 10 seeds evaluated in Exp 5 represent independent optimization initializations on a **fixed dataset and fixed 5-fold cross-validation split**. They quantify algorithmic training stochasticity, not inter-subject biological cohort variation.
- **Null Distribution Structure:** Each seed computes a paired null mean across 15 label permutations sharing a common pre-generated flip-vector bank.
- **Conclusion:** On the pooled domain, single-seed conditional significance at Seed 42 fails to generalize across initializations due to boundary extrapolation noise. Restricting evaluation to empirical common support isolates consistent positive excess curvature across 9/10 initializations, demonstrating that scientific conclusions depend decisively on estimand specification.
