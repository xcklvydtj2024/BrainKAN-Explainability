# Anticipated Theoretical & Methodological Questions
### BrainKAN Interpretability Benchmark

This document addresses key theoretical and statistical questions regarding the BrainKAN Interpretability Benchmark. It clarifies the boundaries between conditional optimization artifacts, reference-domain dependencies, and population-level scientific claims.

---

## Q1: Why does null calibration not equate to identifying biological truth?
**Answer:** It strictly does **not** identify in-vivo biological mechanisms. 
What the null calibration isolates is the *excess structural deviation relative to the empirical null expectation under the specific optimization procedure*. The null model (trained on label-permuted data) preserves the graph topology and architectural flexibility while destroying task supervision. A residual metric indicates whether the learned geometry is associated with the training labels. It proves how the computational model routes information in-silico, not that the biological brain executes this exact mathematical function locally.

---

## Q2: Why can a single seed (e.g., Seed 42) achieve $p = 0.006$, yet fail to generalize across seeds?
**Answer:** Because of an **Estimand Mismatch** (Conditional significance without population-level generalization). 
Seed 42's result is not a computational mistake or a statistical false positive. It validly tests the conditional null hypothesis:
$$H_0^{(s=42)}: \Delta_{s=42} = 0$$
For that specific initialization basin, the real-label model converged to higher curvature than its paired null models. However, the scientific hypothesis of interest is the *marginal population estimand*:
$$H_0: \mathbb{E}_{s \sim \mathcal{S}} [\Delta_s] \le 0$$
When marginalizing across 10 independent initialization seeds under the pooled domain, 9 out of 10 seeds yield negative differences ($\bar{\Delta}_{\mathrm{pool}} = -0.00058$, $p = 0.9513$). By contrast, when evaluated strictly on empirical common support ($\mathcal{S}_{\mathrm{common}}$), 9 out of 10 seeds consistently yield positive excess curvature ($\bar{\Delta}_{\mathrm{common}} = +0.00090$, paired $t = 5.186, p = 0.0003$). This estimand duality demonstrates that single-seed inference on pooled domains reflects boundary extrapolation artifacts rather than a stable invariant.

---

## Q3: Why is nonlinearity measurement reference-domain dependent?
**Answer:** Because curvature and Deviation-From-Linearity (DFL) are operators conditioned on the evaluation support. 
As demonstrated in Synthetic Calibration (Exp 2-B), evaluating the exact same learned model (fitted to $f^*(x) = x + 0.5\sin(2x)$) over shifted reference domains causes apparent DFL to jump by $+0.0816 \pm 0.0084$—an artifactual inflation over 40 times larger than real model differences. Unless evaluation is strictly bounded to the empirical common support ($[q_1^{(0)}, q_{99}^{(0)}] \cap [q_1^{(1)}, q_{99}^{(1)}]$), evaluation domain shifts will generate spurious nonlinear interpretations.

---

## Q4: What is the true invariant learned by BrainKAN?
**Answer:** The intrinsic baseline curvature budget ($T \approx 0.179 \pm 0.001$). 
Regardless of initialization seeds, cross-validation folds, or label permutations, the edge B-splines consistently allocate a substantial degree of functional nonlinearity to route input node features. The model's capacity naturally utilizes curved representations, but task supervision does not drive excess curvature beyond this baseline.

---

## Q5: What is the recommended standard for future interpretable KAN research?
**Answer:** The Seed-Marginalized Permutation Testing (SMPT) with Common Support protocol. 
Future studies evaluating spline nonlinearity must: (1) enforce strict common-support domain bounds to prevent extrapolation artifacts, (2) compute paired permutation nulls, and (3) report the seed-marginalized effect $\mathbb{E}_s[\Delta_s]$ across multiple random initializations.
