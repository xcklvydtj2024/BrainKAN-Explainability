# Statistical & Theoretical Assumptions
### BrainKAN Interpretability Benchmark

This document details the formal mathematical and statistical assumptions underlying the BrainKAN Interpretability Benchmark, defining the conditions required for valid non-linear interpretability.

---

## Assumption 1: Exchangeability of Label Permutations
**Statement:** Under the null hypothesis $H_0$ (no task-associated non-linear structure), the distribution of the geometric test statistic $G(f, X)$ is invariant under the permutation of task labels $Y$:
$$(X_i, Y_i)_{i=1}^n \overset{d}{=} (X_i, \pi(Y)_i)_{i=1}^n$$
**Condition:** Labels are randomized globally at the subject level via paired sign-flips to preserve subject-level graph correlation structures.

---

## Assumption 2: Null Preserves Nuisance Geometry
**Statement:** The label-shuffled null model $f_{\mathrm{null}}$ successfully captures the baseline geometric deformation induced by model expressivity and input feature routing:
$$G(f_{\mathrm{null}}, X) = D(X, f_{\mathrm{null}})$$
**Empirical Validation:** BrainKAN models demonstrate an invariant baseline curvature budget of $T \approx 0.179 \pm 0.001$ across both real and null labels, confirming that the network inherently consumes a fixed non-linear capacity budget.

---

## Assumption 3: Common Support Evaluation Domain
**Statement:** Spline evaluations must be restricted strictly to the common support between task classes:
$$\mathcal{S}_{\mathrm{common}} = \operatorname{supp}(X_0) \cap \operatorname{supp}(X_1)$$
**Violation Consequence:** As demonstrated in Exp 2-B, evaluating outside the common support causes artificial DFL inflation up to $+0.1195$, invalidating any biological interpretation.

---

## Assumption 4: Optimization Marginalization (SMPT)
**Statement:** Valid population-level inference regarding task-associated function properties requires marginalizing over the optimization distribution $\mathcal{S}$:
$$\Delta_{\mathrm{marginal}} = \mathbb{E}_{s \sim \mathcal{S}} \left[ G(f_{\mathrm{real}}^{(s)}, X) - \mathbb{E}_\pi [G(f_{\mathrm{null}}^{(s)}, X)] \right]$$
**Violation Consequence:** Conditioning on a single initialization seed yields conditional significance (e.g., Seed 42, $p = 0.0060$) that fails to generalize across independent initializations (Exp 5: $\bar{\Delta} = -0.00058$, $p = 0.9513$).
