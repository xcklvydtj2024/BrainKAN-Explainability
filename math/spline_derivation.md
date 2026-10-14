# Mathematical Foundations of Spline Geometry & Benchmark Estimands

This document details the theoretical formulation of spline geometric metrics (Deviation-From-Linearity, DFL) and establishes the distinction between conditional and marginal estimands in KAN interpretability.

---

## 1. Deviation-From-Linearity (DFL) & Invariance Properties

Let the empirical computation along an edge $(i, j)$ in a Kolmogorov–Arnold Network (KAN) be parameterized by learnable B-splines:
$$\Phi_{ij}(x) = W_{ij}^{\mathrm{base}}\mathrm{SiLU}(x) + \sum_{k=1}^K w_{ij,k} B_k(x)$$

When evaluating the non-linearity of $\Phi_{ij}$ over a discrete evaluation grid $X = \{t_1, \dots, t_M\} \subset \mathbb{R}$, we compute the Deviation-From-Linearity (DFL):
$$\mathrm{DFL}(\Phi_{ij}, X) \equiv \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}}} = 1 - R^2 = \frac{\sum_{m=1}^M (\Phi_{ij}(t_m) - \hat{y}(t_m))^2}{\sum_{m=1}^M (\Phi_{ij}(t_m) - \bar{y})^2}$$
where $\hat{y}(t) = \hat{\beta}_1 t + \hat{\beta}_0$ is the ordinary least squares (OLS) linear projection of $\Phi_{ij}(t)$ on $t$, and $\bar{y} = \frac{1}{M}\sum_{m=1}^M \Phi_{ij}(t_m)$.

### Affine Scale Invariance Lemma
For any non-zero scalar $a \neq 0$ and constant $b \in \mathbb{R}$, let $\tilde{\Phi}(t) = a \Phi(t) + b$. Because OLS linear regression is scale-equivariant ($\hat{\beta}_1(\tilde{\Phi}) = a \hat{\beta}_1(\Phi)$ and $\hat{\beta}_0(\tilde{\Phi}) = a \hat{\beta}_0(\Phi) + b$), the residuals and total sum of squares scale identically by $a^2$:
$$\mathrm{SS}_{\mathrm{res}}(\tilde{\Phi}) = a^2 \mathrm{SS}_{\mathrm{res}}(\Phi), \qquad \mathrm{SS}_{\mathrm{tot}}(\tilde{\Phi}) = a^2 \mathrm{SS}_{\mathrm{tot}}(\Phi)$$
Consequently, the variance ratio is strictly invariant:
$$\mathrm{DFL}(a \Phi + b, X) = \mathrm{DFL}(\Phi, X)$$
*Remark:* This affine invariance is a fundamental property of the coefficient of determination ($R^2$), ensuring that global linear gains and DC offsets do not register as non-linear curvature.

---

## 2. Anchored Operational Decomposition of Domain Sensitivity

Because $\mathrm{DFL}$ depends on the integration interval, evaluating a function over different domains alters the measured metric. 

### Identifiability Through Anchor Gauge Fixing
Without an explicit anchor, an unconstrained additive decomposition $G(f, X) = S(f) + D(X, f)$ is unidentifiable: for any arbitrary functional $h(f)$, one could set $S' = S + h$ and $D' = D - h$ without changing the sum.

To render the decomposition operationally well-defined and identifiable, we fix a **canonical reference anchor** $\mathcal{S}_0 \subset \mathbb{R}$ (e.g., standard uniform interval $[-1, 1]$ or common support $\mathcal{S}_{\mathrm{common}}$):

1. **Anchored Profile:**
   $$S_{\mathcal{S}_0}(f) \equiv \mathrm{DFL}(f, \mathcal{S}_0)$$
2. **Domain-Shift Discrepancy:**
   $$D(X; \mathcal{S}_0, f) \equiv \mathrm{DFL}(f, X) - \mathrm{DFL}(f, \mathcal{S}_0)$$

Under this anchored gauge, the decomposition is uniquely defined:
$$\mathrm{DFL}(f, X) = S_{\mathcal{S}_0}(f) + D(X; \mathcal{S}_0, f)$$

### Operational Diagnostic Value
This formulation provides an operational diagnostic:
- If a model is evaluated across two conditions using their empirical data distributions $X_A$ and $X_B$, the apparent metric difference is:
  $$\Delta_{\mathrm{apparent}} = \mathrm{DFL}(f_A, X_A) - \mathrm{DFL}(f_B, X_B) = [S_{\mathcal{S}_0}(f_A) - S_{\mathcal{S}_0}(f_B)] + [D(X_A; \mathcal{S}_0, f_A) - D(X_B; \mathcal{S}_0, f_B)]$$
- When $f_A = f_B$ has non-vanishing second or higher derivatives across the shifted domain ($\|f''\|_{L^2(X_A \cup X_B)} > 0$), a domain shift $X_A \neq X_B$ generally induces a non-zero artifactual discrepancy:
  $$\Delta_{\mathrm{apparent}} = D(X_A; \mathcal{S}_0, f) - D(X_B; \mathcal{S}_0, f)$$
  *(Note: For strictly affine functions $f(x) = ax + b$, $\mathrm{DFL}$ remains identically zero on any non-degenerate interval, producing zero domain artifact).*
- As empirically measured in Exp 2-B (`synthetic_summary.csv`) on a fixed model fitted to $f^*(x) = x + 0.5\sin(2x)$, shifting the reference domain alone inflates measured $\Delta_{\mathrm{DFL}}$ by **$+0.0816 \pm 0.0084$** ($t = 136.6$), which is $>40\times$ larger than typical real task condition differences.

---

## 3. Optimization Stochasticity and Estimand Duality

Deep neural networks navigate non-convex loss landscapes. The learned parameters $w_{ij,k}(s)$ are stochastic realizations dependent on the random seed $s \sim \mathcal{S}$ (initialization and mini-batch sequence).

### 1. Conditional Estimand (Fixed Seed $s$)
For a fixed seed $s$, the conditional excess curvature is:
$$\Delta(s) = T_{\mathrm{real}}(s) - \bar{T}_{\mathrm{null}}(s)$$
Testing $H_0^{(s)}: \Delta(s) \le 0$ evaluates whether real labels induced greater curvature *within the specific optimization basin reached by seed $s$*.
- In Exp 1 ($B=500$ permutations, Seed 42), the model achieved conditional significance with $\Delta = +0.00162$ ($p = 0.00599 \approx 0.0060$).
- In Exp 5's 15-permutation screening, Seed 42 yielded $\Delta(42) = +0.00172$.

### 2. Optimization-Marginalized Estimand
The scientific hypothesis of interest across optimization randomness is the marginal expectation:
$$\bar{\Delta} = \mathbb{E}_{s \sim \mathcal{S}} [\Delta(s)] = \mathbb{E}_s [T_{\mathrm{real}}(s)] - \mathbb{E}_{s} [\bar{T}_{\mathrm{null}}(s)]$$

### Empirical Benchmark Findings (Exp 5: Dual Estimand Contrast)
Evaluating across 10 independent initialization seeds under a fixed data split:
1. **Pooled Reference Domain ($\mathcal{S}_{\mathrm{pooled}}$):**
   - Mean effect: $\bar{\Delta}_{\mathrm{pool}} = -0.00058 \pm 0.00100$ ($\mathrm{SE} = 0.00032$)
   - Student's $t$-test ($df=9$): $t = -1.849$, one-sided $p = 0.9513$ (95% CI: $[-0.00130, +0.00013]$)
   - Sign consistency: Only 1 of 10 seeds (Seed 42) exhibited $\Delta > 0$. Boundary extrapolation noise dominates.
2. **Empirical Common Support ($\mathcal{S}_{\mathrm{common}} = [q_1^{(0)}, q_{99}^{(0)}] \cap [q_1^{(1)}, q_{99}^{(1)}]$):**
   - Mean effect: $\bar{\Delta}_{\mathrm{common}} = +0.00090 \pm 0.00055$ ($\mathrm{SE} = 0.00017$)
   - Student's $t$-test ($df=9$): $t = +5.186$, one-sided $p = 0.0003$ (95% CI: $[+0.00051, +0.00129]$)
   - Sign consistency: 9 of 10 seeds exhibited $\Delta > 0$. Filtering out unobserved support domains reveals consistent positive curvature.

### Inferential Qualifications
1. **Sampling Scope:** The 10 seeds evaluate algorithmic optimization variability under a fixed dataset; they do not represent 10 independent biological cohorts.
2. **Finite Permutations & Shared Vectors:** Each seed's null mean is estimated from $B=15$ permutations sharing identical subject sign-flip vectors.
3. **Estimand Sensitivity:** Scientific conclusions depend critically on the choice of reference domain: on the pooled domain, tail extrapolation artifacts produce negative expectations, while restricting evaluations to empirical common support yields consistent positive excess curvature across initializations.
