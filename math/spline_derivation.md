# Mathematical Foundations of Spline Geometry & Benchmark Estimands

This document details the theoretical formulation of spline geometric metrics (Deviation-From-Linearity, DFL) and establishes the distinction between conditional and marginal estimands in KAN interpretability.

---

## Proposition 1: Reference-Induced Deformation Decomposition

Let the empirical computation along an edge $(i, j)$ in a Kolmogorov–Arnold Network be parameterized by learnable B-splines:
$$\Phi_{ij}(x) = W_{ij}^{base}\operatorname{SiLU}(x) + \sum_{k=1}^K w_{ij,k} B_k(x)$$

When evaluating the non-linearity of this function over a specified evaluation domain $X \subset \mathbb{R}$, we compute the Deviation-From-Linearity (DFL):
$$\operatorname{DFL}(\Phi_{ij}, X) = 1 - R^2 = \frac{\sum_{t \in X} (\Phi_{ij}(t) - \hat{y}(t))^2}{\sum_{t \in X} (\Phi_{ij}(t) - \bar{y})^2}$$

### The Geometric Decomposition
The measured geometric curvature over support $X$ decomposes into two components:
$$G(f, X) = S(f) + D(X, f)$$
where $S(f)$ is the intrinsic mathematical transformation, and $D(X, f)$ is the domain-dependent deformation induced by evaluating over support $X$. As proven in Exp 2-B, a domain shift alone can inflate $D(X, f)$ by $+0.1195$ without altering the underlying function.

---

## Proposition 2: Optimization Randomness and Estimand Duality

Because deep neural networks navigate non-convex loss landscapes, the learned spline parameters $w_{ij,k}(s)$ are stochastic realizations dependent on the random seed $s \sim \mathcal{S}$ (initialization and mini-batch sequence).

### 1. Conditional Estimand
For a fixed seed $s$, the conditional excess nonlinearity is defined as:
$$\Delta(s) = G(f_{\mathrm{real}}^{(s)}, X) - \mathbb{E}_\pi [G(f_{\mathrm{null}(\pi)}^{(s)}, X)]$$
Testing $H_0^{(s)}: \Delta(s) \le 0$ evaluates whether real labels induced greater curvature *within the specific optimization basin reached by seed $s$*. For seed $s=42$, $\Delta(42) = +0.0017$ ($p = 0.0060$).

### 2. Marginal Population Estimand
The scientific hypothesis of interest across the optimization distribution is the marginal expectation:
$$\Delta_{\mathrm{marginal}} = \mathbb{E}_{s \sim \mathcal{S}} [\Delta(s)] = \mathbb{E}_s [G(f_{\mathrm{real}}^{(s)}, X)] - \mathbb{E}_{s, \pi} [G(f_{\mathrm{null}}^{(s)}, X)]$$

### Empirical Benchmark Finding
Evaluating over 10 independent initialization seeds reveals:
$$\Delta_{\mathrm{marginal}} = -0.00058 \le 0 \quad (\text{95\% CI: } [-0.00120, +0.00003], \quad p = 0.9513)$$

**Conclusion:** Under the predefined one-sided alternative, the observed marginal effect does not support rejection of the null hypothesis. Single-seed conditional inference does not generalize across optimization randomness.
