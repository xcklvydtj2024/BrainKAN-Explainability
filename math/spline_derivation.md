# Mathematical Foundations of Spline Geometry Analysis

This document provides the theoretical justification for why we use the geometric metric $G_e$ (based on Deviation-From-Linearity and Curvature) to interpret KAN edge computations, and under what conditions it isolates biological signals.

## Proposition 1: Reference-Induced Deformation Decomposition

Let the empirical computation along an edge $(i, j)$ be defined as:
$$ \Phi_{ij}(x) = W_{ij}^{base}\text{SiLU}(x) + \sum_k w_{ij,k}B_k(x) $$

When evaluating the non-linearity of this function, we compute a geometric test statistic $G(f, X)$ (e.g., integrated squared second derivative or deviation from linear fit) over a defined input distribution $X$.

**The Decomposition Hypothesis:**
We assert that the observed geometry can be linearly decomposed into two components:
$$ G(f, X) = S(f) + D(X, f) $$
Where:
1. $S(f)$ is the **intrinsic structural component**. This represents the true, task-associated mathematical transformation (e.g., thresholding, saturation) required to solve the biological objective.
2. $D(X, f)$ is the **reference/input-dependent deformation**. This is the artifactual geometry induced by how the function $f$ interpolates the specific input density $X$, constrained by the flexibility of the B-splines.

## The Role of Null Calibration

When evaluating $G(f, X)$ across two different cohorts (e.g., HCP $\to$ CHCP), a naive evaluation yields $G(f_{real}, X_{CHCP})$. If this value changes, it is mathematically impossible to distinguish whether the true task mechanism $S(f)$ changed, or merely the input density $X$ shifted, altering $D(X, f)$.

To resolve this identifiability failure, we train an empirical null model $f_{null}$ on label-shuffled data. Because $Y_{null}$ contains no task information, the null model requires no intrinsic biological mechanism:
$$ S(f_{null}) \approx 0 $$

However, $f_{null}$ is trained on the exact same input distribution $X$, utilizing the identical spline flexibility and regularization dynamics. Thus, it fully absorbs the input-dependent deformation:
$$ G(f_{null}, X) = D(X, f_{null}) $$

By constructing the **Standardized Separation** estimator (or simply evaluating the difference):
$$ \Psi = G(f_{real}, X) - G(f_{null}, X) $$
$$ \Psi = [S(f_{real}) + D(X, f_{real})] - D(X, f_{null}) $$

**Crucial Theoretical Guarantee:**
Assuming the baseline deformation is statistically independent of the true signal mechanism (i.e., $D(X, f_{real}) \approx D(X, f_{null})$ in expectation), the calibration yields:
$$ \Psi \approx S(f_{real}) $$

This proves that the calibrated metric isolates the intrinsic structural behavior relative to the null expectation, removing the confounding effects of $X$. If $\Psi > 0$, the edge exhibits non-linearity exceeding what is forced by the mere routing of the input distribution.
