# BrainKAN Analysis Plan v1.0

This document locks in the pre-specified methodology and statistical evaluation criteria for the BrainKAN Explainability framework before executing the formal computational pipeline. This ensures that the primary statistical analysis is rigorous, protected from post-hoc selection bias, and clearly bounds its scientific claims.

## 0.1 Primary Study Objective
**Objective:** To evaluate whether edge-level nonlinear geometry learned by task-fMRI graph models exhibits label-associated excess deviation from linearity relative to a design-consistent permutation baseline, and assess its sensitivity to reference-domain and model-training choices.

*Note: This strictly evaluates label-associated nonlinear geometry, and does not directly measure true in-vivo biological mechanisms.*

## 0.2 Primary Endpoint: Full-Graph Average DFL ($\widehat{\Delta}_{\mathrm{task}}$)
We pre-define the edge set $E$ to include all valid structural edges in the graph. For the $k$-th cross-validation fold, the Deviation-From-Linearity (DFL) of a trained function $f_{e,k}$ over the fixed intra-fold reference domain $\mathcal{R}_k$ is:
$$ G_{e,k} = 1 - R^2_{e,k} $$
*(Note on numerical degeneracy: If a function evaluates to a near-constant output over $\mathcal{R}_k$, rendering $R^2$ undefined, we define a strict variance tolerance threshold. Such failures will be explicitly counted and identical fallback rules applied to both real and null models. They will not be silently dropped.)*

The fold-level global non-linearity is:
$$ T_k = \frac{1}{|E|} \sum_{e \in E} G_{e,k} $$

The global cross-fold statistic $T_{\mathrm{real}} = \frac{1}{K}\sum_k T_{\mathrm{real},k}$ is computed. The identical process is repeated for $B$ full-pipeline label permutations yielding $T_{\mathrm{null},b}$.
The primary effect size is defined as:
$$ \widehat{\Delta}_{\mathrm{task}} = T_{\mathrm{real}} - \frac{1}{B}\sum_{b=1}^{B}T_{\mathrm{null},b} $$

## 0.3 Primary Null Hypothesis and Inference
**$H_0$:** $T_{\mathrm{real}}$ is not higher than the level expected by a design-consistent label-permutation null model.

We utilize a one-sided permutation test:
$$ p = \frac{1 + \sum_{b=1}^B \mathbf{1}(T_{\mathrm{null}, b} \ge T_{\mathrm{real}})}{B + 1} $$
*Constraint: This relies on the assumption of exchangeability of 0-back and 2-back labels within subjects. If task block designs violate exchangeability, the permutation distribution will be reported as an empirical baseline rather than a design-guaranteed exact test.*

## 0.4 Pre-specified Interpretation of Results
| Primary Result | Supported Claim |
| -------------- | --------------- |
| $\widehat{\Delta}_{\mathrm{task}} > 0$ and $p < \alpha$ | Real-label training induces excess average nonlinearity beyond the null baseline. |
| Effect near zero or non-significant $p$ | The current analysis provides no evidence that average nonlinearity exceeds the null baseline. |
| Effect changes drastically across model capacity or reference domain | The nonlinear interpretation is highly sensitive to model and domain constraints. |

*(Under no circumstance will "significant" be equated to "true brain mechanism discovered", nor "non-significant" to "nonlinearity does not exist in the brain".)*

## 0.5 Cross-Validation and Reference Domain Rules
- **Data Partitioning:** Strictly subject-level 5-fold cross validation. 0-back and 2-back scans for the same subject MUST remain in the same fold to preserve pairing and prevent train-test leakage.
- **Reference Domain ($\mathcal{R}_k$):** Constructed exclusively using the inputs from the **training** subjects of fold $k$. It will be frozen, and the real model and all null permutations for fold $k$ will be evaluated on this identical grid.
- **Null Protocol:** Every null permutation must repeat the entire training protocol, using the same architecture, budget, and initialization seeds.
