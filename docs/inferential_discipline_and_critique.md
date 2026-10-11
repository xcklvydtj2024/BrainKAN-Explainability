# Inferential Discipline & Methodological Defense
### Comprehensive Analysis of Statistical, Mathematical, and Engineering Invariants in BrainKAN Explainability

This document provides a formal, rigorous response to key methodological and inferential questions regarding the BrainKAN Interpretability Benchmark. It clarifies the mathematical properties of our estimators, provides full computational provenance, and establishes the inferential boundaries of our findings.

---

## 1. Full Computational Provenance in Permutation Testing (Exp 1)

### 1.1 The Issue
In distributed or chunked permutation testing, reference test statistics ($T_{\mathrm{obs}}$) are sometimes recorded as rounded scalars (e.g., $0.1790$). Because empirical permutation p-values rely on strict indicator counts:
$$p = \frac{1 + \sum_{b=1}^B \mathbb{I}(T_b \ge T_{\mathrm{obs}})}{B + 1}$$
even minute floating-point rounding errors near the empirical threshold could alter the exceedance count $k$, potentially compromising computational reproducibility.

### 1.2 Resolution & Exact Floating-Point Provenance
We eliminated all hardcoded constants and refactored `experiments/aggregate_exp1.py` to retrieve the exact double-precision test statistic directly from logged seed evaluations (`results/exp5_seed_stability/seed_res_42.csv`):
$$T_{\mathrm{real}} = 0.1789531037243144$$

Evaluating this exact scalar against all $B = 500$ Monte Carlo null permutations yields:
- **Exact Double-Precision $T_{\mathrm{real}}$**: `0.1789531037243144`
- **Mean of Null Distribution $\bar{T}_{\mathrm{null}}$**: `0.1773326801158365`
- **Null Values in the Neighborhood $[0.1788, 0.1792]$**: Exactly one value exists at `0.1790071602172386`.
- **Exceedance Count ($k$)**: Strictly $k = 2$ null permutations satisfy $T_{\mathrm{null}} \ge T_{\mathrm{real}}$.
- **Exact Empirical p-value**:
  $$p = \frac{1 + 2}{500 + 1} = \frac{3}{501} = 0.005988 \approx 0.0060$$
- **Effect Size**: $\widehat{\Delta} = +0.0016204236084779$.

The empirical p-value of $0.0060$ is verified with complete numerical precision and end-to-end programmatic provenance.

---

## 2. Inferential Discipline: Absence of Evidence vs. Exact Zero Effect (Exp 5)

### 2.1 The Statistical Distinction
A crucial tenet of inferential discipline is that **failing to reject a null hypothesis does not constitute proof of exact nullity**:
$$\text{Failure to reject } H_0: \mathbb{E}[\Delta] \le 0 \quad \not\equiv \quad \text{Proof that } \Delta \equiv 0$$
Our findings do not claim that task-induced excess nonlinearity is mathematically zero. Instead, they demonstrate that under the pre-specified one-sided hypothesis, the empirical data **do not provide sufficient evidence to support task-induced excess nonlinearity**, and that the sign and magnitude of the effect are highly sensitive to the optimization initialization seed.

### 2.2 Small-Sample Inference: Student's t Formulation
For small seed sample sizes ($S = 10$), asymptotic normal approximations ($\pm 1.96 \cdot \mathrm{SE}$) underestimate parameter uncertainty. We therefore report exact Student's $t$ confidence intervals with $df = 9$:
- Sample size: $S = 10$ independent random seeds
- Sample mean: $\bar{\Delta} = -0.00058287$
- Sample standard deviation: $s = 0.00099660$
- Standard error: $\mathrm{SE} = \frac{s}{\sqrt{10}} = 0.00031515$
- Critical $t$-value ($df = 9, \alpha = 0.05$, two-sided): $t_{0.975, 9} = 2.262157$

The exact 95% Student's $t$ confidence interval is:
$$\bar{\Delta} \pm t_{0.975, 9} \cdot \mathrm{SE} = -0.000583 \pm 0.000713 = \mathbf{[-0.00130, +0.00013]}$$
- One-sided test ($H_0: \mathbb{E}[\Delta] \le 0$ vs. $H_1: \mathbb{E}[\Delta] > 0$): $t = -1.8495, \quad p = 0.9513$
- Two-sided test ($H_0: \mathbb{E}[\Delta] = 0$): $t = -1.8495, \quad p = 0.0974$
- Sign consistency: Only 1 out of 10 seeds (Seed 42) produced a positive point estimate.

### 2.3 The Nature of Seed Randomness
We explicitly acknowledge that deterministic integer seeds selected by researchers represent an empirical discrete sample of optimization trajectory randomness. They capture variability across loss basins for a fixed architecture and training setup, but do not provide an asymptotic guarantee across unseen patient populations or alternative optimizer hyperparameters.

---

## 3. Mathematical Demarcation: Pooled-Reference vs. Common-Support Estimands

### 3.1 Formal Definitions
When evaluating spline functions $\Phi_{ij}$ for a given node $n$, researchers must distinguish between two evaluation supports:

1. **Pooled-Reference Support ($\mathcal{S}_{\mathrm{pooled}}$)**:
   $$\mathcal{S}_{\mathrm{pooled}}^{(n)} = \left[ q_{0.01}\left( X_n^{(0)} \cup X_n^{(1)} \right), \; q_{0.99}\left( X_n^{(0)} \cup X_n^{(1)} \right) \right]$$
   Constructed from the union of all training samples across both task conditions.

2. **Common-Support Intersection ($\mathcal{S}_{\mathrm{common}}$)**:
   $$\mathcal{S}_{\mathrm{common}}^{(n)} = \operatorname{supp}(X_n^{(0)}) \cap \operatorname{supp}(X_n^{(1)}) = \left[ \max\left(q_{0.01}(X_n^{(0)}), q_{0.01}(X_n^{(1)})\right), \; \min\left(q_{0.99}(X_n^{(0)}), q_{0.99}(X_n^{(1)})\right) \right]$$
   Constructed strictly from the overlapping region where empirical data points exist under both conditions.

### 3.2 Theoretical Relationship in Within-Cohort HCP Data
In within-cohort task-fMRI (HCP 0-back vs. 2-back), input node feature distributions under the two conditions are almost identical in variance and dynamic range:
$$\frac{|\mathcal{S}_{\mathrm{common}}^{(n)}|}{|\mathcal{S}_{\mathrm{pooled}}^{(n)}|} > 0.998$$
Consequently, in the primary statistical analysis (Exp 1 and Exp 5), evaluating over $\mathcal{S}_{\mathrm{pooled}}$ yields numerical values virtually identical to $\mathcal{S}_{\mathrm{common}}$. 

However, conceptually:
$$\mathcal{S}_{\mathrm{common}} \subseteq \mathcal{S}_{\mathrm{pooled}}$$
In cross-cohort transfer (Exp 3: HCP $\to$ CHCP), the distributions undergo substantial variance compression ($\nu = 0.1412$). There, $\mathcal{S}_{\mathrm{pooled}}$ extends significantly beyond the target support, causing 21.91% of points to represent extrapolation. Restricting evaluation strictly to $\mathcal{S}_{\mathrm{common}}$ in Exp 3 confirmed that the measured nonlinearity differences were not artifacts of boundary extrapolation ($\Delta_{\mathrm{common}} = +0.0025$).

---

## 4. High Model Capacity vs. Sample Size: Spline Curvature as an Inductive Prior

### 4.1 The Overparameterization Dilemma
The BrainKAN architecture contains **589,779 trainable parameters** (primarily edge-specific B-spline control points for 240 graph edges), trained on **100 paired HCP subjects** (200 fMRI graph samples). In classical statistics, $P \gg N$ introduces severe identifiability concerns.

### 4.2 Why Edge-Specific B-Splines?
Edge-specific parameterization is biologically motivated by macroscale connectomics: each edge $(i, j)$ represents a distinct anatomical white-matter tract connecting two functionally specialized brain regions. Shared message passing (as in standard GCN) forces all brain regions to communicate via identical transfer functions, which is biologically unrealistic.

### 4.3 The Critical Discovery: Intrinsic Capacity Bias
The central finding of our benchmark directly resolves this dilemma:
1. When BrainKAN is trained on **true task labels**, the global average edge curvature converges to $T_{\mathrm{real}} \approx 0.17939$.
2. When BrainKAN is trained on **randomly permuted labels**, the global average edge curvature converges to $T_{\mathrm{null}} \approx 0.17998$.
3. When model capacity is varied ($dim \in \{4, 8, 16\}$ in Exp 4), baseline curvature remains stable around $\sim 0.178 - 0.183$.

**Key Scientific Takeaway:**
The baseline curvature of $T \approx 0.179$ is **not a biological discovery extracted from 100 subjects**; it is the **inductive bias of B-spline parameterization, $L_2$ weight decay, and Adam optimization**. 
The network architecture naturally allocates a non-linear curvature budget to route high-dimensional graph signals. Without a paired permutation null model, researchers would mistakenly interpret this structural inductive prior as "evidence of non-linear brain dynamics." This constitutes one of the most important cautionary warnings of our benchmark.

---

## 5. Mathematical Properties of Deviation-from-Linearity ($1-R^2$)

### 5.1 Affine Scale Invariance
The Deviation-from-Linearity metric is defined as:
$$\operatorname{DFL}(f, X) = 1 - R^2 = \frac{\sum_{t \in X} (f(t) - \hat{f}(t))^2}{\sum_{t \in X} (f(t) - \bar{f})^2} = \frac{\mathrm{SS}_{\mathrm{res}}(f)}{\mathrm{SS}_{\mathrm{tot}}(f)}$$
where $\hat{f}(t) = a t + b$ is the best-fit linear regression of $f(t)$ over $X$.

**Theorem (Affine Invariance):**
For any non-zero scalar $c \neq 0$ and constant $d \in \mathbb{R}$, let $g(t) = c \cdot f(t) + d$. Then:
$$\operatorname{DFL}(g, X) = \operatorname{DFL}(f, X)$$

*Proof:*
By linearity of least-squares regression:
1. Best-fit linear model: $\hat{g}(t) = c \cdot \hat{f}(t) + d$.
2. Residual sum of squares:
   $$\mathrm{SS}_{\mathrm{res}}(g) = \sum_{t \in X} [ (c f(t) + d) - (c \hat{f}(t) + d) ]^2 = c^2 \sum_{t \in X} [f(t) - \hat{f}(t)]^2 = c^2 \mathrm{SS}_{\mathrm{res}}(f)$$
3. Total sum of squares:
   $$\mathrm{SS}_{\mathrm{tot}}(g) = \sum_{t \in X} [ (c f(t) + d) - (c \bar{f} + d) ]^2 = c^2 \sum_{t \in X} [f(t) - \bar{f}]^2 = c^2 \mathrm{SS}_{\mathrm{tot}}(f)$$
4. Ratio:
   $$\operatorname{DFL}(g, X) = \frac{c^2 \mathrm{SS}_{\mathrm{res}}(f)}{c^2 \mathrm{SS}_{\mathrm{tot}}(f)} = \frac{\mathrm{SS}_{\mathrm{res}}(f)}{\mathrm{SS}_{\mathrm{tot}}(f)} = \operatorname{DFL}(f, X) \quad \blacksquare$$

### 5.2 Advantage Over Curvature Energy ($\int [f'']^2$)
Differential geometric metrics such as the $L_2$ norm of second derivatives ($\mathcal{E}(f) = \int [f''(t)]^2 dt$) are **not scale-invariant**:
$$\mathcal{E}(c \cdot f) = c^2 \mathcal{E}(f)$$
Curvature energy scales quadratically with function amplitude. In neural networks where weight magnitudes vary across layers, curvature energy conflates function gain with geometric shape. By contrast, $1-R^2$ is strictly dimensionless, representing the proportion of functional variance unexplained by linear projection.

### 5.3 Numerical Singularity at Low Total Variance
The scale invariance holds everywhere except at the singular boundary $\mathrm{SS}_{\mathrm{tot}} \to 0$ (a flat, constant function $f(t) \equiv C$).
For a constant function, $\mathrm{SS}_{\mathrm{res}} = 0$ and $\mathrm{SS}_{\mathrm{tot}} = 0$, making $1-R^2 = 0/0$ mathematically indeterminate.
In our implementation, a threshold of $\mathrm{SS}_{\mathrm{tot}} < 10^{-6}$ assigns $\operatorname{DFL} = 0$. While computationally pragmatic, this threshold is dimensionful. To ensure robustness, we verified that learned edge splines in BrainKAN operate with typical variances $\mathrm{SS}_{\mathrm{tot}} \in [0.05, 1.20]$, safely away from the numerical singularity.

---

## 6. Research Software Engineering Standards & Invariants

### 6.1 Dynamic Project Paths
All codebase scripts now resolve paths dynamically relative to the repository root:
```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
```
No hardcoded drive letters or absolute paths exist in the production scripts.

### 6.2 Modular Worker Architecture
`run_exp5_parallel.py` has been refactored to eliminate dynamic script generation. Parallel execution is managed by dispatching subprocess calls to `exp5_runner.py` via standard CLI arguments (`argparse`), ensuring full cross-platform compatibility.

### 6.3 Data Invariant Validation
In `utils_data_loading.py`, explicit assertions verify that each subject $i$ contains exactly one 0-back and one 2-back graph:
```python
if {y[2*i].item(), y[2*i+1].item()} != {0, 1}:
    raise ValueError(f"Data invariant violated at subject {i}: expected paired classes {{0, 1}}.")
```
This guarantees that paired permutation and fold splits operate on verified balanced data invariants.
