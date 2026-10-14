# Mathematical Research Evidence: Methodological Analysis from an Independent AI Research Project

**Author:** Xin Qi  
**Project:** BrainKAN Explainability Benchmark  
**Date:** October 2026  
**Document Classification:** Candidate Research Evidence Dossier for Graduate Admissions  

**Author Contribution:** Designed the methodological evaluation framework, implemented the end-to-end experimental and statistical auditing pipeline in PyTorch, conducted all permutation and sensitivity analyses, and formulated the inferential boundaries.

---

## 1. Research Motivation & Core Question

In neuroimaging graph learning, edge-specific Kolmogorov–Arnold Networks (KANs) parameterize message-passing transformations via learnable univariate B-splines $\Phi_{ij}(x)$, enabling direct visual inspection of learned functional forms. Researchers often interpret learned spline non-linearities as evidence of task-specific biological neural computations.

**The Core Mathematical Question:**  
*Do apparent spline non-linearities reflect task-specific computational mechanisms, or are they mathematical artifacts of operational metric definitions, evaluation domains, and non-convex optimization randomness?*

This dossier presents three concrete methodological case studies demonstrating research capability in mathematical modeling, permutation hypothesis testing, and estimand-driven sensitivity analysis.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Methodological Pipeline                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  Mathematical Question                                                                 │
│  "Does learned edge spline curvature reflect task mechanisms or evaluation artifacts?" │
│                                          │                                             │
│                                          ▼                                             │
│  Operational Definition                                                                │
│  Deviation-From-Linearity (DFL = 1 - R²) over finite evaluation grid D                 │
│                                          │                                             │
│                                          ▼                                             │
│  Synthetic Model Audit (Exp 2-B)                                                       │
│  Fixed model on f*(x)=x+0.5sin(2x) yields Δ ≈ 0.0816 from domain shift                 │
│                                          │                                             │
│                                          ▼                                             │
│  Statistical Validation                                                                │
│  Paired topology-preserving permutation null models (B = 500, Monte Carlo p ≈ 0.00599) │
│                                          │                                             │
│                                          ▼                                             │
│  Inference Boundary                                                                    │
│  Estimand-dependent sign reversal (Pooled: -0.00058 vs. Empirical Overlap: +0.00090)   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Case 1: Empirical Metric Definition & Reference Domain Sensitivity of DFL

### 2.1 Mathematical Formulation
Let $f: \mathbb{R} \to \mathbb{R}$ denote a learned univariate transformation evaluated on a discrete probe grid $D = \{x_1, \dots, x_K\} \subset \mathbb{R}$ with $K \ge 3$ distinct points. 

To quantify the empirical deviation of sampled function values from their best affine projection over a finite evaluation grid $D$, we fit an ordinary least-squares affine model $\hat{y}(x) = a^* x + b^*$:

$$(a^*, b^*) = \arg\min_{a, b \in \mathbb{R}} \sum_{i=1}^K [f(x_i) - (a x_i + b)]^2$$

The **Deviation-From-Linearity ($\mathrm{DFL}$)** metric is defined as the normalized residual sum of squares:

$$\boxed{\mathrm{DFL}(f; D) = \frac{\min_{a, b} \sum_{i=1}^K [f(x_i) - (a x_i + b)]^2}{\sum_{i=1}^K [f(x_i) - \bar{f}_D]^2} = \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}}} = 1 - R^2}$$

where $\bar{f}_D = \frac{1}{K} \sum_{i=1}^K f(x_i)$.

### 2.2 Analytical Nuances & Discrete Limitations
1. Directionality: $\mathrm{DFL}$ measures the fraction of sample variance unexplained by the affine model. $\mathrm{DFL} = 0$ corresponds to perfect collinearity on $D$; $\mathrm{DFL} = 1$ indicates that the affine hypothesis class explains zero variance on the sampled grid ($R^2 = 0$).
2. Degeneracy Convention: The definition requires $\mathrm{SS}_{\mathrm{tot}} > 0$. If $f$ is constant on $D$ ($f(x_i) \equiv c$), $\mathrm{SS}_{\mathrm{tot}} = 0$, yielding an indeterminate $0/0$. In our codebase, constant functions are assigned $\mathrm{DFL}(f; D) = 0$ by operational convention.
3. Discrete Grid Limitation: $\mathrm{DFL}(f; D) = 0$ guarantees collinearity only at the discrete sample points $\{x_i\}_{i=1}^K$. It does not mathematically guarantee that the continuous spline is affine in the continuum between probe points.

### 2.3 Synthetic Model Calibration & Domain Sensitivity (Exp 2-B)
To audit whether $\mathrm{DFL}$ behaves as an intrinsic invariant of a learned mechanism or depends on the choice of evaluation grid, we conducted a controlled synthetic audit (`exp2_synthetic_calibration.py`):
- Data Generating Process: Ground-truth target $y = f^*(x) + \epsilon$, where $f^*(x) = x + 0.5\sin(2x)$, $x \sim \mathcal{N}(0, 1)$, and $\epsilon \sim \mathcal{N}(0, 0.05^2)$.
- Learned Representation: A B-spline KAN layer (`SyntheticEdgeKAN`) was fitted to this target, yielding a parameter-frozen learned model $\hat{f}$.
- Evaluation Domain Shift: Holding the identical learned model $\hat{f}$ completely fixed, we evaluated $\mathrm{DFL}(\hat{f}; D)$ over two 100-point uniform probe grids:
  - $D_1 = [-2.0, 2.0]$ (full symmetric interval, where the linear drift $x$ accounts for most sample variance, yielding mean $\mathrm{DFL} \approx 0.0388 \pm 0.0024$).
  - $D_2 = [0.0, 2.0]$ (restricted half-cycle, where the oscillatory harmonic $0.5\sin(2x)$ constitutes a larger fraction of variance relative to the linear trend, yielding mean $\mathrm{DFL} \approx 0.1204 \pm 0.0084$).
- Finding: Across $N = 200$ independent simulation trials, evaluating the exact same learned model on $D_2$ versus $D_1$ produced an artifactual metric inflation of $\Delta \mathrm{DFL} = +0.0816 \pm 0.0084$ ($t = 136.6$), which is $>40$ times larger than typical task condition differences in real fMRI networks ($\sim 0.0009$).
- Methodological Qualification: $D_1 = [-2, 2]$ and $D_2 = [0, 2]$ differ simultaneously in their center ($0$ vs. $1$) and support length ($4$ vs. $2$). This demonstrates that even for a fixed learned representation, $\mathrm{DFL}$ reflects the joint sensitivity of the metric to evaluation location and scale rather than an invariant property of the underlying network function.

---

## 3. Case 2: Permutation Null Design & Inferential Limits

### 3.1 Null Hypothesis Formulation & Within-Subject Exchangeability
In our task-fMRI connectomics benchmark, each human subject underwent both a 0-Back and a 2-Back working memory block in a paired design ($N = 100$ subjects, 200 graph samples).

- Null Hypothesis ($H_0$): Task condition labels $Y \in \{0, 1\}$ are exchangeable within each subject $i$:
  $$\mathcal{L}\left((G_{i,0}, G_{i,1}) \mid Y\right) = \mathcal{L}\left((G_{i,0}, G_{i,1}) \mid \pi_i(Y)\right), \quad \forall i \in \{1, \dots, N\}$$
- Exchangeability Structure: The permutation distribution is generated via subject-level sign flips: $\pi_i \in \{(0, 1), (1, 0)\}$ with equal probability $0.5$. This preserves individual anatomy ($48$ ROIs) and functional connectome topology ($240$ edges) under the null, breaking strictly the association between learned weights and task supervision.

### 3.2 Monte Carlo Permutation p-value
With network-level test statistic $T = \frac{1}{|E|} \sum_{e \in E} \mathrm{DFL}_e$ ($T_{\mathrm{obs}} = 0.17895$ for the real model), evaluating $B = 500$ independent paired permutation null models $\{T_b\}_{b=1}^{500}$ (Exp 1) yields the **Monte Carlo permutation $p$-value**:

$$\boxed{p = \frac{1 + \#\{T_b \ge T_{\mathrm{obs}}\}}{B + 1} = \frac{1 + 2}{500 + 1} = \frac{3}{501} \approx 0.00599}$$

- Monte Carlo Precision: The 500 null values yielded sample mean $\bar{T}_{\mathrm{null}} = 0.17733$ and sample standard deviation $\sigma_{\mathrm{null}} = 0.000585$. The standard error of the estimated null mean was approximately $\mathrm{SE}(\bar{T}_{\mathrm{null}}) = \frac{0.000585}{\sqrt{500}} \approx 2.6 \times 10^{-5}$. This characterizes the precision of the location parameter of the null distribution, and should not be confused with the Monte Carlo uncertainty of the permutation $p$-value itself ($\mathrm{SE}(\hat{p}) \approx \sqrt{\hat{p}(1-\hat{p})/B} \approx 0.0035$).

### 3.3 Inferential Boundaries
Rather than stopping at reporting statistical significance, we explicitly articulate the inferential boundaries:
1. The $p$-value tests label exchangeability under the specific architecture, optimization regime, and fixed benchmark cohort $\mathcal{D}$.
2. Rejecting $H_0$ indicates that the network weights encode task-associated variation. However, due to neural network overparameterization and parameter non-identifiability, this statistical rejection does not prove that individual learned edge splines correspond to in-vivo biological mechanisms.
3. This single-seed test on a fixed cohort does not replace external validation across independent clinical cohorts.

---

## 4. Case 3: Sensitivity Analysis & Estimand-Dependent Sign Reversal

### 4.1 The Critical Phenomenon: Domain-Dependent Directional Change
The most compelling methodological finding is an estimand-dependent directional change: the observed excess curvature inverts in sign depending on whether the evaluation domain accounts for empirical support overlap.

In Exp 5, across $S = 10$ independent random optimization seeds ($w_0 \sim \mathcal{S}$), we evaluated the paired excess curvature $\Delta_s = T(Y; s) - \frac{1}{15} \sum_{b=1}^{15} T(\pi_b(Y); s)$:

| Evaluation Estimand | Mathematical Domain Definition | Mean Difference ($\bar{\Delta}$) | Seed Consistency | Exploratory Seed $t$ ($df=9$) | Statistical Note |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Pooled Domain ($\mathcal{S}_{\mathrm{pooled}}$)** | $\text{Quantile}_{1\%-99\%}([X_0, X_1])$ | **$-0.000583$** ($\pm 0.00100$) | 1 / 10 Positive (Seed 42) | $t = -1.849$ | 9/10 seeds negative; Seed 42 conditional finding does not generalize across seeds. |
| **Empirical Common Support ($\mathcal{S}_{\mathrm{common}}$)** | $[q_1^{(0)}, q_{99}^{(0)}] \cap [q_1^{(1)}, q_{99}^{(1)}]$ | **$+0.000896$** ($\pm 0.00055$) | 9 / 10 Positive | $t = +5.186$ | Exploratory; condition-dependent grid limits confirmatory statistical interpretation. |

### 4.2 Methodological Diagnosis
1. Sensitivity to Support Coverage: Across 48 ROIs on HCP fMRI data, the Jaccard overlap between 0-Back and 2-Back empirical distributions averages $38.52\% \pm 5.08\%$ (covering $43.47\%$ of the pooled span). On $\mathcal{S}_{\mathrm{pooled}}$, splines are evaluated across domains where empirical density is sparse or condition-asymmetric, yielding negative expectations for 9 out of 10 initializations.
2. Isolation of the Empirical Overlap Region: Restricting evaluation to the empirical overlap region $[q_1^{(0)}, q_{99}^{(0)}] \cap [q_1^{(1)}, q_{99}^{(1)}]$ yields consistent positive excess curvature across 9 out of 10 seeds. However, because this grid is constructed conditionally on original task labels and evaluated against a fixed-grid null, it cannot be treated as a fully calibrated subject-level permutation test.

### 4.3 Research Boundary Demarcation
- Optimization Stochasticity vs. Population Generalization: The exploratory paired $t$-statistic ($t = 5.186$) evaluates initialization consistency across 10 seeds conditional on the fixed cohort sample and fixed common support grid. It does not constitute a subject-level permutation test and does not establish generalizability across biological subject populations ($\mathcal{D} \sim \mathcal{P}$).
- Shared Permutation Vectors: The 10 seeds shared the same 15 permutation flip vectors, leaving common baseline error unmodeled across seeds ($\mathrm{SE}_{\mathrm{MC}} \approx 0.00015$ by illustrative order-of-magnitude proxy).
- Label-Dependent Domain Selection: The empirical overlap region was constructed using empirical condition distributions. While holding this grid fixed provides measurement comparability across real and null models, formal finite-sample type I error rate control under label-dependent domain selection remains an open statistical problem.

---

## 5. Evidence of Research Capability

This benchmark illustrates core methodological competencies essential for advanced graduate research:

1. **Metric Scrutiny over Black-Box Metrics:** Rather than treating $\mathrm{DFL} = 1 - R^2$ as an intrinsic measure of biological curvature, we formalized it as an empirical projection residual against an affine hypothesis class and demonstrated its joint location-scale sensitivity via controlled synthetic calibration.
2. **Principled Statistical Inference:** Rather than celebrating an empirical $p \approx 0.006$ as "proof of neural mechanisms," we explicitly articulated the conditioning on label exchangeability, model capacity, and dataset-specific baselines.
3. **Estimand Awareness & Scientific Restraint:** Discovering that the sign of the effect inverts (from $-0.00058$ to $+0.00090$) between the pooled domain and the empirical overlap region prevented false scientific conclusions. The observed direction of the estimated effect can depend critically on the chosen evaluation domain, underscoring the need to distinguish measurement sensitivity from biological interpretation.
4. **Research Philosophy:** The objective of this work was not to maximize model complexity, but to develop reliable criteria for interpreting learned representations. If extending this project, the primary priority is establishing formal subject-level uncertainty quantification (via hierarchical Bayesian modeling or nested subject-level resampling), rather than scaling architectural complexity.
