# Mathematical Research Evidence: Three Methodological Case Studies from BrainKAN

**Author:** Xin Qi  
**Project:** BrainKAN Explainability Benchmark  
**Date:** October 2026  
**Document Classification:** Mathematical & Statistical Research Dossier

---

## Overview

This dossier presents three concrete methodological case studies from the BrainKAN benchmark, specifically selected to demonstrate competence in:
1. **Mathematical modeling and function space analysis** (metric formalization and counterexample construction);
2. **Probability, hypothesis testing, and inferential boundaries** (permutation null design and exchangeability);
3. **Sensitivity analysis and statistical critique** (estimand definition and sign-inversion under domain constraints).

Rather than showcasing high predictive accuracy ($93\%$) or textbook derivations of B-splines, these cases demonstrate the ability to interrogate operational metrics, identify statistical failure modes, and delineate the exact boundaries of empirical claims.

---

## Case 1: Mathematical Modeling & Metric Definition: Reference Domain Sensitivity of DFL

### 1.1 Core Mathematical Definition
Let $f: \mathbb{R} \to \mathbb{R}$ denote a univariate functional transformation (such as a learned edge B-spline in a Kolmogorov–Arnold Network), evaluated on a discrete probe grid $D = \{x_1, \dots, x_K\} \subset \mathbb{R}$ with $K \ge 3$ distinct points. 

To quantify the deviation of $f$ from the subspace of affine linear functions $\mathcal{P}_1 = \{ax + b : a, b \in \mathbb{R}\}$, we compute the best linear approximation in the $\ell_2(D)$ norm via ordinary least squares:

$$\hat{y}(x) = a^* x + b^*, \quad (a^*, b^*) = \arg\min_{a, b \in \mathbb{R}} \sum_{i=1}^K [f(x_i) - (a x_i + b)]^2$$

The **Deviation-From-Linearity ($\operatorname{DFL}$)** metric is defined as the normalized residual sum of squares:

$$\boxed{\operatorname{DFL}(f; D) = \frac{\min_{a, b} \sum_{i=1}^K [f(x_i) - (a x_i + b)]^2}{\sum_{i=1}^K [f(x_i) - \bar{f}_D]^2} = \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}}} = 1 - R^2}$$

where $\bar{f}_D = \frac{1}{K} \sum_{i=1}^K f(x_i)$.

### 1.2 Mathematical Properties & Analytical Nuances
1. **Scale and Shift Invariance (Output Space):**  
   For any affine transformation of the output $g(x) = \alpha f(x) + \beta$ ($\alpha \ne 0, \beta \in \mathbb{R}$), we have:
   $$\operatorname{DFL}(\alpha f + \beta; D) = \operatorname{DFL}(f; D)$$
   The metric is dimensionless and scale-invariant with respect to output magnitude.
2. **Directionality:**  
   Under this definition, $\operatorname{DFL}$ measures the *fraction of total functional variance orthogonal to the affine subspace*. A larger $\operatorname{DFL}$ signifies greater deviation from linearity ($\operatorname{DFL} = 0$ corresponds to perfect linearity; $\operatorname{DFL} = 1$ indicates that the linear regression slope is zero and all variance is non-linear).
3. **Non-Degeneracy Condition & Operational Convention:**  
   The metric requires $\mathrm{SS}_{\mathrm{tot}} > 0$. If $f$ is a constant function on $D$ ($f(x_i) \equiv c$), $\mathrm{SS}_{\mathrm{tot}} = 0$, leading to an indeterminate $0/0$. In our implementation, constant functions are assigned $\operatorname{DFL}(f; D) = 0$ by operational convention.
4. **Discrete Grid vs. Continuous Function Space:**  
   $\operatorname{DFL}(f; D) = 0$ guarantees collinearity only at the discrete evaluation points $\{x_i\}_{i=1}^K$. It does not mathematically guarantee that the continuous spline is affine in the continuum between probe points.

### 1.3 Counterexample & Synthetic Domain Sensitivity
To audit whether $\operatorname{DFL}$ is an intrinsic property of $f$ or an artifact of the evaluation domain, we constructed a controlled synthetic experiment (Exp 2):
- **Setup:** A fixed, static target function $f(x) = \tanh(x)$ was evaluated across two reference domains: $D_1 = [-2, 2]$ and $D_2 = [0, 2]$.
- **Finding:** Evaluating the exact same function over $D_1$ versus $D_2$ yielded an apparent metric jump:
  $$\Delta \operatorname{DFL} \approx 0.0816 \pm 0.0084$$
  This domain-induced shift is **over 40 times larger** than the subtle task-associated differences observed in real fMRI networks ($\Delta \sim 0.0009$).
- **Methodological Qualification:**  
  Crucially, $D_1 = [-2, 2]$ and $D_2 = [0, 2]$ differ not only in their center/location ($0$ vs. $1$), but also in their interval width/support length ($4$ vs. $2$). Therefore, this discrepancy cannot be characterized as a pure translational effect; it reflects the joint sensitivity of the metric to domain location and scale.

*Competence Demonstrated:* Mathematical modeling in function spaces, rigorous metric specification ($1-R^2 = \mathrm{SS}_{\mathrm{res}}/\mathrm{SS}_{\mathrm{tot}}$), and designing synthetic counterexamples to audit operational definitions.

---

## Case 2: Probability, Hypothesis Testing & Scientific Inference: Permutation Null Design and Boundaries

### 2.1 Formal Formulation of the Permutation Null
In our task-fMRI connectomics benchmark, each human subject underwent both a 0-Back working memory block and a 2-Back working memory block in a paired design. 

Let $\mathcal{D} = \{(G_{i,0}, G_{i,1})\}_{i=1}^N$ denote the paired graph datasets for $N = 100$ subjects. We formulate the permutation test as follows:
- **Null Hypothesis ($H_0$):**  
  The task condition labels $Y \in \{0, 1\}$ (0-Back vs. 2-Back) are exchangeable within each subject $i$:
  $$\mathcal{L}\left((G_{i,0}, G_{i,1}) \mid Y\right) = \mathcal{L}\left((G_{i,0}, G_{i,1}) \mid \pi_i(Y)\right), \quad \forall i \in \{1, \dots, N\}$$
- **Exchangeability & Invariance Structure:**  
  Because each subject performed both conditions, the null distribution is generated via **subject-level sign flips**:
  $$\pi_i \in \{(0, 1), (1, 0)\} \quad \text{with equal probability } 0.5$$
  This preserves the individual neuroanatomy, static anatomical parcellation ($48$ ROIs), and topological connectome structure ($240$ functional edges) under the null, breaking strictly the association between learned weights and task supervision.

### 2.2 Test Statistic & Exact Empirical p-value
Let the network-level test statistic be the average DFL across all functional edges $E$:
$$T = \frac{1}{|E|} \sum_{e \in E} \operatorname{DFL}_e$$
For the real model, the observed statistic was $T_{\mathrm{obs}} = 0.17895$.

We generated $B = 500$ independent paired permutation null models $\{T_b\}_{b=1}^{500}$ (Exp 1), yielding the exact non-parametric permutation $p$-value:

$$\boxed{p = \frac{1 + \#\{T_b \ge T_{\mathrm{obs}}\}}{B + 1} = \frac{1 + 2}{500 + 1} = \frac{3}{501} \approx 0.005988 \approx 0.00599}$$

- **Monte Carlo Precision:**  
  The 500 null evaluations yielded empirical mean $\bar{T}_{\mathrm{null}} = 0.17733$ and sample standard deviation $\sigma_{\mathrm{null}} = 0.000585$. The Monte Carlo standard error of the null mean is:
  $$\mathrm{SE}_{\mathrm{MC}} = \frac{\sigma_{\mathrm{null}}}{\sqrt{B}} = \frac{0.000585}{\sqrt{500}} \approx 0.000026$$
  confirming high numerical precision for the empirical null mean.

### 2.3 Scientific Demarcation & Inferential Boundaries
Rather than stopping at reporting "$p \approx 0.006$ is statistically significant," we explicitly delineate what this result does and does not establish:
1. **Conditional Validity:**  
   The $p$-value validly tests the exchangeability of task labels *under the specific model architecture, optimization regime, and sample cohort $\mathcal{D}$*.
2. **Non-Equivalence to Biological Mechanism:**  
   Rejecting $H_0$ demonstrates that the learned network weights encode information non-exchangeable with respect to task labels. However, because overparameterized neural networks are non-identifiable in parameter space, this statistical rejection **does not prove that individual learned edge splines correspond to in-vivo biological mechanisms**.
3. **Sample Independence:**  
   The single-seed pooled-domain permutation test in Exp 1 evaluates a specific model instance on a fixed dataset. It cannot replace out-of-distribution validation on external clinical cohorts.

*Competence Demonstrated:* Formal probability formulation of exchangeability in paired experimental designs, finite-sample permutation test derivation, Monte Carlo error quantification, and intellectual restraint in scientific inference.

---

## Case 3: Sensitivity Analysis & Estimand Duality: The Sign Inversion under Common Support

### 3.1 The Critical Phenomenon: Directional Inversion of Results
The most intellectually compelling finding in this benchmark is that the scientific conclusion **inverts in sign** when changing the evaluation reference domain.

In Exp 5, we evaluated the stability of the task effect across $S = 10$ independent random optimization seeds ($w_0 \sim \mathcal{S}$). For each seed $s$, we computed the paired excess curvature:
$$\Delta_s = T(Y; s) - \frac{1}{15} \sum_{b=1}^{15} T(\pi_b(Y); s)$$

The empirical results diverged depending on the choice of evaluation domain:

| Evaluation Estimand | Mathematical Domain | Mean Difference ($\bar{\Delta}$) | Seed Consistency | Student's $t$ ($df=9$) | One-Sided $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Pooled Domain ($\mathcal{S}_{\mathrm{pooled}}$)** | $\text{Quantile}_{1\%-99\%}([X_0, X_1])$ | **$-0.000583$** ($\pm 0.00100$) | **1 / 10 Positive** (Seed 42 only) | $t = -1.849$ | $p = 0.9513$ |
| **Strict Common Support ($\mathcal{S}_{\mathrm{common}}$)** | $\operatorname{supp}(X_0) \cap \operatorname{supp}(X_1)$ | **$+0.000896$** ($\pm 0.00055$) | **9 / 10 Positive** | $t = +5.186$ | $p = 0.0003$ |

### 3.2 Mathematical & Methodological Diagnosis
Why did the effect invert from negative ($\bar{\Delta} = -0.000583$) to positive ($\bar{\Delta} = +0.000896$)?

1. **Extrapolation on the Pooled Grid:**  
   On real HCP fMRI data, the Jaccard overlap between 0-Back and 2-Back empirical distributions across the 48 cortical ROIs averages only $38.52\% \pm 5.08\%$ (covering $43.47\%$ of the pooled quantile span). When evaluating on $\mathcal{S}_{\mathrm{pooled}}$, the spline functions are evaluated outside the empirical range of each respective condition. Boundary B-spline polynomial oscillation dominates the subtle task signal, producing negative expectations for 9 out of 10 initializations.
2. **Isolation of Genuine Mutual Support:**  
   Restricting the evaluation grid to the strict intersection $\mathcal{S}_{\mathrm{common}} = [\max(q_{1,0}, q_{1,1}), \min(q_{99,0}, q_{99,1})]$ filters out extrapolation noise. Under this invariant domain, 9 out of 10 seeds consistently exhibit positive excess curvature.

### 3.3 Critical Thinking & Inferential Discipline
This case demonstrates mature critical thinking: **a seemingly conclusive result ($p = 0.006$ on Seed 42) completely collapsed across seeds on the pooled domain, but emerged consistently when the mathematical domain was properly restricted to common support.**

Furthermore, we explicitly recognize the statistical boundaries of this finding:
- **Optimization Stochasticity vs. Population Generalization:**  
  The paired $t$-statistic across 10 seeds ($t = 5.186, p = 0.0003$) evaluates **initialization consistency conditional on the fixed cohort sample and fixed 15 permutation flip vectors**. It measures stability across optimization trajectories, *not* generalizability across human populations ($\mathcal{D} \sim \mathcal{P}$).
- **Shared Permutation Vectors & Baseline Error:**  
  Because all 10 seeds shared the same 15 permutation flip vectors, the null baseline contains shared Monte Carlo error ($\mathrm{SE}_{\mathrm{MC}} \approx 0.00015$ by illustrative order-of-magnitude proxy). The cross-seed $t$-interval captures initialization variation given this baseline, but does not cover the shared uncertainty of drawing an alternative set of permutation vectors.
- **Label-Dependent Domain Selection:**  
  The common support interval was constructed using empirical condition distributions. While holding this grid fixed provides measurement comparability across real and null models, formal finite-sample type I error rate control under label-dependent domain selection remains an open statistical problem.

*Competence Demonstrated:* Identifying how empirical conclusions depend on mathematical estimands; diagnosing sign inversions driven by support boundaries; and refusing to overclaim optimization stability as population-level proof.

---

## Summary Matrix of Methodological Competencies

| Competency Area | Methodological Challenge | Technical Solution | Demonstrated Skill |
| :--- | :--- | :--- | :--- |
| **Mathematical Modeling** | Defining a scale-invariant metric for functional nonlinearity | Formalized $\operatorname{DFL} = \mathrm{SS}_{\mathrm{res}}/\mathrm{SS}_{\mathrm{tot}} = 1-R^2$ with non-degeneracy conditions | Function space analysis, projection operators, counterexample construction |
| **Probability & Statistics** | Controlling false positive interpretability claims in deep networks | Formulated paired permutation null via within-subject sign flips ($B=500, p \approx 0.00599$) | Exchangeability under paired designs, non-parametric inference, Monte Carlo bounds |
| **Sensitivity & Critical Thinking** | Reconciling single-seed significance with domain-dependent sign inversion | Discovered that $\bar{\Delta}$ flips from $-0.00058$ to $+0.00090$ between pooled and common support | Estimand demarcation, domain sensitivity, distinguishing optimization vs. population inference |
