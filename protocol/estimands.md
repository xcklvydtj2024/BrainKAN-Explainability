# BrainKAN Estimands Definition & Progression Framework

This document formalizes the mathematical estimands evaluated across the BrainKAN benchmark, establishing the direct progression from exploratory edge-level questions (Q3A, Q3B, Q3C) to the formal multi-seed benchmark suite (Exp 1 through Exp 5).

---

## Part 1: Exploratory Edge-Level Estimands (Q1–Q3)

### Q3A: Operating-Regime Sensitivity Estimand
- **Question:** Does apparent edge curvature differ between conditions when evaluated on native, condition-specific input distributions?
- **Estimand:** 
  $$\Delta C_e = C_e(X_{2\mathrm{BK}}) - C_e(X_{0\mathrm{BK}})$$
  where $C_e(X) = \frac{1}{O} \sum_{o=1}^O \mathbb{E}_{x \sim X} [|f''_{e,o}(x)|]$.
- **Methodological Role:** Falsification diagnostic. Demonstrates that naive evaluation under shifted empirical distributions creates apparent widespread significance even without structural remodeling.
- **Artifact:** `results/q3a_pvalues.csv`, `results/q3a_regime_shift.png`.

### Q3B: Common-Reference Structural Geometry Estimand
- **Question:** Does detectable condition-related structural nonlinearity persist when evaluation is standardized onto an aligned reference domain?
- **Estimand:** 
  $$G_e = \frac{1}{O} \sum_{o=1}^O (1 - R^2_{e,o,\mathrm{ref}})$$
  evaluated on intra-fold training reference grids $\mathcal{R}_k = Q_{1\%-99\%}(X_{\mathrm{train}}^{(k)})$.
- **Methodological Role:** Reference standardization. Naive edge-level differences largely collapse when domain shifts are held constant.
- **Artifact:** `results/q3b_null_summary.csv`.

### Q3C: Effective Structural Gain & Identifiability Estimand
- **Question:** Is there a detectable condition-related change in local sensitivity (first derivative $|f'(x)|$), and does the spline parameterization yield genuine structural identifiability?
- **Estimand:** 
  $$G_{e,c} = \frac{1}{O} \sum_{o=1}^O \mathbb{E}_{x \sim X_c} [|f'_{e,o}(x)|], \quad dG_e = G_e^{2\mathrm{BK}} - G_e^{0\mathrm{BK}}$$
- **Methodological Role:** Sensitivity & confusability benchmark against linear and capacity baselines.
- **Artifact:** `results/q3c_final_table.csv`, `results/synthetic_confusability_results.csv`.

---

## Part 2: Production Benchmark Suite Estimands (Exp 1–5)

To address the failure modes identified in Q3A/Q3B/Q3C, the formal benchmark operationalizes full-graph Deviation-From-Linearity ($\mathrm{DFL} = 1 - R^2$) under design-consistent paired null models and multi-seed marginalization:

### Exp 1: High-Precision Single-Seed Permutation Estimand ($N=500$, Seed 42)
- **Estimand:** 
  $$\widehat{\Delta}_{\mathrm{task}, s=42} = T_{\mathrm{real}}(s=42) - \frac{1}{500}\sum_{b=1}^{500} T_{\mathrm{null}, b}(s=42)$$
  evaluated over the pooled empirical quantile grid $\mathcal{S}_{\mathrm{pooled}}$.
- **Result:** $T_{\mathrm{real}} = 0.17895$, $\bar{T}_{\mathrm{null}} = 0.17733$, $\Delta = +0.00162$, Monte Carlo $p = 0.0060$ ($k=2/500$).
- **Limitation:** Conditional inference for seed 42 only; does not generalize across optimization initializations.
- **Directory:** `results/exp1_high_precision/`

### Exp 2: Synthetic Calibration & Gauge Sensitivity Estimand ($N=200$)
- **Estimand:** Apparent $\Delta \mathrm{DFL}$ of fixed ground-truth models ($f^*(x) = x + 0.5\sin(2x)$) under support transformations:
  - Exp 2-A (Uniform fixed): $\Delta = -0.0022 \pm 0.0076$
  - Exp 2-B (Support shift): $\Delta = +0.0816 \pm 0.0084$ (40× inflation artifact)
  - Exp 2-C (Density weighting): $\Delta = +0.0175 \pm 0.0083$
- **Directory:** `results/exp2_synthetic/`

### Exp 3: Common Support Control & Cross-Cohort Transportability Estimand
- **Estimand:** 5-fold cross-validation transfer from HCP to CHCP evaluated on strictly bounded common support.
- **Result:** $\bar{\Delta}_{\mathrm{common}} = -0.000302$ (difference indistinguishable from zero).
- **Directory:** `results/exp3_ref_domain/`, `results/cross_cohort/`

### Exp 4: Multi-Seed Capacity Calibration Estimand
- **Estimand:** Performance and parameter matching ($P_{\mathrm{spline}} = 1,776$ vs. $P_{\mathrm{MLP}} = 1,776$).
- **Directory:** `results/exp4_multi_seed/`, `results/model_parameter_counts.csv`

### Exp 5: Seed-Marginalized Permutation Testing (SMPT) Dual Estimands ($S=10$ Seeds)
Addressing the core question that arose from Q3A/Q3B/Exp 1: Does the signal generalize across optimization landscapes?

$$\bar{\Delta} = \frac{1}{S} \sum_{s=1}^S \left( T(Y; s) - \frac{1}{B} \sum_{b=1}^B T(\pi_b(Y); s) \right)$$
where $S = 10$ seeds, $B = 15$ design-consistent paired permutations per seed.

- **Estimand 1 — Pooled Quantile Support ($\mathcal{S}_{\mathrm{pooled}} = Q_{1\%-99\%}([X_0, X_1])$):**
  - $\bar{\Delta}_{\mathrm{pool}} = -0.00058 \pm 0.00100$
  - 95% Student's $t$ CI ($df=9$): $[-0.00130, +0.00013]$
  - $t = -1.8493$, $p_{\mathrm{one-sided}} = 0.9513$
  - **Behavior:** 9 out of 10 seeds yield negative differences (Seed 42 is the sole positive outlier, $\Delta = +0.00172$). Tail extrapolation noise dominates.

- **Estimand 2 — Strict Common Support Intersection ($\mathcal{S}_{\mathrm{common}} = \operatorname{supp}(X_0) \cap \operatorname{supp}(X_1)$):**
  - $\bar{\Delta}_{\mathrm{common}} = +0.00090 \pm 0.00055$
  - 95% Student's $t$ CI ($df=9$): $[+0.00051, +0.00129]$
  - $t = 5.1862$, $p_{\mathrm{one-sided}} = 0.0003$, $p_{\mathrm{two-sided}} = 0.0006$
  - **Behavior:** 9 out of 10 seeds yield positive excess curvature. Restricting evaluation to the empirical overlap eliminates extrapolation artifacts.
- **Directory:** `results/exp5_seed_stability/`
