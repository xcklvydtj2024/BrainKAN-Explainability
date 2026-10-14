# BrainKAN Analysis Plan v2

## Primary Methodological Question
Under task-related input distribution shifts, can edge-level nonlinear changes in BrainKAN be distinguished from changes induced solely by the operating regime?

## Overall Workflow
The analysis protocol is structured around diagnosing and calibrating apparent edge-level function remodeling:
1. **F0 (Predictive Adequacy):** Validate that the model accurately learns the task signal (Accuracy > Chance).
2. **Q1 & Q2 (Function Characterization):** Document the presence and stability (across folds) of learned nonlinearities using Deviation-From-Linearity (DFL).
3. **Q3A (Operating-Regime Sensitivity):** Evaluate apparent curvature shift under the native task distributions. This is diagnostic/falsification-oriented (`01_q1_q2_q3a_naive_analysis.py`).
4. **Q3B (Common-Reference Geometry):** Evaluate intrinsic structural remodeling under a fixed, condition-independent reference domain (`02_q3b_common_reference_geometry.py`).
5. **Q3C (Effective Gain):** Evaluate condition-related changes in local sensitivity (`04_q3c_effective_gain.py`).
6. **Synthetic Factorial Benchmark:** A 2xK factorial design crossing input shifts with true structural mechanisms to validate the estimators (`03_synthetic_identifiability_benchmark.py`).
7. **CHCP External Validation:** Evaluate transportability of the operational estimand under an independent cohort distribution shift (`chcp_external_validation.py`).

---

## Production Benchmark Formalization (Experiments 1–5)

To provide formal statistical hypothesis testing with rigorous uncertainty quantification, the exploratory workflow was codified into 5 reproducible benchmark experiments:

| Benchmark Experiment | Core Focus | Corresponding Exploratory Protocol | Key Output Path |
| :--- | :--- | :--- | :--- |
| **Exp 1: High-Precision Permutation** | Full-graph DFL permutation baseline ($N=500$, Seed 42) | Formalizes Q3B on global graph scale | `results/exp1_high_precision/` |
| **Exp 2: Synthetic Calibration** | Ground-truth reference domain sensitivity ($f^*(x) = x + 0.5\sin(2x)$) | Rigorous realization of Synthetic Benchmark | `results/exp2_synthetic/` |
| **Exp 3: Common Support Control** | Cross-cohort transfer (HCP $\to$ CHCP) under strictly bounded support | Formalizes CHCP External Validation | `results/exp3_ref_domain/`, `results/cross_cohort/` |
| **Exp 4: Capacity Sensitivity** | Multi-seed capacity sensitivity across BrainKAN hidden dimensions (4, 8, 16), with parameter counts reported separately | Formalizes F0 / Model Capacity control | `results/exp4_multi_seed/` |
| **Exp 5: Multi-Seed SMPT** | 10-seed paired null marginalization under dual estimands | Resolves Q3A vs. Q3B under optimization randomness | `results/exp5_seed_stability/` |
