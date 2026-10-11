# BrainKAN Analysis Plan v2

## Primary Methodological Question
Under task-related input distribution shifts, can edge-level nonlinear changes in BrainKAN be distinguished from changes induced solely by the operating regime?

## Overall Workflow
The analysis protocol is structured around diagnosing and calibrating apparent edge-level function remodeling:
1. **F0 (Predictive Adequacy):** Validate that the model accurately learns the task signal (Accuracy > Chance).
2. **Q1 & Q2 (Function Characterization):** Document the presence and stability (across folds) of learned nonlinearities using Deviation-From-Linearity (DFL).
3. **Q3A (Operating-Regime Sensitivity):** Evaluate apparent curvature shift under the native task distributions. This is diagnostic/falsification-oriented.
4. **Q3B (Common-Reference Geometry):** Evaluate intrinsic structural remodeling under a fixed, condition-independent reference domain.
5. **Q3C (Effective Gain):** Evaluate condition-related changes in local sensitivity.
6. **Synthetic Factorial Benchmark:** A 2xK factorial design crossing input shifts with true structural mechanisms to validate the estimators.
7. **CHCP External Validation:** Evaluate transportability of the operational estimand under an independent cohort distribution shift.
