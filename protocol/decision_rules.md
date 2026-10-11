# BrainKAN Decision Rules & Interpretation

## Falsification Hierarchy

- **F0 (Predictive adequacy):** Accuracy > Chance, reported with 5-fold variance and confusion matrix.
- **F1 (Label-shuffled null):** If $\Delta C_{real} \approx \Delta C_{null}$, the apparent Q3A effect does not support computational remodeling.
- **F2 (Input-shift synthetic):** The naive estimator generates an apparent effect without structural changes.
- **F3 (Positive-control synthetic):** The calibrated estimator must detect true effects under a preset strong regime.

## Q3 Interpretation Matrix

| Q3A | Q3B | Interpretation |
| :--- | :--- | :--- |
| Significant | Null | apparent remodeling not supported under structural estimand |
| Significant | Significant | evidence consistent with structural remodeling |
| Null | Significant | Q3A may be insensitive; structural signal detectable |
| Null | Null | no detectable evidence under tested estimands |

*Note: Q3C is reported independently of Q3B, focusing strictly on local sensitivity (effective gain) rather than geometry.*
