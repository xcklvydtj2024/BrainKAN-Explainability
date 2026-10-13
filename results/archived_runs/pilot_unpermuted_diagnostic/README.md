# Historical Pre-Test Diagnostic Artifacts (Archived Pilot Run)

## Status: ARCHIVED / SUPERSEDED
- **Original Run Date:** October 10–11, 2026
- **Originating Scripts:** `experiments/06_end_to_end_trial.py`, `experiments/07_primary_statistical_analysis.py` (preliminary pilot mode)
- **Superseded By Canonical Results In:** 
  - [`results/exp1_high_precision/`](../exp1_high_precision/) (Exp 1: Full 500-permutation statistical test, Seed 42)
  - [`results/exp5_seed_stability/`](../exp5_seed_stability/) (Exp 5: 10-seed SMPT stability audit, dual estimands)

---

## Technical Context

These archived files were generated during early end-to-end dry-run sanity checks of pipeline mechanics (verifying array shapes, cross-validation folds, and dynamic per-fold reference grid construction) prior to executing production permutation rounds.

Specifically:
- `primary_analysis_config.json`: Ran with `permutations: 0`, which yielded uncomputed effect sizes (`effect_size: NaN`) and trivial default unpermuted p-value (`p_value: 1.0`).
- `primary_null_distribution.csv`: Empty table header (`Permutation,T_null`) reflecting 0 permutations executed in the pilot run.
- `primary_endpoint_by_fold.csv`: Preliminary 5-fold real test statistics ($T_{\mathrm{real}} \approx 0.1789$) recorded without paired permutation comparisons.
- `trial_real.csv`: Pilot trial log ($N=2$ trial permutations).

## Canonical Experimental Results Reference

These diagnostic outputs do **not** represent the production experimental results of the BrainKAN benchmark. All formal conclusions are derived from the production runs:
1. **High-Precision Single-Seed Permutation (Exp 1):** 
   - $N = 500$ permutations at Seed 42: $T_{\mathrm{real}} = 0.17895$, $\bar{T}_{\mathrm{null}} = 0.17733$, $\Delta = +0.00162$, $p = 0.0060$.
   - Output Path: [`results/exp1_high_precision/`](../exp1_high_precision/)
2. **Multi-Seed Optimization Marginalization & SMPT (Exp 5):** 
   - $S = 10$ seeds $\times$ $B = 15$ permutations per seed under dual estimands (Pooled Domain vs. Strict Common Support).
   - Output Path: [`results/exp5_seed_stability/`](../exp5_seed_stability/)
3. **Synthetic Ground-Truth Calibration (Exp 2):**
   - Reference-domain sensitivity scan ($N = 200$ simulations).
   - Output Path: [`results/exp2_synthetic/`](../exp2_synthetic/)
4. **Reference Domain & Cross-Cohort Control (Exp 3):**
   - Output Path: [`results/exp3_ref_domain/`](../exp3_ref_domain/) & [`results/cross_cohort/`](../cross_cohort/)
