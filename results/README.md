# BrainKAN Benchmark Results Directory

This directory contains experimental artifacts, statistical evaluation matrices, and publication diagnostic figures for the BrainKAN Explainability Benchmark.

See [`manifest.json`](./manifest.json) for environment specifications, full Git commit SHA, and exact experimental summary estimands.

---

## 🏛️ Directory Architecture & Canonical Result Entrances

| Directory / File | Benchmark Role | Key Artifacts & Estimands |
| :--- | :--- | :--- |
| **[`exp1_high_precision/`](./exp1_high_precision/)** | **Primary Permutation Baseline (Exp 1)** | $N = 500$ permutations at fixed Seed 42 on pooled grid: $T_{\mathrm{real}} = 0.17895, \bar{T}_{\mathrm{null}} = 0.17733, \Delta_{\mathrm{pool}} = +0.00162, p = 0.0060$. Contains chunked nulls and `aggregated_nulls.csv`. |
| **[`exp5_seed_stability/`](./exp5_seed_stability/)** | **Primary Multi-Seed SMPT (Exp 5)** | $S = 10$ seeds $\times$ $B = 15$ permutations per seed under dual estimands:<br>• **Pooled Support:** $\bar{\Delta}_{\mathrm{pool}} = -0.00058$ ($p = 0.9513$, 9/10 seeds negative).<br>• **Strict Common Support:** $\bar{\Delta}_{\mathrm{common}} = +0.00090$ ($p = 0.0003$, 9/10 seeds positive). |
| **[`exp2_synthetic/`](./exp2_synthetic/)** | **Synthetic Calibration (Exp 2)** | Ground-truth calibration on $f^*(x) = x + 0.5\sin(2x)$ ($N = 200$ simulations). Demonstrates gauge/grid dependence: domain shift inflates DFL by $+0.0816 \pm 0.0084$. |
| **[`exp3_ref_domain/`](./exp3_ref_domain/)** | **Reference Domain Control (Exp 3)** | 5-fold cross-validation evaluating impact of support bounding. Common support difference collapses to zero ($\bar{\Delta} = -0.0003$). |
| **[`cross_cohort/`](./cross_cohort/)** | **Cross-Cohort Transfer (HCP $\to$ CHCP)** | Evaluation of trained representations under independent external cohort distribution shift (`chcp_results_full.npz`). |
| **[`exp4_multi_seed/`](./exp4_multi_seed/)** | **Model Capacity Scan (Exp 4)** | Evaluation across hidden dimensions and parameter budgets. |
| **[`primary_analysis/`](./primary_analysis/)** | **Empirical Node Support Audit** | Full empirical support overlap metrics across all 48 cortical ROIs on HCP fMRI (`node_support_overlap_stats.csv`). Mean condition overlap = $38.52\% \pm 5.08\%$. |
| **[`archived_runs/`](./archived_runs/)** | **Archived Historical Diagnostic Runs** | Pre-test pilot pipeline verification runs ($N_{\mathrm{permutations}} = 0$, dry runs) preserved for provenance transparency in [`pilot_unpermuted_diagnostic/`](./archived_runs/pilot_unpermuted_diagnostic/). |

---

## 🔬 Unified Nomenclature: Exploratory Analyses (Q1–Q3) vs. Formal Benchmark (Exp 1–5)

To maintain clarity across project phases, the experimental progression is structured as follows:

1. **Exploratory Characterization Phase (Q1–Q3):**
   - **Q1 / Q2 (Function Characterization & Taxonomy):** Documenting learned edge B-spline forms, non-linear heatmaps, and functional clustering stability (`q1_function_gallery.png`, `q2_cluster_stability.csv`).
   - **Q3A (Operating-Regime Sensitivity):** Naive edge-level tests under unstandardized condition inputs (`q3a_pvalues.csv`), revealing apparent widespread curvature differences caused by input domain divergence.
   - **Q3B (Common-Reference Geometry):** Standardizing evaluation onto aligned reference domains (`q3b_null_summary.csv`), showing substantial shrinkage of naive edge differences.
   - **Q3C (Effective Structural Gain & Identifiability):** Testing whether spline parameters provide structural identifiability beyond affine baselines (`q3c_final_table.csv`, `synthetic_confusability_results.csv`).

2. **Formal Production Benchmark Suite (Exp 1–5):**
   - **Exp 1 (High-Precision Permutation Test):** High-precision single-seed baseline ($N=500$, seed=42) on pooled domain.
   - **Exp 2 (Synthetic Calibration):** Rigorous ground-truth sensitivity test of DFL gauge dependence.
   - **Exp 3 (Common Support Control):** Elimination of boundary extrapolation artifacts on real fMRI graphs.
   - **Exp 4 (Capacity Calibration):** Parameter matching (1,776 params) against linear and MLP baselines.
   - **Exp 5 (Multi-Seed Optimization Marginalization / SMPT):** Definitive 10-seed paired null test addressing optimization stochasticity and establishing the dual-estimand inferential boundary.
