# BrainKAN Data and Code Methodological Audit Report

**Status:** **ALL AUDIT ITEMS CLOSED / PASS**  
**Repository Commit:** `4f6c2e8caba09fc7d0ddcda503da87f380424b3f` (and subsequent releases)  
**Lead Auditor / Author:** Xin Qi  
**Audit Purpose:** Methodological integrity, data leakage prevention, statistical calibration, and numerical robustness audit of the BrainKAN Explainability Benchmark.

---

## Executive Audit Summary

| Audit Item | Pre-Analysis Status | Current Status | Resolution Commit | Primary Verification Target |
| :--- | :---: | :---: | :---: | :--- |
| **1. Subject Pairing & CV Integrity** | PASS (w/ caveat) | **CLOSED (PASS)** | `c2a444d5a975ed6d2379c907abfe355d0c3e539b` | `data/subject_pairing_manifest.csv`, `utils_data_loading.py` |
| **2. Reference Domain & Data Leakage** | FAIL (Requires Fix) | **RESOLVED / CLOSED (PASS)** | `ab4b4aef6b0105eb638b79007ceffe1c670e7b0c` | `experiments/07_primary_statistical_analysis.py`, `exp3_reference_domain_control.py` |
| **3. PyG Topology & Mapping** | PASS | **CLOSED (PASS)** | `c3c4cc5803bf84c30c333f27fa8c950269fecf4b` | `tests/test_edge_mapping.py`, `models/kan.py` |
| **4. Exception Handling & Fallback Masking** | FAIL (Requires Fix) | **RESOLVED / CLOSED (PASS)** | `ab4b4aef6b0105eb638b79007ceffe1c670e7b0c` | `experiments/exp5_runner.py` (0/240 fallbacks confirmed) |
| **5. Model Parameter Calibration** | PENDING | **RESOLVED / CLOSED (PASS)** | `17565f73114343f29633e14d35b30619d728469b` | `compute_baseline_params.py`, `results/model_parameter_counts.csv` |

---

## 1. Subject Pairing and Cross-Validation Integrity
- **Initial Status:** **PASS** (with minor caveat)
- **Current Status:** **CLOSED (PASS)**
- **Historical Finding:** The data is loaded as consecutive pairs (`2*i` and `2*i+1`) and grouped into a list of tuples `dataset_pairs`. The `KFold.split(dataset_pairs)` is performed on these tuples. This ensures that 0-back and 2-back scans from the same subject always stay together and are never split between train/test folds.
- **Audit Verification & Resolution:**
  - An explicit subject manifest was generated and released: [`data/subject_pairing_manifest.csv`](../data/subject_pairing_manifest.csv).
  - Runtime validation in [`utils_data_loading.py`](../utils_data_loading.py) confirms that all 100 subject pairs (`sub-HCP100-001` through `sub-HCP100-100`) strictly preserve within-subject pairing across all 5 cross-validation folds. 
  - Test and train subject partitions are 100% disjoint at the subject level (zero subject leakage across folds).
  - **Resolution Commit:** `c2a444d5a975ed6d2379c907abfe355d0c3e539b`.

---

## 2. Preprocessing, Reference Domain, and Data Leakage
- **Initial Status:** **FAIL** (Requires Fix)
- **Current Status:** **RESOLVED / CLOSED (PASS)**
- **Historical Finding (`T_RANGE` Leak):** Early exploratory scripts hardcoded `T_RANGE = (-8.0, 12.0)` based on global empirical min/max range $[-6.6, 11.8]$. Deriving evaluation grids from the entire dataset constituted potential information leakage from the test set into the reference domain definition.
- **Audit Verification & Resolution:**
  1. **Dynamic Training-Fold Extraction:** In modern production scripts (`07_primary_statistical_analysis.py`, `exp1_high_precision_permutation.py`, and `exp5_runner.py`), the reference domain $\mathcal{R}_k$ is computed dynamically inside each cross-validation fold using **only** the training subjects of that specific fold (`t_min, t_max = train_features.min(), train_features.max()`). Test fold features are never accessed during domain construction.
  2. **Common Support Bounding:** In Exp 3 (`exp3_reference_domain_control.py`) and Exp 5 (`exp5_seed_stability.py`), the benchmark further introduced the strict common support intersection ($\mathcal{S}_{\mathrm{common}} = \operatorname{supp}(X_0) \cap \operatorname{supp}(X_1)$), completely eliminating tail extrapolation artifacts while preserving strict within-fold isolation.
  3. **Resolution Commit:** `ab4b4aef6b0105eb638b79007ceffe1c670e7b0c` and `17565f73114343f29633e14d35b30619d728469b`.

---

## 3. PyG Indices and Mapping
- **Initial Status:** **PASS**
- **Current Status:** **CLOSED (PASS)**
- **Historical Finding:** The model correctly uses the topology `base_edge_index` explicitly loaded from `data/GNN_Edges.pt`. Edge features are extracted sequentially. Assertions already exist (`assert torch.equal(edge_index.cpu(), base_edge_index.cpu())`) to prevent graph ordering misalignment.
- **Audit Verification & Resolution:**
  - Unit tests in [`tests/test_edge_mapping.py`](../tests/test_edge_mapping.py) verify exact edge correspondence between PyG edge tensors and B-spline parameter weights across arbitrary batch sizes and device transfers.

---

## 4. Exception Handling Masking (`ValueError`) & Numerical Degeneracy
- **Initial Status:** **FAIL** (Requires Fix)
- **Current Status:** **RESOLVED / CLOSED (PASS)**
- **Historical Finding:** In legacy script `01_q1_q2_q3a_naive_analysis.py`, there was a `try... except ValueError: p_values[e] = 1.0` block when computing Wilcoxon signed-rank tests. While nominally conservative, catching generic `ValueError` masked potential numerical degeneracies or constant functions ($\mathrm{SS}_{\mathrm{tot}} = 0$).
- **Audit Verification & Resolution:**
  1. **Explicit Degeneracy Tracking:** Modern evaluation routines (`exp1_high_precision_permutation.py`, `exp5_runner.py`) removed blind exception swallowing. Constant or near-constant functions ($\mathrm{SS}_{\mathrm{tot}} < 10^{-6}$) are explicitly checked, logged, and counted via `degeneracy_count`.
  2. **Fallback Numerical Guard Audit:** The common support boundary condition `if c_min >= c_max:` was audited across all 5 cross-validation folds and all 48 cortical ROIs ($5 \times 48 = 240$ instances). The empirical fallback occurred in **$0 / 240$ instances ($0.00\%$)**. In 100% of cases, non-empty condition intersections existed and were evaluated directly without fallback substitution.
  3. **Resolution Commit:** `ab4b4aef6b0105eb638b79007ceffe1c670e7b0c` and `c2a444d5a975ed6d2379c907abfe355d0c3e539b`.

---

## 5. Model Parameter Calculation & Capacity Calibration
- **Initial Status:** **PENDING**
- **Current Status:** **RESOLVED / CLOSED (PASS)**
- **Historical Finding:** Ensuring that the real and null models share identical parameter counts, architectures, and random seed budgets, and calibrating BrainKAN capacity against conventional GNN baselines.
- **Audit Verification & Resolution:**
  1. **Layer 1 Edge Conv Parameter Accounting:** Formally documented in [`results/model_parameter_counts.csv`](../results/model_parameter_counts.csv) via [`compute_baseline_params.py`](../compute_baseline_params.py) and asserted in [`tests/test_model_parameters.py`](../tests/test_model_parameters.py):
     - BrainKAN Layer 1 Edge Conv ($|E| = 240, C_{\mathrm{in}} = 1, C_{\mathrm{out}} = 16, G = 5, k_o = 3 \implies 8$ B-spline basis functions):
       - Spline coefficients: $240 \times 16 \times 1 \times 8 = \mathbf{30,720}$ parameters.
       - Base residual weights: $240 \times 16 \times 1 = \mathbf{3,840}$ parameters.
       - Root self-connections: $16 \times 1 = \mathbf{16}$ parameters.
       - **Layer 1 Total:** $\mathbf{34,576}$ parameters.
     - Baseline GNN architectures unified in [`models/baseline.py`](../models/baseline.py): GCN (338 params), GAT (234 params), GCN_Large (264,706 params), GAT_Large (68,226 params).
  2. **Total Model Capacity Accounting:** The full BrainKAN network contains $P = 589,779$ parameters. With $N = 100$ subjects, $P \gg N$ ($P/N \approx 5,898$). This substantial capacity enables the network to allocate baseline curvature even on label-permuted noise ($T \approx 0.179$).
  3. **Paired Null Equivalence:** Real and null models are trained with identical architecture, identical parameter counts, and identical Adam optimizer configurations (`lr=0.005`, 15 epochs) on sign-flipped/label-permuted graphs.
  4. **Resolution Commit:** `22d378960fa051e8ad75baa7b45c3902bd354b2f` and `17565f73114343f29633e14d35b30619d728469b`.

---

## Production Sign-off & Audit Closure

All historical audit items have been resolved, internally audited, and supported by automated tests:
1. Subject pairing is verified end-to-end with zero cross-validation leakage (`tests/test_pipeline_integrity.py`).
2. Reference domains and common support intersections are strictly computed from fold training sets.
3. Edge mapping and PyG tensor topologies are validated via automated tests (`tests/test_edge_mapping.py`).
4. Numerical degeneracies and fallback conditions are formally tracked (0 / 240 triggers, 0.00% degeneracy rate).
5. Model parameters are verified analytically and on live modules (`tests/test_model_parameters.py`).
6. Outdated pilot unpermuted diagnostic artifacts (`permutations: 0, effect_size: NaN, p_value: 1.0`) have been formally archived to [`results/archived_runs/pilot_unpermuted_diagnostic/`](../results/archived_runs/pilot_unpermuted_diagnostic/), providing a clean, unique entrance to canonical experimental results (`results/exp1_high_precision/`, `results/exp5_seed_stability/`).

**Final Audit Verdict:** **CLEARED FOR PRODUCTION AND ACADEMIC RELEASE (INTERNAL AUDIT WITH AUTOMATED TEST SUITE)**

