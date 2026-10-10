# BrainKAN Data and Code Audit Report

## 1. Subject Pairing and Cross-Validation Integrity
- **Status:** **PASS** (with minor caveat)
- **Finding:** The data is loaded as consecutive pairs (`2*i` and `2*i+1`) and grouped into a list of tuples `dataset_pairs`. The `KFold.split(dataset_pairs)` is performed on these tuples. This ensures that 0-back and 2-back from the same subject always stay together and are never split between train/test folds.
- **Caveat:** The pairing assumes the underlying `.npz` arrays are strictly ordered as `[Subj1_0bk, Subj1_2bk, Subj2_0bk, Subj2_2bk...]`. An explicit check (`tests/test_pair_integrity.py`) will verify this if metadata is available.

## 2. Preprocessing, Reference Domain, and Data Leakage
- **Status:** **FAIL** (Requires Fix)
- **Finding (`T_RANGE` Leak):** `T_RANGE` is hardcoded as `(-8.0, 12.0)` based on the "empirical range [-6.6, 11.8]". If this empirical range was derived from the entire dataset, it constitutes information leakage from the test set into the reference domain definition.
- **Finding (`StandardScaler`):** Used only in Q2 (Function Taxonomy) to standardize signatures post-hoc, not during model training. No train-test leakage here.
- **Action Required:** `T_RANGE` (reference domain $\mathcal{R}_k$) must be computed dynamically inside each fold loop using ONLY the training subjects of that fold (e.g., using 1st and 99th percentiles of the training node features).

## 3. PyG Indices and Mapping
- **Status:** **PASS**
- **Finding:** The model correctly uses the topology `base_edge_index` explicitly loaded from `GNN_Edges.pt`. Edge features are extracted sequentially. Assertions already exist (`assert torch.equal(edge_index.cpu(), base_edge_index.cpu())`) to prevent graph ordering misalignment.

## 4. Exception Handling Masking (`ValueError`)
- **Status:** **FAIL** (Requires Fix)
- **Finding:** In `01_q1_q2_q3a_naive_analysis.py`, there is a `try... except ValueError: p_values[e] = 1.0` block when computing the Wilcoxon signed-rank test. A `ValueError` typically occurs if the differences are entirely zero or ties prevent proper ranking. While mapping to `p=1.0` is nominally conservative, it silently masks numerical degeneracy or constant functions.
- **Action Required:** Failed rank tests or degenerate $R^2$ computations (e.g., flat functions) must be explicitly recorded and counted, not silently swallowed. 

## 5. Model Parameter Calculation
- **Status:** **PENDING**
- **Finding:** Must ensure that the real and null models share identical parameter counts, architectures, and random seed budgets.

## Next Steps Before Main Analysis
1. Rewrite the evaluation loop to dynamically compute $\mathcal{R}_k$ based on the `train_idx` data for each fold.
2. Implement explicit logging for degenerate numerical cases rather than catching `ValueError`.
3. Proceed to Stage 2 (Minimal Synthetic Validation).
