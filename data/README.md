# BrainKNN Data Directory

This directory is intended for storing the processed HCP working-memory task fMRI data.

Due to data privacy and size constraints, the `.npz` files are not included in the repository. You must generate or download `HCP_GNN_features.npz` and place it here to run the codebase.

## Data Format Requirements

The expected format for `HCP_GNN_features.npz` is a dictionary-like archive containing:
- `x_...`: Node features (e.g., BOLD signal vectors)
- `edge_index_...`: Edge connectivity matrices
- `y_...`: Task labels (0 for 0BK, 1 for 2BK)
- `subject_id_...`: An array of subject IDs to ensure proper paired grouping during cross-validation.

Ensure you configure the correct path in `config/paths.yaml` if you place the data elsewhere.
