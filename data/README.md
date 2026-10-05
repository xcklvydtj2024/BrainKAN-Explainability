# BrainKNN Data Directory

This directory stores the processed HCP working-memory task fMRI data.
A lightweight subset (`HCP_GNN_features.npz` and `GNN_Edges.pt`) is included in the repository for reproducibility.

## Data Format Requirements

The code expects two files:
1. `HCP_GNN_features.npz`:
   - `X`: Node features tensor (Shape: `[N_graphs, N_nodes, N_features]`)
   - `y`: Task labels tensor (Shape: `[N_graphs]`, e.g., 0 for 0BK, 1 for 2BK)
2. `GNN_Edges.pt`: PyTorch tensor containing the edge connectivity matrix.

## Important Note on Subject Pairing
The current data loader (`utils_data_loading.py`) assumes strict sequential pairing: indices `[2*i]` and `[2*i + 1]` correspond to the 0-back and 2-back conditions for subject `i`. If you replace this dataset, you must strictly maintain this paired ordering, as our 5-fold cross-validation and paired-permutation tests fundamentally rely on this nested paired structure.
