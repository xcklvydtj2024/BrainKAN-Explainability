import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import time
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kan import BrainKAN, _b_spline_basis
from utils_data_loading import load_real_hcp_data

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "primary_analysis")
os.makedirs(ARTIFACTS, exist_ok=True)

SEED = 42
N_SPLITS = 5
N_PROBE = 100
N_PERMUTATIONS = 100
EPOCHS = 15

def extract_fold_dfl(model, t_values):
    """
    Computes DFL (1-R^2) for each edge over the fixed fold grid.
    Returns: array of length num_edges, and number of degeneracies.
    """
    conv1 = model.conv1
    t_np = t_values.numpy()
    batch_size = len(t_values)
    
    num_edges = conv1.num_edges
    in_channels = conv1.in_channels
    out_channels = conv1.out_channels
    
    dfl_scores = []
    degeneracy_count = 0
    
    for edge_id in range(num_edges):
        spline_weight = conv1.spline_weight[edge_id]
        base_weight = conv1.base_weight[edge_id]
        grid = conv1.grid[edge_id]
        grid_batch = grid.unsqueeze(0).expand(batch_size, -1, -1)
        
        probe_inputs = t_values.unsqueeze(1)
        
        spline_basis = _b_spline_basis(probe_inputs, grid_batch, conv1.spline_order)
        base_out = torch.einsum("Bi,oi->Bo", F.silu(probe_inputs), base_weight)
        spline_out = torch.einsum("Bik,oik->Bo", spline_basis, spline_weight)
        msg = base_out + spline_out # [batch, out_channels]
        
        edge_ch_dfl = []
        for o in range(out_channels):
            y = msg[:, o].detach().numpy()
            ss_tot = np.sum((y - np.mean(y))**2)
            
            if ss_tot < 1e-6:
                degeneracy_count += 1
                edge_ch_dfl.append(0.0) 
            else:
                A = np.vstack([t_np, np.ones(len(t_np))]).T
                w, b = np.linalg.lstsq(A, y, rcond=None)[0]
                y_pred = w * t_np + b
                ss_res = np.sum((y - y_pred)**2)
                r2 = 1 - (ss_res / ss_tot)
                edge_ch_dfl.append(max(0.0, 1 - r2))
                
        dfl_scores.append(np.mean(edge_ch_dfl))
        
    return np.array(dfl_scores), degeneracy_count

def permute_labels(train_data):
    """Subject-level label permutation respecting exchangeability."""
    permuted_data = []
    for pair in train_data:
        g0_new = pair[0].clone()
        g1_new = pair[1].clone()
        if np.random.rand() > 0.5:
            g0_new.y, g1_new.y = pair[1].y.clone(), pair[0].y.clone()
        permuted_data.append(g0_new)
        permuted_data.append(g1_new)
    return permuted_data

def train_model(train_loader, base_edge_index, num_nodes, t_range, seed):
    torch.manual_seed(seed)
    model = BrainKAN(
        base_edge_index=base_edge_index,
        num_nodes=num_nodes,
        in_channels=1,
        hidden_dim=16,
        num_classes=2,
        grid_range=t_range
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    model.train()
    for _ in range(EPOCHS):
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
    return model

def main():
    print("=== Stage 3: Primary Statistical Analysis ===")
    dataset_pairs, base_edge_index, num_nodes = load_real_hcp_data()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    real_T = []
    null_T = {b: [] for b in range(N_PERMUTATIONS)}
    config_record = {'fold_ranges': {}}
    
    start_time = time.time()
    
    for fold_i, (train_idx, _) in enumerate(kf.split(dataset_pairs)):
        print(f"\n--- Fold {fold_i+1}/{N_SPLITS} ---")
        train_pairs = [dataset_pairs[i] for i in train_idx]
        
        train_x_vals = []
        for pair in train_pairs:
            train_x_vals.extend(pair[0].x[:,0].numpy().tolist())
            train_x_vals.extend(pair[1].x[:,0].numpy().tolist())
        
        r_min, r_max = np.percentile(train_x_vals, 1), np.percentile(train_x_vals, 99)
        t_range = (r_min, r_max)
        t_values = torch.linspace(r_min, r_max, N_PROBE)
        config_record['fold_ranges'][f'Fold_{fold_i+1}'] = (r_min, r_max)
        print(f"  R_k (1st-99th percentile): [{r_min:.2f}, {r_max:.2f}]")
        
        train_data_real = [d for p in train_pairs for d in p]
        train_loader_real = DataLoader(train_data_real, batch_size=32, shuffle=True)
        
        print("  Training Real Model...")
        model_real = train_model(train_loader_real, base_edge_index, num_nodes, t_range, SEED+fold_i)
        dfl_real, deg_real = extract_fold_dfl(model_real, t_values)
        T_k_real = np.mean(dfl_real)
        real_T.append(T_k_real)
        print(f"    Real T_k: {T_k_real:.4f} (Degenerate channels: {deg_real})")
        
        for b in range(N_PERMUTATIONS):
            np.random.seed(SEED + fold_i + b * 100)
            permuted_list = permute_labels(train_pairs)
            train_loader_null = DataLoader(permuted_list, batch_size=32, shuffle=True)
            
            model_null = train_model(train_loader_null, base_edge_index, num_nodes, t_range, SEED+fold_i)
            dfl_null, _ = extract_fold_dfl(model_null, t_values)
            null_T[b].append(np.mean(dfl_null))
            
            if (b + 1) % 10 == 0:
                print(f"    Completed Null Model {b+1}/{N_PERMUTATIONS}")
                
    # Summarize & Export
    T_real_global = np.mean(real_T)
    T_null_globals = [np.mean(null_T[b]) for b in range(N_PERMUTATIONS)]
    
    delta_task = T_real_global - np.mean(T_null_globals)
    p_value = (1 + sum([1 for tn in T_null_globals if tn >= T_real_global])) / (N_PERMUTATIONS + 1)
    
    print("\n=== Final Statistical Inference ===")
    print(f"T_real: {T_real_global:.4f}")
    print(f"T_null mean: {np.mean(T_null_globals):.4f}")
    print(f"Estimated Delta_task: {delta_task:.4f}")
    print(f"Empirical P-value: {p_value:.4f}")
    print(f"Total time: {time.time() - start_time:.2f}s")
    
    pd.DataFrame({'Fold': range(1, 6), 'T_real': real_T}).to_csv(os.path.join(ARTIFACTS, "primary_endpoint_by_fold.csv"), index=False)
    pd.DataFrame({'Permutation': range(1, N_PERMUTATIONS+1), 'T_null': T_null_globals}).to_csv(os.path.join(ARTIFACTS, "primary_null_distribution.csv"), index=False)
    
    with open(os.path.join(ARTIFACTS, "primary_analysis_config.json"), "w") as f:
        json.dump({
            "seed": SEED,
            "permutations": N_PERMUTATIONS,
            "epochs": EPOCHS,
            "effect_size": delta_task,
            "p_value": p_value,
            **config_record
        }, f, indent=4)
    print(f"Saved artifacts to {ARTIFACTS}/")

if __name__ == "__main__":
    main()
