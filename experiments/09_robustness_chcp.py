import os
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kan import BrainKAN
import importlib
p07 = importlib.import_module("experiments.07_primary_statistical_analysis")
extract_fold_dfl = p07.extract_fold_dfl
permute_labels = p07.permute_labels
train_model = p07.train_model
from utils_data_loading import load_real_hcp_data, load_real_chcp_data

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "robustness")
os.makedirs(ARTIFACTS, exist_ok=True)

SEED = 42
N_SPLITS = 5
N_PROBE = 100
N_PERMUTATIONS = 5

def main():
    print("=== Stage 4.3: CHCP Reference Domain Replacement ===")
    hcp_pairs, base_edge_index, num_nodes = load_real_hcp_data()
    chcp_pairs, _, _ = load_real_chcp_data()
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    # Pre-calculate CHCP reference domain per node
    node_x_chcp = {n: [] for n in range(num_nodes)}
    for pair in chcp_pairs:
        for n in range(num_nodes):
            node_x_chcp[n].append(pair[0].x[n, 0].item())
            node_x_chcp[n].append(pair[1].x[n, 0].item())
            
    node_t_values_chcp = {}
    for n in range(num_nodes):
        r_min = float(np.percentile(node_x_chcp[n], 1))
        r_max = float(np.percentile(node_x_chcp[n], 99))
        node_t_values_chcp[n] = torch.linspace(r_min, r_max, N_PROBE)
        
    global_flip_vectors = []
    for b in range(N_PERMUTATIONS):
        np.random.seed(SEED + 2026 + b)
        global_flip_vectors.append(np.random.randint(0, 2, size=len(hcp_pairs)))
        
    results = []
    
    for fold_i, (train_idx, _) in enumerate(kf.split(hcp_pairs)):
        print(f"\n--- Fold {fold_i+1}/{N_SPLITS} ---")
        train_pairs = [hcp_pairs[i] for i in train_idx]
        
        node_x_hcp = {n: [] for n in range(num_nodes)}
        for pair in train_pairs:
            for n in range(num_nodes):
                node_x_hcp[n].append(pair[0].x[n, 0].item())
                node_x_hcp[n].append(pair[1].x[n, 0].item())
        
        node_t_values_hcp = {}
        global_min, global_max = 0.0, 0.0
        for n in range(num_nodes):
            r_min = float(np.percentile(node_x_hcp[n], 1))
            r_max = float(np.percentile(node_x_hcp[n], 99))
            node_t_values_hcp[n] = torch.linspace(r_min, r_max, N_PROBE)
            if n == 0:
                global_min, global_max = r_min, r_max
            else:
                global_min = min(global_min, r_min)
                global_max = max(global_max, r_max)
                
        t_range_hcp = (global_min, global_max)
        
        train_data_real = [d for p in train_pairs for d in p]
        train_loader_real = DataLoader(train_data_real, batch_size=32, shuffle=True)
        
        model_real = train_model(train_loader_real, base_edge_index, num_nodes, t_range_hcp, SEED+fold_i)
        
        dfl_real_hcp, _ = extract_fold_dfl(model_real, node_t_values_hcp, base_edge_index)
        dfl_real_chcp, _ = extract_fold_dfl(model_real, node_t_values_chcp, base_edge_index)
        
        real_T_hcp = float(np.mean(dfl_real_hcp))
        real_T_chcp = float(np.mean(dfl_real_chcp))
        
        null_T_hcp_list = []
        null_T_chcp_list = []
        
        for b in range(N_PERMUTATIONS):
            permuted_list = permute_labels(train_pairs, train_idx, global_flip_vectors[b])
            train_loader_null = DataLoader(permuted_list, batch_size=32, shuffle=True)
            
            model_null = train_model(train_loader_null, base_edge_index, num_nodes, t_range_hcp, SEED+fold_i)
            
            dfl_null_hcp, _ = extract_fold_dfl(model_null, node_t_values_hcp, base_edge_index)
            dfl_null_chcp, _ = extract_fold_dfl(model_null, node_t_values_chcp, base_edge_index)
            
            null_T_hcp_list.append(float(np.mean(dfl_null_hcp)))
            null_T_chcp_list.append(float(np.mean(dfl_null_chcp)))
            
        results.append({
            'Fold': fold_i + 1,
            'Real_T_HCP': real_T_hcp,
            'Null_T_HCP_Mean': float(np.mean(null_T_hcp_list)),
            'Delta_HCP': real_T_hcp - float(np.mean(null_T_hcp_list)),
            'Real_T_CHCP': real_T_chcp,
            'Null_T_CHCP_Mean': float(np.mean(null_T_chcp_list)),
            'Delta_CHCP': real_T_chcp - float(np.mean(null_T_chcp_list))
        })
        
        print(f"  HCP Eval -> Real: {real_T_hcp:.4f} | Null: {np.mean(null_T_hcp_list):.4f} | Delta: {results[-1]['Delta_HCP']:+.4f}")
        print(f"  CHCP Eval-> Real: {real_T_chcp:.4f} | Null: {np.mean(null_T_chcp_list):.4f} | Delta: {results[-1]['Delta_CHCP']:+.4f}")

    df_res = pd.DataFrame(results)
    
    print("\n=== Summary ===")
    print(f"Mean Delta (Frozen HCP Domain): {df_res['Delta_HCP'].mean():.4f}")
    print(f"Mean Delta (Native CHCP Domain): {df_res['Delta_CHCP'].mean():.4f}")
    
    df_res.to_csv(os.path.join(ARTIFACTS, "chcp_reference_domain_robustness.csv"), index=False)
    print(f"\nSaved CHCP robustness to {ARTIFACTS}/chcp_reference_domain_robustness.csv")

if __name__ == "__main__":
    main()
