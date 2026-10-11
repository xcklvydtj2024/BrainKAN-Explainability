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

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp3_ref_domain")
os.makedirs(ARTIFACTS, exist_ok=True)

SEED = 42
N_SPLITS = 5
N_PROBE = 100
N_PERMUTATIONS = 10 # 10 permutations per fold for domain control evaluation

def main():
    print("=== Experiment 3: Reference Domain Control ===")
    hcp_pairs, base_edge_index, num_nodes = load_real_hcp_data()
    chcp_pairs, _, _ = load_real_chcp_data()
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    # Pre-calculate CHCP per-node reference domain
    node_x_chcp = {n: [] for n in range(num_nodes)}
    for pair in chcp_pairs:
        for n in range(num_nodes):
            node_x_chcp[n].append(pair[0].x[n, 0].item())
            node_x_chcp[n].append(pair[1].x[n, 0].item())
            
    node_t_values_chcp = {}
    r_min_chcp = {}
    r_max_chcp = {}
    for n in range(num_nodes):
        r_min_chcp[n] = float(np.percentile(node_x_chcp[n], 1))
        r_max_chcp[n] = float(np.percentile(node_x_chcp[n], 99))
        node_t_values_chcp[n] = torch.linspace(r_min_chcp[n], r_max_chcp[n], N_PROBE)
        
    print(f"CHCP Native Domain extracted for {num_nodes} nodes (Mean [1%, 99%]: [{np.mean(list(r_min_chcp.values())):.2f}, {np.mean(list(r_max_chcp.values())):.2f}])")
    
    # Pre-generate global flip vectors for reproducible paired permutations across folds
    global_flip_vectors = []
    for b in range(N_PERMUTATIONS):
        np.random.seed(SEED + 2026 + b)
        global_flip_vectors.append(np.random.randint(0, 2, size=len(hcp_pairs)))
    
    results = []
    
    for fold_i, (train_idx, _) in enumerate(kf.split(hcp_pairs)):
        print(f"\n--- Fold {fold_i+1}/{N_SPLITS} ---")
        train_pairs = [hcp_pairs[i] for i in train_idx]
        
        # Per-node empirical domains for HCP fold
        node_x_hcp = {n: [] for n in range(num_nodes)}
        for pair in train_pairs:
            for n in range(num_nodes):
                node_x_hcp[n].append(pair[0].x[n, 0].item())
                node_x_hcp[n].append(pair[1].x[n, 0].item())
        
        node_t_values_hcp = {}
        node_t_values_common = {}
        global_min_hcp, global_max_hcp = 0.0, 0.0
        
        for n in range(num_nodes):
            r_min_h = float(np.percentile(node_x_hcp[n], 1))
            r_max_h = float(np.percentile(node_x_hcp[n], 99))
            node_t_values_hcp[n] = torch.linspace(r_min_h, r_max_h, N_PROBE)
            
            # Common Support per node
            c_min = max(r_min_h, r_min_chcp[n])
            c_max = min(r_max_h, r_max_chcp[n])
            if c_min >= c_max:
                c_min, c_max = r_min_h, r_max_h
            node_t_values_common[n] = torch.linspace(c_min, c_max, N_PROBE)
            
            if n == 0:
                global_min_hcp, global_max_hcp = r_min_h, r_max_h
            else:
                global_min_hcp = min(global_min_hcp, r_min_h)
                global_max_hcp = max(global_max_hcp, r_max_h)
        
        t_range_hcp = (global_min_hcp, global_max_hcp)
        print(f"  HCP Fold Grid Range: [{global_min_hcp:.2f}, {global_max_hcp:.2f}]")
        
        train_data_real = [d for p in train_pairs for d in p]
        train_loader_real = DataLoader(train_data_real, batch_size=32, shuffle=True)
        
        # Train on HCP
        model_real = train_model(train_loader_real, base_edge_index, num_nodes, t_range_hcp, SEED + fold_i)
        
        dfl_real_hcp, _ = extract_fold_dfl(model_real, node_t_values_hcp, base_edge_index)
        dfl_real_chcp, _ = extract_fold_dfl(model_real, node_t_values_chcp, base_edge_index)
        dfl_real_common, _ = extract_fold_dfl(model_real, node_t_values_common, base_edge_index)
        
        real_T_hcp = float(np.mean(dfl_real_hcp))
        real_T_chcp = float(np.mean(dfl_real_chcp))
        real_T_common = float(np.mean(dfl_real_common))
        
        null_T_hcp_list = []
        null_T_chcp_list = []
        null_T_common_list = []
        
        for b in range(N_PERMUTATIONS):
            permuted_list = permute_labels(train_pairs, train_idx, global_flip_vectors[b])
            train_loader_null = DataLoader(permuted_list, batch_size=32, shuffle=True)
            
            model_null = train_model(train_loader_null, base_edge_index, num_nodes, t_range_hcp, SEED + fold_i)
            
            dfl_null_hcp, _ = extract_fold_dfl(model_null, node_t_values_hcp, base_edge_index)
            dfl_null_chcp, _ = extract_fold_dfl(model_null, node_t_values_chcp, base_edge_index)
            dfl_null_common, _ = extract_fold_dfl(model_null, node_t_values_common, base_edge_index)
            
            null_T_hcp_list.append(float(np.mean(dfl_null_hcp)))
            null_T_chcp_list.append(float(np.mean(dfl_null_chcp)))
            null_T_common_list.append(float(np.mean(dfl_null_common)))
            
        results.append({
            'Fold': fold_i + 1,
            'Real_T_HCP': real_T_hcp,
            'Null_T_HCP_Mean': float(np.mean(null_T_hcp_list)),
            'Delta_HCP': real_T_hcp - float(np.mean(null_T_hcp_list)),
            
            'Real_T_CHCP': real_T_chcp,
            'Null_T_CHCP_Mean': float(np.mean(null_T_chcp_list)),
            'Delta_CHCP': real_T_chcp - float(np.mean(null_T_chcp_list)),
            
            'Real_T_Common': real_T_common,
            'Null_T_Common_Mean': float(np.mean(null_T_common_list)),
            'Delta_Common': real_T_common - float(np.mean(null_T_common_list))
        })
        
        print(f"  HCP Eval    -> Real: {real_T_hcp:.4f} | Null: {np.mean(null_T_hcp_list):.4f} | Delta: {results[-1]['Delta_HCP']:+.4f}")
        print(f"  CHCP Eval   -> Real: {real_T_chcp:.4f} | Null: {np.mean(null_T_chcp_list):.4f} | Delta: {results[-1]['Delta_CHCP']:+.4f}")
        print(f"  Common Eval -> Real: {real_T_common:.4f} | Null: {np.mean(null_T_common_list):.4f} | Delta: {results[-1]['Delta_Common']:+.4f}")

    df_res = pd.DataFrame(results)
    
    print("\n=== Summary ===")
    print(f"Mean Delta (HCP Domain): {df_res['Delta_HCP'].mean():.4f}")
    print(f"Mean Delta (CHCP Domain): {df_res['Delta_CHCP'].mean():.4f}")
    print(f"Mean Delta (Common Support): {df_res['Delta_Common'].mean():.4f}")
    
    df_res.to_csv(os.path.join(ARTIFACTS, "reference_domain_control.csv"), index=False)
    print(f"\nSaved Reference Domain Control to {ARTIFACTS}/reference_domain_control.csv")

if __name__ == "__main__":
    main()
