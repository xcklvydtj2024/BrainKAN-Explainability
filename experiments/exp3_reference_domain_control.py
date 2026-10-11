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
N_PERMUTATIONS = 10 # Increase to 10 for better accuracy in this exp

def main():
    print("=== Experiment 3: Reference Domain Control ===")
    hcp_pairs, base_edge_index, num_nodes = load_real_hcp_data()
    chcp_pairs, _, _ = load_real_chcp_data()
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    # Pre-calculate CHCP reference domain globally
    chcp_x_vals = []
    for pair in chcp_pairs:
        chcp_x_vals.extend(pair[0].x[:,0].numpy().tolist())
        chcp_x_vals.extend(pair[1].x[:,0].numpy().tolist())
    r_min_chcp, r_max_chcp = np.percentile(chcp_x_vals, 1), np.percentile(chcp_x_vals, 99)
    t_values_chcp = torch.linspace(r_min_chcp, r_max_chcp, N_PROBE)
    print(f"CHCP Native Domain (R_CHCP): [{r_min_chcp:.2f}, {r_max_chcp:.2f}]")
    
    results = []
    
    for fold_i, (train_idx, _) in enumerate(kf.split(hcp_pairs)):
        print(f"\n--- Fold {fold_i+1}/{N_SPLITS} ---")
        train_pairs = [hcp_pairs[i] for i in train_idx]
        
        train_x_vals = []
        for pair in train_pairs:
            train_x_vals.extend(pair[0].x[:,0].numpy().tolist())
            train_x_vals.extend(pair[1].x[:,0].numpy().tolist())
        
        r_min_hcp, r_max_hcp = np.percentile(train_x_vals, 1), np.percentile(train_x_vals, 99)
        t_range_hcp = (r_min_hcp, r_max_hcp)
        t_values_hcp = torch.linspace(r_min_hcp, r_max_hcp, N_PROBE)
        print(f"  HCP Frozen Domain (R_HCP_fold): [{r_min_hcp:.2f}, {r_max_hcp:.2f}]")
        
        # Calculate Common Support Domain
        r_min_common = max(r_min_hcp, r_min_chcp)
        r_max_common = min(r_max_hcp, r_max_chcp)
        t_values_common = torch.linspace(r_min_common, r_max_common, N_PROBE)
        print(f"  Common Support Domain (R_Common): [{r_min_common:.2f}, {r_max_common:.2f}]")
        
        train_data_real = [d for p in train_pairs for d in p]
        train_loader_real = DataLoader(train_data_real, batch_size=32, shuffle=True)
        
        # Train on HCP!
        model_real = train_model(train_loader_real, base_edge_index, num_nodes, t_range_hcp, SEED+fold_i)
        
        dfl_real_hcp, _ = extract_fold_dfl(model_real, t_values_hcp)
        dfl_real_chcp, _ = extract_fold_dfl(model_real, t_values_chcp)
        dfl_real_common, _ = extract_fold_dfl(model_real, t_values_common)
        
        real_T_hcp = np.mean(dfl_real_hcp)
        real_T_chcp = np.mean(dfl_real_chcp)
        real_T_common = np.mean(dfl_real_common)
        
        null_T_hcp_list = []
        null_T_chcp_list = []
        null_T_common_list = []
        
        for b in range(N_PERMUTATIONS):
            np.random.seed(SEED + fold_i + b * 100)
            permuted_list = permute_labels(train_pairs)
            train_loader_null = DataLoader(permuted_list, batch_size=32, shuffle=True)
            
            # Null model also trained on HCP!
            model_null = train_model(train_loader_null, base_edge_index, num_nodes, t_range_hcp, SEED+fold_i)
            
            dfl_null_hcp, _ = extract_fold_dfl(model_null, t_values_hcp)
            dfl_null_chcp, _ = extract_fold_dfl(model_null, t_values_chcp)
            dfl_null_common, _ = extract_fold_dfl(model_null, t_values_common)
            
            null_T_hcp_list.append(np.mean(dfl_null_hcp))
            null_T_chcp_list.append(np.mean(dfl_null_chcp))
            null_T_common_list.append(np.mean(dfl_null_common))
            
        results.append({
            'Fold': fold_i + 1,
            'Real_T_HCP': real_T_hcp,
            'Null_T_HCP_Mean': np.mean(null_T_hcp_list),
            'Delta_HCP': real_T_hcp - np.mean(null_T_hcp_list),
            
            'Real_T_CHCP': real_T_chcp,
            'Null_T_CHCP_Mean': np.mean(null_T_chcp_list),
            'Delta_CHCP': real_T_chcp - np.mean(null_T_chcp_list),
            
            'Real_T_Common': real_T_common,
            'Null_T_Common_Mean': np.mean(null_T_common_list),
            'Delta_Common': real_T_common - np.mean(null_T_common_list)
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
