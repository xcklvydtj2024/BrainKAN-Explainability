import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.kan import BrainKAN
from utils_data_loading import load_real_hcp_data, load_real_chcp_data
from stats_utils.empirical_null import train_and_eval_brainkan, compute_reference_grid_for_fold
from experiments.q3b_common_reference import compute_G_for_model
from experiments.q3c_effective_gain import compute_effective_gain

R_NULLS = 100

def train_null_model(seed, train_pairs, base_edge_index, num_nodes):
    import copy
    np.random.seed(seed)
    
    null_pairs = []
    for (d0, d2) in train_pairs:
        nd0 = copy.deepcopy(d0)
        nd2 = copy.deepcopy(d2)
        if np.random.rand() > 0.5:
            nd0.y = d2.y.clone()
            nd2.y = d0.y.clone()
        null_pairs.append((nd0, nd2))
        
    train_data = [d for pair in null_pairs for d in pair]
    model = train_and_eval_brainkan(train_data, num_nodes, base_edge_index, seed=seed)
    return model

def run_chcp_validation():
    print("--- CHCP External Validation with Null Models ---")
    hcp_data, base_edge_index, num_nodes = load_real_hcp_data()
    chcp_data, _, _ = load_real_chcp_data()
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Computing HCP Frozen Reference Grid (X_ref_B)...")
    hcp_ref_grid_B = compute_reference_grid_for_fold(hcp_data, base_edge_index)
    
    all_G_native = []
    all_G_frozen = []
    
    all_null_G_native = []
    all_null_G_frozen = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(chcp_data)):
        print(f"--- CHCP Fold {fold+1}/5 ---")
        train_pairs = [chcp_data[i] for i in train_idx]
        test_pairs = [chcp_data[i] for i in test_idx]
        
        # Reference A: Cohort-native
        ref_grid_A = compute_reference_grid_for_fold(train_pairs, base_edge_index)
        
        # Real Model
        print("Training Real Model...")
        train_data = [d for pair in train_pairs for d in pair]
        real_model = train_and_eval_brainkan(train_data, num_nodes, base_edge_index, seed=42)
        
        G_native = compute_G_for_model(real_model, ref_grid_A, base_edge_index)
        all_G_native.append(G_native)
        
        G_frozen = compute_G_for_model(real_model, hcp_ref_grid_B, base_edge_index)
        all_G_frozen.append(G_frozen)
        
        # Null Models
        print(f"Training {R_NULLS} Null Models in Parallel...")
        null_models = Parallel(n_jobs=8)(delayed(train_null_model)(
            seed=3000 + fold*10000 + r, 
            train_pairs=train_pairs, 
            base_edge_index=base_edge_index, 
            num_nodes=num_nodes
        ) for r in range(R_NULLS))
        
        print("Evaluating Null Models...")
        G_null_native_fold = [compute_G_for_model(m, ref_grid_A, base_edge_index) for m in null_models]
        G_null_frozen_fold = [compute_G_for_model(m, hcp_ref_grid_B, base_edge_index) for m in null_models]
        
        all_null_G_native.append(G_null_native_fold)
        all_null_G_frozen.append(G_null_frozen_fold)
        
    avg_G_native = np.mean(all_G_native, axis=0)
    avg_G_frozen = np.mean(all_G_frozen, axis=0)
    
    avg_null_G_native = np.mean(all_null_G_native, axis=0)
    avg_null_G_frozen = np.mean(all_null_G_frozen, axis=0)
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "cross_cohort")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "chcp_results_full.npz"), 
             avg_G_native=avg_G_native, avg_G_frozen=avg_G_frozen,
             avg_null_G_native=avg_null_G_native, avg_null_G_frozen=avg_null_G_frozen)
             
    print("CHCP External Validation with Null Models completed.")

if __name__ == "__main__":
    run_chcp_validation()
