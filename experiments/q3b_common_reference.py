import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.kan import BrainKAN, _b_spline_basis
from utils_data_loading import load_real_hcp_data
from stats_utils.empirical_null import train_and_eval_brainkan, compute_reference_grid_for_fold
from stats_utils.permutation_tests import permutation_test_Q3B
from stats_utils.multiple_testing import apply_fdr

R_NULLS = 100 # Pilot setting for fast execution

def train_null_model(seed, train_pairs, base_edge_index, num_nodes):
    # Deep copy needed components
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

def compute_G_for_model(model, ref_grids, base_edge_index):
    num_edges = base_edge_index.shape[1]
    src_nodes = base_edge_index[0].numpy()
    
    G = np.zeros(num_edges)
    conv = model.conv1
    
    for e in range(num_edges):
        src = src_nodes[e]
        x_ref = ref_grids[src]
        n = len(x_ref)
        probe = torch.tensor(x_ref, dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            grid_e = conv.grid[e]
            grid_batch = grid_e.unsqueeze(0).expand(n, -1, -1)
            basis = _b_spline_basis(probe, grid_batch, conv.spline_order)
            
            base_out = torch.einsum("Bi,oi->Bo", F.silu(probe), conv.base_weight[e])
            spline_out = torch.einsum("Bik,oik->Bo", basis, conv.spline_weight[e])
            y_pred = base_out + spline_out
            
            # Simple linear fit for DFL
            y_np = y_pred.numpy()
            x_np = x_ref.reshape(-1, 1)
            x_pad = np.hstack([x_np, np.ones_like(x_np)])
            
            G_e = 0
            for o in range(16):
                y_o = y_np[:, o]
                beta, _, _, _ = np.linalg.lstsq(x_pad, y_o, rcond=None)
                y_lin = x_pad @ beta
                ss_res = np.sum((y_o - y_lin)**2)
                ss_tot = np.sum((y_o - np.mean(y_o))**2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 1e-8 else 0
                G_e += (1 - max(0, min(1, r2)))
            G[e] = G_e / 16.0
            
    return G

def run_q3b_v2():
    print("Running Q3B: Common-Reference Structural Geometry (V2)")
    dataset, base_edge_index, num_nodes = load_real_hcp_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_real_G = []
    all_null_G = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"--- Fold {fold+1}/5 ---")
        train_pairs = [dataset[i] for i in train_idx]
        
        # 1. Compute X_ref securely (NO DATA LEAKAGE)
        print("Computing secure reference grid...")
        ref_grid_A = compute_reference_grid_for_fold(train_pairs, base_edge_index)
        
        # 2. Train Real Model
        print("Training real model...")
        train_data = [d for pair in train_pairs for d in pair]
        real_model = train_and_eval_brainkan(train_data, num_nodes, base_edge_index, seed=42)
        G_real = compute_G_for_model(real_model, ref_grid_A, base_edge_index)
        all_real_G.append(G_real)
        
        # 3. Train Null Models in parallel
        print(f"Training {R_NULLS} null models...")
        null_models = Parallel(n_jobs=8)(delayed(train_null_model)(
            seed=1000 + fold*10000 + r, 
            train_pairs=train_pairs, 
            base_edge_index=base_edge_index, 
            num_nodes=num_nodes
        ) for r in range(R_NULLS))
        
        print("Computing G for null models...")
        G_nulls = [compute_G_for_model(m, ref_grid_A, base_edge_index) for m in null_models]
        all_null_G.append(G_nulls)
        
    print("Aggregating Folds...")
    avg_real_G = np.mean(all_real_G, axis=0)
    # Average null distributions across folds: (R, N_edges)
    avg_null_G = np.mean(all_null_G, axis=0) 
    
    print("Applying GPD Tail-Aware Permutation Test...")
    emp_p, gpd_p = permutation_test_Q3B(avg_real_G, avg_null_G)
    
    reject, pvals_corrected = apply_fdr(gpd_p)
    
    print(f"Significant Edges after FDR (GPD p-values): {np.sum(reject)} / 240")
    
    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "hcp")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "q3b_results.npz"), 
             avg_real_G=avg_real_G, avg_null_G=avg_null_G, 
             emp_p=emp_p, gpd_p=gpd_p, pvals_corrected=pvals_corrected, reject=reject)
    
    print("Q3B completed successfully.")

if __name__ == "__main__":
    run_q3b_v2()
