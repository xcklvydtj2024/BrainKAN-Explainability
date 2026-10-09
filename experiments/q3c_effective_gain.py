import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.kan import BrainKAN, _b_spline_basis
from utils_data_loading import load_real_hcp_data
from stats_utils.empirical_null import train_and_eval_brainkan
from stats_utils.permutation_tests import permutation_test_Q3B
from stats_utils.multiple_testing import apply_fdr

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

def compute_effective_gain(model, test_pairs, base_edge_index):
    num_edges = base_edge_index.shape[1]
    src_nodes = base_edge_index[0].numpy()
    conv = model.conv1
    
    # Calculate difference in effective gain for each test subject
    T_e_diffs = []
    
    eps = 1e-4
    for d0, d2 in test_pairs:
        x0 = d0.x[:, 0]
        x2 = d2.x[:, 0]
        
        diff_e = np.zeros(num_edges)
        for e in range(num_edges):
            src = src_nodes[e]
            
            def eval_deriv(x_val):
                probe = torch.tensor([x_val], dtype=torch.float32).unsqueeze(1)
                with torch.no_grad():
                    grid_e = conv.grid[e].unsqueeze(0)
                    # + eps
                    probe_p = probe + eps
                    b_p = _b_spline_basis(probe_p, grid_e, conv.spline_order)
                    y_p = torch.einsum("Bi,oi->Bo", F.silu(probe_p), conv.base_weight[e]) + \
                          torch.einsum("Bik,oik->Bo", b_p, conv.spline_weight[e])
                    
                    # - eps
                    probe_m = probe - eps
                    b_m = _b_spline_basis(probe_m, grid_e, conv.spline_order)
                    y_m = torch.einsum("Bi,oi->Bo", F.silu(probe_m), conv.base_weight[e]) + \
                          torch.einsum("Bik,oik->Bo", b_m, conv.spline_weight[e])
                          
                return ((y_p - y_m) / (2*eps)).abs().mean().item()
            
            g0 = eval_deriv(x0[src].item())
            g2 = eval_deriv(x2[src].item())
            diff_e[e] = g2 - g0
            
        T_e_diffs.append(diff_e)
        
    # Return mean difference across test subjects
    return np.mean(T_e_diffs, axis=0)

def run_q3c_v2():
    print("Running Q3C: Effective Gain (V2)")
    dataset, base_edge_index, num_nodes = load_real_hcp_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_real_T = []
    all_null_T = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f"--- Fold {fold+1}/5 ---")
        train_pairs = [dataset[i] for i in train_idx]
        test_pairs = [dataset[i] for i in test_idx]
        
        print("Training real model...")
        train_data = [d for pair in train_pairs for d in pair]
        real_model = train_and_eval_brainkan(train_data, num_nodes, base_edge_index, seed=42)
        T_real = compute_effective_gain(real_model, test_pairs, base_edge_index)
        all_real_T.append(T_real)
        
        print(f"Training {R_NULLS} null models...")
        null_models = Parallel(n_jobs=8)(delayed(train_null_model)(
            seed=2000 + fold*10000 + r, 
            train_pairs=train_pairs, 
            base_edge_index=base_edge_index, 
            num_nodes=num_nodes
        ) for r in range(R_NULLS))
        
        print("Computing T for null models...")
        T_nulls = [compute_effective_gain(m, test_pairs, base_edge_index) for m in null_models]
        all_null_T.append(T_nulls)
        
    print("Aggregating Folds...")
    avg_real_T = np.mean(all_real_T, axis=0)
    avg_null_T = np.mean(all_null_T, axis=0) 
    
    print("Applying GPD Tail-Aware Permutation Test...")
    emp_p, gpd_p = permutation_test_Q3B(avg_real_T, avg_null_T)
    reject, pvals_corrected = apply_fdr(gpd_p)
    print(f"Significant Edges after FDR (GPD p-values): {np.sum(reject)} / 240")
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "hcp")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "q3c_results.npz"), 
             avg_real_T=avg_real_T, avg_null_T=avg_null_T, 
             emp_p=emp_p, gpd_p=gpd_p, pvals_corrected=pvals_corrected, reject=reject)
    
    print("Q3C completed successfully.")

if __name__ == "__main__":
    run_q3c_v2()
