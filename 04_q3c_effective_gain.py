import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.kan import BrainKAN, _b_spline_basis

# Setup paths
WORKDIR = r"D:\BrainKNN"
ARTIFACTS = os.path.join(WORKDIR, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

# Load data and models
def get_analytic_deriv(conv, probe, e):
    eps = 1e-4
    n = len(probe)
    grid_e = conv.grid[e].unsqueeze(0).expand(n, -1, -1)
    
    def eval_f(x_tensor):
        x_probe = x_tensor.unsqueeze(1) # [n, 1]
        
        # B-spline basis
        basis = _b_spline_basis(x_probe, grid_e, conv.spline_order) # [n, 1, num_bases]
            
        base_out = torch.einsum("Bi,oi->Bo", torch.nn.functional.silu(x_probe), conv.base_weight[e])
        spline_out = torch.einsum("Bik,oik->Bo", basis, conv.spline_weight[e])
        return base_out + spline_out
        
    y_plus = eval_f(probe + eps)
    y_minus = eval_f(probe - eps)
    return (y_plus - y_minus) / (2 * eps) # [n, 16]

def evaluate_Q3C(model, test_pairs, ref_grids, base_edge_index):
    conv = model.conv1
    num_edges = conv.num_edges
    
    G_actual_0bk = np.zeros((len(test_pairs), num_edges))
    G_actual_2bk = np.zeros((len(test_pairs), num_edges))
    G_ref = np.zeros(num_edges)
    
    for e in range(num_edges):
        src = base_edge_index[0, e].item()
        x_ref = torch.tensor(ref_grids[src], dtype=torch.float32)
        with torch.no_grad():
            deriv = get_analytic_deriv(conv, x_ref, e)
        G_ref[e] = deriv.abs().mean().item()
        
    for i, (g0, g2) in enumerate(test_pairs):
        x0 = g0.x[:, 0]
        x2 = g2.x[:, 0]
        
        for e in range(num_edges):
            src = base_edge_index[0, e].item()
            val_0 = torch.tensor([x0[src]], dtype=torch.float32)
            val_2 = torch.tensor([x2[src]], dtype=torch.float32)
            
            with torch.no_grad():
                d0 = get_analytic_deriv(conv, val_0, e).abs().mean().item()
                d2 = get_analytic_deriv(conv, val_2, e).abs().mean().item()
                
            G_actual_0bk[i, e] = d0
            G_actual_2bk[i, e] = d2
            
    return G_actual_0bk, G_actual_2bk, G_ref

def train_model(train_pairs, base_edge_index, num_nodes, is_null=False):
    if is_null:
        working_pairs = []
        for (d0, d2) in train_pairs:
            d0_y, d2_y = d0.y.clone(), d2.y.clone()
            if np.random.rand() > 0.5: # Subject-level swap (Authoritative null)
                d0.y, d2.y = d2_y, d0_y
            working_pairs.append((d0, d2))
    else:
        working_pairs = train_pairs
        
    train_data = [d for pair in working_pairs for d in pair]
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    model = BrainKAN(base_edge_index, num_nodes, 1, 16, 2, grid_range=(-8.0, 12.0))
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for _ in range(15):
        for batch in train_loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = torch.nn.functional.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
    model.eval()
    
    # Restore labels if modified
    if is_null:
        for (d0, d2) in train_pairs:
            d0.y = torch.tensor(0, dtype=torch.long)
            d2.y = torch.tensor(1, dtype=torch.long)
            
    return model

def compute_reference_grids(dataset_pairs, base_edge_index):
    num_nodes = dataset_pairs[0][0].x.shape[0]
    all_x = np.zeros((len(dataset_pairs) * 2, num_nodes))
    for i, (d0, d2) in enumerate(dataset_pairs):
        all_x[2*i] = d0.x[:, 0].numpy()
        all_x[2*i+1] = d2.x[:, 0].numpy()
    ref_grids = {}
    src_nodes = base_edge_index[0].numpy()
    q_vals = np.linspace(1, 99, 99)
    for src in np.unique(src_nodes):
        ref_grids[src] = np.percentile(all_x[:, src], q_vals)
    return ref_grids

def main():
    import sys
    sys.path.append(WORKDIR)
    from utils_data_loading import load_real_hcp_data
    from models.kan import BrainKAN
    
    print("Loading Data...")
    res = load_real_hcp_data()
    dataset_pairs = res[0]
    base_edge_index = res[1]
    num_nodes = res[2]
    num_edges = base_edge_index.shape[1]
    
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(dataset_pairs))
    
    real_dG = {e: [] for e in range(num_edges)}
    null_dG = {e: [] for e in range(num_edges)}
    
    real_Gref_all = []
    null_Gref_all = []
    
    print("Evaluating Q3C on Real and Null models (Training on the fly)...")
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Fold {fold_i+1}/5...")
        train_pairs = [dataset_pairs[i] for i in train_idx]
        test_pairs = [dataset_pairs[i] for i in test_idx]
        
        # Per-fold x_ref: compute reference grids strictly on train data to avoid leakage
        ref_grids = compute_reference_grids(train_pairs, base_edge_index)
        
        # Train
        torch.manual_seed(42 + fold_i)
        np.random.seed(42 + fold_i)
        real_model = train_model(train_pairs, base_edge_index, num_nodes, is_null=False)
        null_model = train_model(train_pairs, base_edge_index, num_nodes, is_null=True)
        
        # Evaluate
        G0, G2, Gref = evaluate_Q3C(real_model, test_pairs, ref_grids, base_edge_index)
        dG = G2 - G0 # [n_test, num_edges]
        real_Gref_all.append(Gref)
        
        nG0, nG2, nGref = evaluate_Q3C(null_model, test_pairs, ref_grids, base_edge_index)
        ndG = nG2 - nG0
        null_Gref_all.append(nGref)
        
        for i in range(len(test_pairs)):
            for e in range(num_edges):
                real_dG[e].append(dG[i, e])
                null_dG[e].append(ndG[i, e])
                
    real_Gref = np.mean(real_Gref_all, axis=0)
    null_Gref = np.mean(null_Gref_all, axis=0)
    
    # Statistical Testing: Subject-level Paired Permutation Test
    p_values = []
    n_perms = 5000
    
    for e in range(num_edges):
        # dG_r and dG_n have the same length (n_subjects)
        # We test the hypothesis that mean(dG_r) > mean(dG_n) using permutations
        diff = np.array(real_dG[e]) - np.array(null_dG[e])
        observed_mean = np.mean(diff)
        
        if observed_mean <= 0:
            p_values.append(1.0)
            continue
            
        # Subject-level permutation: flip sign of the difference
        # representing a swap of real and null assignment for that subject
        signs = np.random.choice([-1, 1], size=(n_perms, len(diff)))
        permuted_means = np.mean(signs * diff, axis=1)
        p_val = (np.sum(permuted_means >= observed_mean) + 1) / (n_perms + 1)
        p_values.append(p_val)
        
    # FDR
    from statsmodels.stats.multitest import multipletests
    _, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
    
    sig_edges = np.sum(p_fdr < 0.05)
    
    print(f"\n=== Q3C Effective Gain Results ===")
    print(f"Number of edges with significant Delta G (Real > Null, FDR < 0.05): {sig_edges} / {num_edges}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(null_Gref, real_Gref, alpha=0.5, color='purple')
    plt.plot([0, max(real_Gref)], [0, max(real_Gref)], 'k--')
    plt.xlabel("Null Model Structural Gain ($G^{ref}$)")
    plt.ylabel("Real Model Structural Gain ($G^{ref}$)")
    plt.title("Structural Gain comparison: Real vs Null")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(ARTIFACTS, "q3c_structural_gain.png"), dpi=200)
    
    # Save decision summary
    decision = "C" # Default
    if sig_edges > 0:
        decision = "A"
    else:
        decision = "B"
        
    with open(os.path.join(ARTIFACTS, "final_decision.txt"), "w") as f:
        f.write(f"DECISION: {decision}\n")
        f.write(f"Significant edges: {sig_edges}\n")

if __name__ == "__main__":
    main()
