import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.linear_model import LinearRegression

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_data_loading import load_real_hcp_data
from models.kan import BrainKAN

DATA_DIR = r"D:\BrainKNN\data"
SEED = 42
T_RANGE = (-8.0, 12.0)
R_NULLS = 100 # We can do 100 because we train 1 model per null!

def train_one_model(dataset_pairs, is_null=False, null_seed=None):
    num_nodes = dataset_pairs[0][0].x.shape[0]
    base_edge_index = dataset_pairs[0][0].edge_index
    
    if is_null:
        np.random.seed(null_seed)
        working_pairs = []
        for (d0, d2) in dataset_pairs:
            d0_y = d0.y.clone()
            d2_y = d2.y.clone()
            if np.random.rand() > 0.5:
                d0.y = d2_y
                d2.y = d0_y
            working_pairs.append((d0, d2))
    else:
        working_pairs = dataset_pairs

    train_data = [d for pair in working_pairs for d in pair]
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    torch.manual_seed(SEED if not is_null else null_seed)
    np.random.seed(SEED if not is_null else null_seed)
    
    model = BrainKAN(base_edge_index, num_nodes, 1, 16, 2, grid_range=T_RANGE)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    
    model.train()
    for epoch in range(15):
        for batch in train_loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
            
    model.eval()
    
    if is_null:
        for (d0, d2) in dataset_pairs:
            d0.y = torch.tensor([0], dtype=torch.long)
            d2.y = torch.tensor([1], dtype=torch.long)
            
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

def _b_spline_basis(x: torch.Tensor, grid: torch.Tensor, order: int) -> torch.Tensor:
    x = x.unsqueeze(-1)
    bases = ((x >= grid[..., :-1]) & (x < grid[..., 1:])).float()
    for k in range(1, order + 1):
        left_num = x - grid[..., : -(k + 1)]
        left_den = grid[..., k:-1] - grid[..., : -(k + 1)]
        right_num = grid[..., k + 1 :] - x
        right_den = grid[..., k + 1 :] - grid[..., 1 : (-k if k > 0 else None)]
        left = left_num / (left_den + 1e-8) * bases[..., :-1]
        right = right_num / (right_den + 1e-8) * bases[..., 1:]
        bases = left + right
    return bases

def compute_G(model, ref_grids, base_edge_index):
    num_edges = base_edge_index.shape[1]
    src_nodes = base_edge_index[0].numpy()
    
    G = np.zeros(num_edges)
    conv = model.conv1
    
    for e in range(num_edges):
        src = src_nodes[e]
        x_ref = ref_grids[src]
        n = len(x_ref)
        probe = torch.tensor(x_ref, dtype=torch.float32).unsqueeze(1) # [n, 1]
        
        with torch.no_grad():
            grid_e = conv.grid[e] # [1, G]
            grid_batch = grid_e.unsqueeze(0).expand(n, -1, -1) # [n, 1, G]
            
            basis = _b_spline_basis(probe, grid_batch, conv.spline_order) # [n, 1, num_bases]
            base_out = torch.einsum("Bi,oi->Bo", F.silu(probe), conv.base_weight[e]) # [n, 16]
            spline_out = torch.einsum("Bik,oik->Bo", basis, conv.spline_weight[e]) # [n, 16]
            y = (base_out + spline_out).numpy() # [n, 16]
            
        r2_channels = []
        X_fit = x_ref.reshape(-1, 1)
        for o in range(16):
            Y_fit = y[:, o]
            var_y = np.var(Y_fit)
            if var_y < 1e-10:
                r2_channels.append(1.0)
            else:
                reg = LinearRegression().fit(X_fit, Y_fit)
                r2_channels.append(reg.score(X_fit, Y_fit))
        
        G[e] = np.mean(1.0 - np.array(r2_channels))
        
    return G

def main():
    print("Loading Dataset...")
    res = load_real_hcp_data()
    dataset_pairs = res[0]
    base_edge_index = dataset_pairs[0][0].edge_index
    num_edges = base_edge_index.shape[1]
    
    print("Computing Reference Grids...")
    ref_grids = compute_reference_grids(dataset_pairs, base_edge_index)
    
    print("\n=== Training Real Model ===")
    real_model = train_one_model(dataset_pairs, is_null=False)
    G_real = compute_G(real_model, ref_grids, base_edge_index)
    print(f"Real G_e overall mean: {np.mean(G_real):.4f}")
    
    print(f"\n=== Training {R_NULLS} Null Models ===")
    G_nulls = np.zeros((R_NULLS, num_edges))
    
    for k in range(R_NULLS):
        null_seed = 2000 + k
        if k % 10 == 0:
            print(f"  Null Model {k+1}/{R_NULLS} (seed={null_seed})...")
        null_model = train_one_model(dataset_pairs, is_null=True, null_seed=null_seed)
        G_nulls[k] = compute_G(null_model, ref_grids, base_edge_index)
        
    print("\n=== Q3B Independent-Null Calibration ===")
    
    # Statistic T is directly the non-linearity metric G_e
    p_values = np.zeros(num_edges)
    for e in range(num_edges):
        count = np.sum(G_nulls[:, e] >= G_real[e])
        p_values[e] = (1.0 + count) / (R_NULLS + 1.0)
        
    from statsmodels.stats.multitest import multipletests
    rejected, p_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    sig_count = np.sum(rejected)
    print(f"Edges with empirical p < 0.05 (FDR corrected): {sig_count} / {num_edges}")
    
    print("\nTop 10 edges by empirical p-value:")
    sorted_e = np.argsort(p_values)
    for e in sorted_e[:10]:
        print(f"  Edge {e:>3}: p_emp = {p_values[e]:.4f}, p_fdr = {p_fdr[e]:.4f}, Real G = {G_real[e]:.4f}, Null G mean = {np.mean(G_nulls[:, e]):.4f}, std = {np.std(G_nulls[:, e]):.4f}")

if __name__ == "__main__":
    main()
