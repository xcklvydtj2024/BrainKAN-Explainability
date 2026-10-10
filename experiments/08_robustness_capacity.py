import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kan import BrainKAN
from experiments.07_primary_statistical_analysis import extract_fold_dfl, permute_labels
from utils_data_loading import load_real_hcp_data

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "robustness")
os.makedirs(ARTIFACTS, exist_ok=True)

SEED = 42
N_SPLITS = 5
N_PROBE = 100
N_PERMUTATIONS = 5 # Small number for robustness check
EPOCHS = 15

DIMS = [4, 8, 16]

def train_model(train_loader, test_loader, base_edge_index, num_nodes, t_range, dim, seed):
    torch.manual_seed(seed)
    model = BrainKAN(
        base_edge_index=base_edge_index,
        num_nodes=num_nodes,
        in_channels=1,
        hidden_dim=dim,
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
            
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    acc = correct / total if total > 0 else 0
    return model, acc

def main():
    print("=== Stage 4.1: Model Capacity Robustness ===")
    dataset_pairs, base_edge_index, num_nodes = load_real_hcp_data()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    results = []
    
    for dim in DIMS:
        print(f"\nEvaluating Hidden Dimension = {dim}")
        real_T = []
        real_accs = []
        null_T = {b: [] for b in range(N_PERMUTATIONS)}
        
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(dataset_pairs)):
            train_pairs = [dataset_pairs[i] for i in train_idx]
            test_pairs = [dataset_pairs[i] for i in test_idx]
            
            train_x_vals = []
            for pair in train_pairs:
                train_x_vals.extend(pair[0].x[:,0].numpy().tolist())
                train_x_vals.extend(pair[1].x[:,0].numpy().tolist())
            
            r_min, r_max = np.percentile(train_x_vals, 1), np.percentile(train_x_vals, 99)
            t_range = (r_min, r_max)
            t_values = torch.linspace(r_min, r_max, N_PROBE)
            
            train_data_real = [d for p in train_pairs for d in p]
            test_data_real = [d for p in test_pairs for d in p]
            train_loader_real = DataLoader(train_data_real, batch_size=32, shuffle=True)
            test_loader_real = DataLoader(test_data_real, batch_size=32, shuffle=False)
            
            model_real, acc_real = train_model(train_loader_real, test_loader_real, base_edge_index, num_nodes, t_range, dim, SEED+fold_i)
            dfl_real, _ = extract_fold_dfl(model_real, t_values)
            real_T.append(np.mean(dfl_real))
            real_accs.append(acc_real)
            
            for b in range(N_PERMUTATIONS):
                np.random.seed(SEED + fold_i + b * 100)
                permuted_list = permute_labels(train_pairs)
                train_loader_null = DataLoader(permuted_list, batch_size=32, shuffle=True)
                
                model_null, _ = train_model(train_loader_null, test_loader_real, base_edge_index, num_nodes, t_range, dim, SEED+fold_i)
                dfl_null, _ = extract_fold_dfl(model_null, t_values)
                null_T[b].append(np.mean(dfl_null))
                
        T_real_global = np.mean(real_T)
        T_null_globals = [np.mean(null_T[b]) for b in range(N_PERMUTATIONS)]
        delta_task = T_real_global - np.mean(T_null_globals)
        mean_acc = np.mean(real_accs)
        
        print(f"  Result dim={dim}: Acc={mean_acc:.3f}, T_real={T_real_global:.4f}, Delta={delta_task:.4f}")
        results.append({
            'HiddenDim': dim,
            'Accuracy': mean_acc,
            'T_real': T_real_global,
            'T_null_mean': np.mean(T_null_globals),
            'Delta_task': delta_task
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(ARTIFACTS, "capacity_robustness.csv"), index=False)
    print(f"\nSaved capacity robustness to {ARTIFACTS}/capacity_robustness.csv")

if __name__ == "__main__":
    main()
