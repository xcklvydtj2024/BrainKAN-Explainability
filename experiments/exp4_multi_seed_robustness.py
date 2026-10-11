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
import importlib
p07 = importlib.import_module("experiments.07_primary_statistical_analysis")
extract_fold_dfl = p07.extract_fold_dfl
permute_labels = p07.permute_labels
from utils_data_loading import load_real_hcp_data

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp4_multi_seed")
os.makedirs(ARTIFACTS, exist_ok=True)

# We use 5 seeds
SEEDS = [42, 100, 2023, 888, 777] # 5 seeds for budget
N_SPLITS = 5
N_PROBE = 100
N_PERMUTATIONS = 10 # 10 permutations per fold for Null baseline calibration per seed
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
    return model, correct, total

def main():
    print("=== Experiment 4: Multi-Seed Capacity Robustness ===")
    dataset_pairs, base_edge_index, num_nodes = load_real_hcp_data()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42) # Keep data folds identical!
    
    # Generate global flip vectors
    global_flip_vectors = []
    for b in range(N_PERMUTATIONS):
        np.random.seed(42 + b * 100) # Ensure consistent flips across dimensions and seeds
        global_flip_vectors.append(np.random.randint(0, 2, size=len(dataset_pairs)))
        
    results = []
    
    for dim in DIMS:
        print(f"\nEvaluating Hidden Dimension = {dim}")
        
        for run_seed in SEEDS:
            print(f"  -> Run Seed: {run_seed}")
            real_T = []
            run_correct = 0
            run_total = 0
            null_T = {b: [] for b in range(N_PERMUTATIONS)}
            
            for fold_i, (train_idx, test_idx) in enumerate(kf.split(dataset_pairs)):
                train_pairs = [dataset_pairs[i] for i in train_idx]
                test_pairs = [dataset_pairs[i] for i in test_idx]
                
                node_x_vals = {n: [] for n in range(num_nodes)}
                for pair in train_pairs:
                    for n in range(num_nodes):
                        node_x_vals[n].append(pair[0].x[n,0].item())
                        node_x_vals[n].append(pair[1].x[n,0].item())
                
                node_t_values = {}
                global_min, global_max = 0, 0
                for n in range(num_nodes):
                    r_min = np.percentile(node_x_vals[n], 1)
                    r_max = np.percentile(node_x_vals[n], 99)
                    node_t_values[n] = torch.linspace(r_min, r_max, N_PROBE)
                    if n == 0:
                        global_min, global_max = r_min, r_max
                    else:
                        global_min = min(global_min, r_min)
                        global_max = max(global_max, r_max)
                
                t_range = (global_min, global_max)
                
                train_data_real = [d for p in train_pairs for d in p]
                test_data_real = [d for p in test_pairs for d in p]
                
                # We randomise batch order using the run_seed
                torch.manual_seed(run_seed + fold_i)
                train_loader_real = DataLoader(train_data_real, batch_size=32, shuffle=True)
                test_loader_real = DataLoader(test_data_real, batch_size=32, shuffle=False)
                
                model_real, fold_correct, fold_total = train_model(train_loader_real, test_loader_real, base_edge_index, num_nodes, t_range, dim, run_seed + fold_i)
                run_correct += fold_correct
                run_total += fold_total
                
                dfl_real, _ = extract_fold_dfl(model_real, node_t_values, base_edge_index)
                real_T.append(np.mean(dfl_real))
                
                for b in range(N_PERMUTATIONS):
                    np.random.seed(run_seed + fold_i + b * 100)
                    permuted_list = permute_labels(train_pairs, train_idx, global_flip_vectors[b])
                    train_loader_null = DataLoader(permuted_list, batch_size=32, shuffle=True)
                    
                    model_null, _, _ = train_model(train_loader_null, test_loader_real, base_edge_index, num_nodes, t_range, dim, run_seed + fold_i)
                    dfl_null, _ = extract_fold_dfl(model_null, node_t_values, base_edge_index)
                    null_T[b].append(np.mean(dfl_null))
                    
            T_real_global = np.mean(real_T)
            T_null_globals = [np.mean(null_T[b]) for b in range(N_PERMUTATIONS)]
            delta_task = T_real_global - np.mean(T_null_globals)
            raw_acc = run_correct / run_total
            
            print(f"     Acc={run_correct}/{run_total} ({raw_acc:.5f}), T_real={T_real_global:.4f}, Delta={delta_task:.4f}")
            results.append({
                'HiddenDim': dim,
                'Seed': run_seed,
                'Correct': run_correct,
                'Total': run_total,
                'Accuracy': raw_acc,
                'T_real': T_real_global,
                'T_null_mean': np.mean(T_null_globals),
                'Delta_task': delta_task
            })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(ARTIFACTS, "multi_seed_capacity.csv"), index=False)
    print(f"\nSaved capacity robustness to {ARTIFACTS}/multi_seed_capacity.csv")

if __name__ == "__main__":
    main()
