import os
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import time
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kan import BrainKAN
import importlib
p07 = importlib.import_module("experiments.07_primary_statistical_analysis")
extract_fold_dfl = p07.extract_fold_dfl
permute_labels = p07.permute_labels
train_model = p07.train_model
from utils_data_loading import load_real_hcp_data

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp1_high_precision")
os.makedirs(ARTIFACTS, exist_ok=True)

SEED = 42
N_SPLITS = 5
N_PROBE = 100
N_PERMUTATIONS_GLOBAL = 2000

def run_chunk(chunk_id, chunk_size):
    dataset_pairs, base_edge_index, num_nodes = load_real_hcp_data()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    start_perm = chunk_id * chunk_size
    end_perm = start_perm + chunk_size
    
    print(f"=== Exp 1: High Precision Permutation ===")
    print(f"Executing Chunk {chunk_id+1} (Permutations {start_perm} to {end_perm-1})")
    
    # Generate global flip vectors for ALL permutations to maintain exact alignment
    global_flip_vectors = []
    for b in range(N_PERMUTATIONS_GLOBAL):
        np.random.seed(SEED + b * 100)
        global_flip_vectors.append(np.random.randint(0, 2, size=len(dataset_pairs)))
    
    null_T_global = []
    
    for b in range(start_perm, end_perm):
        fold_null_ts = []
        
        for fold_i, (train_idx, _) in enumerate(kf.split(dataset_pairs)):
            train_pairs = [dataset_pairs[i] for i in train_idx]
            
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
            
            np.random.seed(SEED + fold_i + b * 100)
            permuted_list = permute_labels(train_pairs, train_idx, global_flip_vectors[b])
            train_loader_null = DataLoader(permuted_list, batch_size=32, shuffle=True)
            
            model_null = train_model(train_loader_null, base_edge_index, num_nodes, t_range, SEED+fold_i)
            dfl_null, _ = extract_fold_dfl(model_null, node_t_values, base_edge_index)
            fold_null_ts.append(np.mean(dfl_null))
        
        global_null = np.mean(fold_null_ts)
        print(f"  Permutation {b}: T_null = {global_null:.4f}")
        null_T_global.append({'PermutationID': b, 'T_null': global_null})
        
    df = pd.DataFrame(null_T_global)
    out_file = os.path.join(ARTIFACTS, f"null_chunk_{chunk_id}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved chunk to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_id', type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=50)
    args = parser.parse_args()
    
    run_chunk(args.chunk_id, args.chunk_size)
