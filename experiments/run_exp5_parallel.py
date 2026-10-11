import os
import subprocess
import sys
import time

def run_exp5_parallel(seeds, max_workers=4):
    processes = []
    
    # We will spawn a process for each seed. 
    # To do this cleanly, we write a temporary runner script that takes the seed as an argument.
    runner_code = """
import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.kan import BrainKAN
import importlib
p07 = importlib.import_module("experiments.07_primary_statistical_analysis")
extract_fold_dfl = p07.extract_fold_dfl
permute_labels = p07.permute_labels
from utils_data_loading import load_real_hcp_data

run_seed = int(sys.argv[1])
ARTIFACTS = os.path.join("D:\\\\BrainKNN\\\\results", "exp5_seed_stability")

N_SPLITS = 5
N_PROBE = 100
N_PERMUTATIONS = 15
EPOCHS = 15
DIM = 16

def train_model(train_loader, base_edge_index, num_nodes, t_range, dim, seed):
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
    return model

dataset_pairs, base_edge_index, num_nodes = load_real_hcp_data()
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

global_flip_vectors = []
for b in range(N_PERMUTATIONS):
    np.random.seed(2026 + b)
    global_flip_vectors.append(np.random.randint(0, 2, size=len(dataset_pairs)))

real_T = []
null_T = {b: [] for b in range(N_PERMUTATIONS)}

for fold_i, (train_idx, test_idx) in enumerate(kf.split(dataset_pairs)):
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
    train_data_real = [d for p in train_pairs for d in p]
    
    torch.manual_seed(run_seed + fold_i)
    train_loader_real = DataLoader(train_data_real, batch_size=32, shuffle=True)
    
    model_real = train_model(train_loader_real, base_edge_index, num_nodes, t_range, DIM, run_seed + fold_i)
    dfl_real, _ = extract_fold_dfl(model_real, node_t_values, base_edge_index)
    real_T.append(np.mean(dfl_real))
    
    for b in range(N_PERMUTATIONS):
        np.random.seed(run_seed + fold_i + b * 100)
        permuted_list = permute_labels(train_pairs, train_idx, global_flip_vectors[b])
        train_loader_null = DataLoader(permuted_list, batch_size=32, shuffle=True)
        
        model_null = train_model(train_loader_null, base_edge_index, num_nodes, t_range, DIM, run_seed + fold_i)
        dfl_null, _ = extract_fold_dfl(model_null, node_t_values, base_edge_index)
        null_T[b].append(np.mean(dfl_null))

T_real_global = np.mean(real_T)
T_null_globals = [np.mean(null_T[b]) for b in range(N_PERMUTATIONS)]
delta_task = T_real_global - np.mean(T_null_globals)

pd.DataFrame([{
    'Seed': run_seed,
    'T_real': T_real_global,
    'T_null_mean': np.mean(T_null_globals),
    'Delta_task': delta_task
}]).to_csv(os.path.join(ARTIFACTS, f"seed_res_{run_seed}.csv"), index=False)
"""
    
    with open("experiments/exp5_runner.py", "w") as f:
        f.write(runner_code)
        
    for seed in seeds:
        while len(processes) >= max_workers:
            for p in processes[:]:
                if p.poll() is not None:
                    processes.remove(p)
            time.sleep(2)
        print(f"Starting seed {seed}...")
        p = subprocess.Popen([sys.executable, "-u", "experiments/exp5_runner.py", str(seed)])
        processes.append(p)
        
    for p in processes:
        p.wait()

    import glob
    import pandas as pd
    import numpy as np
    
    ARTIFACTS = os.path.join("D:\\BrainKNN\\results", "exp5_seed_stability")
    files = glob.glob(os.path.join(ARTIFACTS, "seed_res_*.csv"))
    all_df = []
    for f in files:
        all_df.append(pd.read_csv(f))
    final_df = pd.concat(all_df, ignore_index=True)
    final_df.to_csv(os.path.join(ARTIFACTS, "seed_stability_final.csv"), index=False)
    
    mean_delta = final_df['Delta_task'].mean()
    sd_delta = final_df['Delta_task'].std()
    ci_lower = mean_delta - 1.96 * (sd_delta / np.sqrt(len(seeds)))
    ci_upper = mean_delta + 1.96 * (sd_delta / np.sqrt(len(seeds)))
    
    print("\\n=== Final Seed Stability Inference ===")
    print(final_df)
    print(f"Mean Delta: {mean_delta:.5f}")
    print(f"SD Delta: {sd_delta:.5f}")
    print(f"95% CI: [{ci_lower:.5f}, {ci_upper:.5f}]")
    positives = sum(final_df['Delta_task'] > 0)
    print(f"Sign Consistency: {positives}/{len(seeds)} positive Deltas")

if __name__ == "__main__":
    run_exp5_parallel([42, 100, 2023, 888, 777, 1234, 5678, 9999, 1337, 2048], max_workers=5)
