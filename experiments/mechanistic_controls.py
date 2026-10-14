"""
Experiment 2: Mechanistic Control Experiments (Parallel Runner).

Calibrates:
1. Linear Negative Control:
   - Synthetic data containing purely linear mean shifts and Gaussian noise.
   - Evaluates whether BrainKAN's over-parameterization produces raw DFL > 0.
   - Calibrates empirical Type I error rate (False Positive Rate) under H_0 using paired permutation tests.
2. Nonlinear Positive Control:
   - Synthesizes known nonlinear neural coupling mechanisms (e.g. sigmoidal saturation).
   - Tests statistical detection power (1 - beta) across varying coupling strengths.
"""

import os
import sys
from pathlib import Path
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.kan import BrainKAN
from protocol.reference_domains import compute_common_support_domain
import importlib
p07 = importlib.import_module("experiments.07_primary_statistical_analysis")
extract_fold_dfl = p07.extract_fold_dfl
permute_labels = p07.permute_labels

ARTIFACTS = PROJECT_ROOT / "results" / "mechanistic_controls"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_NODES = 48
NUM_PAIRS = 100
N_SPLITS = 3
N_PROBE = 80
N_PERM = 10
EPOCHS = 8


def generate_synthetic_cohort(base_edge_index: torch.Tensor, num_pairs: int = 100, nonlinear_strength: float = 0.0, seed: int = 42):
    rng = np.random.RandomState(seed)
    torch_rng = torch.Generator().manual_seed(seed)
    num_nodes = NUM_NODES
    num_edges = base_edge_index.size(1)

    w_edge_linear = torch.randn(num_edges, generator=torch_rng) * 0.5
    src0 = base_edge_index[0]
    dst0 = base_edge_index[1]

    dataset_pairs = []
    for i in range(num_pairs):
        x0_np = rng.normal(loc=0.0, scale=1.0, size=(num_nodes, 1))
        shift = rng.normal(loc=0.8, scale=0.3, size=(num_nodes, 1))
        x1_np = rng.normal(loc=0.0, scale=1.0, size=(num_nodes, 1)) + shift

        x0 = torch.tensor(x0_np, dtype=torch.float32)
        x1 = torch.tensor(x1_np, dtype=torch.float32)

        msg0_lin = x0[src0, 0] * w_edge_linear
        msg1_lin = x1[src0, 0] * w_edge_linear

        if nonlinear_strength > 0.0:
            msg0 = msg0_lin + nonlinear_strength * torch.tanh(2.5 * x0[src0, 0])
            msg1 = msg1_lin + nonlinear_strength * torch.tanh(2.5 * x1[src0, 0])
        else:
            msg0 = msg0_lin
            msg1 = msg1_lin

        d0 = Data(x=x0, edge_index=base_edge_index, y=torch.tensor(0, dtype=torch.long))
        d1 = Data(x=x1, edge_index=base_edge_index, y=torch.tensor(1, dtype=torch.long))
        dataset_pairs.append((d0, d1))

    return dataset_pairs


def train_fast_model(train_loader, base_edge_index, num_nodes, t_range, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    model = BrainKAN(
        base_edge_index=base_edge_index,
        num_nodes=num_nodes,
        in_channels=1,
        hidden_dim=16,
        num_classes=2,
        grid_range=t_range,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = torch.nn.functional.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()
    return model


def run_single_simulation(exp_name: str, gamma: float, rep: int):
    torch.set_num_threads(2)
    print(f"Starting {exp_name} (Gamma={gamma}, Rep={rep}) on PID {os.getpid()}...", flush=True)

    edges_data = torch.load(os.path.join(PROJECT_ROOT, "data", "GNN_Edges.pt"))
    base_edge_index = edges_data["edge_index_A"]

    sim_seed = 500 + int(gamma * 100) + rep
    cohort = generate_synthetic_cohort(base_edge_index, num_pairs=NUM_PAIRS, nonlinear_strength=gamma, seed=sim_seed)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=sim_seed)

    global_flip_vectors = []
    for b in range(N_PERM):
        np.random.seed(sim_seed + 1000 + b)
        global_flip_vectors.append(np.random.randint(0, 2, size=len(cohort)))

    fold_T_real = []
    fold_T_null = {b: [] for b in range(N_PERM)}

    for fold_i, (train_idx, _) in enumerate(kf.split(cohort)):
        train_pairs = [cohort[i] for i in train_idx]
        dom_common, (g_min, g_max) = compute_common_support_domain(train_pairs, NUM_NODES, n_probe=N_PROBE)
        t_range = (g_min, g_max)

        train_data_real = [d for p in train_pairs for d in p]
        train_loader_real = DataLoader(train_data_real, batch_size=32, shuffle=True)

        model_real = train_fast_model(train_loader_real, base_edge_index, NUM_NODES, t_range, sim_seed + fold_i)
        dfl_real, _ = extract_fold_dfl(model_real, dom_common, base_edge_index)
        fold_T_real.append(float(np.mean(dfl_real)))

        for b in range(N_PERM):
            permuted_list = permute_labels(train_pairs, train_idx, global_flip_vectors[b])
            train_loader_null = DataLoader(permuted_list, batch_size=32, shuffle=True)
            model_null = train_fast_model(train_loader_null, base_edge_index, NUM_NODES, t_range, sim_seed + fold_i)
            dfl_null, _ = extract_fold_dfl(model_null, dom_common, base_edge_index)
            fold_T_null[b].append(float(np.mean(dfl_null)))

    T_real = float(np.mean(fold_T_real))
    null_globals = [float(np.mean(fold_T_null[b])) for b in range(N_PERM)]
    T_null = float(np.mean(null_globals))
    delta = T_real - T_null
    emp_p = float((1.0 + np.sum(np.array(null_globals) >= T_real)) / (N_PERM + 1.0))

    record = {
        "Experiment": exp_name,
        "Nonlinear_Strength_Gamma": gamma,
        "Replicate": rep,
        "T_real": T_real,
        "T_null": T_null,
        "Delta": delta,
        "p_value": emp_p,
        "significant_05": bool(emp_p < 0.05),
    }

    out_file = ARTIFACTS / f"sim_{exp_name}_g{gamma}_r{rep}.csv"
    pd.DataFrame([record]).to_csv(out_file, index=False)
    print(f"[{exp_name} G={gamma} R={rep}] T_real={T_real:.4f}, Delta={delta:+.5f}, p={emp_p:.4f}, Sig={emp_p < 0.05}", flush=True)


def run_parallel_controls(max_workers: int = 6):
    print("=" * 80, flush=True)
    print("=== TASK 2: MECHANISTIC CONTROLS (PARALLEL RUNNER) ===", flush=True)
    print("=" * 80, flush=True)
    start_time = time.time()

    tasks = []
    # 1. Negative control (gamma = 0.0, 10 reps)
    for r in range(1, 11):
        tasks.append(("Linear_Negative_Control", 0.0, r))

    # 2. Positive controls (gamma in [0.2, 0.5, 1.0], 3 reps each)
    for g in [0.2, 0.5, 1.0]:
        for r in range(1, 4):
            tasks.append(("Nonlinear_Positive_Control", g, r))

    processes = []
    task_idx = 0

    while task_idx < len(tasks) or len(processes) > 0:
        # Spawn up to max_workers
        while len(processes) < max_workers and task_idx < len(tasks):
            exp_name, gamma, rep = tasks[task_idx]
            cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--exp",
                exp_name,
                "--gamma",
                str(gamma),
                "--rep",
                str(rep),
            ]
            p = subprocess.Popen(cmd)
            processes.append((exp_name, gamma, rep, p))
            task_idx += 1

        # Check for completed processes
        for item in processes[:]:
            exp_name, gamma, rep, p = item
            if p.poll() is not None:
                if p.returncode != 0:
                    raise RuntimeError(f"Simulation {exp_name} (gamma={gamma}, rep={rep}) failed with code {p.returncode}")
                processes.remove(item)

        time.sleep(1)

    # Aggregate all simulation files
    records = []
    for exp_name, gamma, rep in tasks:
        sim_file = ARTIFACTS / f"sim_{exp_name}_g{gamma}_r{rep}.csv"
        records.append(pd.read_csv(sim_file))

    df = pd.concat(records, ignore_index=True)
    out_csv = ARTIFACTS / "mechanistic_controls_results.csv"
    df.to_csv(out_csv, index=False)

    summary = []
    for (exp, gamma), grp in df.groupby(["Experiment", "Nonlinear_Strength_Gamma"]):
        n_trials = len(grp)
        type1_or_power = float(grp["significant_05"].mean())
        mean_raw_dfl = float(grp["T_real"].mean())
        mean_delta = float(grp["Delta"].mean())
        std_delta = float(grp["Delta"].std(ddof=1)) if n_trials > 1 else 0.0

        summary.append({
            "Experiment": exp,
            "Gamma": gamma,
            "N_Trials": n_trials,
            "Mean_Raw_T_real": mean_raw_dfl,
            "Mean_Delta": mean_delta,
            "Std_Delta": std_delta,
            "Empirical_Rejection_Rate_alpha05": type1_or_power,
            "Metric_Interpretation": "Type I Error Rate (FPR)" if gamma == 0.0 else "Statistical Power (1 - beta)",
        })

    sum_df = pd.DataFrame(summary)
    sum_csv = ARTIFACTS / "mechanistic_controls_summary.csv"
    sum_df.to_csv(sum_csv, index=False)

    print("\n" + "=" * 80, flush=True)
    print("=== MECHANISTIC CONTROLS CALIBRATION SUMMARY ===", flush=True)
    print("=" * 80, flush=True)
    print(sum_df.to_string(index=False), flush=True)
    print("=" * 80, flush=True)
    print(f"Parallel execution completed in {time.time() - start_time:.2f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--rep", type=int, default=None)
    args = parser.parse_args()

    if args.exp is not None and args.gamma is not None and args.rep is not None:
        run_single_simulation(args.exp, args.gamma, args.rep)
    else:
        run_parallel_controls()
