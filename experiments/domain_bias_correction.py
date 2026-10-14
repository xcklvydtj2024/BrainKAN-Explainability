"""
Experiment 1: Reference Domain Evaluation Protocol Comparison (Parallel Multi-Fold Runner).

Resolves domain selection bias by auditing four reference domain regimes:
1. Legacy Fixed Support (label-bound): Common support derived on real labels and held fixed.
2. Label-Blind Domain (pooled empirical quantiles): Evaluated on label-agnostic pooled domain.
3. Independent Sample Support: Evaluated on common support determined by an independent split of subjects.
4. Full-Pipeline Re-enactment: Common support re-computed dynamically on permuted labels under every permutation.
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
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.kan import BrainKAN
from protocol.reference_domains import (
    compute_common_support_domain,
    compute_label_blind_domain,
    compute_independent_split_domain,
)
from utils_data_loading import load_real_hcp_data
import importlib
p07 = importlib.import_module("experiments.07_primary_statistical_analysis")
extract_fold_dfl = p07.extract_fold_dfl
permute_labels = p07.permute_labels

ARTIFACTS = PROJECT_ROOT / "results" / "domain_bias_correction"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5
N_PROBE = 100
N_PERMUTATIONS = 10
EPOCHS = 10


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


def run_single_fold(fold_k: int):
    torch.set_num_threads(2)
    print(f"Starting Fold {fold_k + 1}/{N_SPLITS} on PID {os.getpid()}...", flush=True)

    dataset_pairs, base_edge_index, num_nodes = load_real_hcp_data()

    # Pre-generate global paired flip vectors for permutations
    global_flip_vectors = []
    for b in range(N_PERMUTATIONS):
        np.random.seed(SEED + 1000 + b)
        global_flip_vectors.append(np.random.randint(0, 2, size=len(dataset_pairs)))

    # Independent split domain: first 20 pairs reserved
    ref_split_pairs = dataset_pairs[:20]
    frozen_indep_dom, _ = compute_independent_split_domain(ref_split_pairs, num_nodes, n_probe=N_PROBE)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(kf.split(dataset_pairs))
    train_idx, _ = splits[fold_k]

    train_pairs = [dataset_pairs[i] for i in train_idx]

    # Reference domains
    dom_label_blind, (g_min_blind, g_max_blind) = compute_label_blind_domain(
        train_pairs, num_nodes, n_probe=N_PROBE
    )
    dom_legacy, _ = compute_common_support_domain(train_pairs, num_nodes, n_probe=N_PROBE)
    t_range = (g_min_blind, g_max_blind)

    train_data_real = [d for p in train_pairs for d in p]
    train_loader_real = DataLoader(train_data_real, batch_size=32, shuffle=True)

    print(f"[Fold {fold_k + 1}] Training real model...", flush=True)
    model_real = train_fast_model(train_loader_real, base_edge_index, num_nodes, t_range, SEED + fold_k)

    dfl_real_legacy, _ = extract_fold_dfl(model_real, dom_legacy, base_edge_index)
    dfl_real_blind, _ = extract_fold_dfl(model_real, dom_label_blind, base_edge_index)
    dfl_real_indep, _ = extract_fold_dfl(model_real, frozen_indep_dom, base_edge_index)
    dfl_real_reenact = dfl_real_legacy

    T_real_legacy = float(np.mean(dfl_real_legacy))
    T_real_blind = float(np.mean(dfl_real_blind))
    T_real_indep = float(np.mean(dfl_real_indep))
    T_real_reenact = float(np.mean(dfl_real_reenact))

    null_Ts = {
        "legacy_fixed": [],
        "label_blind": [],
        "independent_split": [],
        "full_reenactment": [],
    }

    print(f"[Fold {fold_k + 1}] Evaluating {N_PERMUTATIONS} paired nulls...", flush=True)
    for b in range(N_PERMUTATIONS):
        permuted_list = permute_labels(train_pairs, train_idx, global_flip_vectors[b])
        train_loader_null = DataLoader(permuted_list, batch_size=32, shuffle=True)

        model_null = train_fast_model(train_loader_null, base_edge_index, num_nodes, t_range, SEED + fold_k)

        dfl_null_leg, _ = extract_fold_dfl(model_null, dom_legacy, base_edge_index)
        null_Ts["legacy_fixed"].append(float(np.mean(dfl_null_leg)))

        dfl_null_bld, _ = extract_fold_dfl(model_null, dom_label_blind, base_edge_index)
        null_Ts["label_blind"].append(float(np.mean(dfl_null_bld)))

        dfl_null_ind, _ = extract_fold_dfl(model_null, frozen_indep_dom, base_edge_index)
        null_Ts["independent_split"].append(float(np.mean(dfl_null_ind)))

        permuted_labels_b = []
        for pair, s_idx in zip(train_pairs, train_idx):
            if global_flip_vectors[b][s_idx] == 1:
                permuted_labels_b.append((int(pair[1].y.item()), int(pair[0].y.item())))
            else:
                permuted_labels_b.append((int(pair[0].y.item()), int(pair[1].y.item())))
        dom_permuted, _ = compute_common_support_domain(
            train_pairs, num_nodes, n_probe=N_PROBE, override_labels=permuted_labels_b
        )
        dfl_null_ree, _ = extract_fold_dfl(model_null, dom_permuted, base_edge_index)
        null_Ts["full_reenactment"].append(float(np.mean(dfl_null_ree)))

    records = [
        {
            "Fold": fold_k + 1,
            "Protocol": "Legacy_Fixed_Support",
            "T_real": T_real_legacy,
            "T_null_mean": float(np.mean(null_Ts["legacy_fixed"])),
            "Delta": T_real_legacy - float(np.mean(null_Ts["legacy_fixed"])),
        },
        {
            "Fold": fold_k + 1,
            "Protocol": "Label_Blind_Domain",
            "T_real": T_real_blind,
            "T_null_mean": float(np.mean(null_Ts["label_blind"])),
            "Delta": T_real_blind - float(np.mean(null_Ts["label_blind"])),
        },
        {
            "Fold": fold_k + 1,
            "Protocol": "Independent_Split_Domain",
            "T_real": T_real_indep,
            "T_null_mean": float(np.mean(null_Ts["independent_split"])),
            "Delta": T_real_indep - float(np.mean(null_Ts["independent_split"])),
        },
        {
            "Fold": fold_k + 1,
            "Protocol": "Full_Pipeline_Reenactment",
            "T_real": T_real_reenact,
            "T_null_mean": float(np.mean(null_Ts["full_reenactment"])),
            "Delta": T_real_reenact - float(np.mean(null_Ts["full_reenactment"])),
        },
    ]

    fold_df = pd.DataFrame(records)
    fold_file = ARTIFACTS / f"fold_{fold_k}.csv"
    fold_df.to_csv(fold_file, index=False)
    print(f"[Fold {fold_k + 1}] Finished! Saved to {fold_file}", flush=True)


def run_parallel_benchmark():
    print("=" * 80, flush=True)
    print("=== TASK 1: UNBIASED REFERENCE DOMAIN PROTOCOLS (5 FOLDS PARALLEL) ===", flush=True)
    print("=" * 80, flush=True)
    start_time = time.time()

    processes = []
    for k in range(N_SPLITS):
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--fold",
            str(k),
        ]
        p = subprocess.Popen(cmd)
        processes.append((k, p))

    print(f"Spawned {len(processes)} fold subprocesses in parallel across CPU cores.", flush=True)

    for k, p in processes:
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"Fold {k + 1} subprocess failed with exit code {p.returncode}")

    # Aggregate fold files
    all_dfs = []
    for k in range(N_SPLITS):
        fold_file = ARTIFACTS / f"fold_{k}.csv"
        all_dfs.append(pd.read_csv(fold_file))
    final_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = ARTIFACTS / "domain_protocol_comparison.csv"
    final_df.to_csv(out_csv, index=False)

    summary = []
    for proto, group in final_df.groupby("Protocol"):
        from scipy import stats
        t_res = stats.ttest_1samp(group["Delta"], 0.0)
        summary.append({
            "Protocol": proto,
            "Mean_T_real": float(group["T_real"].mean()),
            "Mean_T_null": float(group["T_null_mean"].mean()),
            "Mean_Delta": float(group["Delta"].mean()),
            "Std_Delta": float(group["Delta"].std(ddof=1)),
            "Positive_Folds": int((group["Delta"] > 0).sum()),
            "Total_Folds": len(group),
            "t_statistic": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
        })
    sum_df = pd.DataFrame(summary)
    sum_csv = ARTIFACTS / "domain_protocol_summary.csv"
    sum_df.to_csv(sum_csv, index=False)

    print("\n" + "=" * 80, flush=True)
    print("=== TASK 1: DOMAIN EVALUATION PROTOCOL BENCHMARK SUMMARY ===", flush=True)
    print("=" * 80, flush=True)
    print(sum_df.to_string(index=False), flush=True)
    print("=" * 80, flush=True)
    print(f"Parallel execution completed in {time.time() - start_time:.2f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=-1, help="Specific fold index to run (0-4), or -1 for parallel all.")
    args = parser.parse_args()

    if args.fold >= 0:
        run_single_fold(args.fold)
    else:
        run_parallel_benchmark()
