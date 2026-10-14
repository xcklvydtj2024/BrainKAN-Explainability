r"""
Experiment 1: Reference Domain Evaluation Protocol Comparison (5 Folds Parallel Runner).

Audits and resolves reference domain selection bias under four protocols:
1. Legacy Fixed Support (label-bound):
   Common support derived on real labels and held fixed for both real and null models.
   Vulnerable to condition label leakage into the evaluation grid.
2. Label-Blind Domain (Scheme A - Primary Protocol):
   Reference domain derived exclusively on each fold's training cohort, completely ignoring
   condition labels (pooled empirical quantiles). Both real and null models are evaluated
   on this identical, label-agnostic reference domain.
3. Independent Reference Set (Scheme B):
   20 subject pairs are permanently held out exclusively for reference domain estimation,
   never participating in any model training or testing. The remaining 80 subject pairs
   undergo 5-fold cross-validation. The intersection between reference subjects and CV subjects
   is strictly empty: ref_subjects \cap cv_subjects = \emptyset.
4. Full-Pipeline Re-enactment:
   Common support is dynamically re-derived from scratch under every label permutation,
   treating domain estimation as an endogenous step of the inference pipeline.

Statistical Significance:
   Replaces the flawed, pseudoreplicated 5-fold Student's t-test (which produced an invalid p=0.000085)
   with the proper full-pipeline permutation test:
       T_real = \frac{1}{K} \sum_{k=1}^K T_{k, real}
       T_b^{null} = \frac{1}{K} \sum_{k=1}^K T_{k, b}^{null}
       p = \frac{1 + \sum_{b=1}^B 1(T_b^{null} >= T_real)}{B + 1}
"""

import os
import sys
from pathlib import Path
import time
import argparse
import json
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
N_INDEP_REF_PAIRS = 20


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


def run_single_fold(fold_k: int, n_permutations: int = N_PERMUTATIONS):
    torch.set_num_threads(2)
    print(f"Starting Fold {fold_k + 1}/{N_SPLITS} on PID {os.getpid()}...", flush=True)

    dataset_pairs, base_edge_index, num_nodes = load_real_hcp_data()
    n_total_pairs = len(dataset_pairs)

    # -------------------------------------------------------------------------
    # Scheme B Partition: Strictly disjoint Reference and CV sets
    # -------------------------------------------------------------------------
    # First 20 pairs reserved exclusively for reference domain estimation
    ref_split_pairs = dataset_pairs[:N_INDEP_REF_PAIRS]
    # Remaining 80 pairs used exclusively for cross-validation
    cv_dataset_pairs = dataset_pairs[N_INDEP_REF_PAIRS:]
    n_cv_pairs = len(cv_dataset_pairs)

    # Verify disjointness
    ref_indices = set(range(N_INDEP_REF_PAIRS))
    cv_indices = set(range(N_INDEP_REF_PAIRS, n_total_pairs))
    assert len(ref_indices.intersection(cv_indices)) == 0, "Reference and CV subject sets must be disjoint!"

    # Derive Scheme B frozen independent reference domain once from held-out cohort
    frozen_indep_dom, _ = compute_independent_split_domain(
        ref_split_pairs, num_nodes, n_probe=N_PROBE
    )

    # Pre-generate global flip vectors for permutations to preserve subject pairing
    global_flip_vectors = []
    for b in range(n_permutations):
        np.random.seed(SEED + 1000 + b)
        global_flip_vectors.append(np.random.randint(0, 2, size=n_total_pairs))

    # Perform K-Fold CV on the 80 CV pairs
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(kf.split(cv_dataset_pairs))
    train_sub_idx, test_sub_idx = splits[fold_k]

    # Convert local cv indices to original global indices for permutation vector lookup
    train_global_idx = [i + N_INDEP_REF_PAIRS for i in train_sub_idx]
    train_pairs = [cv_dataset_pairs[i] for i in train_sub_idx]

    # -------------------------------------------------------------------------
    # Reference Domains for Fold k
    # -------------------------------------------------------------------------
    # Scheme A (Primary): Label-blind domain derived strictly on this fold's training cohort
    dom_label_blind, (g_min_blind, g_max_blind) = compute_label_blind_domain(
        train_pairs, num_nodes, n_probe=N_PROBE
    )
    # Legacy fixed support derived on real labels
    dom_legacy, _ = compute_common_support_domain(train_pairs, num_nodes, n_probe=N_PROBE)
    t_range = (g_min_blind, g_max_blind)

    train_data_real = [d for p in train_pairs for d in p]
    train_loader_real = DataLoader(train_data_real, batch_size=32, shuffle=True)

    print(f"[Fold {fold_k + 1}] Training real model (N_train = {len(train_pairs)} pairs)...", flush=True)
    model_real = train_fast_model(train_loader_real, base_edge_index, num_nodes, t_range, SEED + fold_k)

    # Evaluate real model under each protocol
    dfl_real_legacy, _ = extract_fold_dfl(model_real, dom_legacy, base_edge_index)
    dfl_real_blind, _ = extract_fold_dfl(model_real, dom_label_blind, base_edge_index)
    dfl_real_indep, _ = extract_fold_dfl(model_real, frozen_indep_dom, base_edge_index)
    dfl_real_reenact = dfl_real_legacy

    T_real_legacy = float(np.mean(dfl_real_legacy))
    T_real_blind = float(np.mean(dfl_real_blind))
    T_real_indep = float(np.mean(dfl_real_indep))
    T_real_reenact = float(np.mean(dfl_real_reenact))

    null_Ts = {
        "Legacy_Fixed_Support": [],
        "Label_Blind_Domain": [],
        "Independent_Split_Domain": [],
        "Full_Pipeline_Reenactment": [],
    }

    print(f"[Fold {fold_k + 1}] Evaluating {n_permutations} paired nulls...", flush=True)
    for b in range(n_permutations):
        # Permute labels respecting subject pairing
        permuted_list = permute_labels(train_pairs, train_global_idx, global_flip_vectors[b])
        train_loader_null = DataLoader(permuted_list, batch_size=32, shuffle=True)

        model_null = train_fast_model(train_loader_null, base_edge_index, num_nodes, t_range, SEED + fold_k)

        # 1. Legacy fixed support
        dfl_null_leg, _ = extract_fold_dfl(model_null, dom_legacy, base_edge_index)
        null_Ts["Legacy_Fixed_Support"].append(float(np.mean(dfl_null_leg)))

        # 2. Scheme A: Label-blind domain (identical grid evaluated for null model)
        dfl_null_bld, _ = extract_fold_dfl(model_null, dom_label_blind, base_edge_index)
        null_Ts["Label_Blind_Domain"].append(float(np.mean(dfl_null_bld)))

        # 3. Scheme B: Independent split domain (frozen on held-out reference cohort)
        dfl_null_ind, _ = extract_fold_dfl(model_null, frozen_indep_dom, base_edge_index)
        null_Ts["Independent_Split_Domain"].append(float(np.mean(dfl_null_ind)))

        # 4. Full-pipeline re-enactment: re-derive common support under permuted labels
        permuted_labels_b = []
        for pair, s_idx in zip(train_pairs, train_global_idx):
            if global_flip_vectors[b][s_idx] == 1:
                permuted_labels_b.append((int(pair[1].y.item()), int(pair[0].y.item())))
            else:
                permuted_labels_b.append((int(pair[0].y.item()), int(pair[1].y.item())))
        dom_permuted, _ = compute_common_support_domain(
            train_pairs, num_nodes, n_probe=N_PROBE, override_labels=permuted_labels_b
        )
        dfl_null_ree, _ = extract_fold_dfl(model_null, dom_permuted, base_edge_index)
        null_Ts["Full_Pipeline_Reenactment"].append(float(np.mean(dfl_null_ree)))

    records = [
        {
            "Fold": fold_k + 1,
            "Protocol": "Legacy_Fixed_Support",
            "T_real": T_real_legacy,
            "T_null_mean": float(np.mean(null_Ts["Legacy_Fixed_Support"])),
            "Delta": T_real_legacy - float(np.mean(null_Ts["Legacy_Fixed_Support"])),
            "null_Ts_json": json.dumps(null_Ts["Legacy_Fixed_Support"]),
        },
        {
            "Fold": fold_k + 1,
            "Protocol": "Label_Blind_Domain",
            "T_real": T_real_blind,
            "T_null_mean": float(np.mean(null_Ts["Label_Blind_Domain"])),
            "Delta": T_real_blind - float(np.mean(null_Ts["Label_Blind_Domain"])),
            "null_Ts_json": json.dumps(null_Ts["Label_Blind_Domain"]),
        },
        {
            "Fold": fold_k + 1,
            "Protocol": "Independent_Split_Domain",
            "T_real": T_real_indep,
            "T_null_mean": float(np.mean(null_Ts["Independent_Split_Domain"])),
            "Delta": T_real_indep - float(np.mean(null_Ts["Independent_Split_Domain"])),
            "null_Ts_json": json.dumps(null_Ts["Independent_Split_Domain"]),
        },
        {
            "Fold": fold_k + 1,
            "Protocol": "Full_Pipeline_Reenactment",
            "T_real": T_real_reenact,
            "T_null_mean": float(np.mean(null_Ts["Full_Pipeline_Reenactment"])),
            "Delta": T_real_reenact - float(np.mean(null_Ts["Full_Pipeline_Reenactment"])),
            "null_Ts_json": json.dumps(null_Ts["Full_Pipeline_Reenactment"]),
        },
    ]

    fold_df = pd.DataFrame(records)
    fold_file = ARTIFACTS / f"fold_{fold_k}.csv"
    fold_df.to_csv(fold_file, index=False)
    print(f"[Fold {fold_k + 1}] Finished! Saved to {fold_file}", flush=True)


def aggregate_and_summarize_protocols():
    """Aggregates all fold files and executes proper full-pipeline permutation testing."""
    all_dfs = []
    for k in range(N_SPLITS):
        fold_file = ARTIFACTS / f"fold_{k}.csv"
        if not fold_file.exists():
            raise FileNotFoundError(f"Missing fold results file: {fold_file}")
        all_dfs.append(pd.read_csv(fold_file))
    final_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = ARTIFACTS / "domain_protocol_comparison.csv"
    final_df.to_csv(out_csv, index=False)

    summary = []
    for proto, group in final_df.groupby("Protocol"):
        # 1. Fold-averaged observed test statistic
        mean_t_real = float(group["T_real"].mean())
        mean_t_null = float(group["T_null_mean"].mean())
        mean_delta = float(group["Delta"].mean())
        std_delta = float(group["Delta"].std(ddof=1))
        n_pos_folds = int((group["Delta"] > 0).sum())

        # 2. Exact Full-Pipeline Permutation Test across folds:
        # For each permutation b: T_b^{null} = 1/K \sum_{k=1}^K T_{k, b}^{null}
        null_matrices = [json.loads(row["null_Ts_json"]) for _, row in group.iterrows()]
        # null_matrices is [K folds, B permutations]
        null_arr = np.array(null_matrices)  # shape [K, B]
        global_null_Ts = np.mean(null_arr, axis=0)  # shape [B]

        B = len(global_null_Ts)
        perm_pvalue = float((1.0 + np.sum(global_null_Ts >= mean_t_real)) / (B + 1.0))

        summary.append({
            "Protocol": proto,
            "Mean_T_real": mean_t_real,
            "Mean_T_null": mean_t_null,
            "Mean_Delta": mean_delta,
            "Std_Delta": std_delta,
            "Positive_Folds": n_pos_folds,
            "Total_Folds": len(group),
            "Permutation_B": B,
            "Permutation_p_value": perm_pvalue,
            "Statistical_Status": "Significant (p < 0.05)" if perm_pvalue < 0.05 else "Not significant",
        })

    sum_df = pd.DataFrame(summary)
    sum_csv = ARTIFACTS / "domain_protocol_summary.csv"
    sum_df.to_csv(sum_csv, index=False)

    print("\n" + "=" * 80, flush=True)
    print("=== TASK 1: DOMAIN EVALUATION PROTOCOL BENCHMARK SUMMARY ===", flush=True)
    print("=" * 80, flush=True)
    print(sum_df.to_string(index=False), flush=True)
    print("=" * 80, flush=True)
    print("Methodological Note:")
    print("  Previous p-values (e.g. p=0.000085) based on a 1-sample Student's t-test across 5 correlated CV folds")
    print("  have been completely replaced with exact full-pipeline permutation p-values.")
    return sum_df


def run_parallel_benchmark(n_permutations: int = N_PERMUTATIONS):
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
            "--n_perm",
            str(n_permutations),
        ]
        p = subprocess.Popen(cmd)
        processes.append((k, p))

    print(f"Spawned {len(processes)} fold subprocesses in parallel across CPU cores (B={n_permutations} nulls/fold).", flush=True)

    for k, p in processes:
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"Fold {k + 1} subprocess failed with exit code {p.returncode}")

    aggregate_and_summarize_protocols()
    print(f"Parallel execution completed in {time.time() - start_time:.2f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=-1, help="Specific fold index to run (0-4), -1 for all parallel, -2 for summarize only.")
    parser.add_argument("--n_perm", type=int, default=10, help="Number of paired permutations per fold (default: 10)")
    args = parser.parse_args()

    if args.fold >= 0:
        run_single_fold(args.fold, n_permutations=args.n_perm)
    elif args.fold == -2:
        aggregate_and_summarize_protocols()
    else:
        run_parallel_benchmark(n_permutations=args.n_perm)
