"""
Experiment 3: Curvature Regularization & Incremental Predictive Validation (Parallel Runner).

Evaluates:
1. Second-derivative roughness penalty \\lambda \\int |f''(t)|^2 dt added to classification loss.
2. Incremental predictive ablation sweep across \\lambda in {0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0}.
3. Observes whether forcibly straightening model internal curves degrades test accuracy,
   identifying whether learned nonlinearities are functionally indispensable or epiphenomenal.
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
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.kan import BrainKAN
from protocol.reference_domains import compute_common_support_domain
from utils_data_loading import load_real_hcp_data
import importlib
p07 = importlib.import_module("experiments.07_primary_statistical_analysis")
extract_fold_dfl = p07.extract_fold_dfl

ARTIFACTS = PROJECT_ROOT / "results" / "curvature_ablation"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5
N_PROBE = 100
EPOCHS = 10
LAMBDA_GRID = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def train_regularized_model(
    train_loader: DataLoader,
    base_edge_index: torch.Tensor,
    num_nodes: int,
    t_range: tuple,
    reg_lambda: float,
    seed: int,
    epochs: int = EPOCHS,
) -> BrainKAN:
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
            loss_ce = F.cross_entropy(out, batch.y)

            if reg_lambda > 0.0:
                curv_penalty = model.compute_curvature_penalty(num_points=40, grid_range=t_range, layer="conv1")
                total_loss = loss_ce + reg_lambda * curv_penalty
            else:
                total_loss = loss_ce

            total_loss.backward()
            optimizer.step()

    return model


def evaluate_model_accuracy(model: BrainKAN, test_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            preds = out.argmax(dim=-1)
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)
    return float(correct / total) if total > 0 else 0.0


def run_single_lambda(lam: float):
    torch.set_num_threads(2)
    print(f"Starting Lambda = {lam} on PID {os.getpid()}...", flush=True)

    dataset_pairs, base_edge_index, num_nodes = load_real_hcp_data()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_records = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(dataset_pairs)):
        train_pairs = [dataset_pairs[i] for i in train_idx]
        test_pairs = [dataset_pairs[i] for i in test_idx]

        dom_common, (g_min, g_max) = compute_common_support_domain(train_pairs, num_nodes, n_probe=N_PROBE)
        t_range = (g_min, g_max)

        train_data = [d for p in train_pairs for d in p]
        test_data = [d for p in test_pairs for d in p]

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        model = train_regularized_model(
            train_loader, base_edge_index, num_nodes, t_range, reg_lambda=lam, seed=SEED + fold_i
        )

        acc = evaluate_model_accuracy(model, test_loader)
        dfl_scores, _ = extract_fold_dfl(model, dom_common, base_edge_index)
        mean_dfl = float(np.mean(dfl_scores))

        with torch.no_grad():
            curv_val = model.compute_curvature_penalty(num_points=40, grid_range=t_range, layer="conv1").item()

        fold_records.append({
            "Lambda": lam,
            "Fold": fold_i + 1,
            "Test_Accuracy": acc,
            "Mean_DFL": mean_dfl,
            "Curvature_Energy": curv_val,
        })
        print(f"[Lambda={lam}] Fold {fold_i + 1}: Acc = {acc * 100:.1f}%, DFL = {mean_dfl:.4f}, Curv = {curv_val:.4f}", flush=True)

    df = pd.DataFrame(fold_records)
    lam_tag = str(lam).replace(".", "_")
    out_file = ARTIFACTS / f"lambda_{lam_tag}.csv"
    df.to_csv(out_file, index=False)
    print(f"[Lambda={lam}] Completed! Saved to {out_file}", flush=True)


def run_parallel_ablation():
    print("=" * 80, flush=True)
    print("=== TASK 3: CURVATURE REGULARIZATION & INCREMENTAL PREDICTIVE ABLATION (PARALLEL) ===", flush=True)
    print("=" * 80, flush=True)
    start_time = time.time()

    processes = []
    for lam in LAMBDA_GRID:
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--lam",
            str(lam),
        ]
        p = subprocess.Popen(cmd)
        processes.append((lam, p))

    print(f"Spawned {len(processes)} lambda workers across CPU cores.", flush=True)

    for lam, p in processes:
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"Lambda {lam} subprocess failed with exit code {p.returncode}")

    # Aggregate results
    all_dfs = []
    for lam in LAMBDA_GRID:
        lam_tag = str(lam).replace(".", "_")
        out_file = ARTIFACTS / f"lambda_{lam_tag}.csv"
        all_dfs.append(pd.read_csv(out_file))

    df = pd.concat(all_dfs, ignore_index=True)
    out_csv = ARTIFACTS / "curvature_ablation_folds.csv"
    df.to_csv(out_csv, index=False)

    baseline_accs = df[df["Lambda"] == 0.0]["Test_Accuracy"].values
    summary = []

    for lam in LAMBDA_GRID:
        sub = df[df["Lambda"] == lam]
        accs = sub["Test_Accuracy"].values
        dfls = sub["Mean_DFL"].values
        curvs = sub["Curvature_Energy"].values

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs, ddof=1))
        mean_dfl = float(np.mean(dfls))
        std_dfl = float(np.std(dfls, ddof=1))
        mean_curv = float(np.mean(curvs))

        delta_acc = mean_acc - float(np.mean(baseline_accs))
        t_stat, p_val = np.nan, np.nan
        if lam > 0.0:
            diff = accs - baseline_accs
            if np.std(diff, ddof=1) > 1e-8:
                t_res = stats.ttest_rel(accs, baseline_accs)
                t_stat = float(t_res.statistic)
                p_val = float(t_res.pvalue)

        # Decision rule:
        if lam == 0.0:
            verdict = "Baseline (Unregularized)"
        elif not np.isnan(p_val) and p_val < 0.05 and delta_acc < -0.02:
            verdict = "Nonlinearity Indispensable (Significant Acc Drop)"
        else:
            verdict = "Accuracy Preserved (Nonlinearity Non-Essential)"

        summary.append({
            "Lambda": lam,
            "Mean_Accuracy": mean_acc,
            "Std_Accuracy": std_acc,
            "Mean_DFL": mean_dfl,
            "Std_DFL": std_dfl,
            "Mean_Curvature_Energy": mean_curv,
            "Delta_Acc_vs_Baseline": delta_acc,
            "p_value_vs_Baseline": p_val,
            "Ablation_Verdict": verdict,
        })

    sum_df = pd.DataFrame(summary)
    sum_csv = ARTIFACTS / "curvature_ablation_summary.csv"
    sum_df.to_csv(sum_csv, index=False)

    print("\n" + "=" * 80, flush=True)
    print("=== INCREMENTAL PREDICTIVE VALIDATION SUMMARY ===", flush=True)
    print("=" * 80, flush=True)
    print(sum_df.to_string(index=False), flush=True)
    print("=" * 80, flush=True)
    print(f"Parallel execution completed in {time.time() - start_time:.2f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, default=-1.0, help="Specific lambda value to run, or -1 for parallel all.")
    args = parser.parse_args()

    if args.lam >= 0.0:
        run_single_lambda(args.lam)
    else:
        run_parallel_ablation()
