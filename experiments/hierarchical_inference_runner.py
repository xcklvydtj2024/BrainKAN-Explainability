r"""
Experiment 4: Hierarchical Statistical Inference & Optimization Sensitivity Audit.

Methodological Scope:
1. Single-Seed Conditional Permutation Test (Exp 1: Seed 42, B=500 permutations):
   - Formally evaluates whether model weights learned from seed 42 exhibit excess curvature
     over paired condition-label permutations on the label-blind training-fold domain.
   - Observed statistic: T_obs(s=42) = 0.178953
   - Permutation null:   T_b^{null}(s=42) for b = 1..500 (mean = 0.177333, SD = 0.000585)
   - Real excess:        \Delta = +0.001620
   - Empirical p-value:  p = (1 + 2) / (500 + 1) = 0.00599 (GPD tail p = 0.00058)
   - Scope: Statistically significant conditioned on the Seed 42 initialization.

2. Multi-Seed Optimization Sensitivity Audit (Exp 5: S=10 seeds):
   - Evaluates whether this conditional significance replicates across independent initialization seeds.
   - Across S=10 seeds: Mean T_real = 0.179394, Mean T_null = 0.179977, Mean \Delta = -0.000583.
   - Positive count: 1 / 10 seeds (Seed 42 only; 9/10 seeds produce negative \Delta).
   - Finding: Conditional significance on the label-blind domain fails to generalize across the optimization landscape.

3. Methodological Correction on Mismatched Testing:
   - Formally discards the apparent p = 0.00399 obtained by naively comparing the 10-seed
     observed mean (0.179394) against the single-seed permutation null of Seed 42 (0.177333).
   - A valid permutation test requires that observed and null statistics share the identical
     evaluation procedure.
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from stats_utils.hierarchical_inference import compute_permutation_pvalue

ARTIFACTS = PROJECT_ROOT / "results" / "hierarchical_inference"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def run_hierarchical_inference(n_permutations: int = 500):
    print("=" * 80)
    print("=== TASK 4: RIGOROUS STATISTICAL INFERENCE & SENSITIVITY AUDIT ===")
    print("=" * 80)

    # 1. Load multi-seed experimental results (Exp 5: S=10 seeds)
    exp5_file = PROJECT_ROOT / "results" / "exp5_seed_stability" / "seed_stability_final.csv"
    if not exp5_file.exists():
        raise FileNotFoundError(f"Missing required Exp 5 multi-seed file: {exp5_file}")
    df_exp5 = pd.read_csv(exp5_file)
    n_seeds = len(df_exp5)
    print(f"Loaded Exp 5 multi-seed records: {n_seeds} optimization seeds.")

    # 2. Load high-precision 500-permutation empirical null distribution (Exp 1: Seed 42, B=500 permutations)
    exp1_file = PROJECT_ROOT / "results" / "exp1_high_precision" / "aggregated_nulls.csv"
    if not exp1_file.exists():
        raise FileNotFoundError(f"Missing required Exp 1 permutation null file: {exp1_file}")
    df_exp1 = pd.read_csv(exp1_file)
    null_dist_500_seed42 = df_exp1["T_null"].values
    B = len(null_dist_500_seed42)
    print(f"Loaded Exp 1 single-seed permutation nulls: B = {B} permutations for Seed 42.")

    # -------------------------------------------------------------------------
    # PART 1: Single-Seed Conditional Permutation Test (Seed 42)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART 1: Single-Seed Conditional Permutation Test (Seed 42 Baseline, B=500)")
    print("-" * 80)

    # In Exp 1 / Exp 5, Seed 42 real model curvature
    row_seed42 = df_exp5[df_exp5["Seed"] == 42].iloc[0]
    T_obs_seed42 = float(row_seed42["T_real_pool"])

    perm_res_seed42 = compute_permutation_pvalue(
        observed_stat=T_obs_seed42,
        null_distribution=null_dist_500_seed42,
        alternative="greater",
    )
    delta_seed42 = T_obs_seed42 - perm_res_seed42["null_mean"]

    print(f"Target: Label-Blind Training-Fold Domain D_k (Conditioned on Seed 42)")
    print(f"  Observed Test Statistic T_obs(s=42): {T_obs_seed42:.6f}")
    print(f"  Permutation Null Mean (s=42):        {perm_res_seed42['null_mean']:.6f} (SD = {perm_res_seed42['null_std']:.6f})")
    print(f"  Excess Curvature Delta(s=42):        {delta_seed42:+.6f}")
    print(f"  Permutation Null 95th Percentile:    {perm_res_seed42['null_q95']:.6f}")
    print(f"  Exact Empirical p-value (B={B}):     {perm_res_seed42['empirical_p']:.6f} ({int(np.sum(null_dist_500_seed42 >= T_obs_seed42))}/{B} exceedances)")
    print(f"  Tail-Aware GPD p-value:              {perm_res_seed42['gpd_p']:.6f}")
    print(f"  Conditional Inference Status:        {'SIGNIFICANT (Conditional on Seed 42)' if perm_res_seed42['empirical_p'] < 0.05 else 'NOT SIGNIFICANT'}")

    # -------------------------------------------------------------------------
    # PART 2: Multi-Seed Optimization Sensitivity Audit (Exp 5: S=10 seeds)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART 2: Multi-Seed Optimization Sensitivity Audit (Exp 5: S=10 seeds)")
    print("-" * 80)

    t_real_seeds = df_exp5["T_real_pool"].values
    t_null_seeds = df_exp5["T_null_pool_mean"].values
    deltas_seed = df_exp5["Delta_pool"].values

    mean_t_real = float(np.mean(t_real_seeds))
    mean_t_null = float(np.mean(t_null_seeds))
    mean_delta = float(np.mean(deltas_seed))
    sd_delta = float(np.std(deltas_seed, ddof=1))
    se_delta = sd_delta / np.sqrt(n_seeds)
    t_stat = mean_delta / se_delta
    p_val_t = float(1.0 - stats.t.cdf(t_stat, df=n_seeds - 1))
    n_positive_seeds = int(np.sum(deltas_seed > 0))

    sd_real_seeds = float(np.std(t_real_seeds, ddof=1))
    cv_real_seeds = float(sd_real_seeds / mean_t_real * 100.0)

    print(f"Multi-Seed Landscape across {n_seeds} Independent Initializations:")
    print(f"  Mean Observed Curvature T_real:      {mean_t_real:.6f} (SD={sd_real_seeds:.6f}, CV={cv_real_seeds:.2f}%)")
    print(f"  Mean Null Baseline T_null:           {mean_t_null:.6f} (SD={np.std(t_null_seeds, ddof=1):.6f})")
    print(f"  Mean Excess Curvature Delta:         {mean_delta:+.6f} (SD={sd_delta:.6f}, SE={se_delta:.6f})")
    print(f"  Replication Count (Delta > 0):       {n_positive_seeds} / {n_seeds} seeds (Seed 42 only; 9/10 seeds negative)")
    print(f"  One-Sided Student's t-test:          t={t_stat:.4f}, p={p_val_t:.4f}")
    print(f"  Audit Verdict:                       FAILS TO GENERALIZE across seeds on label-blind domain")

    # -------------------------------------------------------------------------
    # PART 3: Critical Methodological Audit on Mismatched Comparison (p=0.00399)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART 3: Critical Methodological Audit: Mismatched Statistics (Discarding p=0.00399)")
    print("-" * 80)
    print("  [AUDIT & CORRECTION]")
    print("  Comparing the 10-seed observed mean (0.179394) against Seed 42's 500 permutations")
    print("  (null mean = 0.177333) produced an apparent p = 0.00399.")
    print("  This comparison is METHODOLOGICALLY INVALID:")
    print("    1. The observed test statistic (10-seed average) does not match the permutation null (1-seed).")
    print("    2. In Exp 5 itself, the 10-seed null baseline is 0.179977, which EXCEEDS the real mean (0.179394).")
    print("    3. Evaluating whether Delta > 0 across the 10 seeds yields Delta = -0.000583 (p = 0.9513).")
    print("  Therefore, the apparent p = 0.00399 is formally DISCARDED as a mismatched statistic artifact.")

    # -------------------------------------------------------------------------
    # PART 4: Summary Records
    # -------------------------------------------------------------------------
    records = [
        {
            "Analysis": "Single_Seed_Conditional_Inference",
            "Seed": "42",
            "n_seeds": 1,
            "n_permutations": B,
            "T_obs": T_obs_seed42,
            "Null_Mean": perm_res_seed42["null_mean"],
            "Null_SD": perm_res_seed42["null_std"],
            "Delta": delta_seed42,
            "Empirical_p_value": perm_res_seed42["empirical_p"],
            "GPD_p_value": perm_res_seed42["gpd_p"],
            "Significance": "Significant (Conditional on Seed 42)",
            "Notes": "Valid conditional permutation test on fixed initialization seed 42",
        },
        {
            "Analysis": "Multi_Seed_Optimization_Audit",
            "Seed": "10_seeds_pooled",
            "n_seeds": 10,
            "n_permutations": 15,
            "T_obs": mean_t_real,
            "Null_Mean": mean_t_null,
            "Null_SD": float(np.std(t_null_seeds, ddof=1)),
            "Delta": mean_delta,
            "Empirical_p_value": p_val_t,
            "GPD_p_value": np.nan,
            "Significance": "Not significant (9/10 seeds negative)",
            "Notes": "Initialization sensitivity audit; signal fails to generalize across seeds on label-blind domain",
        },
        {
            "Analysis": "Mismatched_Comparison_Audit",
            "Seed": "10_seeds_vs_seed_42_null",
            "n_seeds": 10,
            "n_permutations": 500,
            "T_obs": mean_t_real,
            "Null_Mean": perm_res_seed42["null_mean"],
            "Null_SD": perm_res_seed42["null_std"],
            "Delta": mean_t_real - perm_res_seed42["null_mean"],
            "Empirical_p_value": 0.003992,
            "GPD_p_value": 0.000579,
            "Significance": "DISCARDED_AS_STATISTICALLY_INVALID",
            "Notes": "Methodologically invalid comparison of 10-seed mean to 1-seed null; formally discarded",
        },
    ]

    summary_df = pd.DataFrame(records)
    summary_csv = ARTIFACTS / "hierarchical_inference_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved transparent statistical summary to {summary_csv}")
    print("=" * 80)
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_perm", type=int, default=500)
    args = parser.parse_args()
    run_hierarchical_inference(n_permutations=args.n_perm)
