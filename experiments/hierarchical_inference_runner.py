r"""
Experiment 4: Hierarchical Statistical Inference Framework.

Refactored to enforce rigorous, defensible statistical inference:
1. Primary Inferential Estimand: Label-blind Training-Fold Reference Domain.
   For fold k in 1..K and initialization seed s in 1..S:
       T(D, s) = \frac{1}{K} \sum_{k=1}^K \frac{1}{|E|} \sum_{e \in E} DFL(f_e^{(k, s)}; D_k)
   The optimization-marginalized observed summary statistic is:
       T_obs = \frac{1}{S} \sum_{s=1}^S T(D, s)

2. Large-Scale Finite-Sample Permutation Test (B >= 499):
   Under paired condition label permutation \pi_b (preserving subject pair exchangeability):
       T_b^{null} = \frac{1}{S} \sum_{s=1}^S T(\pi_b D, s)
   Exact non-parametric empirical p-value:
       p = \frac{1 + \sum_{b=1}^B 1(T_b^{null} >= T_obs)}{B + 1}

3. Integrity of Null Distributions:
   - Evaluates label-blind observed statistics strictly against the label-blind permutation distribution.
   - Eliminates invalid scaling of single-seed pooled nulls to approximate multi-seed common support nulls.
   - Notes that common-support inference requires dynamic re-determination under every permutation.

4. Deletion of Fabricated Bootstrap Claims:
   - Completely removes synthetic normal delta simulations (subject_deltas = rng.normal(...)).
   - Transparently refrains from claiming population-level bootstrap CIs without full cohort resampling.
   - Reports well-defined conditional estimates and initialization stability across independent seeds.
"""

import os
import sys
from pathlib import Path
import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from stats_utils.hierarchical_inference import compute_permutation_pvalue
from utils_data_loading import load_real_hcp_data

ARTIFACTS = PROJECT_ROOT / "results" / "hierarchical_inference"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def run_hierarchical_inference(n_permutations: int = 500):
    print("=" * 80)
    print("=== TASK 4: RIGOROUS STATISTICAL INFERENCE FRAMEWORK ===")
    print("=" * 80)

    # 1. Load multi-seed experimental results (Exp 5: S=10 seeds)
    exp5_file = PROJECT_ROOT / "results" / "exp5_seed_stability" / "seed_stability_final.csv"
    if not exp5_file.exists():
        raise FileNotFoundError(f"Missing required Exp 5 multi-seed file: {exp5_file}")
    df_exp5 = pd.read_csv(exp5_file)
    n_seeds = len(df_exp5)
    print(f"Loaded Exp 5 multi-seed records: {n_seeds} optimization seeds.")

    # 2. Load high-precision 500-permutation empirical null distribution (Exp 1: B=500 permutations)
    exp1_file = PROJECT_ROOT / "results" / "exp1_high_precision" / "aggregated_nulls.csv"
    if not exp1_file.exists():
        raise FileNotFoundError(f"Missing required Exp 1 permutation null file: {exp1_file}")
    df_exp1 = pd.read_csv(exp1_file)
    null_dist_500 = df_exp1["T_null"].values
    B = len(null_dist_500)
    print(f"Loaded Exp 1 large-scale permutation nulls: B = {B} permutations (evaluated on label-blind training-fold domain).")

    # -------------------------------------------------------------------------
    # PART 1: Optimization-Marginalized Permutation Inference on Primary Target
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART 1: Optimization-Marginalized Permutation Test (Label-Blind Domain)")
    print("-" * 80)

    # Primary estimand: Label-blind Training-Fold Domain
    # T(D, s) across S=10 seeds
    t_real_seeds_pool = df_exp5["T_real_pool"].values
    T_obs_pool = float(np.mean(t_real_seeds_pool))
    sd_seeds_pool = float(np.std(t_real_seeds_pool, ddof=1))
    cv_seeds_pool = float(sd_seeds_pool / T_obs_pool * 100.0)

    print(f"Primary Target: Label-Blind Training-Fold Domain D_k")
    print(f"  Observed Marginal Test Statistic T_obs: {T_obs_pool:.5f}")
    print(f"  Initialization Seeds (S={n_seeds}):      Mean={T_obs_pool:.5f}, SD={sd_seeds_pool:.5f} (CV={cv_seeds_pool:.2f}%)")
    print(f"  Seed Range: [{np.min(t_real_seeds_pool):.5f}, {np.max(t_real_seeds_pool):.5f}]")

    # Permutation test on label-blind domain
    perm_res_pool = compute_permutation_pvalue(
        observed_stat=T_obs_pool,
        null_distribution=null_dist_500,
        alternative="greater",
    )

    print(f"\nEmpirical Non-Parametric Permutation Test (B = {B}):")
    print(f"  Observed Test Statistic T_obs: {T_obs_pool:.5f}")
    print(f"  Permutation Null Mean:         {perm_res_pool['null_mean']:.5f} (SD = {perm_res_pool['null_std']:.5f})")
    print(f"  Permutation Null Median:       {perm_res_pool['null_median']:.5f}")
    print(f"  Permutation Null 95th Pct:     {perm_res_pool['null_q95']:.5f}")
    print(f"  Exact Empirical p-value:       {perm_res_pool['empirical_p']:.5f} (count={int(np.sum(null_dist_500 >= T_obs_pool))}/{B})")
    print(f"  GPD Tail-aware p-value:        {perm_res_pool['gpd_p']:.5f}")
    print(f"  Statistical Significance:      {'Significant at alpha=0.05' if perm_res_pool['empirical_p'] < 0.05 else 'Not significant'}")

    # Contrast with historical flawed pseudoreplicated Student's t-test
    deltas_seed = df_exp5["Delta_pool"].values
    mean_delta_seed = float(np.mean(deltas_seed))
    sd_delta_seed = float(np.std(deltas_seed, ddof=1))
    se_delta_seed = sd_delta_seed / np.sqrt(n_seeds)
    t_stat_legacy = mean_delta_seed / se_delta_seed
    p_val_legacy_t = float(1.0 - stats.t.cdf(t_stat_legacy, df=n_seeds - 1))

    print(f"\nMethodological Contrast: Pseudoreplicated Seed-Level t-Test (Audited & Discarded):")
    print(f"  t-statistic across 10 seeds: {t_stat_legacy:.4f}, p_value: {p_val_legacy_t:.4e}")
    print(f"  Scientific Verdict: Discarded as pseudoreplication. Random initialization seeds")
    print(f"  on a fixed sample do not constitute independent statistical units of population inference.")

    # -------------------------------------------------------------------------
    # PART 2: Algorithmic Initialization Stability & Scope of Inference
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART 2: Algorithmic Initialization Stability (Optimization Landscape)")
    print("-" * 80)

    # Report conditional seed-level metrics transparently
    seed_summary = []
    for idx, row in df_exp5.iterrows():
        seed_summary.append({
            "Seed": int(row["Seed"]),
            "T_real_pool": float(row["T_real_pool"]),
            "T_null_pool_mean": float(row["T_null_pool_mean"]),
            "Delta_pool": float(row["Delta_pool"]),
        })
    df_seeds = pd.DataFrame(seed_summary)
    print(df_seeds.to_string(index=False))

    var_optimization_seed = float(np.var(t_real_seeds_pool, ddof=1))
    print(f"\nOptimization Variance across Seeds Var(T_real; s): {var_optimization_seed:.8f}")
    print(f"Standard Error of Optimization Mean SE_opt:         {sd_seeds_pool / np.sqrt(n_seeds):.5f}")

    # -------------------------------------------------------------------------
    # PART 3: Methodological Integrity Statement regarding Population Bootstrap
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("PART 3: Methodological Integrity Statement: Population Uncertainty & Bootstrap")
    print("-" * 80)
    print("  [METHODOLOGICAL CORRECTION]")
    print("  Previous iterations of this runner synthesized artificial subject deltas via:")
    print("    subject_deltas = rng.normal(loc=mean_delta_seed, scale=0.0025, size=n_subjects)")
    print("  This practice has been completely removed.")
    print("  A mathematically valid population bootstrap requires directly resampling original")
    print("  subject pairs with replacement and executing full cross-validation and model fitting")
    print("  on each bootstrap replicate cohort.")
    print("  Because full re-training across 1000 bootstrap replicates is computationally prohibitive,")
    print("  we refrain from claiming an unsubstantiated population-level bootstrap CI or an artificial")
    print("  variance decomposition. The reported inferences are conditioned on the observed cohort")
    print("  and quantify finite-sample label exchangeability and initialization stability.")

    # Save comprehensive results
    summary_df = pd.DataFrame([{
        "Primary_Target": "Label_Blind_Training_Fold_Domain",
        "n_seeds": n_seeds,
        "n_permutations": B,
        "T_obs_marginal": T_obs_pool,
        "perm_null_mean": perm_res_pool["null_mean"],
        "perm_null_std": perm_res_pool["null_std"],
        "perm_null_q95": perm_res_pool["null_q95"],
        "perm_empirical_pvalue": perm_res_pool["empirical_p"],
        "perm_gpd_pvalue": perm_res_pool["gpd_p"],
        "significant_05": bool(perm_res_pool["empirical_p"] < 0.05),
        "seed_optimization_sd": sd_seeds_pool,
        "seed_optimization_cv_pct": cv_seeds_pool,
        "seed_optimization_var": var_optimization_seed,
        "legacy_pseudoreplicated_t_p": p_val_legacy_t,
        "synthetic_bootstrap_status": "REMOVED_IN_FAVOR_OF_RIGOR",
    }])
    summary_csv = ARTIFACTS / "hierarchical_inference_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved defensible statistical summary to {summary_csv}")
    print("=" * 80)
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_perm", type=int, default=500)
    args = parser.parse_args()
    run_hierarchical_inference(n_permutations=args.n_perm)
