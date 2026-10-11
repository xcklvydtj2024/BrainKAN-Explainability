import os
import subprocess
import sys
import time
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP5_DIR = PROJECT_ROOT / "results" / "exp5_seed_stability"
RUNNER_SCRIPT = PROJECT_ROOT / "experiments" / "exp5_runner.py"

def run_exp5_parallel(seeds, max_workers=4):
    EXP5_DIR.mkdir(parents=True, exist_ok=True)
    processes = []
    
    for seed in seeds:
        while len(processes) >= max_workers:
            for p in processes[:]:
                if p.poll() is not None:
                    processes.remove(p)
            time.sleep(2)
        print(f"Starting seed {seed}...")
        p = subprocess.Popen([
            sys.executable, "-u", str(RUNNER_SCRIPT),
            "--seed", str(seed),
            "--out_dir", str(EXP5_DIR)
        ])
        processes.append(p)
        
    for p in processes:
        p.wait()

    # Aggregate results
    files = glob.glob(str(EXP5_DIR / "seed_res_*.csv"))
    all_df = []
    for f in files:
        all_df.append(pd.read_csv(f))
    final_df = pd.concat(all_df, ignore_index=True).drop_duplicates(subset=['Seed']).sort_values('Seed').reset_index(drop=True)
    final_df.to_csv(EXP5_DIR / "seed_stability_final.csv", index=False)
    
    # Statistical analysis with Student's t critical values
    n_seeds = len(final_df)
    mean_delta = float(final_df['Delta_task'].mean())
    sd_delta = float(final_df['Delta_task'].std(ddof=1))
    se_delta = sd_delta / np.sqrt(n_seeds)
    
    # Degrees of freedom = n - 1
    df = n_seeds - 1
    t_crit = float(stats.t.ppf(0.975, df=df))
    ci_lower = mean_delta - t_crit * se_delta
    ci_upper = mean_delta + t_crit * se_delta
    
    t_stat, p_two_sided = stats.ttest_1samp(final_df['Delta_task'], 0.0)
    # One-sided test (H1: Delta > 0)
    p_greater = 1.0 - stats.t.cdf(t_stat, df=df)
    
    print("\n=== Final Seed Stability Inference (Student's t Formulation) ===")
    print(final_df.to_string(index=False))
    print(f"\nSample Size (Seeds): {n_seeds}")
    print(f"Mean T_real:         {final_df['T_real'].mean():.5f}")
    print(f"Mean T_null:         {final_df['T_null_mean'].mean():.5f}")
    print(f"Mean Delta:          {mean_delta:+.5f}")
    print(f"SD Delta:            {sd_delta:.5f}")
    print(f"SE Delta:            {se_delta:.5f}")
    print(f"Student's t Critical (df={df}, 95%): {t_crit:.3f}")
    print(f"95% Confidence Interval (Student's t): [{ci_lower:+.5f}, {ci_upper:+.5f}]")
    print(f"t-statistic:         {t_stat:.4f}")
    print(f"p-value (two-sided): {p_two_sided:.4f}")
    print(f"p-value (H1: Delta > 0, one-sided): {p_greater:.4f}")
    positives = int(np.sum(final_df['Delta_task'] > 0))
    print(f"Sign Consistency:    {positives}/{n_seeds} positive Deltas (Seed 42 only)")
    
    summary_dict = {
        'n_seeds': n_seeds,
        'mean_t_real': float(final_df['T_real'].mean()),
        'mean_t_null': float(final_df['T_null_mean'].mean()),
        'mean_delta': mean_delta,
        'sd_delta': sd_delta,
        'se_delta': se_delta,
        't_crit_95': t_crit,
        'ci_lower_t': ci_lower,
        'ci_upper_t': ci_upper,
        't_statistic': float(t_stat),
        'p_two_sided': float(p_two_sided),
        'p_one_sided_greater': float(p_greater),
        'positive_count': positives
    }
    pd.DataFrame([summary_dict]).to_csv(EXP5_DIR / "exp5_statistical_summary.csv", index=False)
    print(f"Saved complete statistical summary to {EXP5_DIR / 'exp5_statistical_summary.csv'}")

if __name__ == "__main__":
    seeds_to_run = [42, 100, 2023, 888, 777, 1234, 5678, 9999, 1337, 2048]
    # Check if files already exist to allow fast summarization if desired
    existing_files = glob.glob(str(EXP5_DIR / "seed_res_*.csv"))
    if len(existing_files) >= len(seeds_to_run):
        print(f"Found {len(existing_files)} existing seed result files. Running statistical aggregation...")
        # Run aggregation directly without re-training
        all_df = [pd.read_csv(f) for f in existing_files]
        final_df = pd.concat(all_df, ignore_index=True).drop_duplicates(subset=['Seed']).sort_values('Seed').reset_index(drop=True)
        final_df.to_csv(EXP5_DIR / "seed_stability_final.csv", index=False)
        
        n_seeds = len(final_df)
        mean_delta = float(final_df['Delta_task'].mean())
        sd_delta = float(final_df['Delta_task'].std(ddof=1))
        se_delta = sd_delta / np.sqrt(n_seeds)
        df = n_seeds - 1
        t_crit = float(stats.t.ppf(0.975, df=df))
        ci_lower = mean_delta - t_crit * se_delta
        ci_upper = mean_delta + t_crit * se_delta
        t_stat, p_two_sided = stats.ttest_1samp(final_df['Delta_task'], 0.0)
        p_greater = 1.0 - stats.t.cdf(t_stat, df=df)
        
        print("\n=== Final Seed Stability Inference (Student's t Formulation) ===")
        print(final_df.to_string(index=False))
        print(f"\nSample Size (Seeds): {n_seeds}")
        print(f"Mean T_real:         {final_df['T_real'].mean():.5f}")
        print(f"Mean T_null:         {final_df['T_null_mean'].mean():.5f}")
        print(f"Mean Delta:          {mean_delta:+.5f}")
        print(f"SD Delta:            {sd_delta:.5f}")
        print(f"SE Delta:            {se_delta:.5f}")
        print(f"Student's t Critical (df={df}, 95%): {t_crit:.3f}")
        print(f"95% Confidence Interval (Student's t): [{ci_lower:+.5f}, {ci_upper:+.5f}]")
        print(f"t-statistic:         {t_stat:.4f}")
        print(f"p-value (two-sided): {p_two_sided:.4f}")
        print(f"p-value (H1: Delta > 0, one-sided): {p_greater:.4f}")
        positives = int(np.sum(final_df['Delta_task'] > 0))
        print(f"Sign Consistency:    {positives}/{n_seeds} positive Deltas (Seed 42 only)")
        
        summary_dict = {
            'n_seeds': n_seeds,
            'mean_t_real': float(final_df['T_real'].mean()),
            'mean_t_null': float(final_df['T_null_mean'].mean()),
            'mean_delta': mean_delta,
            'sd_delta': sd_delta,
            'se_delta': se_delta,
            't_crit_95': t_crit,
            'ci_lower_t': ci_lower,
            'ci_upper_t': ci_upper,
            't_statistic': float(t_stat),
            'p_two_sided': float(p_two_sided),
            'p_one_sided_greater': float(p_greater),
            'positive_count': positives
        }
        pd.DataFrame([summary_dict]).to_csv(EXP5_DIR / "exp5_statistical_summary.csv", index=False)
    else:
        run_exp5_parallel(seeds_to_run, max_workers=5)
