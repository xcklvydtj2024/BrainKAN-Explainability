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

def compute_domain_stats(deltas: pd.Series, prefix: str):
    n_seeds = len(deltas)
    mean_val = float(deltas.mean())
    sd_val = float(deltas.std(ddof=1))
    se_val = sd_val / np.sqrt(n_seeds)
    df = n_seeds - 1
    t_crit = float(stats.t.ppf(0.975, df=df))
    ci_lower = mean_val - t_crit * se_val
    ci_upper = mean_val + t_crit * se_val
    
    t_stat, p_two_sided = stats.ttest_1samp(deltas, 0.0)
    p_greater = 1.0 - stats.t.cdf(t_stat, df=df)
    positives = int(np.sum(deltas > 0))
    
    return {
        f'{prefix}_mean': mean_val,
        f'{prefix}_sd': sd_val,
        f'{prefix}_se': se_val,
        f'{prefix}_ci_lower_95': ci_lower,
        f'{prefix}_ci_upper_95': ci_upper,
        f'{prefix}_t_stat': float(t_stat),
        f'{prefix}_p_two_sided': float(p_two_sided),
        f'{prefix}_p_one_sided': float(p_greater),
        f'{prefix}_positive_count': positives
    }

def aggregate_and_report(final_df: pd.DataFrame):
    EXP5_DIR.mkdir(parents=True, exist_ok=True)
    final_df = final_df.sort_values('Seed').reset_index(drop=True)
    final_df.to_csv(EXP5_DIR / "seed_stability_final.csv", index=False)
    
    n_seeds = len(final_df)
    df = n_seeds - 1
    
    # 1. Pooled Support stats
    pool_stats = compute_domain_stats(final_df['Delta_pool'], 'pool')
    # 2. Common Support stats
    common_stats = compute_domain_stats(final_df['Delta_common'], 'common')
    
    print("\n" + "="*80)
    print("=== DUAL-SUPPORT MULTI-SEED STABILITY AUDIT (EXP 5: POOLED vs. COMMON SUPPORT) ===")
    print("="*80)
    cols_display = ['Seed', 'T_real_pool', 'T_null_pool_mean', 'Delta_pool',
                    'T_real_common', 'T_null_common_mean', 'Delta_common']
    print(final_df[[c for c in cols_display if c in final_df.columns]].to_string(index=False))
    
    print(f"\nSample Size: {n_seeds} Seeds (df = {df})")
    print("-" * 80)
    print("ESTIMAND 1: POOLED 1%-99% QUANTILE SUPPORT (S_pooled = Quantile([X_0, X_1]))")
    print(f"  Mean T_real:      {final_df['T_real_pool'].mean():.5f}")
    print(f"  Mean T_null:      {final_df['T_null_pool_mean'].mean():.5f}")
    print(f"  Mean Delta_pool:  {pool_stats['pool_mean']:+.5f} (SD={pool_stats['pool_sd']:.5f}, SE={pool_stats['pool_se']:.5f})")
    print(f"  95% CI (t-dist):  [{pool_stats['pool_ci_lower_95']:+.5f}, {pool_stats['pool_ci_upper_95']:+.5f}]")
    print(f"  t-statistic:      {pool_stats['pool_t_stat']:.4f} (p_two_sided={pool_stats['pool_p_two_sided']:.4f}, p_one_sided={pool_stats['pool_p_one_sided']:.4f})")
    print(f"  Sign Consistency: {pool_stats['pool_positive_count']}/{n_seeds} seeds positive")
    
    print("-" * 80)
    print("ESTIMAND 2: STRICT COMMON SUPPORT INTERSECTION (S_common = supp(X_0) ∩ supp(X_1))")
    print(f"  Mean T_real:      {final_df['T_real_common'].mean():.5f}")
    print(f"  Mean T_null:      {final_df['T_null_common_mean'].mean():.5f}")
    print(f"  Mean Delta_common:{common_stats['common_mean']:+.5f} (SD={common_stats['common_sd']:.5f}, SE={common_stats['common_se']:.5f})")
    print(f"  95% CI (t-dist):  [{common_stats['common_ci_lower_95']:+.5f}, {common_stats['common_ci_upper_95']:+.5f}]")
    print(f"  t-statistic:      {common_stats['common_t_stat']:.4f} (p_two_sided={common_stats['common_p_two_sided']:.4f}, p_one_sided={common_stats['common_p_one_sided']:.4f})")
    print(f"  Sign Consistency: {common_stats['common_positive_count']}/{n_seeds} seeds positive")
    print("=" * 80)
    
    summary_dict = {
        'n_seeds': n_seeds,
        'mean_t_real_pool': float(final_df['T_real_pool'].mean()),
        'mean_t_null_pool': float(final_df['T_null_pool_mean'].mean()),
        **pool_stats,
        'mean_t_real_common': float(final_df['T_real_common'].mean()),
        'mean_t_null_common': float(final_df['T_null_common_mean'].mean()),
        **common_stats
    }
    pd.DataFrame([summary_dict]).to_csv(EXP5_DIR / "exp5_statistical_summary.csv", index=False)
    print(f"Saved complete dual-support statistical summary to {EXP5_DIR / 'exp5_statistical_summary.csv'}")

def run_exp5_parallel(seeds, max_workers=6):
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
    all_df = [pd.read_csv(f) for f in files]
    final_df = pd.concat(all_df, ignore_index=True).drop_duplicates(subset=['Seed'])
    aggregate_and_report(final_df)

if __name__ == "__main__":
    seeds_to_run = [42, 100, 2023, 888, 777, 1234, 5678, 9999, 1337, 2048]
    
    # Check which seeds need running (must exist AND have 'Delta_common')
    seeds_needed = []
    complete_dfs = []
    for s in seeds_to_run:
        fpath = EXP5_DIR / f"seed_res_{s}.csv"
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                if 'Delta_common' in df.columns and len(df) > 0:
                    complete_dfs.append(df)
                    continue
            except Exception:
                pass
        seeds_needed.append(s)
        
    if not seeds_needed:
        print("All seeds already have complete dual-support results. Aggregating directly...")
        final_df = pd.concat(complete_dfs, ignore_index=True).drop_duplicates(subset=['Seed'])
        aggregate_and_report(final_df)
    else:
        print(f"Seeds needing execution ({len(seeds_needed)}/{len(seeds_to_run)}): {seeds_needed}")
        run_exp5_parallel(seeds_needed, max_workers=6)
