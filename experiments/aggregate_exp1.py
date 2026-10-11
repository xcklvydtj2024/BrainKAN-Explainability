import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

# Dynamic project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP1_DIR = PROJECT_ROOT / "results" / "exp1_high_precision"
SEED42_FILE = PROJECT_ROOT / "results" / "exp5_seed_stability" / "seed_res_42.csv"
CAPACITY_FILE = PROJECT_ROOT / "results" / "exp4_multi_seed" / "multi_seed_capacity.csv"

def aggregate_exp1():
    chunk_files = sorted(glob.glob(str(EXP1_DIR / "null_chunk_*.csv")))
    if not chunk_files:
        raise FileNotFoundError(f"No null chunk files found in {EXP1_DIR}")
        
    all_nulls = []
    for f in chunk_files:
        df = pd.read_csv(f)
        all_nulls.extend(df['T_null'].tolist())
        
    all_nulls = np.array(all_nulls, dtype=np.float64)
    B = len(all_nulls)
    mean_null = float(np.mean(all_nulls))
    
    # Provenance-tracked T_real extraction:
    # Seed 42 is the exact baseline initialization used in Exp 1.
    # We retrieve the exact double-precision value from logged seed evaluations.
    if SEED42_FILE.exists():
        seed_df = pd.read_csv(SEED42_FILE)
        T_real = float(seed_df.loc[seed_df['Seed'] == 42, 'T_real'].values[0])
        source = f"seed_res_42.csv"
    elif CAPACITY_FILE.exists():
        cap_df = pd.read_csv(CAPACITY_FILE)
        T_real = float(cap_df.loc[(cap_df['HiddenDim'] == 16) & (cap_df['Seed'] == 42), 'T_real'].values[0])
        source = f"multi_seed_capacity.csv (dim=16, seed=42)"
    else:
        raise FileNotFoundError("Could not find exact double-precision record for Seed 42 T_real")
        
    # Exact permutation counting
    k = int(np.sum(all_nulls >= T_real))
    p_val = float((1 + k) / (B + 1))
    delta_task = float(T_real - mean_null)
    
    print("=== Exp 1 High-Precision Permutation Aggregation ===")
    print(f"Total Null Permutations (B): {B}")
    print(f"T_real (Exact double-precision): {T_real:.16f} (Source: {source})")
    print(f"Mean T_null:                     {mean_null:.16f}")
    print(f"K (Null >= T_real):              {k}")
    print(f"Empirical p-value:               {p_val:.6f}")
    print(f"Estimated Delta:                 {delta_task:+.16f}")
    
    # Save aggregated null distribution
    out_df = pd.DataFrame({
        'T_null': all_nulls
    })
    out_df.to_csv(EXP1_DIR / "aggregated_nulls.csv", index=False)
    
    summary_df = pd.DataFrame([{
        'T_real': T_real,
        'Mean_T_null': mean_null,
        'K_exceed': k,
        'Total_B': B,
        'P_value': p_val,
        'Delta_task': delta_task
    }])
    summary_df.to_csv(EXP1_DIR / "exp1_summary.csv", index=False)
    print(f"Saved aggregated nulls and summary to {EXP1_DIR}")

if __name__ == "__main__":
    aggregate_exp1()
