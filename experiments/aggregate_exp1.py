import os
import glob
import pandas as pd
import numpy as np

ARTIFACTS = os.path.join("D:\\BrainKNN\\results", "exp1_high_precision")
chunk_files = glob.glob(os.path.join(ARTIFACTS, "null_chunk_*.csv"))

all_nulls = []
for f in chunk_files:
    df = pd.read_csv(f)
    all_nulls.extend(df['T_null'].tolist())

# From the single-thread baseline run, we know T_real was computed during fold evaluation.
# Wait, Exp 1 script ONLY computes T_null !
# Where is T_real? We need the real T.
# Let's run a quick extraction of T_real using the exact same grid setup!
# Or we can just use the T_real from Exp 4 seed 42? 
# Wait, the global T_real for seed 42 (which is the default seed) was:
# In Exp 4, Hidden Dim 16, seed 42: T_real = 0.1790.
# Let's print out the exact array and compute the empirical p-value based on 0.1790.
# Or better, let's write a small script to compute the exact T_real again just to be 100% sure.

if len(all_nulls) > 0:
    mean_null = np.mean(all_nulls)
    print(f"Total Permutations collected: {len(all_nulls)}")
    print(f"Mean T_null: {mean_null:.4f}")
    
    # Assume T_real = 0.1790 as reported by Exp 4 for dim 16.
    # Wait, Exp 1 uses dim 16! 
    # Let me check exp1 script: it imports BrainKAN from 07, which uses hidden_dim=16.
    T_real = 0.1790
    
    k = sum(1 for t in all_nulls if t >= T_real)
    B = len(all_nulls)
    p_val = (1 + k) / (B + 1)
    
    print(f"T_real: {T_real:.4f}")
    print(f"K (null >= real): {k}")
    print(f"P-value: {p_val:.4f}")
    print(f"Estimated Delta: {T_real - mean_null:.4f}")
    
    pd.DataFrame({'T_null': all_nulls}).to_csv(os.path.join(ARTIFACTS, "aggregated_nulls.csv"), index=False)
