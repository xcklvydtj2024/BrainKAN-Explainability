import os
import torch
import numpy as np
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_data_loading import load_real_hcp_data, load_real_chcp_data

def main():
    print("=== Computing Per-Node Empirical Support Overlap Statistics ===")
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "primary_analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    hcp_pairs, _, num_nodes = load_real_hcp_data()
    chcp_pairs, _, _ = load_real_chcp_data()
    
    print(f"Loaded {len(hcp_pairs)} HCP pairs, {len(chcp_pairs)} CHCP pairs, {num_nodes} ROIs.")
    
    # 1. Collect values per node for HCP Condition 0 (0-Back), Condition 1 (2-Back), and CHCP
    hcp_cond0 = {n: [] for n in range(num_nodes)}
    hcp_cond1 = {n: [] for n in range(num_nodes)}
    chcp_all  = {n: [] for n in range(num_nodes)}
    
    for pair in hcp_pairs:
        # pair[0] is one condition, pair[1] is the other
        for n in range(num_nodes):
            val0 = pair[0].x[n, 0].item()
            val1 = pair[1].x[n, 0].item()
            y0 = pair[0].y.item() if hasattr(pair[0], 'y') else 0
            y1 = pair[1].y.item() if hasattr(pair[1], 'y') else 1
            if y0 == 0:
                hcp_cond0[n].append(val0)
                hcp_cond1[n].append(val1)
            else:
                hcp_cond1[n].append(val0)
                hcp_cond0[n].append(val1)
                
    for pair in chcp_pairs:
        for n in range(num_nodes):
            chcp_all[n].append(pair[0].x[n, 0].item())
            chcp_all[n].append(pair[1].x[n, 0].item())
            
    rows = []
    
    for n in range(num_nodes):
        c0_vals = np.array(hcp_cond0[n])
        c1_vals = np.array(hcp_cond1[n])
        pooled_vals = np.concatenate([c0_vals, c1_vals])
        chcp_vals = np.array(chcp_all[n])
        
        # 1% - 99% empirical quantile intervals
        q1_c0, q99_c0 = np.percentile(c0_vals, 1), np.percentile(c0_vals, 99)
        q1_c1, q99_c1 = np.percentile(c1_vals, 1), np.percentile(c1_vals, 99)
        q1_pool, q99_pool = np.percentile(pooled_vals, 1), np.percentile(pooled_vals, 99)
        q1_chcp, q99_chcp = np.percentile(chcp_vals, 1), np.percentile(chcp_vals, 99)
        
        # Intersection & Union between 0-Back and 2-Back (Within-HCP task conditions)
        inter_min_task = max(q1_c0, q1_c1)
        inter_max_task = min(q99_c0, q99_c1)
        union_min_task = min(q1_c0, q1_c1)
        union_max_task = max(q99_c0, q99_c1)
        
        task_inter_len = max(0.0, inter_max_task - inter_min_task)
        task_union_len = max(1e-8, union_max_task - union_min_task)
        pool_len = max(1e-8, q99_pool - q1_pool)
        
        task_jaccard = task_inter_len / task_union_len
        task_pool_coverage = task_inter_len / pool_len
        
        # Intersection & Union between HCP pooled and CHCP
        inter_min_cross = max(q1_pool, q1_chcp)
        inter_max_cross = min(q99_pool, q99_chcp)
        union_min_cross = min(q1_pool, q1_chcp)
        union_max_cross = max(q99_pool, q99_chcp)
        
        cross_inter_len = max(0.0, inter_max_cross - inter_min_cross)
        cross_union_len = max(1e-8, union_max_cross - union_min_cross)
        cross_jaccard = cross_inter_len / cross_union_len
        
        rows.append({
            'node_id': n,
            # Condition 0
            'c0_q01': q1_c0, 'c0_q99': q99_c0,
            # Condition 1
            'c1_q01': q1_c1, 'c1_q99': q99_c1,
            # Pooled
            'pool_q01': q1_pool, 'pool_q99': q99_pool,
            # Common Support
            'inter_q01': inter_min_task, 'inter_q99': inter_max_task,
            'task_jaccard_overlap': task_jaccard,
            'task_pool_coverage': task_pool_coverage,
            # Cross dataset
            'chcp_q01': q1_chcp, 'chcp_q99': q99_chcp,
            'cross_dataset_jaccard': cross_jaccard
        })
        
    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "node_support_overlap_stats.csv")
    df.to_csv(out_csv, index=False)
    
    print(f"\n=== Summary Statistics Across {num_nodes} Brain ROIs (240 Functional Edges) ===")
    print("1. Cohort Condition Support (0-Back vs 2-Back across 100 subjects):")
    print(f"   Mean Jaccard Overlap:        {df['task_jaccard_overlap'].mean()*100:.2f}% (± {df['task_jaccard_overlap'].std()*100:.2f}%)")
    print(f"   Min Jaccard Overlap:         {df['task_jaccard_overlap'].min()*100:.2f}%")
    print(f"   Median Jaccard Overlap:      {df['task_jaccard_overlap'].median()*100:.2f}%")
    print(f"   Max Jaccard Overlap:         {df['task_jaccard_overlap'].max()*100:.2f}%")
    print(f"   Mean Pooled Coverage:        {df['task_pool_coverage'].mean()*100:.2f}%")
    
    print("\n2. Cross-Dataset Support (HCP vs CHCP):")
    print(f"   Mean Jaccard Overlap:        {df['cross_dataset_jaccard'].mean()*100:.2f}% (± {df['cross_dataset_jaccard'].std()*100:.2f}%)")
    print(f"   Min Jaccard Overlap:         {df['cross_dataset_jaccard'].min()*100:.2f}%")
    print(f"   Median Jaccard Overlap:      {df['cross_dataset_jaccard'].median()*100:.2f}%")
    
    print(f"\nSaved per-node empirical support analysis to: {out_csv}")

if __name__ == "__main__":
    main()
