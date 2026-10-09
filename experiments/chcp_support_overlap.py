import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_data_loading import load_real_hcp_data, load_real_chcp_data
from stats_utils.empirical_null import compute_reference_grid_for_fold

def run_overlap_analysis():
    print("Running Reference Support Overlap Analysis...")
    hcp_data, base_edge_index, num_nodes = load_real_hcp_data()
    chcp_data, _, _ = load_real_chcp_data()
    
    # 1. Compute full HCP reference (X_ref_B)
    hcp_ref_grid_B = compute_reference_grid_for_fold(hcp_data, base_edge_index)
    
    # 2. Extract CHCP native support bounds [q1, q99]
    chcp_vals = np.zeros((len(chcp_data) * 2, num_nodes))
    for i, (d0, d2) in enumerate(chcp_data):
        chcp_vals[2*i] = d0.x[:, 0].numpy()
        chcp_vals[2*i+1] = d2.x[:, 0].numpy()
        
    src_nodes = np.unique(base_edge_index[0].numpy())
    overlaps = []
    
    for src in src_nodes:
        # CHCP 1% and 99% quantiles for this ROI
        q_low, q_high = np.percentile(chcp_vals[:, src], [1, 99])
        
        # HCP reference points for this ROI
        hcp_points = hcp_ref_grid_B[src]
        
        # Calculate Overlap
        in_support = (hcp_points >= q_low) & (hcp_points <= q_high)
        overlap_frac = np.mean(in_support)
        overlaps.append(overlap_frac)
        
    overlaps = np.array(overlaps)
    
    # 3. Report
    print("--- Support Overlap Statistics (O_j) ---")
    print(f"Median Overlap: {np.median(overlaps):.4f}")
    print(f"Minimum Overlap: {np.min(overlaps):.4f}")
    print(f"10th Percentile: {np.percentile(overlaps, 10):.4f}")
    print(f"Proportion of ROIs with O_j < 0.8: {np.mean(overlaps < 0.8):.4f}")
    print(f"Global Proportion of Ref Points outside CHCP support: {1.0 - np.mean(overlaps):.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.hist(overlaps, bins=20, edgecolor='black', alpha=0.7)
    plt.title("HCP Reference Support Overlap on CHCP ($O_j$)")
    plt.xlabel("Overlap Proportion ($O_j$)")
    plt.ylabel("Number of ROIs")
    plt.grid(True, alpha=0.3)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "cross_cohort")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "reference_support_overlap.png"))
    print("Saved histogram to results/cross_cohort/reference_support_overlap.png")

if __name__ == "__main__":
    run_overlap_analysis()
