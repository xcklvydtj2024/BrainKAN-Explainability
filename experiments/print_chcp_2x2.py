import os
import numpy as np

def print_2x2_results():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "cross_cohort", "chcp_results_full.npz")
    if not os.path.exists(path):
        print("Results file not found yet.")
        return
        
    data = np.load(path)
    avg_G_native = data['avg_G_native']
    avg_G_frozen = data['avg_G_frozen']
    avg_null_G_native = data['avg_null_G_native'] # Shape: (10, num_edges)
    avg_null_G_frozen = data['avg_null_G_frozen'] # Shape: (10, num_edges)
    
    # We report the median (or mean) across all 240 edges for the report
    G_R_A = np.mean(avg_G_native)
    G_R_B = np.mean(avg_G_frozen)
    
    G_N_A = np.mean(avg_null_G_native) # mean over (10, 240) -> scalar
    G_N_B = np.mean(avg_null_G_frozen)
    
    print("--- 2x2 Mean Nonlinearity (G) ---")
    print(f"Native CHCP ref (A): Real={G_R_A:.4f}, Null={G_N_A:.4f}")
    print(f"Frozen HCP ref (B) : Real={G_R_B:.4f}, Null={G_N_B:.4f}")
    print(f"Sep (R-N) A: {G_R_A - G_N_A:.4f}")
    print(f"Sep (R-N) B: {G_R_B - G_N_B:.4f}")
    
    # Standardized Separation Z
    # Z_e = (G_e^{real} - mu(G_e^{null})) / sigma(G_e^{null})
    mu_N_A = np.mean(avg_null_G_native, axis=0) # (240,)
    sig_N_A = np.std(avg_null_G_native, axis=0) + 1e-8
    Z_A = (avg_G_native - mu_N_A) / sig_N_A
    
    mu_N_B = np.mean(avg_null_G_frozen, axis=0)
    sig_N_B = np.std(avg_null_G_frozen, axis=0) + 1e-8
    Z_B = (avg_G_frozen - mu_N_B) / sig_N_B
    
    print("--- Standardized Separation Z (Mean across edges) ---")
    print(f"Z_A (Native): {np.mean(Z_A):.4f}")
    print(f"Z_B (Frozen): {np.mean(Z_B):.4f}")
    
if __name__ == "__main__":
    print_2x2_results()
