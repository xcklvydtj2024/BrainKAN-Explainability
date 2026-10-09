import numpy as np
from stats_utils.gpd_tail import compute_tail_aware_pvalue

def permutation_test_Q3B(real_stats, null_stats_matrix):
    """
    real_stats: (N_edges,)
    null_stats_matrix: (R, N_edges)
    Returns: empirical_p, gpd_p for each edge
    """
    N_edges = real_stats.shape[0]
    R = null_stats_matrix.shape[0]
    
    emp_p = np.zeros(N_edges)
    gpd_p = np.zeros(N_edges)
    
    for e in range(N_edges):
        T_real = real_stats[e]
        T_nulls = null_stats_matrix[:, e]
        e_p, g_p = compute_tail_aware_pvalue(T_real, T_nulls)
        emp_p[e] = e_p
        gpd_p[e] = g_p
        
    return emp_p, gpd_p

def compute_q3c_statistic(G_2bk, G_0bk):
    """
    Computes T_e^{real} = 1/N * sum_i (G_{i,e}^{2BK} - G_{i,e}^{0BK})
    G_2bk, G_0bk: (N_subjects, N_edges)
    """
    diffs = G_2bk - G_0bk
    return np.mean(diffs, axis=0) # (N_edges,)
