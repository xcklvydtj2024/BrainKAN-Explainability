import numpy as np
from scipy.stats import genpareto

def compute_tail_aware_pvalue(t_stat, null_dist, tail_fraction=0.1):
    """
    Computes tail-aware p-value using generalized-Pareto (GPD) extrapolation.
    t_stat: scalar, observed test statistic
    null_dist: array, empirical null distribution
    tail_fraction: fraction of largest null values to use for fitting the tail (e.g., 0.1 for top 10%)
    """
    R = len(null_dist)
    empirical_p = (1 + np.sum(null_dist >= t_stat)) / (R + 1)
    
    # If the statistic is not extreme, just return empirical p-value
    if empirical_p > tail_fraction:
        return empirical_p, empirical_p
    
    # Sort null distribution
    sorted_null = np.sort(null_dist)
    
    # Exceedances threshold
    N_tail = int(R * tail_fraction)
    threshold = sorted_null[-N_tail]
    
    # Exceedances above threshold
    exceedances = sorted_null[-N_tail:] - threshold
    
    try:
        # Fit GPD to exceedances
        c, loc, scale = genpareto.fit(exceedances, floc=0)
        
        # Calculate exceedance probability under GPD
        if t_stat > threshold:
            tail_prob = genpareto.sf(t_stat - threshold, c, loc=loc, scale=scale)
            gpd_p = (N_tail / R) * tail_prob
        else:
            gpd_p = empirical_p
            
        return empirical_p, max(gpd_p, 1e-10) # avoid exact zero
    except Exception as e:
        # Fallback to empirical if fitting fails
        return empirical_p, empirical_p
