import numpy as np
from statsmodels.stats.multitest import multipletests

def apply_fdr(p_values, alpha=0.05):
    """
    Applies Benjamini-Hochberg FDR correction.
    """
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return reject, pvals_corrected
