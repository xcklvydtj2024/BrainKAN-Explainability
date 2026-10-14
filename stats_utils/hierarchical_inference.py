"""
Hierarchical Statistical Inference Framework for BrainKAN.

This module provides:
1. Replacement of seed-level Student's t_9 tests with large-scale label permutation
   tests (B >= 500) directly evaluating the optimization-marginalized summary statistic
   E_s[T(Y; s)] against the empirical permutation distribution.
2. Clustered / Hierarchical bootstrap at the Subject-Pair level to quantify biological
   population sampling uncertainty and decouple it from algorithmic optimization randomness.
3. Variance decomposition into biological population variance and optimization variance.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.stats import genpareto

from stats_utils.gpd_tail import compute_tail_aware_pvalue


def compute_permutation_pvalue(
    observed_stat: float,
    null_distribution: np.ndarray,
    alternative: str = "greater",
    use_gpd_tail: bool = True,
    tail_fraction: float = 0.1,
) -> Dict[str, float]:
    """Compute exact empirical non-parametric p-value from permutation distribution.

    Args:
        observed_stat: The test statistic on real data (e.g. \bar{T}_{real} or \bar{\Delta})
        null_distribution: 1D array of null test statistics from B permutations
        alternative: 'greater' (excess curvature), 'less', or 'two-sided'
        use_gpd_tail: Whether to apply semi-parametric GPD tail extrapolation
        tail_fraction: Fraction of tail values used for GPD fitting

    Returns:
        Dict with empirical_p, gpd_p, null_mean, null_std, and quantiles
    """
    null_dist = np.asarray(null_distribution, dtype=float)
    B = len(null_dist)

    if alternative == "greater":
        count = np.sum(null_dist >= observed_stat)
    elif alternative == "less":
        count = np.sum(null_dist <= observed_stat)
    elif alternative == "two-sided":
        mean_null = np.mean(null_dist)
        count = np.sum(np.abs(null_dist - mean_null) >= np.abs(observed_stat - mean_null))
    else:
        raise ValueError(f"Unknown alternative {alternative}")

    emp_p = float((1.0 + count) / (B + 1.0))
    gpd_p = emp_p

    if use_gpd_tail and alternative == "greater":
        emp_val, gpd_val = compute_tail_aware_pvalue(
            observed_stat, null_dist, tail_fraction=tail_fraction
        )
        emp_p = float(emp_val)
        gpd_p = float(gpd_val)

    # Permutation quantiles
    q05, q25, q50, q75, q95 = np.percentile(null_dist, [5, 25, 50, 75, 95])

    return {
        "observed_stat": float(observed_stat),
        "empirical_p": emp_p,
        "gpd_p": gpd_p,
        "n_permutations": B,
        "null_mean": float(np.mean(null_dist)),
        "null_std": float(np.std(null_dist, ddof=1)),
        "null_median": float(np.median(null_dist)),
        "null_q05": float(q05),
        "null_q95": float(q95),
    }


def clustered_subject_bootstrap_ci(
    subject_estimates: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Subject-pair level clustered bootstrap confidence interval.

    Args:
        subject_estimates: 1D array of per-subject-pair metric contributions
                           e.g. \Delta_i for i in 1..N
        n_bootstrap: Number of bootstrap iterations (e.g. 1000)
        confidence_level: Confidence level (default 0.95)
        seed: Random seed

    Returns:
        Dict with mean, standard error, percentile CI, and normal CI
    """
    rng = np.random.RandomState(seed)
    N = len(subject_estimates)
    boot_means = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        sample_indices = rng.choice(N, size=N, replace=True)
        boot_means[b] = np.mean(subject_estimates[sample_indices])

    alpha = 1.0 - confidence_level
    ci_lower_pct = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_upper_pct = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))

    se_boot = float(np.std(boot_means, ddof=1))
    mean_boot = float(np.mean(boot_means))
    obs_mean = float(np.mean(subject_estimates))

    # Bias estimate
    bias = mean_boot - obs_mean

    # BCa acceleration & bias-correction if feasible
    z0 = stats.norm.ppf(np.mean(boot_means < obs_mean)) if np.mean(boot_means < obs_mean) not in [0, 1] else 0.0

    return {
        "observed_mean": obs_mean,
        "bootstrap_mean": mean_boot,
        "bootstrap_se": se_boot,
        "ci_lower_percentile": ci_lower_pct,
        "ci_upper_percentile": ci_upper_pct,
        "bootstrap_bias": float(bias),
        "n_subjects": N,
        "n_bootstrap": n_bootstrap,
    }


def decompose_hierarchical_variance(
    seed_estimates: np.ndarray,
    subject_bootstrap_estimates: np.ndarray,
) -> Dict[str, float]:
    """Decompose total inferential variance into biological population and algorithmic optimization components.

    Var_total = Var_subject (population sampling) + Var_seed (optimization landscape)
    """
    var_seed = float(np.var(seed_estimates, ddof=1)) if len(seed_estimates) > 1 else 0.0
    var_subject = float(np.var(subject_bootstrap_estimates, ddof=1))

    var_total = var_subject + var_seed
    pct_subject = (var_subject / var_total * 100.0) if var_total > 0 else 0.0
    pct_seed = (var_seed / var_total * 100.0) if var_total > 0 else 0.0

    return {
        "var_subject_population": var_subject,
        "se_subject_population": float(np.sqrt(var_subject)),
        "var_optimization_seed": var_seed,
        "se_optimization_seed": float(np.sqrt(var_seed)),
        "var_total_decoupled": var_total,
        "se_total_decoupled": float(np.sqrt(var_total)),
        "pct_population_variance": float(pct_subject),
        "pct_optimization_variance": float(pct_seed),
    }
