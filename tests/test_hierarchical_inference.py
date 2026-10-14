import pytest
import numpy as np

from stats_utils.hierarchical_inference import (
    compute_permutation_pvalue,
    clustered_subject_bootstrap_ci,
    decompose_hierarchical_variance,
)


def test_compute_permutation_pvalue_extreme():
    # Null distribution centered at 0 with small variance
    null_dist = np.random.normal(loc=0.0, scale=1.0, size=500)
    # Observed stat far out in the tail
    obs_stat = 5.0

    res = compute_permutation_pvalue(obs_stat, null_dist, alternative="greater")
    assert res["empirical_p"] <= (1.0 / 501.0) + 1e-6
    assert res["gpd_p"] <= res["empirical_p"]
    assert res["observed_stat"] == 5.0
    assert res["n_permutations"] == 500


def test_compute_permutation_pvalue_null():
    # Observed stat in middle of null
    null_dist = np.random.normal(loc=0.0, scale=1.0, size=500)
    obs_stat = 0.0

    res = compute_permutation_pvalue(obs_stat, null_dist, alternative="greater")
    # Should be around 0.5
    assert 0.35 <= res["empirical_p"] <= 0.65


def test_clustered_subject_bootstrap_ci():
    rng = np.random.RandomState(42)
    # Simulated per-subject effect sizes for N=100 subjects
    subject_effects = rng.normal(loc=0.005, scale=0.01, size=100)

    res = clustered_subject_bootstrap_ci(subject_effects, n_bootstrap=500, seed=42)
    assert res["n_subjects"] == 100
    assert res["ci_lower_percentile"] < res["observed_mean"] < res["ci_upper_percentile"]
    assert res["bootstrap_se"] > 0


def test_decompose_hierarchical_variance():
    rng = np.random.RandomState(42)
    seed_estimates = rng.normal(loc=0.005, scale=0.002, size=10) # 10 seeds
    subject_boot_estimates = rng.normal(loc=0.005, scale=0.005, size=500)

    decomp = decompose_hierarchical_variance(seed_estimates, subject_boot_estimates)
    assert decomp["var_total_decoupled"] == decomp["var_subject_population"] + decomp["var_optimization_seed"]
    assert np.isclose(decomp["pct_population_variance"] + decomp["pct_optimization_variance"], 100.0)


def test_multi_seed_permutation_inference_runner():
    """Verify that hierarchical_inference_runner executes and produces valid summary output under Scheme A."""
    from experiments.hierarchical_inference_runner import run_hierarchical_inference
    summary_df = run_hierarchical_inference(n_permutations=500)
    assert len(summary_df) == 3

    # Row 0: Single-seed conditional test
    assert summary_df["Analysis"].iloc[0] == "Single_Seed_Conditional_Inference"
    assert summary_df["n_seeds"].iloc[0] == 1
    assert summary_df["n_permutations"].iloc[0] == 500
    assert np.isclose(summary_df["Empirical_p_value"].iloc[0], 0.005988, atol=1e-4)

    # Row 1: Multi-seed optimization audit
    assert summary_df["Analysis"].iloc[1] == "Multi_Seed_Optimization_Audit"
    assert summary_df["n_seeds"].iloc[1] == 10
    assert summary_df["Delta"].iloc[1] < 0  # 9/10 seeds negative

    # Row 2: Formally discarded mismatched comparison
    assert summary_df["Analysis"].iloc[2] == "Mismatched_Comparison_Audit"
    assert summary_df["Significance"].iloc[2] == "DISCARDED_AS_STATISTICALLY_INVALID"
