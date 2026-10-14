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
    """Verify that hierarchical_inference_runner executes and produces valid summary output."""
    from experiments.hierarchical_inference_runner import run_hierarchical_inference
    summary_df = run_hierarchical_inference(n_permutations=500)
    assert summary_df["Primary_Target"].iloc[0] == "Label_Blind_Training_Fold_Domain"
    assert summary_df["n_seeds"].iloc[0] == 10
    assert summary_df["n_permutations"].iloc[0] == 500
    assert 0.0 <= summary_df["perm_empirical_pvalue"].iloc[0] <= 1.0
    assert summary_df["synthetic_bootstrap_status"].iloc[0] == "REMOVED_IN_FAVOR_OF_RIGOR"
