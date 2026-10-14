import pytest
import numpy as np
import torch

from experiments.mechanistic_controls import (
    get_base_topology,
    generate_counterfactual_synthetic_cohort,
    verify_data_injection,
    compute_edge_dfl_identifiable,
    run_single_identifiable_simulation,
    PLANTED_EDGE_INDICES,
    NUM_NODES,
)


def test_data_injection_and_counterfactual_integrity():
    """Verify that changing gamma strictly modifies planted edge target nodes and leaves others intact."""
    assert verify_data_injection() is True


def test_linear_null_hypothesis_gamma_zero():
    """Verify that when gamma = 0, the relationship is purely linear in both conditions."""
    base_edge_index, planted_edges = get_base_topology()
    _, x0, x1 = generate_counterfactual_synthetic_cohort(
        base_edge_index, planted_edges, gamma=0.0, seed=42, num_pairs=80
    )

    for u, v in planted_edges:
        # Check condition 0: xv = 0.8 * xu + eps
        diff_0 = x0[:, v] - 0.8 * x0[:, u]
        # Should be purely Gaussian noise with std ~ 0.1
        assert np.std(diff_0) < 0.15
        assert np.abs(np.mean(diff_0)) < 0.05

        # Check condition 1: xv = 0.8 * xu + eps
        diff_1 = x1[:, v] - 0.8 * x1[:, u]
        assert np.std(diff_1) < 0.15


def test_planted_edge_recovery_high_gamma():
    """Verify that under high gamma (e.g. 1.0), planted edges are detected with high AUC and precision."""
    res = run_single_identifiable_simulation(gamma=1.0, seed=42, n_pairs=100, n_perm=50)
    assert res["p_value"] <= 0.05
    assert res["AUC"] >= 0.85
    assert res["Recall"] >= 0.50
    assert res["Precision"] >= 0.50


def test_null_calibration_single_run():
    """Verify that under gamma = 0, the simulation completes and does not produce spurious high Delta."""
    res = run_single_identifiable_simulation(gamma=0.0, seed=123, n_pairs=80, n_perm=50)
    assert -0.01 <= res["Delta"] <= 0.01
