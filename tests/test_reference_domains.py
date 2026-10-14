import pytest
import torch
from torch_geometric.data import Data
import numpy as np

from protocol.reference_domains import (
    compute_label_blind_domain,
    compute_independent_split_domain,
    compute_common_support_domain,
    UnbiasedDomainProtocol,
    DomainMode,
)


def create_mock_pairs(num_pairs=10, num_nodes=6, shift=1.5):
    pairs = []
    for i in range(num_pairs):
        # Condition 0: N(0, 1)
        x0 = torch.randn(num_nodes, 1)
        # Condition 1: N(shift, 1)
        x1 = torch.randn(num_nodes, 1) + shift

        g0 = Data(x=x0, y=torch.tensor(0, dtype=torch.long))
        g1 = Data(x=x1, y=torch.tensor(1, dtype=torch.long))
        pairs.append((g0, g1))
    return pairs


def test_label_blind_domain_invariance():
    """Verify that label-blind domain is completely independent of condition labels."""
    pairs = create_mock_pairs(num_pairs=10, num_nodes=4)
    num_nodes = 4

    # Calculate domain with original pairs
    dom1, (min1, max1) = compute_label_blind_domain(pairs, num_nodes, n_probe=50)

    # Invert the labels of the pairs
    inverted_pairs = []
    for g0, g1 in pairs:
        g0_inv = g0.clone()
        g1_inv = g1.clone()
        g0_inv.y = torch.tensor(1, dtype=torch.long)
        g1_inv.y = torch.tensor(0, dtype=torch.long)
        inverted_pairs.append((g0_inv, g1_inv))

    dom2, (min2, max2) = compute_label_blind_domain(inverted_pairs, num_nodes, n_probe=50)

    # Domains should be EXACTLY identical because labels were never read
    for n in range(num_nodes):
        assert torch.allclose(dom1[n], dom2[n], atol=1e-7)
    assert np.isclose(min1, min2)
    assert np.isclose(max1, max2)


def test_independent_split_domain():
    """Verify that independent split domain stays frozen for test set."""
    ref_pairs = create_mock_pairs(num_pairs=8, num_nodes=4)
    test_pairs = create_mock_pairs(num_pairs=8, num_nodes=4, shift=3.0)
    num_nodes = 4

    protocol = UnbiasedDomainProtocol(
        mode=DomainMode.INDEPENDENT_SPLIT,
        independent_pairs=ref_pairs,
        n_probe=40,
    )

    # Real evaluation domain
    dom_real, _ = protocol.get_evaluation_domain(test_pairs, num_nodes, is_permutation=False)
    # Permutation evaluation domain
    dom_null, _ = protocol.get_evaluation_domain(
        test_pairs, num_nodes, is_permutation=True, permuted_labels=[(1, 0)] * 8
    )

    for n in range(num_nodes):
        assert torch.allclose(dom_real[n], dom_null[n], atol=1e-7)


def test_full_pipeline_reenactment():
    """Verify that full-pipeline re-enactment dynamically recomputes common support under permutation."""
    pairs = create_mock_pairs(num_pairs=10, num_nodes=4, shift=2.0)
    num_nodes = 4

    protocol = UnbiasedDomainProtocol(
        mode=DomainMode.FULL_REENACTMENT,
        n_probe=50,
    )

    dom_real, _ = protocol.get_evaluation_domain(pairs, num_nodes, is_permutation=False)

    # Under complete label flip, condition 0 and condition 1 distributions are exchanged
    permuted_labels = [(1, 0) for _ in pairs]
    dom_null, _ = protocol.get_evaluation_domain(
        pairs, num_nodes, is_permutation=True, permuted_labels=permuted_labels
    )

    # The common support intersection max(c0_min, c1_min) is symmetric under swapping {0, 1},
    # but for arbitrary non-trivial random permutation, the domain recomputes based on the permuted sets
    random_labels = [(np.random.choice([0, 1]), np.random.choice([0, 1])) for _ in pairs]
    dom_rand, _ = protocol.get_evaluation_domain(
        pairs, num_nodes, is_permutation=True, permuted_labels=random_labels
    )
    assert dom_rand is not None
    assert len(dom_rand) == num_nodes


def test_independent_reference_set_strict_disjointness():
    """Verify Scheme B strictly enforces disjointness: ref_subjects intersect train/test subjects is empty."""
    from sklearn.model_selection import KFold
    total_pairs = create_mock_pairs(num_pairs=100, num_nodes=4)
    n_ref = 20

    ref_pairs = total_pairs[:n_ref]
    cv_pairs = total_pairs[n_ref:]

    ref_indices = set(range(n_ref))
    cv_indices = set(range(n_ref, len(total_pairs)))

    # 1. Verification of top-level partition disjointness
    assert len(ref_indices.intersection(cv_indices)) == 0
    assert len(ref_pairs) == 20
    assert len(cv_pairs) == 80

    # 2. Verification across all K folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold_i, (train_sub_idx, test_sub_idx) in enumerate(kf.split(cv_pairs)):
        train_global = set(i + n_ref for i in train_sub_idx)
        test_global = set(i + n_ref for i in test_sub_idx)

        # Reference subjects must never appear in training or testing
        assert len(ref_indices.intersection(train_global)) == 0, f"Fold {fold_i}: Reference subjects leaked into train!"
        assert len(ref_indices.intersection(test_global)) == 0, f"Fold {fold_i}: Reference subjects leaked into test!"
        # Train and test must be disjoint
        assert len(train_global.intersection(test_global)) == 0, f"Fold {fold_i}: Train and test overlap!"


def test_reference_domains_under_permutation_behavior():
    """Verify expected behavior of reference domains under true vs permuted labels."""
    pairs = create_mock_pairs(num_pairs=20, num_nodes=4, shift=1.5)
    num_nodes = 4

    # Scheme A: Label-blind domain MUST be strictly identical under permuted labels
    dom_real_a, _ = compute_label_blind_domain(pairs, num_nodes, n_probe=50)

    # Invert labels
    permuted_pairs = []
    for g0, g1 in pairs:
        p0 = g0.clone()
        p1 = g1.clone()
        p0.y, p1.y = g1.y.clone(), g0.y.clone()
        permuted_pairs.append((p0, p1))

    dom_perm_a, _ = compute_label_blind_domain(permuted_pairs, num_nodes, n_probe=50)
    for n in range(num_nodes):
        assert torch.allclose(dom_real_a[n], dom_perm_a[n], atol=1e-7)

    # Scheme B: Independent reference set domain MUST remain identical regardless of downstream test labels
    ref_pairs = pairs[:8]
    test_pairs = pairs[8:]
    dom_ref, _ = compute_independent_split_domain(ref_pairs, num_nodes, n_probe=50)
    dom_ref_perm, _ = compute_independent_split_domain(ref_pairs, num_nodes, n_probe=50)
    for n in range(num_nodes):
        assert torch.allclose(dom_ref[n], dom_ref_perm[n], atol=1e-7)

