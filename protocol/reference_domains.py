"""
Reference Domain Evaluation Protocols for BrainKAN.

This module provides unbiased reference-domain estimation protocols that eliminate
domain selection bias and prevent condition labels from leaking into or back-binding
the evaluation grid during permutation null testing.

Protocols implemented:
1. Label-blind domain (标签盲化域):
   Derives a fixed reference grid independent of condition labels (e.g. pooled empirical
   quantiles across all samples or fixed physical interval [-3.0, 3.0]).
2. Independent sample support (独立样本确定域):
   Derives condition-dependent common support on an independent partition of subjects,
   freezing this interval for all real and null inferences on the test/inferential cohort.
3. Full-pipeline re-enactment (全流程算法重演):
   Treats common support derivation as an endogenous step of the inference pipeline;
   for every label permutation, the intersection support is re-derived from scratch
   under the permuted condition assignments.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class DomainMode(str, Enum):
    LABEL_BLIND = "label_blind"
    INDEPENDENT_SPLIT = "independent_split"
    FULL_REENACTMENT = "full_reenactment"
    LEGACY_FIXED = "legacy_fixed"


def compute_label_blind_domain(
    data_pairs: List[Tuple[any, any]],
    num_nodes: int,
    n_probe: int = 100,
    q_range: Tuple[float, float] = (1.0, 99.0),
    preset_range: Optional[Tuple[float, float]] = None,
) -> Tuple[Dict[int, torch.Tensor], Tuple[float, float]]:
    """Compute reference domain strictly agnostic to task condition labels.

    Args:
        data_pairs: List of subject graph pairs [(g0, g1), ...]
        num_nodes: Number of graph nodes (ROIs)
        n_probe: Number of evaluation probe points
        q_range: Lower and upper percentiles for pooled quantile support
        preset_range: Optional hardcoded physical range, e.g. (-3.0, 3.0)

    Returns:
        node_t_values: Dict mapping node_idx -> probe grid Tensor [n_probe]
        global_range: (min_val, max_val) across all nodes
    """
    if preset_range is not None:
        p_min, p_max = preset_range
        probe = torch.linspace(float(p_min), float(p_max), n_probe)
        node_t_values = {n: probe.clone() for n in range(num_nodes)}
        return node_t_values, (float(p_min), float(p_max))

    # Pooled empirical quantile domain: all nodes, all samples, zero label access
    node_x_pool: Dict[int, List[float]] = {n: [] for n in range(num_nodes)}
    for pair in data_pairs:
        for n in range(num_nodes):
            x0 = float(pair[0].x[n, 0].item())
            x1 = float(pair[1].x[n, 0].item())
            node_x_pool[n].append(x0)
            node_x_pool[n].append(x1)

    node_t_values: Dict[int, torch.Tensor] = {}
    global_min = float("inf")
    global_max = float("-inf")

    for n in range(num_nodes):
        r_min = float(np.percentile(node_x_pool[n], q_range[0]))
        r_max = float(np.percentile(node_x_pool[n], q_range[1]))
        if r_min >= r_max:
            r_max = r_min + 1e-4
        node_t_values[n] = torch.linspace(r_min, r_max, n_probe)
        global_min = min(global_min, r_min)
        global_max = max(global_max, r_max)

    return node_t_values, (global_min, global_max)


def compute_common_support_domain(
    data_pairs: List[Tuple[any, any]],
    num_nodes: int,
    n_probe: int = 100,
    q_range: Tuple[float, float] = (1.0, 99.0),
    override_labels: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[Dict[int, torch.Tensor], Tuple[float, float]]:
    """Compute condition common support intersection: supp(X_0) ∩ supp(X_1).

    If override_labels is supplied (e.g. from label permutation), the common support
    is calculated on the permuted label partition (Full-pipeline re-enactment).

    Args:
        data_pairs: List of subject pairs [(g0, g1), ...]
        num_nodes: Number of ROIs
        n_probe: Probe points per grid
        q_range: Percentile boundaries
        override_labels: Optional list of (y0, y1) per pair reflecting permuted labels

    Returns:
        node_t_values: Dict mapping node_idx -> probe grid Tensor [n_probe]
        global_range: (global_min, global_max)
    """
    node_x_c0: Dict[int, List[float]] = {n: [] for n in range(num_nodes)}
    node_x_c1: Dict[int, List[float]] = {n: [] for n in range(num_nodes)}
    node_x_all: Dict[int, List[float]] = {n: [] for n in range(num_nodes)}

    for idx, pair in enumerate(data_pairs):
        if override_labels is not None:
            y0, y1 = override_labels[idx]
        else:
            y0 = int(pair[0].y.item())
            y1 = int(pair[1].y.item())

        for n in range(num_nodes):
            x0 = float(pair[0].x[n, 0].item())
            x1 = float(pair[1].x[n, 0].item())
            node_x_all[n].extend([x0, x1])

            if y0 == 0:
                node_x_c0[n].append(x0)
                node_x_c1[n].append(x1)
            else:
                node_x_c1[n].append(x0)
                node_x_c0[n].append(x1)

    node_t_values: Dict[int, torch.Tensor] = {}
    global_min = float("inf")
    global_max = float("-inf")

    for n in range(num_nodes):
        c0_min = float(np.percentile(node_x_c0[n], q_range[0]))
        c0_max = float(np.percentile(node_x_c0[n], q_range[1]))
        c1_min = float(np.percentile(node_x_c1[n], q_range[0]))
        c1_max = float(np.percentile(node_x_c1[n], q_range[1]))

        c_min = max(c0_min, c1_min)
        c_max = min(c0_max, c1_max)

        # Defensive fallback if disjoint (audit demonstrated 0/240 on HCP fMRI)
        if c_min >= c_max:
            c_min = float(np.percentile(node_x_all[n], q_range[0]))
            c_max = float(np.percentile(node_x_all[n], q_range[1]))
            if c_min >= c_max:
                c_max = c_min + 1e-4

        node_t_values[n] = torch.linspace(c_min, c_max, n_probe)
        global_min = min(global_min, c_min)
        global_max = max(global_max, c_max)

    return node_t_values, (global_min, global_max)


def compute_independent_split_domain(
    reference_pairs: List[Tuple[any, any]],
    num_nodes: int,
    n_probe: int = 100,
    q_range: Tuple[float, float] = (1.0, 99.0),
) -> Tuple[Dict[int, torch.Tensor], Tuple[float, float]]:
    """Derive common support from an independent split of subjects and freeze it.

    The returned domain is computed ONCE on reference_pairs and held immutable
    for all inferences (real and permuted) on downstream evaluation subjects.
    """
    return compute_common_support_domain(
        data_pairs=reference_pairs,
        num_nodes=num_nodes,
        n_probe=n_probe,
        q_range=q_range,
        override_labels=None,
    )


class UnbiasedDomainProtocol:
    """Orchestrator for managing domain evaluation strategies."""

    def __init__(
        self,
        mode: Union[DomainMode, str] = DomainMode.LABEL_BLIND,
        n_probe: int = 100,
        preset_range: Optional[Tuple[float, float]] = None,
        independent_pairs: Optional[List[Tuple[any, any]]] = None,
    ):
        self.mode = DomainMode(mode)
        self.n_probe = n_probe
        self.preset_range = preset_range
        self.independent_pairs = independent_pairs
        self._frozen_independent_domain: Optional[Tuple[Dict[int, torch.Tensor], Tuple[float, float]]] = None

        if self.mode == DomainMode.INDEPENDENT_SPLIT and self.independent_pairs is not None:
            # Precompute and freeze the domain once
            num_nodes = self.independent_pairs[0][0].x.shape[0]
            self._frozen_independent_domain = compute_independent_split_domain(
                self.independent_pairs, num_nodes, n_probe=self.n_probe
            )

    def get_evaluation_domain(
        self,
        train_pairs: List[Tuple[any, any]],
        num_nodes: int,
        is_permutation: bool = False,
        permuted_labels: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[Dict[int, torch.Tensor], Tuple[float, float]]:
        """Retrieve evaluation reference domain for real or null models."""
        if self.mode == DomainMode.LABEL_BLIND:
            return compute_label_blind_domain(
                train_pairs,
                num_nodes,
                n_probe=self.n_probe,
                preset_range=self.preset_range,
            )

        if self.mode == DomainMode.INDEPENDENT_SPLIT:
            if self._frozen_independent_domain is not None:
                return self._frozen_independent_domain
            raise ValueError(
                "INDEPENDENT_SPLIT mode requires independent_pairs provided at initialization."
            )

        if self.mode == DomainMode.FULL_REENACTMENT:
            # Under null permutation, re-calculate common support based on the permuted labels
            if is_permutation and permuted_labels is not None:
                return compute_common_support_domain(
                    train_pairs,
                    num_nodes,
                    n_probe=self.n_probe,
                    override_labels=permuted_labels,
                )
            # For real model, calculate on real labels
            return compute_common_support_domain(
                train_pairs,
                num_nodes,
                n_probe=self.n_probe,
                override_labels=None,
            )

        if self.mode == DomainMode.LEGACY_FIXED:
            # Legacy mode: derive once from real labels and fix for null
            return compute_common_support_domain(
                train_pairs,
                num_nodes,
                n_probe=self.n_probe,
                override_labels=None,
            )

        raise NotImplementedError(f"Unsupported mode {self.mode}")
