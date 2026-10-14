import pytest
import torch
import torch.nn.functional as F
import numpy as np

from models.kan import BrainKAN, EdgeSpecificKANConv


def test_curvature_penalty_computation():
    base_edge_index = torch.stack([torch.arange(10), torch.arange(1, 11) % 10], dim=0)
    model = BrainKAN(
        base_edge_index,
        num_nodes=10,
        in_channels=1,
        hidden_dim=8,
        num_classes=2,
        grid_range=(-3.0, 3.0),
    )

    penalty = model.compute_curvature_penalty(num_points=40, layer="conv1")
    assert penalty.item() >= 0.0
    assert torch.isfinite(penalty)

    # Test backpropagation
    penalty.backward()
    assert model.conv1.spline_weight.grad is not None
    assert model.conv1.base_weight.grad is not None
    assert model.conv1.spline_weight.grad.norm().item() > 0.0


def test_curvature_penalty_monotonicity():
    """Higher oscillation in spline weights should strictly yield higher curvature penalty."""
    base_edge_index = torch.stack([torch.arange(4), torch.arange(1, 5) % 4], dim=0)
    conv = EdgeSpecificKANConv(
        in_channels=1,
        out_channels=2,
        base_edge_index=base_edge_index,
        num_nodes=4,
        grid_range=(-2.0, 2.0),
    )

    # Set spline weights to zero
    with torch.no_grad():
        conv.spline_weight.zero_()
        conv.base_weight.zero_()
    pen_zero = conv.compute_curvature_penalty(num_points=30).item()
    assert np.isclose(pen_zero, 0.0, atol=1e-5)

    # Set smooth weights
    with torch.no_grad():
        conv.spline_weight.fill_(0.1)
    pen_smooth = conv.compute_curvature_penalty(num_points=30).item()

    # Set oscillating weights (+1, -1, +1, -1...)
    with torch.no_grad():
        for b in range(conv.spline_weight.size(-1)):
            conv.spline_weight[..., b] = ((-1.0) ** b) * 2.0
    pen_oscillating = conv.compute_curvature_penalty(num_points=30).item()

    assert pen_oscillating > pen_smooth >= pen_zero
