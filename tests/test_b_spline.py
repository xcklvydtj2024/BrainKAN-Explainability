import pytest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.kan import EdgeSpecificKANConv, _b_spline_basis

def test_b_spline_basis():
    """Test that B-spline basis returns expected shapes and values."""
    # Create a dummy base_edge_index with 10 edges
    base_edge_index = torch.stack([torch.arange(10), torch.arange(1, 11) % 10], dim=0)
    conv = EdgeSpecificKANConv(in_channels=1, out_channels=1, base_edge_index=base_edge_index, num_nodes=10, grid_size=5, spline_order=3)
    
    # Input x with shape [N, 1]
    x = torch.linspace(-1, 1, steps=20).unsqueeze(1)
    
    # The input x must match the actual node count during call, wait, this is just testing the _b_spline_basis
    # which takes [N, in_channels]
    basis = _b_spline_basis(x, conv.grid[0], conv.spline_order)
    
    # Expected shape: [N, 1, grid_size + spline_order] -> [20, 1, 8]
    assert basis.shape == (20, 1, 5 + 3)
    
    # B-splines partition of unity: sum of basis functions should be 1 across the grid at any point
    # Since our grid covers [-1, 1], points inside should sum to 1.
    basis_sum = basis.sum(dim=-1).squeeze(-1)
    assert torch.allclose(basis_sum, torch.ones_like(basis_sum), atol=1e-4)
