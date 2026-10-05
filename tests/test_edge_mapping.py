import pytest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.kan import EdgeSpecificKANConv

def test_edge_mapping_crosstalk():
    """Test that EdgeSpecificKANConv does not mix edge mappings."""
    num_edges = 10
    # Create a dummy base_edge_index with 10 edges
    base_edge_index = torch.stack([torch.arange(10), torch.arange(1, 11) % 10], dim=0)
    conv = EdgeSpecificKANConv(in_channels=1, out_channels=1, base_edge_index=base_edge_index, num_nodes=10, grid_size=3)
    
    # We will bypass PyG message passing and just call the message function directly.
    # Because PyG aggregates, we want to isolate edge mappings.
    for test_edge in range(num_edges):
        # N=1 message to be passed on test_edge
        x_j = torch.tensor([[1.0]])
        # source, destination shape for edge_indices: [2, E_batch]
        edge_indices = torch.tensor([[test_edge], [(test_edge + 1) % 10]])
        
        # Output should be exclusively determined by the weights of test_edge
        edge_weight = None
        # We manually call message() which is what propagates features.
        out = conv.message(x_j, edge_indices, edge_weight)
        
        # If we manually compute what the output should be using the weights for test_edge
        base_out = torch.einsum("ei,eoi->eo", torch.nn.functional.silu(x_j), conv.base_weight[[test_edge]])
        
        from models.kan import _b_spline_basis
        spline_basis = _b_spline_basis(x_j, conv.grid[[test_edge]], conv.spline_order)
        spline_out = torch.einsum("eib,eoib->eo", spline_basis, conv.spline_weight[[test_edge]])
        
        expected_out = base_out + spline_out
        
        assert torch.allclose(out, expected_out, atol=1e-5), f"Edge {test_edge} computation mismatched! Cross-talk detected."
