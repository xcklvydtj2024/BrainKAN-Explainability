import pytest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.kan import BrainKAN

def test_batch_invariance():
    """Test that attention pooling does not induce batch size scaling bugs."""
    # Fully connected
    num_nodes = 48
    edge_index = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes)).T
    
    model = BrainKAN(base_edge_index=edge_index, num_nodes=num_nodes, in_channels=1, hidden_dim=8, num_classes=2)
    model.eval()
    
    # Create single graph
    x_single = torch.randn(num_nodes, 1)
    batch_single = torch.zeros(num_nodes, dtype=torch.long)
    
    # Compute for single graph
    with torch.no_grad():
        out_single = model(x_single, edge_index, batch=batch_single)
        
    # Create batched graph (e.g. batch size = 4)
    B = 4
    x_batched = x_single.repeat(B, 1)
    
    # Shift edge_index for batched blocks
    edge_indices = []
    for i in range(B):
        edge_indices.append(edge_index + i * num_nodes)
    edge_index_batched = torch.cat(edge_indices, dim=1)
    
    # Create batch assignments
    batch_batched = torch.arange(B).repeat_interleave(num_nodes)
    
    # Compute for batched graphs
    with torch.no_grad():
        out_batched = model(x_batched, edge_index_batched, batch=batch_batched)
        
    # The output for each graph in the batch should be identical to the single graph output
    for i in range(B):
        assert torch.allclose(out_single[0], out_batched[i], atol=1e-5), \
            f"Batch invariance failed at index {i}. out_single={out_single[0]}, out_batched={out_batched[i]}"
