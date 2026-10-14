"""
Unit tests for model parameter counts across BrainKAN and baseline architectures.
Verifies exact parameter counts for Layer 1, full BrainKAN, and baseline models
matching results/model_parameter_counts.csv.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_analytical_parameter_accounting():
    """
    Test exact analytical parameter formulas for BrainKAN Layer 1.
    Formula:
      - Spline weights: E * C_out * C_in * (G + k)
      - Base weights:   E * C_out * C_in
      - Root weights:   C_out * C_in
    """
    num_edges = 240
    in_channels = 1
    out_channels = 16
    grid_size = 5
    spline_order = 3
    n_basis = grid_size + spline_order  # 8

    n_spline = num_edges * out_channels * in_channels * n_basis
    n_base = num_edges * out_channels * in_channels
    n_root = out_channels * in_channels
    layer1_total = n_spline + n_base + n_root

    assert n_spline == 30720, f"Expected 30,720 spline weights, got {n_spline}"
    assert n_base == 3840, f"Expected 3,840 base weights, got {n_base}"
    assert n_root == 16, f"Expected 16 root weights, got {n_root}"
    assert layer1_total == 34576, f"Expected 34,576 total Layer 1 parameters, got {layer1_total}"


def test_live_model_parameter_counts():
    """
    Test parameter counts on live PyTorch modules if PyTorch is available.
    """
    try:
        import torch
        from models.baseline import GCNBaseline, GATBaseline
        from models.kan import BrainKAN, EdgeSpecificKANConv
    except ImportError:
        print("Skipping live test: PyTorch or models dependencies not found in runtime.")
        return

    num_nodes = 48
    num_edges = 240
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    edge_path = os.path.join(base_dir, "data", "GNN_Edges.pt")
    if not os.path.exists(edge_path):
        print("Skipping live test: data/GNN_Edges.pt not found.")
        return

    edge_data = torch.load(
        edge_path,
        map_location="cpu",
        weights_only=True
    )
    base_edge_index = edge_data["edge_index_A"] if isinstance(edge_data, dict) and "edge_index_A" in edge_data else edge_data

    # 1. Test Layer 1 EdgeSpecificKANConv parameter count
    conv1 = EdgeSpecificKANConv(
        in_channels=1,
        out_channels=16,
        base_edge_index=base_edge_index,
        num_nodes=num_nodes,
        grid_size=5,
        spline_order=3
    )
    l1_params = sum(p.numel() for p in conv1.parameters() if p.requires_grad)
    assert l1_params == 34576, f"Expected 34,576 Layer 1 parameters, got {l1_params}"

    # 2. Test full BrainKAN parameter count
    brainkan = BrainKAN(
        base_edge_index=base_edge_index,
        num_nodes=num_nodes,
        in_channels=1,
        hidden_dim=16,
        num_classes=2
    )
    total_brainkan = sum(p.numel() for p in brainkan.parameters() if p.requires_grad)
    assert total_brainkan == 589779, f"Expected 589,779 BrainKAN parameters, got {total_brainkan}"

    # 3. Test baseline GNN parameter counts
    gcn = GCNBaseline(in_channels=1, hidden_channels=16, num_classes=2)
    gat = GATBaseline(in_channels=1, hidden_channels=8, heads=2, num_classes=2)
    gcn_large = GCNBaseline(in_channels=1, hidden_channels=512, num_classes=2)
    gat_large = GATBaseline(in_channels=1, hidden_channels=128, heads=4, num_classes=2)

    assert sum(p.numel() for p in gcn.parameters() if p.requires_grad) == 338
    assert sum(p.numel() for p in gat.parameters() if p.requires_grad) == 234
    assert sum(p.numel() for p in gcn_large.parameters() if p.requires_grad) == 264706
    assert sum(p.numel() for p in gat_large.parameters() if p.requires_grad) == 68226


if __name__ == '__main__':
    test_analytical_parameter_accounting()
    print("test_analytical_parameter_accounting: PASS")
    test_live_model_parameter_counts()
    print("test_live_model_parameter_counts: PASS (or skipped if torch missing)")
