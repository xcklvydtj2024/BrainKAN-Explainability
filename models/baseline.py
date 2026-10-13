"""
Baseline GNN architectures (GCN and GAT) for the BrainKAN benchmark.
Provides canonical GCNBaseline and GATBaseline models matching experiments/00_baseline_models.py,
with aliases GCNClassifier and GATClassifier for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class GCNBaseline(nn.Module):
    """
    Standard 2-layer GCN baseline matching experiments/00_baseline_models.py.
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 16, num_classes: int = 2, num_nodes: int = 48, out_channels: int = None):
        super(GCNBaseline, self).__init__()
        # Handle out_channels alias if passed
        out_dim = out_channels if out_channels is not None else num_classes
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_dim)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class GATBaseline(nn.Module):
    """
    Standard 2-layer GAT baseline matching experiments/00_baseline_models.py.
    """
    def __init__(self, in_channels: int = 1, hidden_channels: int = 8, heads: int = 2, num_classes: int = 2, num_nodes: int = 48, out_channels: int = None):
        super(GATBaseline, self).__init__()
        out_dim = out_channels if out_channels is not None else num_classes
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.lin = nn.Linear(hidden_channels, out_dim)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


# Aliases for backward compatibility
GCNClassifier = GCNBaseline
GATClassifier = GATBaseline
