"""
Multi-scale graph readout (pooling) strategies.

Instead of relying on a single global_mean_pool, we combine multiple pooling
perspectives—mean, max, and attention-weighted—to produce a richer graph-level
representation.  This is a standard trick in competitive GNN papers and
significantly boosts representational capacity.

Reference:
    Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019
    Li et al., "DeeperGCN", arXiv:2006.07739
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class AttentionPooling(nn.Module):
    """Soft-attention graph-level readout.

    Each node embedding is scored by a learnable gating vector, then
    aggregated via a weighted sum.  This allows the model to attend to
    the most informative nodes when producing the graph representation.

    Args:
        in_channels: Dimension of node embeddings.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, 1),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted pooling.

        Args:
            x: Node embeddings of shape ``[N, D]``.
            batch: Batch assignment vector of shape ``[N]``.

        Returns:
            Graph-level embeddings of shape ``[B, D]``.
        """
        gate_scores = self.gate(x)  # [N, 1]

        # Softmax within each graph in the batch
        from torch_geometric.utils import softmax
        gate_scores = softmax(gate_scores, batch)

        weighted = x * gate_scores  # [N, D]
        return global_add_pool(weighted, batch)


class MultiScaleReadout(nn.Module):
    """Concatenation of mean, max, and attention-weighted pooling.

    Produces a ``3 * in_channels`` dimensional graph-level embedding
    that captures average activation (mean), extreme activation (max),
    and task-relevant activation (attention).

    Args:
        in_channels: Dimension of node embeddings.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.attention_pool = AttentionPooling(in_channels)
        self.out_channels = 3 * in_channels

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Aggregate node embeddings into graph-level representations.

        Args:
            x: Node embeddings ``[N, D]``.
            batch: Batch assignment ``[N]``.

        Returns:
            Graph embeddings ``[B, 3D]``.
        """
        h_mean = global_mean_pool(x, batch)
        h_max = global_max_pool(x, batch)
        h_att = self.attention_pool(x, batch)
        return torch.cat([h_mean, h_max, h_att], dim=-1)
