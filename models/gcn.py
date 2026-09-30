"""
Deep Graph Convolutional Network with residual connections and batch
normalisation.

Architecture improvements over the original 2-layer SimpleGCN:

1. **Residual connections** — gradient highways that prevent information
   loss in deeper architectures (He et al., 2016).
2. **Batch normalisation** — stabilises hidden representations across the
   mini-batch, accelerating convergence and acting as a mild regulariser.
3. **Multi-scale readout** — concatenation of mean-pool, max-pool, and
   attention-pool to capture diverse graph-level statistics.
4. **Configurable depth** — default 3 layers (64→64→32) instead of the
   original 2 layers (1→16→16).

Reference:
    Kipf & Welling, "Semi-Supervised Classification with GCNs", ICLR 2017
    Li et al., "DeeperGCN: All You Need to Train Deeper GCNs", 2020
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .readout import MultiScaleReadout


class DeepGCN(nn.Module):
    """Graph Convolutional Network with residual connections.

    Args:
        in_channels: Dimension of input node features.
        hidden_dims: List of hidden-layer widths.  Length determines depth.
        num_classes: Number of output classes.
        dropout: Dropout probability applied after each layer.
        use_residual: Whether to add skip connections (requires matching dims).
        use_batchnorm: Whether to apply BatchNorm after each GCN layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_residual: bool = True,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 32]

        self.use_residual = use_residual
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        # Build GCN backbone
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        dims = [in_channels] + hidden_dims
        for i in range(len(hidden_dims)):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(dims[i + 1]))
            # Projection layer for residual when dimensions change
            if use_residual and dims[i] != dims[i + 1]:
                self.residual_projs.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            else:
                self.residual_projs.append(None)

        # Multi-scale readout + classifier
        self.readout = MultiScaleReadout(hidden_dims[-1])
        self.classifier = nn.Sequential(
            nn.Linear(self.readout.out_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, edge_index, batch=None):
        """Forward pass.

        Args:
            x: Node feature matrix ``[N, D]``.
            edge_index: Edge connectivity ``[2, E]``.
            batch: Batch vector ``[N]``.  Defaults to single-graph batch.

        Returns:
            Class logits ``[B, num_classes]``.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i, conv in enumerate(self.convs):
            identity = x
            x = conv(x, edge_index)
            if self.use_batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection
            if self.use_residual:
                proj = self.residual_projs[i]
                if proj is not None:
                    identity = proj(identity)
                x = x + identity

        # Graph-level readout
        h_graph = self.readout(x, batch)
        return self.classifier(h_graph)
