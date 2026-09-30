"""
Multi-Head Graph Attention Network with residual connections.

Architecture improvements over the original single-head, 2-layer BrainGAT:

1. **Multi-head attention** — 4 independent attention heads per layer,
   each learning a distinct notion of neighbourhood importance.  Outputs
   are concatenated (layer 1) or averaged (final layer), following
   Veličković et al. (2018).
2. **Residual connections** — skip connections around each GAT block to
   stabilise gradient flow and allow deeper architectures.
3. **ELU activation** — preserves negative gradients (unlike ReLU), which
   empirically helps attention-based GNNs.
4. **Native attention extraction** — attention weights from every head of
   every layer are cached during forward passes and can be retrieved for
   explainability analysis without any external explainer.

Reference:
    Veličković et al., "Graph Attention Networks", ICLR 2018
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

from .readout import MultiScaleReadout


class MultiHeadGAT(nn.Module):
    """Multi-head Graph Attention Network for graph classification.

    Args:
        in_channels: Dimension of input node features.
        hidden_dim: Per-head hidden dimension.
        num_heads: List of attention heads per layer (length = depth).
        num_classes: Number of output classes.
        dropout: Dropout probability.
        use_residual: Whether to use residual connections.
        add_self_loops: Whether GAT adds self-loops internally.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 16,
        num_heads: Optional[List[int]] = None,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_residual: bool = True,
        add_self_loops: bool = False,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = [4, 4]

        self.use_residual = use_residual
        self.dropout = dropout
        self.num_layers = len(num_heads)

        self.convs = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        # Layer 1: in_channels → hidden_dim * heads[0]  (concatenate heads)
        self.convs.append(
            GATConv(
                in_channels,
                hidden_dim,
                heads=num_heads[0],
                concat=True,
                add_self_loops=add_self_loops,
                dropout=dropout,
            )
        )
        in_after_l1 = hidden_dim * num_heads[0]
        if use_residual and in_channels != in_after_l1:
            self.residual_projs.append(nn.Linear(in_channels, in_after_l1, bias=False))
        else:
            self.residual_projs.append(None)

        # Intermediate + final layers
        for layer_idx in range(1, len(num_heads)):
            is_last = layer_idx == len(num_heads) - 1
            in_dim = hidden_dim * num_heads[layer_idx - 1]
            out_dim = hidden_dim
            concat = not is_last  # Average heads on the last layer

            self.convs.append(
                GATConv(
                    in_dim,
                    out_dim,
                    heads=num_heads[layer_idx],
                    concat=concat,
                    add_self_loops=add_self_loops,
                    dropout=dropout,
                )
            )
            out_after = out_dim * num_heads[layer_idx] if concat else out_dim
            if use_residual and in_dim != out_after:
                self.residual_projs.append(nn.Linear(in_dim, out_after, bias=False))
            else:
                self.residual_projs.append(None)

        # Readout + classifier
        final_node_dim = hidden_dim  # last layer averages heads
        self.readout = MultiScaleReadout(final_node_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.readout.out_channels, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        # Cache for attention weights (populated during forward)
        self._attention_weights: List[torch.Tensor] = []

    @property
    def attention_weights(self) -> List[torch.Tensor]:
        """Return cached attention weights from the most recent forward pass.

        Returns:
            List of tensors, one per layer.  Each has shape ``[E, H]`` where
            ``E`` is the number of edges and ``H`` the number of heads.
        """
        return self._attention_weights

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass with optional attention weight extraction.

        Args:
            x: Node features ``[N, D]``.
            edge_index: Graph edges ``[2, E]``.
            batch: Batch vector ``[N]``.
            return_attention: If ``True``, returns attention weights as the
                second element of the output tuple.

        Returns:
            ``logits`` or ``(logits, attention_list)``
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        self._attention_weights = []

        for i, conv in enumerate(self.convs):
            identity = x
            x, (edge_idx_out, alpha) = conv(
                x, edge_index, return_attention_weights=True
            )
            self._attention_weights.append(alpha)
            x = F.elu(x)

            if self.use_residual:
                proj = self.residual_projs[i]
                if proj is not None:
                    identity = proj(identity)
                x = x + identity

        h_graph = self.readout(x, batch)
        logits = self.classifier(h_graph)

        if return_attention:
            return logits, self._attention_weights
        return logits
