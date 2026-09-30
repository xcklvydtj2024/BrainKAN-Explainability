import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from .readout import MultiScaleReadout


def _b_spline_basis(x: torch.Tensor, grid: torch.Tensor, order: int) -> torch.Tensor:
    """Compute B-spline basis values for edge-specific inputs.

    Args:
        x: Input values [E, in_channels]
        grid: Knot vectors [E, in_channels, grid_size + 2 * order + 1]
        order: Spline order (degree)

    Returns:
        Basis values [E, in_channels, grid_size + order]
    """
    x = x.unsqueeze(-1)
    bases = ((x >= grid[..., :-1]) & (x < grid[..., 1:])).float()

    for k in range(1, order + 1):
        left_num = x - grid[..., : -(k + 1)]
        left_den = grid[..., k:-1] - grid[..., : -(k + 1)]
        right_num = grid[..., k + 1 :] - x
        right_den = grid[..., k + 1 :] - grid[..., 1 : (-k if k > 0 else None)]

        left = left_num / (left_den + 1e-8) * bases[..., :-1]
        right = right_num / (right_den + 1e-8) * bases[..., 1:]
        bases = left + right

    return bases


class EdgeSpecificKANConv(MessagePassing):
    """
    Edge-Specific Kolmogorov-Arnold Network Convolution.
    
    This layer allocates independent B-Spline parameters ONLY for the actual
    edges present in the base graph, preventing N^2 parameter explosion.
    
    Mathematical formulation:
        h'_i = Root(h_i) + \sum_{j \in N(i)} A_{ij} * Φ_{ij}(h_j)
    
    where Φ_{ij} is the edge-specific nonlinear transformation matrix.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_edge_index: torch.Tensor,
        num_nodes: int = 48,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-3.0, 3.0),
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.grid_size = grid_size
        self.spline_order = spline_order
        n_basis = grid_size + spline_order

        # 1. Edge ID mapping
        # Create a mapping from (dst, src) -> edge_id [0, E-1]
        self.num_edges = base_edge_index.size(1)
        mapping = torch.full((num_nodes, num_nodes), -1, dtype=torch.long)
        mapping[base_edge_index[1], base_edge_index[0]] = torch.arange(self.num_edges)
        self.register_buffer("edge_mapping", mapping)

        # 2. Allocate edge-specific parameters ONLY for existing edges [E, out, in, basis]
        # This reduces params from O(N^2) to O(E).
        self.spline_weight = nn.Parameter(
            torch.randn(self.num_edges, out_channels, in_channels, n_basis)
            * (1.0 / math.sqrt(in_channels * n_basis))
        )
        self.base_weight = nn.Parameter(
            torch.randn(self.num_edges, out_channels, in_channels)
            * (1.0 / math.sqrt(in_channels))
        )

        # 3. Root connection (self-loop transformation)
        self.root_weight = nn.Parameter(
            torch.randn(out_channels, in_channels)
            * (1.0 / math.sqrt(in_channels))
        )

        # 4. Grid initialization
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid_vals = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer(
            "grid", 
            grid_vals.view(1, 1, -1).expand(self.num_edges, in_channels, -1)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Propagate messages along edges
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
        # Add root (self) connection: Root(h_i)
        root_out = F.linear(F.silu(x), self.root_weight)
        
        return out + root_out

    def message(
        self, 
        x_j: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_weight: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        x_j: Features of the source nodes, shape [E_batch, in_channels]
        edge_index: Graph connectivity, shape [2, E_batch]
        edge_weight: Optional connection strength, shape [E_batch]
        """
        src, dst = edge_index
        
        # PyG batching shifts node IDs (e.g. 0 to B*N-1). 
        # Modulo logic recovers base node IDs to lookup the template edge ID.
        src_base = src % self.num_nodes
        dst_base = dst % self.num_nodes
        
        edge_ids = self.edge_mapping[dst_base, src_base]
        
        if (edge_ids < 0).any():
            raise RuntimeError("Forward graph contains edges not present in the base_edge_index topology.")
        
        # Retrieve edge-specific parameters for this batch
        edge_spline_weight = self.spline_weight[edge_ids]  # [E_batch, out, in, n_basis]
        edge_base_weight = self.base_weight[edge_ids]      # [E_batch, out, in]
        edge_grid = self.grid[edge_ids]                    # [E_batch, out, in, G]

        # Base SiLU function (residual)
        base_out = torch.einsum("ei,eoi->eo", F.silu(x_j), edge_base_weight)

        # B-Spline specific function
        spline_basis = _b_spline_basis(x_j, edge_grid, self.spline_order)
        spline_out = torch.einsum("eib,eoib->eo", spline_basis, edge_spline_weight)

        msg = base_out + spline_out
        
        # Apply physical edge weight modulation if provided
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
            
        return msg


class BrainKAN(nn.Module):
    """
    Graph-level KAN classifier using Edge-Specific KAN convolutions.
    """
    def __init__(
        self,
        base_edge_index: torch.Tensor,
        num_nodes: int = 48,
        in_channels: int = 1,
        hidden_dim: int = 16, # Reduced hidden dim due to high param count per edge
        num_classes: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple = (-3.0, 3.0),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = EdgeSpecificKANConv(
            in_channels, hidden_dim, base_edge_index, num_nodes, grid_size, spline_order, grid_range=grid_range
        )
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.conv2 = EdgeSpecificKANConv(
            hidden_dim, hidden_dim, base_edge_index, num_nodes, grid_size, spline_order, grid_range=grid_range
        )
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.dropout = dropout
        self.readout = MultiScaleReadout(hidden_dim)
        
        # The readout classifier remains a standard feature-specific BSplineKANLinear 
        # because the graph has been pooled. We'll use a simple MLP for the final classification
        # to ensure the focus (and parameter budget) remains on the edge functions.
        self.classifier = nn.Sequential(
            nn.Linear(self.readout.out_channels, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.ln1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.ln2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        h_graph = self.readout(x, batch)
        return self.classifier(h_graph)
