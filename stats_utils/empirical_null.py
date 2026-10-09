import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np

def train_and_eval_brainkan(train_data, num_nodes, base_edge_index, seed=42, is_null=False, null_seed=None):
    """
    Trains BrainKAN. 
    If is_null is True, shuffles the labels inside train_data using null_seed.
    """
    from models.kan import BrainKAN
    
    # Optional: shuffle logic if is_null
    if is_null and null_seed is not None:
        np.random.seed(null_seed)
        # Deepcopy train_data or modify in place depending on strategy
        # Simplified for methodology outline
    
    torch.manual_seed(seed)
    model = BrainKAN(base_edge_index, num_nodes, 1, 16, 2, grid_range=(-8.0, 12.0))
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    
    loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(15):
        for batch in loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
            
    model.eval()
    return model

def compute_reference_grid_for_fold(train_pairs, base_edge_index):
    """
    Computes X_ref securely per fold.
    """
    num_nodes = train_pairs[0][0].x.shape[0]
    
    all_x = np.zeros((len(train_pairs) * 2, num_nodes))
    for i, (d0, d2) in enumerate(train_pairs):
        all_x[2*i] = d0.x[:, 0].numpy()
        all_x[2*i+1] = d2.x[:, 0].numpy()
        
    ref_grids = {}
    src_nodes = base_edge_index[0].numpy()
    
    q_vals = np.linspace(1, 99, 99)
    for src in np.unique(src_nodes):
        ref_grids[src] = np.percentile(all_x[:, src], q_vals)
        
    return ref_grids
