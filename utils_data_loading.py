import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os

from models.kan import BrainKAN, _b_spline_basis

ARTIFACTS_DIR = r"D:\BrainKNN\results"

def load_real_hcp_data():
    print("Loading Real HCP Data from F:\\hcp_processed...")
    data = np.load(r"D:\BrainKNN\data\HCP_GNN_features.npz")
    X_np = data['X']
    y_np = data['y']
    
    num_samples, num_nodes = X_np.shape
    X = torch.tensor(X_np, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y_np, dtype=torch.long)
    
    edges_data = torch.load(r"D:\BrainKNN\data\GNN_Edges.pt")
    base_edge_index = edges_data['edge_index_A']
    
    dataset = []
    # Group by subject: i-th subject has index 2*i (0bk) and 2*i+1 (2bk)
    num_subjects = num_samples // 2
    for i in range(num_subjects):
        dataset.append((
            Data(x=X[2*i], edge_index=base_edge_index, y=y[2*i]),
            Data(x=X[2*i+1], edge_index=base_edge_index, y=y[2*i+1])
        ))
        
    return dataset, base_edge_index, num_nodes

def extract_nonlinearity(model, t_values):
    """Extracts nonlinearity index (1-R^2) correctly across input & output channels."""
    conv2 = model.conv2
    t_np = t_values.numpy()
    batch_size = len(t_values)
    
    num_edges = conv2.num_edges
    in_channels = conv2.in_channels
    out_channels = conv2.out_channels
    
    nonlinearity_scores = []
    
    for edge_id in range(num_edges):
        spline_weight = conv2.spline_weight[edge_id]
        base_weight = conv2.base_weight[edge_id]
        grid = conv2.grid[edge_id]
        grid_batch = grid.unsqueeze(0).expand(batch_size, -1, -1)
        
        edge_nl_scores = []
        for k in range(in_channels):
            probe_inputs = torch.zeros(batch_size, in_channels)
            probe_inputs[:, k] = t_values
            
            spline_basis = _b_spline_basis(probe_inputs, grid_batch, conv2.spline_order)
            base_out = torch.einsum("Bi,oi->Bo", F.silu(probe_inputs), base_weight)
            spline_out = torch.einsum("Bik,oik->Bo", spline_basis, spline_weight)
            msg = base_out + spline_out # [batch, out_channels]
            
            for o in range(out_channels):
                y = msg[:, o].numpy()
                
                # Linear fit
                A = np.vstack([t_np, np.ones(len(t_np))]).T
                w, b = np.linalg.lstsq(A, y, rcond=None)[0]
                y_pred = w * t_np + b
                
                ss_res = np.sum((y - y_pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                if ss_tot > 1e-6:
                    r2 = 1 - (ss_res / ss_tot)
                else:
                    r2 = 1.0 # Flat line
                    
                edge_nl_scores.append(1 - r2)
                
        # Aggregate nonlinearity for the edge across all input-output channel pairs
        nonlinearity_scores.append(np.mean(edge_nl_scores))
        
    return np.array(nonlinearity_scores)

def plot_scatter(nl_init, nl_trained):
    plt.figure(figsize=(8, 8))
    plt.scatter(nl_init, nl_trained, alpha=0.6, edgecolors='k', color='royalblue')
    
    # y=x reference line
    min_val = min(nl_init.min(), nl_trained.min()) - 0.05
    max_val = max(nl_init.max(), nl_trained.max()) + 0.05
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y = x (No change)')
    
    plt.xlabel('Initial Nonlinearity ($NL^{init}$)', fontsize=14)
    plt.ylabel('Trained Nonlinearity ($NL^{trained}$)', fontsize=14)
    plt.title('Learning-Induced Edge Specialization\n($\Delta NL = NL^{trained} - NL^{init}$)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    out_path = os.path.join(ARTIFACTS_DIR, "learning_induced_nl_scatter.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

def main():
    print("=== Formal Experiment (Phase A & B) ===")
    dataset_pairs, base_edge_index, num_nodes = load_real_hcp_data()
    
    # Subject-level 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold = 1
    t_values = torch.linspace(-3, 3, 100)
    
    for train_idx, test_idx in kf.split(dataset_pairs):
        print(f"\n--- Phase A: Fold {fold}/5 ---")
        # Flatten the pairs for PyG DataLoader
        train_data = [d for i in train_idx for d in dataset_pairs[i]]
        test_data = [d for i in test_idx for d in dataset_pairs[i]]
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        model = BrainKAN(
            base_edge_index=base_edge_index,
            num_nodes=num_nodes,
            in_channels=1,
            hidden_dim=16,
            num_classes=2
        )
        
        if fold == 1:
            print("[Phase B] Extracting Random Init NL for Fold 1...")
            model.eval()
            with torch.no_grad():
                nl_init = extract_nonlinearity(model, t_values)
                
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        model.train()
        for epoch in range(15):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch=batch.batch)
                loss = F.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        # Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.x, batch.edge_index, batch=batch.batch)
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
        acc = correct / len(test_data)
        print(f"   Test Accuracy (Fold {fold}): {acc:.4f}")
        
        if fold == 1:
            print("\n[Phase B] Extracting Trained NL for Fold 1...")
            with torch.no_grad():
                nl_trained = extract_nonlinearity(model, t_values)
            
            # Identify highest induced non-linearity
            delta_nl = nl_trained - nl_init
            top_edges = np.argsort(delta_nl)[-5:][::-1]
            bottom_edges = np.argsort(delta_nl)[:5]
            
            print("\n>>> Top 5 edges with highest Learning-Induced Nonlinearity (ΔNL > 0):")
            for e in top_edges:
                src, dst = base_edge_index[0, e].item(), base_edge_index[1, e].item()
                print(f"   Edge ID {e} (ROI {src} -> ROI {dst}): Init {nl_init[e]:.4f} -> Trained {nl_trained[e]:.4f} (ΔNL = {delta_nl[e]:.4f})")
                
            print("\n>>> Top 5 edges with highest Learning-Suppressed Nonlinearity (ΔNL < 0):")
            for e in bottom_edges:
                src, dst = base_edge_index[0, e].item(), base_edge_index[1, e].item()
                print(f"   Edge ID {e} (ROI {src} -> ROI {dst}): Init {nl_init[e]:.4f} -> Trained {nl_trained[e]:.4f} (ΔNL = {delta_nl[e]:.4f})")
                
            print("\n[Phase B] Generating Scatter Plot...")
            out_path = plot_scatter(nl_init, nl_trained)
            print(f"Scatter plot successfully saved to: {out_path}")
            
        fold += 1
        
    print("\n🎉 Phase A & Phase B Experiment Complete!")

if __name__ == "__main__":
    main()
