import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_data_loading import load_real_hcp_data
from models.kan import BrainKAN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

class GCNBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCNBaseline, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

class GATBaseline(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4):
        super(GATBaseline, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.lin = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

def train_and_eval(model_class, dataset_pairs, epochs=15):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(dataset_pairs))
    
    accuracies = []
    
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        train_pairs = [dataset_pairs[i] for i in train_idx]
        test_pairs = [dataset_pairs[i] for i in test_idx]
        
        train_data = [d for pair in train_pairs for d in pair]
        test_data = [d for pair in test_pairs for d in pair]
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        torch.manual_seed(42 + fold_i)
        if model_class == GCNBaseline:
            model = GCNBaseline(train_data[0].x.shape[1], 16)
        elif model_class == GATBaseline:
            model = GATBaseline(train_data[0].x.shape[1], 8, heads=2)
        elif model_class == 'GCNBaseline_Large':
            model = GCNBaseline(train_data[0].x.shape[1], 512)
        elif model_class == 'GATBaseline_Large':
            model = GATBaseline(train_data[0].x.shape[1], 128, heads=4)
            
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        
        for _ in range(epochs):
            model.train()
            for batch in train_loader:
                opt.zero_grad()
                out = model(batch.x, batch.edge_index, batch=batch.batch)
                loss = F.cross_entropy(out, batch.y)
                loss.backward()
                opt.step()
                
        model.eval()
        correct = 0
        total = 0
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            
        accuracies.append(correct / total)
        
    return accuracies

def main():
    print("Loading Real HCP Data...")
    res = load_real_hcp_data()
    dataset_pairs = res[0]
    
    print("Evaluating GCN...")
    gcn_accs = train_and_eval(GCNBaseline, dataset_pairs, epochs=15)
    
    print("Evaluating GAT...")
    gat_accs = train_and_eval(GATBaseline, dataset_pairs, epochs=15)

    print("Evaluating GCN (Large)...")
    gcn_large_accs = train_and_eval('GCNBaseline_Large', dataset_pairs, epochs=15)
    
    print("Evaluating GAT (Large)...")
    gat_large_accs = train_and_eval('GATBaseline_Large', dataset_pairs, epochs=15)
    
    print(f"GCN Mean Accuracy: {np.mean(gcn_accs):.3f}")
    print(f"GAT Mean Accuracy: {np.mean(gat_accs):.3f}")
    print(f"GCN_Large Mean Accuracy: {np.mean(gcn_large_accs):.3f}")
    print(f"GAT_Large Mean Accuracy: {np.mean(gat_large_accs):.3f}")
    
    df = pd.DataFrame({
        'Fold': [1, 2, 3, 4, 5],
        'GCN_Accuracy': gcn_accs,
        'GAT_Accuracy': gat_accs,
        'GCN_Large_Accuracy': gcn_large_accs,
        'GAT_Large_Accuracy': gat_large_accs
    })
    df.to_csv(os.path.join(ARTIFACTS_DIR, "baseline_accuracies.csv"), index=False)
    print("Saved baseline accuracies to results/baseline_accuracies.csv")
    
    # Save parameter counts
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    num_nodes = 48
    base_edge_index = dataset_pairs[0][0].edge_index
    gcn = GCNBaseline(1, 16)
    gat = GATBaseline(1, 8, heads=2)
    gcn_large = GCNBaseline(1, 512)
    gat_large = GATBaseline(1, 128, heads=4)
    brainkan = BrainKAN(base_edge_index, num_nodes=num_nodes, in_channels=1, hidden_dim=16, num_classes=2)
    
    df_params = pd.DataFrame({
        'Model': ['GCN', 'GAT', 'GCN_Large', 'GAT_Large', 'BrainKAN'],
        'Parameters': [count_parameters(gcn), count_parameters(gat), count_parameters(gcn_large), count_parameters(gat_large), count_parameters(brainkan)]
    })
    df_params.to_csv(os.path.join(ARTIFACTS_DIR, "model_parameter_counts.csv"), index=False)
    print("Saved model_parameter_counts.csv")

if __name__ == "__main__":
    main()
