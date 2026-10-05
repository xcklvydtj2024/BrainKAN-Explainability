import torch
import pandas as pd
import os
from models.baseline import GCNClassifier, GATClassifier
from models.kan import BrainKAN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Initialize models with same parameters as 00_baseline_models.py
    num_nodes = 48
    base_edge_index = torch.cartesian_prod(torch.arange(num_nodes), torch.arange(num_nodes)).T
    
    gcn = GCNClassifier(num_nodes=48, in_channels=1, hidden_channels=16, out_channels=2)
    gat = GATClassifier(num_nodes=48, in_channels=1, hidden_channels=8, heads=2, out_channels=2)
    brainkan = BrainKAN(base_edge_index, num_nodes=48, in_channels=1, hidden_dim=16, num_classes=2)
    
    data = {
        'Model': ['GCN', 'GAT', 'BrainKAN'],
        'Parameters': [count_parameters(gcn), count_parameters(gat), count_parameters(brainkan)]
    }
    
    df = pd.DataFrame(data)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/model_parameter_counts.csv', index=False)
    print("Saved model_parameter_counts.csv:")
    print(df)
