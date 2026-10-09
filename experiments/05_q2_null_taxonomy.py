import os
import numpy as np
import torch
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils_data_loading import load_real_hcp_data
from models.kan import BrainKAN
from importlib import import_module

q1_module = import_module("01_q1_q2_q3a_naive_analysis")
extract_layer_curves = q1_module.extract_layer_curves
compute_function_signatures = q1_module.compute_function_signatures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "results")
SEED = 2000
N_SPLITS = 5
T_RANGE = (-8.0, 12.0)
N_PROBE = 500

def train_null_fold(train_pairs, base_edge_index, num_nodes, fold_seed):
    working_pairs = []
    # Null shuffle strategy
    np.random.seed(fold_seed)
    for (d0, d2) in train_pairs:
        d0_y, d2_y = d0.y.clone(), d2.y.clone()
        if np.random.rand() > 0.5:
            d0.y, d2.y = d2_y, d0_y
        working_pairs.append((d0, d2))
        
    train_data = [d for pair in working_pairs for d in pair]
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    torch.manual_seed(fold_seed)
    model = BrainKAN(base_edge_index, num_nodes, 1, 16, 2, grid_range=T_RANGE)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for _ in range(15):
        for batch in train_loader:
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = torch.nn.functional.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
    model.eval()
    return model

def main():
    print("=" * 70)
    print("Q2_Null: Null Function Clusters Taxonomy")
    print("=" * 70)
    
    res = load_real_hcp_data()
    dataset_pairs = res[0]
    base_edge_index = res[1]
    num_nodes = res[2]
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    splits = list(kf.split(dataset_pairs))
    
    t_values = torch.linspace(T_RANGE[0], T_RANGE[1], N_PROBE)
    
    null_models = []
    all_sigs = []
    
    print("\n--- Training 5-fold Null Models ---")
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        print(f"  Training Null Fold {fold_i+1}...")
        train_pairs = [dataset_pairs[i] for i in train_idx]
        model = train_null_fold(train_pairs, base_edge_index, num_nodes, SEED + fold_i)
        null_models.append(model)
        
        with torch.no_grad():
            curves = extract_layer_curves(model.conv1, t_values)
        sigs = compute_function_signatures(curves, t_values)
        all_sigs.append(sigs)
        
    # Evaluate clustering consistency (ARI) across folds
    feature_names = ['nl', 'slope', 'curvature', 'monotonicity', 'turning_points']
    scaler = StandardScaler()
    
    fold_labels_list = []
    for sigs in all_sigs:
        X_fold = np.column_stack([sigs[f] for f in feature_names])
        X_fold_sc = scaler.fit_transform(X_fold)
        clust = AgglomerativeClustering(n_clusters=4, linkage='ward') # Best k from real is 4
        labels = clust.fit_predict(X_fold_sc)
        fold_labels_list.append(labels)
        
    ari_scores = []
    for i in range(len(fold_labels_list)):
        for j in range(i+1, len(fold_labels_list)):
            ari = adjusted_rand_score(fold_labels_list[i], fold_labels_list[j])
            ari_scores.append(ari)
            
    print(f"\nNull Models Cross-Fold Cluster Stability (ARI):")
    print(f"  Null models ARI: {np.mean(ari_scores):.4f} +/- {np.std(ari_scores):.4f}")
    
    df_res = pd.DataFrame({'ARI': ari_scores})
    df_res.to_csv(os.path.join(ARTIFACTS_DIR, "q2_null_ari.csv"), index=False)
    print("Saved Null ARI distribution to results/q2_null_ari.csv")
    
if __name__ == "__main__":
    main()
