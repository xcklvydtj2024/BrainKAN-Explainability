import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.kan import _b_spline_basis

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "exp2_synthetic")
os.makedirs(ARTIFACTS, exist_ok=True)

class SyntheticEdgeKAN(torch.nn.Module):
    def __init__(self, in_features=1, out_features=1, grid_size=5, spline_order=3, grid_range=(-3, 3)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * step + grid_range[0]
        self.register_buffer('grid', grid.expand(in_features, -1))
        
        self.base_weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.spline_weight = torch.nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order) * 0.1)
        
    def forward(self, x):
        base_out = F.linear(F.silu(x), self.base_weight)
        grid_batch = self.grid.unsqueeze(0).expand(x.size(0), -1, -1)
        basis = _b_spline_basis(x, grid_batch, self.spline_order)
        spline_out = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        return base_out + spline_out

def compute_dfl(model, eval_range, num_points=100):
    t_values = torch.linspace(eval_range[0], eval_range[1], num_points).unsqueeze(1)
    with torch.no_grad():
        y_pred = model(t_values).numpy().flatten()
    
    t_np = t_values.numpy().flatten()
    ss_tot = np.sum((y_pred - np.mean(y_pred))**2)
    if ss_tot < 1e-6:
        return 0.0
    
    A = np.vstack([t_np, np.ones(len(t_np))]).T
    w, b = np.linalg.lstsq(A, y_pred, rcond=None)[0]
    y_lin = w * t_np + b
    ss_res = np.sum((y_pred - y_lin)**2)
    return max(0.0, 1 - (ss_res / ss_tot))

def f0(x):
    return 1.0 * x + 0.5 * torch.sin(2 * x)

def f1_change(x):
    return f0(x) + 0.8 * (x ** 2)

def train_synthetic_model(X, Y, grid_range):
    model = SyntheticEdgeKAN(grid_range=grid_range)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for _ in range(50): # Enough epochs for simple 1D curve fitting
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = F.mse_loss(out, batch_y)
            loss.backward()
            optimizer.step()
    return model

def main():
    print("=== Experiment 2: Synthetic Data Calibration ===")
    
    N_SIMULATIONS = 500
    N_SAMPLES = 1000
    NOISE = 0.2
    
    results = []
    start_time = time.time()
    
    for i in range(N_SIMULATIONS):
        if (i+1) % 50 == 0:
            print(f"Running simulation {i+1}/{N_SIMULATIONS}...")
            
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        
        # P_train = N(0, 1)
        X_train = torch.randn(N_SAMPLES, 1)
        r_min, r_max = -2.5, 2.5 # ~99th percentile of N(0, 1)
        grid_range = (r_min, r_max)
        
        # Scenario 1: Function SAME (Real trains on f0, Null trains on f0)
        Y_null = f0(X_train) + torch.randn(N_SAMPLES, 1) * NOISE
        Y_real_same = f0(X_train) + torch.randn(N_SAMPLES, 1) * NOISE
        
        model_null_same = train_synthetic_model(X_train, Y_null, grid_range)
        model_real_same = train_synthetic_model(X_train, Y_real_same, grid_range)
        
        # Scenario 2: Function CHANGED (Real trains on f1, Null trains on f0)
        Y_real_changed = f1_change(X_train) + torch.randn(N_SAMPLES, 1) * NOISE
        model_real_changed = train_synthetic_model(X_train, Y_real_changed, grid_range)
        
        # Domains
        domain_same = (-2.5, 2.5) # Matches P_train
        domain_shift = (0.0, 3.0) # Matches a shifted P_eval ~ N(1.5, 0.5)
        
        # Evaluate A: Func Same, Dist Same
        t_real = compute_dfl(model_real_same, domain_same)
        t_null = compute_dfl(model_null_same, domain_same)
        results.append({'Scenario': 'A (Func Same, Dist Same)', 'Sim': i, 'T_real': t_real, 'T_null': t_null, 'Delta': t_real - t_null})
        
        # Evaluate B: Func Same, Dist Shift
        t_real = compute_dfl(model_real_same, domain_shift)
        t_null = compute_dfl(model_null_same, domain_shift)
        results.append({'Scenario': 'B (Func Same, Dist Shift)', 'Sim': i, 'T_real': t_real, 'T_null': t_null, 'Delta': t_real - t_null})
        
        # Evaluate C: Func Change, Dist Same
        t_real = compute_dfl(model_real_changed, domain_same)
        t_null = compute_dfl(model_null_same, domain_same)
        results.append({'Scenario': 'C (Func Change, Dist Same)', 'Sim': i, 'T_real': t_real, 'T_null': t_null, 'Delta': t_real - t_null})
        
        # Evaluate D: Func Change, Dist Shift
        t_real = compute_dfl(model_real_changed, domain_shift)
        t_null = compute_dfl(model_null_same, domain_shift)
        results.append({'Scenario': 'D (Func Change, Dist Shift)', 'Sim': i, 'T_real': t_real, 'T_null': t_null, 'Delta': t_real - t_null})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(ARTIFACTS, "synthetic_calibration.csv"), index=False)
    
    print("\n=== Synthetic Calibration Summary ===")
    for scenario in sorted(df['Scenario'].unique()):
        sub_df = df[df['Scenario'] == scenario]
        mean_delta = sub_df['Delta'].mean()
        std_delta = sub_df['Delta'].std()
        
        # Power / False Positive calculation (is Delta > 0 at p<0.05 level? Here we just count how many times Delta > 0.005)
        # We define a threshold for "detection"
        detected = (sub_df['Delta'] > 0.05).mean() * 100
        
        print(f"[{scenario}]")
        print(f"  Mean Delta: {mean_delta:+.4f} (±{std_delta:.4f})")
        print(f"  Detection Rate (Delta > 0.05): {detected:.1f}%")
        
    print(f"\nTotal Time: {time.time() - start_time:.2f}s")
    print(f"Saved artifacts to {ARTIFACTS}/synthetic_calibration.csv")

if __name__ == "__main__":
    main()
