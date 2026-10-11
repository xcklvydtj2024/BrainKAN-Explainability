import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
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
        # Fix einsum issue here
        spline_out = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        return base_out + spline_out

def compute_dfl(model, eval_points):
    t_values = eval_points.unsqueeze(1)
    with torch.no_grad():
        y_pred = model(t_values).numpy().flatten()
    
    t_np = t_values.numpy().flatten()
    ss_tot = np.sum((y_pred - np.mean(y_pred))**2)
    if ss_tot < 1e-6:
        return 0.0
    
    A = np.vstack([t_np, np.ones(len(t_np))]).T
    w, b = np.linalg.lstsq(A, y_pred, rcond=None)[0]
    y_lin = w * t_np + b
    dfl = ss_res / ss_tot
    return float(np.clip(dfl, 0.0, 1.0))

def f_true(x):
    return 1.0 * x + 0.5 * torch.sin(2 * x)

def train_synthetic_model(X, Y, grid_range):
    model = SyntheticEdgeKAN(grid_range=grid_range)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for _ in range(100): # Ensure convergence
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = F.mse_loss(out, batch_y)
            loss.backward()
            optimizer.step()
    return model

def main():
    print("=== Experiment 2: Synthetic Calibration Trial ===")
    
    N_SIMULATIONS = 500
    N_SAMPLES = 1000
    NOISE = 0.05
    results = []
    
    for i in range(N_SIMULATIONS):
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        
        # Two input distributions
        X_A = torch.randn(N_SAMPLES, 1) # N(0, 1)
        X_B = torch.randn(N_SAMPLES, 1) * 0.5 + 1.0 # N(1, 0.5)
        
        Y_A = f_true(X_A) + torch.randn(N_SAMPLES, 1) * NOISE
        Y_B = f_true(X_B) + torch.randn(N_SAMPLES, 1) * NOISE
        
        # Grid range for spline setup (must cover both to avoid extrapolation failure)
        grid_range = (-3.0, 3.0)
        
        model_A = train_synthetic_model(X_A, Y_A, grid_range)
        model_B = train_synthetic_model(X_B, Y_B, grid_range)
        
        # --- Exp2-A: Fixed function, Fixed uniform grid, Different input training distribution ---
        eval_grid_uniform_common = torch.linspace(-1.0, 2.0, 100) # Region where both have data
        dfl_A_common = compute_dfl(model_A, eval_grid_uniform_common)
        dfl_B_common = compute_dfl(model_B, eval_grid_uniform_common)
        results.append({'SubExp': 'Exp2-A', 'Sim': i, 'DFL_A': dfl_A_common, 'DFL_B': dfl_B_common, 'Delta': dfl_A_common - dfl_B_common})
        
        # --- Exp2-B: Fixed function (model A), Changing reference interval (Uniform grids) ---
        eval_grid_1 = torch.linspace(-2.0, 2.0, 100)
        eval_grid_2 = torch.linspace(0.0, 2.0, 100)
        dfl_A_grid1 = compute_dfl(model_A, eval_grid_1)
        dfl_A_grid2 = compute_dfl(model_A, eval_grid_2)
        results.append({'SubExp': 'Exp2-B', 'Sim': i, 'DFL_A': dfl_A_grid1, 'DFL_B': dfl_A_grid2, 'Delta': dfl_A_grid1 - dfl_A_grid2})
        
        # --- Exp2-C: Fixed function (model A), Fixed interval [-2, 2], Changing Grid Type (Uniform vs Quantile) ---
        eval_grid_uniform = torch.linspace(-2.0, 2.0, 100)
        quantiles = torch.linspace(0.01, 0.99, 100)
        eval_grid_quantile = torch.quantile(X_A.flatten(), quantiles)
        # Ensure it's bounded similarly
        dfl_A_unif = compute_dfl(model_A, eval_grid_uniform)
        dfl_A_quant = compute_dfl(model_A, eval_grid_quantile)
        results.append({'SubExp': 'Exp2-C', 'Sim': i, 'DFL_A': dfl_A_unif, 'DFL_B': dfl_A_quant, 'Delta': dfl_A_unif - dfl_A_quant})

    df = pd.DataFrame(results)
    
    print("\n=== Summary ===")
    for subexp in sorted(df['SubExp'].unique()):
        sub = df[df['SubExp'] == subexp]
        print(f"[{subexp}]")
        print(f"  Mean DFL A: {sub['DFL_A'].mean():.4f}")
        print(f"  Mean DFL B: {sub['DFL_B'].mean():.4f}")
        print(f"  Mean Delta: {sub['Delta'].mean():+.4f} (± {sub['Delta'].std():.4f})")

if __name__ == "__main__":
    main()
