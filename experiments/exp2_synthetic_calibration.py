import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.kan import _b_spline_basis

ARTIFACTS = PROJECT_ROOT / "results" / "exp2_synthetic"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

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

def compute_dfl(model, eval_points):
    """
    Computes Deviation-From-Linearity (DFL = 1 - R^2 = SS_res / SS_tot).
    Higher values strictly indicate greater nonlinearity.
    """
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
    ss_res = np.sum((y_pred - y_lin)**2)
    
    dfl = ss_res / ss_tot
    return float(np.clip(dfl, 0.0, 1.0))

def f_true(x):
    return 1.0 * x + 0.5 * torch.sin(2.0 * x)

def train_synthetic_model(X, Y, grid_range):
    model = SyntheticEdgeKAN(grid_range=grid_range)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        out = model(X)
        loss = F.mse_loss(out, Y)
        loss.backward()
        optimizer.step()
    return model

def main():
    print("=== Experiment 2: Synthetic Data Calibration (Corrected DFL Metric) ===")
    
    N_SIMULATIONS = 200
    N_SAMPLES = 1000
    NOISE = 0.05
    results = []
    start_time = time.time()
    
    for i in range(N_SIMULATIONS):
        if (i + 1) % 50 == 0:
            print(f"Running simulation {i+1}/{N_SIMULATIONS}...")
            
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        
        # Two input distributions
        X_A = torch.randn(N_SAMPLES, 1) # N(0, 1)
        X_B = torch.randn(N_SAMPLES, 1) * 0.5 + 1.0 # N(1, 0.5)
        
        Y_A = f_true(X_A) + torch.randn(N_SAMPLES, 1) * NOISE
        Y_B = f_true(X_B) + torch.randn(N_SAMPLES, 1) * NOISE
        
        grid_range = (-3.5, 3.5)
        
        model_A = train_synthetic_model(X_A, Y_A, grid_range)
        model_B = train_synthetic_model(X_B, Y_B, grid_range)
        
        # --- Exp2-A: Fixed function, Fixed uniform grid, Different input training distribution ---
        eval_grid_uniform_common = torch.linspace(-1.0, 2.0, 100)
        dfl_A_common = compute_dfl(model_A, eval_grid_uniform_common)
        dfl_B_common = compute_dfl(model_B, eval_grid_uniform_common)
        results.append({
            'SubExp': 'Exp2-A', 'Sim': i,
            'Condition_A': 'Model_A_CommonGrid', 'DFL_A': dfl_A_common,
            'Condition_B': 'Model_B_CommonGrid', 'DFL_B': dfl_B_common,
            'Delta': dfl_A_common - dfl_B_common
        })
        
        # --- Exp2-B: Fixed function (model A), Shifted reference intervals ---
        # Domain 1: [-2.0, 2.0] (symmetric, linear trend dominates)
        # Domain 2: [0.0, 2.0] (restricted half-cycle, curvature dominates)
        eval_grid_1 = torch.linspace(-2.0, 2.0, 100)
        eval_grid_2 = torch.linspace(0.0, 2.0, 100)
        dfl_A_grid1 = compute_dfl(model_A, eval_grid_1)
        dfl_A_grid2 = compute_dfl(model_A, eval_grid_2)
        # Measure domain-induced inflation: DFL(Domain 2) - DFL(Domain 1)
        results.append({
            'SubExp': 'Exp2-B', 'Sim': i,
            'Condition_A': 'Domain_HalfCycle', 'DFL_A': dfl_A_grid2,
            'Condition_B': 'Domain_FullSymmetric', 'DFL_B': dfl_A_grid1,
            'Delta': dfl_A_grid2 - dfl_A_grid1
        })
        
        # --- Exp2-C: Fixed function (model A), Fixed interval [-2, 2], Changing Grid Type ---
        eval_grid_uniform = torch.linspace(-2.0, 2.0, 100)
        quantiles = torch.linspace(0.01, 0.99, 100)
        eval_grid_quantile = torch.quantile(X_A.flatten(), quantiles)
        dfl_A_unif = compute_dfl(model_A, eval_grid_uniform)
        dfl_A_quant = compute_dfl(model_A, eval_grid_quantile)
        results.append({
            'SubExp': 'Exp2-C', 'Sim': i,
            'Condition_A': 'Quantile_Grid', 'DFL_A': dfl_A_quant,
            'Condition_B': 'Uniform_Grid', 'DFL_B': dfl_A_unif,
            'Delta': dfl_A_quant - dfl_A_unif
        })

    df = pd.DataFrame(results)
    out_csv = ARTIFACTS / "synthetic_calibration.csv"
    df.to_csv(out_csv, index=False)
    
    print("\n=== Synthetic Calibration Summary (DFL = SS_res / SS_tot) ===")
    summary_rows = []
    for subexp in sorted(df['SubExp'].unique()):
        sub = df[df['SubExp'] == subexp]
        mean_dfl_a = float(sub['DFL_A'].mean())
        mean_dfl_b = float(sub['DFL_B'].mean())
        mean_delta = float(sub['Delta'].mean())
        std_delta = float(sub['Delta'].std(ddof=1))
        se_delta = std_delta / np.sqrt(len(sub))
        
        print(f"[{subexp}]")
        print(f"  Mean DFL A: {mean_dfl_a:.4f}")
        print(f"  Mean DFL B: {mean_dfl_b:.4f}")
        print(f"  Mean Delta: {mean_delta:+.4f} (± {std_delta:.4f}, SE={se_delta:.4f})")
        summary_rows.append({
            'SubExp': subexp,
            'Mean_DFL_A': mean_dfl_a,
            'Mean_DFL_B': mean_dfl_b,
            'Mean_Delta': mean_delta,
            'Std_Delta': std_delta,
            'SE_Delta': se_delta
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(ARTIFACTS / "synthetic_summary.csv", index=False)
    print(f"\nCompleted in {time.time() - start_time:.2f}s")
    print(f"Saved verified synthetic results to {out_csv}")

if __name__ == "__main__":
    main()
