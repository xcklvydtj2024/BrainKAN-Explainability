import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import time
from scipy.interpolate import LSQUnivariateSpline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "results")

def generate_data(n, snr, mechanism, strength, input_shift=False):
    X_0 = np.random.randn(n) * 1.0
    shift_val = 1.5 if input_shift else 0.0
    X_2 = np.random.randn(n) * 1.0 + shift_val
    
    def f_base(x): return np.tanh(x)
    
    Y_0_clean = f_base(X_0)
    
    if mechanism == 'S0': Y_2_clean = f_base(X_2)
    elif mechanism == 'S1': Y_2_clean = f_base(X_2)
    elif mechanism == 'S2': Y_2_clean = (1.0 + strength) * f_base(X_2)
    elif mechanism == 'S3': Y_2_clean = f_base(X_2) / (1.0 + strength)
    elif mechanism == 'S4': Y_2_clean = f_base(X_2) * (1.0 - strength)
    elif mechanism == 'S5': Y_2_clean = f_base(X_2) + strength * (X_2**2)
    
    Y_0 = Y_0_clean + np.random.randn(n) / snr
    Y_2 = Y_2_clean + np.random.randn(n) / snr
    
    return X_0, Y_0, X_2, Y_2

def fit_spline(X, Y):
    idx = np.argsort(X)
    X_s, Y_s = X[idx], Y[idx]
    t = np.linspace(-5, 5, 7)[1:-1]
    try:
        spl = LSQUnivariateSpline(X_s, Y_s, t, k=3, ext=0)
    except:
        spl = np.poly1d(np.polyfit(X_s, Y_s, 1))
    return spl

def compute_G(f, X_ref):
    # Compute integral of (f'(x))^2 over X_ref
    eps = 1e-4
    derivs = (f(X_ref + eps) - f(X_ref - eps)) / (2 * eps)
    return np.mean(derivs**2)

def evaluate_sim(args):
    n, snr, mech, strength, rep, input_shift = args
    import hashlib
    seed_str = f"{mech}_{n}_{snr}_{strength}_{rep}_{input_shift}"
    np.random.seed(int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32))
    
    X_0, Y_0, X_2, Y_2 = generate_data(n, snr, mech, strength, input_shift=input_shift)
    
    f_0 = fit_spline(X_0, Y_0)
    f_2 = fit_spline(X_2, Y_2)
    
    # Common support distance D_e
    X_cap_min = max(np.min(X_0), np.min(X_2))
    X_cap_max = min(np.max(X_0), np.max(X_2))
    if X_cap_min < X_cap_max:
        X_cap = np.linspace(X_cap_min, X_cap_max, 50)
        D_e = np.max(np.abs(f_0(X_cap) - f_2(X_cap)))
    else:
        D_e = 0.0
        
    # Structural Nonlinearity G and Delta G
    # X_0 is treated as the reference domain
    X_ref = np.linspace(np.min(X_0), np.max(X_0), 100)
    G_0 = compute_G(f_0, X_ref)
    G_2 = compute_G(f_2, X_ref)
    delta_G = G_2 - G_0
        
    return {
        'n': n, 'snr': snr, 'mech': mech, 'strength': strength, 'rep': rep,
        'input_shift': input_shift,
        'D_e': D_e,
        'delta_G': delta_G
    }

def main():
    print("Starting full grid search for Synthetic Identifiability Benchmark...")
    mechanisms = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
    SNRs = [0.5, 1, 2, 5]
    Ns = [50, 200, 1000, 5000]
    strengths = [0.2, 0.5, 1.0] # Weak, Med, Strong
    reps = 10
    input_shifts = [False, True]
    
    tasks = []
    for mech in mechanisms:
        for snr in SNRs:
            for n in Ns:
                for s in strengths:
                    for rep in range(reps):
                        for shift in input_shifts:
                            tasks.append((n, snr, mech, s, rep, shift))
                        
    print(f"Total simulations to run: {len(tasks)}")
    
    t0 = time.time()
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(evaluate_sim, tasks)
        
    df = pd.DataFrame(results)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(ARTIFACTS_DIR, "synthetic_confusability_results.csv"), index=False)
    print(f"Finished in {time.time()-t0:.1f} seconds. Saved to CSV.")

if __name__ == "__main__":
    main()
