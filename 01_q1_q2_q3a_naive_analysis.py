"""
Q1 -> Q2 -> Q3A: Core Neuroscience Analysis
=============================================
Uses cross-fitted BrainKAN models trained under the prespecified analysis protocol (split seed=42; fold initialization seeds=42-46) to answer:

Q1: What do the 240 edge functions actually look like? (Function Characterization)
Q2: Do they form reproducible computational types? (Function Taxonomy)
Q3A: Do 0BK vs 2BK inputs operate in different regimes of these functions?

Layer 1 only (input: 1 -> output: 16). Layer 2 is deferred.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

from utils_data_loading import load_real_hcp_data
from models.kan import BrainKAN, _b_spline_basis

ARTIFACTS = r"D:\BrainKNN\results"
SEED = 42
N_SPLITS = 5
N_PROBE = 500
T_RANGE = (-8.0, 12.0) # Expanded from (-3, 3) to cover empirical range [-6.6, 11.8]


# ============================================================
# Utility: extract full function curves for a conv layer
# ============================================================
def extract_layer_curves(conv, t_values, layer_name="conv1"):
    """
    For each edge, probe the transformation function f_{ij,o,k}(t).
    
    For Layer 1: in_channels=1, out_channels=16
      -> 240 edges x 1 input channel x 16 output channels = 240 x 16 curves
    
    Returns:
        curves: np.array [num_edges, out_channels, n_probe]
            curves[e, o, :] = f_{e, o, k=0}(t) for the single input channel
    """
    num_edges = conv.num_edges
    in_ch = conv.in_channels
    out_ch = conv.out_channels
    n = len(t_values)
    
    # Safety: this function assumes single input channel (Layer 1)
    assert in_ch == 1, f"extract_layer_curves expects in_channels=1 for Layer 1 analysis, got {in_ch}"
    
    curves = np.zeros((num_edges, out_ch, n))
    
    for e in range(num_edges):
        grid_e = conv.grid[e]  # [in_ch, G]
        grid_batch = grid_e.unsqueeze(0).expand(n, -1, -1)  # [n, in_ch, G]
        
        # For layer 1: in_ch=1, so k=0 only
        for k in range(in_ch):
            probe = torch.zeros(n, in_ch)
            probe[:, k] = t_values
            
            basis = _b_spline_basis(probe, grid_batch, conv.spline_order)
            base_out = torch.einsum("Bi,oi->Bo", F.silu(probe), conv.base_weight[e])
            spline_out = torch.einsum("Bik,oik->Bo", basis, conv.spline_weight[e])
            f = (base_out + spline_out).numpy()  # [n, out_ch]
            
            curves[e, :, :] = f.T  # [out_ch, n]
    
    return curves


def compute_function_signatures(curves, t_values):
    """
    For each edge, compute a function signature vector from its curves.
    
    curves: [num_edges, out_channels, n_probe]
    
    Returns dict of per-edge metrics:
        nl: [num_edges] nonlinearity (1-R^2), averaged across output channels
        slope: [num_edges] mean absolute slope
        curvature: [num_edges] mean max|f''|
        monotonicity: [num_edges] fraction of channels that are monotonic
        turning_points: [num_edges] mean number of turning points
        range_ratio: [num_edges] mean (max-min of f) / (max-min of linear fit)
    """
    num_edges, out_ch, n = curves.shape
    t_np = t_values.numpy()
    dt = (t_np[-1] - t_np[0]) / (n - 1)
    
    # Derivative magnitude threshold for turning point detection
    # Ignore sign changes where |f'| < DERIV_THRESH (numerical noise)
    DERIV_THRESH = 0.01
    
    nl = np.zeros(num_edges)
    slope = np.zeros(num_edges)
    curvature = np.zeros(num_edges)
    monotonicity = np.zeros(num_edges)
    turning_points = np.zeros(num_edges)
    active_fraction = np.zeros(num_edges)
    
    for e in range(num_edges):
        e_nl, e_slope, e_curv, e_mono, e_tp = [], [], [], [], []
        active_channels = 0
        
        for o in range(out_ch):
            y = curves[e, o, :]
            if np.max(y) - np.min(y) > 1e-4:
                active_channels += 1
            
            # NL (1 - R^2 of linear fit)
            A = np.vstack([t_np, np.ones(n)]).T
            w, b = np.linalg.lstsq(A, y, rcond=None)[0]
            y_lin = w * t_np + b
            ss_res = np.sum((y - y_lin) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12) if ss_tot > 1e-6 else 1.0
            e_nl.append(max(0, 1 - r2))
            
            # Slope (absolute value of linear fit slope)
            e_slope.append(abs(w))
            
            # Curvature: max|f''|
            dy = np.diff(y) / dt
            d2y = np.diff(dy) / dt
            e_curv.append(np.max(np.abs(d2y)) if len(d2y) > 0 else 0)
            
            # Monotonicity: only consider regions where |f'| > threshold
            dy_significant = dy[np.abs(dy) > DERIV_THRESH]
            if len(dy_significant) > 0:
                is_mono = np.all(dy_significant > 0) or np.all(dy_significant < 0)
            else:
                is_mono = True  # effectively flat
            e_mono.append(float(is_mono))
            
            # Turning points: sign changes in f' (only where |f'| > threshold)
            # Mask out near-zero derivative regions before counting
            dy_masked = dy.copy()
            dy_masked[np.abs(dy_masked) < DERIV_THRESH] = 0
            sign_changes = np.sum(np.abs(np.diff(np.sign(dy_masked[dy_masked != 0]))) > 0) \
                if np.sum(dy_masked != 0) > 1 else 0
            e_tp.append(sign_changes)
        
        nl[e] = np.mean(e_nl)
        slope[e] = np.mean(e_slope)
        curvature[e] = np.mean(e_curv)
        monotonicity[e] = np.mean(e_mono)
        turning_points[e] = np.mean(e_tp)
        active_fraction[e] = active_channels / out_ch
    
    return {
        'nl': nl, 'slope': slope, 'curvature': curvature,
        'monotonicity': monotonicity, 'turning_points': turning_points, 'active_fraction': active_fraction
    }


# ============================================================
# Q1: Function Shape Gallery
# ============================================================
def run_q1(models, base_edge_index, t_values):
    print("\n" + "=" * 70)
    print("Q1: What do the 240 edge functions look like?")
    print("=" * 70)
    
    # Extract curves from all 5 fold models and average
    all_curves = []
    all_sigs = []
    for fold_i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            curves = extract_layer_curves(model.conv1, t_values, "conv1")
        all_curves.append(curves)
        sigs = compute_function_signatures(curves, t_values)
        all_sigs.append(sigs)
        print(f"  Fold {fold_i+1}: curves extracted ({curves.shape})")
    
    # Average signatures across folds
    avg_sigs = {}
    for key in all_sigs[0]:
        avg_sigs[key] = np.mean([s[key] for s in all_sigs], axis=0)
    
    t_np = t_values.numpy()
    
    # ---- Plot 1: NL distribution ----
    metrics = ['nl', 'slope', 'curvature', 'monotonicity', 'turning_points', 'active_fraction']
    titles = ['Nonlinearity (1-R^2)', 'Mean |Slope|', 'Max |f\'\'(t)|',
              'Monotonicity (fraction)', 'Turning Points', 'Active Channels Fraction']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Q1: Layer-1 Edge Function Signatures (240 edges, averaged across 5 folds)", fontsize=14)
    
    for idx, (m, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        ax.hist(avg_sigs[m], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(m)
        ax.set_ylabel("Count")
        ax.axvline(np.mean(avg_sigs[m]), color='red', ls='--', label=f"mean={np.mean(avg_sigs[m]):.3f}")
        ax.legend(fontsize=8)
    
    # Hide unused 6th subplot (now used)
    plt.tight_layout()
    path1 = os.path.join(ARTIFACTS, "q1_signature_distributions.png")
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  [Q1] Signature distributions saved: {path1}")
    
    # ---- Plot 2: Function Gallery (show 12 representative edges, 4 channels each) ----
    # Pick edges spanning the NL range
    nl_sorted = np.argsort(avg_sigs['nl'])
    gallery_indices = np.linspace(0, len(nl_sorted) - 1, 12, dtype=int)
    selected_edges = nl_sorted[gallery_indices]
    
    # Show 4 representative output channels
    show_channels = [0, 4, 8, 12]
    ch_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle("Q1: Edge Function Gallery (Layer 1, 4 output channels)\n"
                 "Sorted by increasing nonlinearity. Thin=per-fold, bold=representative fold 0.", fontsize=13)
    
    for plot_i, edge_id in enumerate(selected_edges):
        ax = axes[plot_i // 3, plot_i % 3]
        src = base_edge_index[0, edge_id].item()
        dst = base_edge_index[1, edge_id].item()
        
        for ch_i, ch in enumerate(show_channels):
            # Per-fold thin lines
            for fold_i in range(len(all_curves)):
                y = all_curves[fold_i][edge_id, ch, :]
                ax.plot(t_np, y, alpha=0.15, color=ch_colors[ch_i], linewidth=0.5)
            
            # Representative Fold (fold 0) bold line (avoiding across-fold channel averaging)
            y_rep = all_curves[0][edge_id, ch, :]
            ax.plot(t_np, y_rep, color=ch_colors[ch_i], linewidth=1.8,
                    label=f"ch{ch}" if plot_i == 0 else None)
        
        ax.set_title(f"Edge {edge_id} (ROI {src}->{dst})\nNL={avg_sigs['nl'][edge_id]:.3f}", fontsize=9)
        ax.set_xlabel("t", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Add legend to first subplot
    axes[0, 0].legend(fontsize=7, loc='upper left')
    
    plt.tight_layout()
    path2 = os.path.join(ARTIFACTS, "q1_function_gallery.png")
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [Q1] Function gallery saved: {path2}")
    
    # ---- Plot 3: 48x48 Edge NL Heatmap ----
    nl_matrix = np.full((48, 48), np.nan)
    for e in range(240):
        src = base_edge_index[0, e].item()
        dst = base_edge_index[1, e].item()
        nl_matrix[dst, src] = avg_sigs['nl'][e]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(nl_matrix, cmap='RdYlBu_r', aspect='equal', interpolation='nearest')
    ax.set_title("Q1: Edge Nonlinearity Heatmap (Layer 1)\nROI(dst) x ROI(src)", fontsize=14)
    ax.set_xlabel("Source ROI")
    ax.set_ylabel("Destination ROI")
    plt.colorbar(im, ax=ax, label="NL (1-R^2)")
    
    path3 = os.path.join(ARTIFACTS, "q1_nl_heatmap.png")
    plt.savefig(path3, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [Q1] NL heatmap saved: {path3}")
    
    # Print summary statistics
    print(f"\n  Q1 Summary Statistics (Layer 1, averaged across 5 folds):")
    for m in metrics:
        vals = avg_sigs[m]
        print(f"    {m:>16s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")
    
    return all_curves, all_sigs, avg_sigs


# ============================================================
# Q2: Function Taxonomy (Clustering)
# ============================================================
def run_q2(all_curves, all_sigs, avg_sigs, base_edge_index, t_values):
    print("\n" + "=" * 70)
    print("Q2: Do edge functions form reproducible computational types?")
    print("=" * 70)
    
    # Build feature matrix from averaged signatures
    # NOTE: range_ratio excluded from primary clustering (unstable near-zero denominator).
    # It is reported separately as supplementary.
    feature_names = ['nl', 'slope', 'curvature', 'monotonicity', 'turning_points']
    X = np.column_stack([avg_sigs[f] for f in feature_names])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different numbers of clusters, pick best silhouette
    sil_scores = {}
    for k in range(2, 8):
        clust = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clust.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        sil_scores[k] = sil
        print(f"  k={k}: silhouette={sil:.4f}")
    
    best_k = max(sil_scores, key=sil_scores.get)
    print(f"\n  Best k={best_k} (silhouette={sil_scores[best_k]:.4f})")
    
    # Final clustering with best k
    clust_final = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    labels_avg = clust_final.fit_predict(X_scaled)
    
    # ---- Cross-fold stability: cluster each fold independently, compare ----
    print("\n  Cross-fold cluster stability:")
    fold_labels_list = []
    for fold_i, sigs in enumerate(all_sigs):
        X_fold = np.column_stack([sigs[f] for f in feature_names])
        X_fold_sc = scaler.transform(X_fold)
        clust_fold = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
        fold_labels = clust_fold.fit_predict(X_fold_sc)
        fold_labels_list.append(fold_labels)
    
    # For each edge, count how many folds assign it to the same cluster as the average
    # (Since cluster IDs may permute, we use a contingency-based approach)
    # Instead: for each pair of folds, compute adjusted Rand index
    from sklearn.metrics import adjusted_rand_score
    ari_scores = []
    for i in range(len(fold_labels_list)):
        for j in range(i+1, len(fold_labels_list)):
            ari = adjusted_rand_score(fold_labels_list[i], fold_labels_list[j])
            ari_scores.append(ari)
    print(f"    Pairwise Adjusted Rand Index: {np.mean(ari_scores):.4f} +/- {np.std(ari_scores):.4f}")
    print(f"    (1.0 = perfect agreement, 0.0 = random)")
    
    # ---- Plot: Cluster profiles ----
    t_np = t_values.numpy()
    cluster_colors = plt.cm.Set1(np.linspace(0, 1, best_k))
    
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, best_k + 1, width_ratios=[1]*best_k + [0.8])
    
    # Top row: representative curves per cluster (output channel 0)
    for c in range(best_k):
        ax = fig.add_subplot(gs[0, c])
        edges_in_c = np.where(labels_avg == c)[0]
        
        # Plot all edges in this cluster (thin lines)
        for e in edges_in_c:
            y = np.mean([curves[e, 0, :] for curves in all_curves], axis=0)
            ax.plot(t_np, y, alpha=0.15, color=cluster_colors[c], linewidth=0.5)
        
        # Plot cluster centroid (use fold 0 curves to avoid inter-fold channel mixing)
        ys = [all_curves[0][e, 0, :] for e in edges_in_c]
        if len(ys) > 0:
            centroid = np.mean(ys, axis=0)
            ax.plot(t_np, centroid, color='black', linewidth=2.5)
        
        ax.set_title(f"Cluster {c} (n={len(edges_in_c)})", fontsize=11, fontweight='bold')
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)
    
    # Top row last column: cluster signature radar/bar
    ax_bar = fig.add_subplot(gs[0, best_k])
    cluster_means = []
    for c in range(best_k):
        edges_in_c = np.where(labels_avg == c)[0]
        means = [np.mean(avg_sigs[f][edges_in_c]) for f in feature_names]
        cluster_means.append(means)
    
    x_pos = np.arange(len(feature_names))
    width = 0.8 / best_k
    for c in range(best_k):
        ax_bar.barh(x_pos + c * width, cluster_means[c], height=width,
                    color=cluster_colors[c], label=f"C{c}", alpha=0.8)
    ax_bar.set_yticks(x_pos + width * (best_k - 1) / 2)
    ax_bar.set_yticklabels(feature_names, fontsize=9)
    ax_bar.set_title("Cluster Profiles\n(standardized)", fontsize=10)
    ax_bar.legend(fontsize=8)
    
    # Bottom row: 48x48 heatmap colored by cluster
    ax_map = fig.add_subplot(gs[1, :])
    cluster_matrix = np.full((48, 48), np.nan)
    for e in range(240):
        src = base_edge_index[0, e].item()
        dst = base_edge_index[1, e].item()
        cluster_matrix[dst, src] = labels_avg[e]
    
    # Custom colormap for discrete clusters
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap_disc = ListedColormap(cluster_colors[:best_k])
    bounds = np.arange(-0.5, best_k + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap_disc.N)
    
    im = ax_map.imshow(cluster_matrix, cmap=cmap_disc, norm=norm, aspect='equal', interpolation='nearest')
    ax_map.set_title("Q2: Edge Function Clusters in Brain Network (ROI x ROI)", fontsize=13)
    ax_map.set_xlabel("Source ROI")
    ax_map.set_ylabel("Destination ROI")
    cbar = plt.colorbar(im, ax=ax_map, ticks=range(best_k))
    cbar.set_label("Cluster ID")
    
    plt.suptitle(f"Q2: Function Taxonomy ({best_k} clusters, Silhouette={sil_scores[best_k]:.3f}, "
                 f"ARI={np.mean(ari_scores):.3f})", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(ARTIFACTS, "q2_function_taxonomy.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  [Q2] Taxonomy plot saved: {path}")
    
    # Print cluster summary
    print(f"\n  Q2 Cluster Summary (k={best_k}):")
    for c in range(best_k):
        edges_in_c = np.where(labels_avg == c)[0]
        profile = {f: np.mean(avg_sigs[f][edges_in_c]) for f in feature_names}
        print(f"    Cluster {c} ({len(edges_in_c):>3d} edges): "
              f"NL={profile['nl']:.3f}  slope={profile['slope']:.3f}  "
              f"curv={profile['curvature']:.3f}  mono={profile['monotonicity']:.2f}  "
              f"TP={profile['turning_points']:.1f}")
    
    return labels_avg, best_k, sil_scores, ari_scores


# ============================================================
# Q3A: Operating Regime Shift (0BK vs 2BK)
# ============================================================
def run_q3a(models, dataset_pairs, base_edge_index, t_values):
    print("\n" + "=" * 70)
    print("Q3A: Do 0BK and 2BK inputs operate in different regimes?")
    print("=" * 70)
    
    t_np = t_values.numpy()
    dt = (t_np[-1] - t_np[0]) / (len(t_np) - 1)
    num_edges = 240
    num_nodes = 48
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    # ---- Pre-check: Layer-1 input range ----
    print("\n  Pre-check: Layer-1 raw input feature range (x_j)...")
    # Per-condition and per-edge out-of-range tracking
    x_vals_0bk_all, x_vals_2bk_all = [], []
    for subj_pair in dataset_pairs:
        x_vals_0bk_all.append(subj_pair[0].x[:, 0].numpy())
        x_vals_2bk_all.append(subj_pair[1].x[:, 0].numpy())
    x_0bk_flat = np.concatenate(x_vals_0bk_all)
    x_2bk_flat = np.concatenate(x_vals_2bk_all)
    all_x_vals = np.concatenate([x_0bk_flat, x_2bk_flat])
    
    x_min, x_max = np.min(all_x_vals), np.max(all_x_vals)
    probe_lo, probe_hi = T_RANGE[0], T_RANGE[1]
    oor_total = (all_x_vals < probe_lo) | (all_x_vals > probe_hi)
    oor_0bk = (x_0bk_flat < probe_lo) | (x_0bk_flat > probe_hi)
    oor_2bk = (x_2bk_flat < probe_lo) | (x_2bk_flat > probe_hi)
    
    pct_outside_total = np.mean(oor_total) * 100
    pct_outside_0bk = np.mean(oor_0bk) * 100
    pct_outside_2bk = np.mean(oor_2bk) * 100
    print(f"    x range: [{x_min:.4f}, {x_max:.4f}]")
    print(f"    mean={np.mean(all_x_vals):.4f}, std={np.std(all_x_vals):.4f}")
    print(f"    Probing range: [{probe_lo:.3f}, {probe_hi:.3f}]")
    print(f"    % outside probing range: total={pct_outside_total:.2f}%, 0BK={pct_outside_0bk:.2f}%, 2BK={pct_outside_2bk:.2f}%")
    
    # Per-edge out-of-range rate
    edge_oor_count = np.zeros(num_edges)
    edge_total_count = np.zeros(num_edges)
    for subj_pair in dataset_pairs:
        for d in subj_pair:
            src_nodes = d.edge_index[0]
            for e in range(num_edges):
                val = d.x[src_nodes[e].item(), 0].item()
                edge_total_count[e] += 1
                if val < probe_lo or val > probe_hi:
                    edge_oor_count[e] += 1
    edge_oor_rate = edge_oor_count / (edge_total_count + 1e-8)
    max_oor_edge = np.argmax(edge_oor_rate)
    print(f"    Max per-edge OOR rate: Edge {max_oor_edge} = {edge_oor_rate[max_oor_edge]*100:.2f}%")
    
    if pct_outside_total > 1.0:
        print("    WARNING: >1% of inputs outside probing range! Consider expanding T_RANGE.")
    else:
        print("    OK: probing interval adequately covers the input distribution.")
    
    # For each edge: collect per-subject paired dC values
    # delta_C[edge_id] = list of (C_2bk - C_0bk) per held-out subject
    delta_C_all = {e: [] for e in range(num_edges)}
    delta_x_all = {e: [] for e in range(num_edges)}
    # Normalized curvature (sensitivity analysis)
    delta_C_norm_all = {e: [] for e in range(num_edges)}
    # Track which fold each subject belongs to
    subj_fold_map = {}
    fold_dC_all = {e: [[] for _ in range(N_SPLITS)] for e in range(num_edges)}
    
    # Also collect operating points for visualization
    op_points_0bk = {e: [] for e in range(num_edges)}
    op_points_2bk = {e: [] for e in range(num_edges)}
    
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(dataset_pairs)):
        model = models[fold_i]
        model.eval()
        conv1 = model.conv1
        
        for si in test_idx:
            subj_fold_map[si] = fold_i
        
        # Precompute f''(t) and range(f) for all edges in this fold's model
        with torch.no_grad():
            curves = extract_layer_curves(conv1, t_values, "conv1")
        
        # Compute |f''(t)| and normalized |f''(t)| / range(f)
        d2f_all = np.zeros((num_edges, len(t_np) - 2))
        d2f_norm_all = np.zeros((num_edges, len(t_np) - 2))
        for e in range(num_edges):
            d2f_channels = []
            d2f_norm_channels = []
            for o in range(curves.shape[1]):  # out_channels
                dy = np.diff(curves[e, o, :]) / dt
                d2y = np.diff(dy) / dt
                abs_d2y = np.abs(d2y)
                d2f_channels.append(abs_d2y)
                # Normalized curvature: |f''| / range(f)
                f_range = np.max(curves[e, o, :]) - np.min(curves[e, o, :]) + 1e-8
                d2f_norm_channels.append(abs_d2y / f_range)
            d2f_all[e] = np.mean(d2f_channels, axis=0)
            d2f_norm_all[e] = np.mean(d2f_norm_channels, axis=0)
        
        t_d2f = t_np[1:-1]
        
        for subj_i in test_idx:
            data_0bk, data_2bk = dataset_pairs[subj_i]
            x_0bk = data_0bk.x
            x_2bk = data_2bk.x
            edge_index = data_0bk.edge_index
            
            # CRITICAL: verify edge ordering matches the base topology
            assert torch.equal(edge_index.cpu(), base_edge_index.cpu()), \
                f"Edge ordering mismatch for subject {subj_i}! Q3A results would be invalid."
            
            src_nodes = edge_index[0]
            
            for e in range(num_edges):
                src = src_nodes[e].item()
                x_val_0bk = x_0bk[src, 0].item()
                x_val_2bk = x_2bk[src, 0].item()
                
                op_points_0bk[e].append(x_val_0bk)
                op_points_2bk[e].append(x_val_2bk)
                delta_x_all[e].append(x_val_2bk - x_val_0bk)
                
                # Primary metric: |f''| (raw curvature)
                c_0bk = np.interp(x_val_0bk, t_d2f, d2f_all[e], left=np.nan, right=np.nan)
                c_2bk = np.interp(x_val_2bk, t_d2f, d2f_all[e], left=np.nan, right=np.nan)
                delta_c = c_2bk - c_0bk
                delta_C_all[e].append(delta_c)
                fold_dC_all[e][fold_i].append(delta_c)
                
                # Sensitivity metric: |f''| / range(f) (normalized curvature)
                cn_0bk = np.interp(x_val_0bk, t_d2f, d2f_norm_all[e], left=np.nan, right=np.nan)
                cn_2bk = np.interp(x_val_2bk, t_d2f, d2f_norm_all[e], left=np.nan, right=np.nan)
                delta_C_norm_all[e].append(cn_2bk - cn_0bk)
        
        print(f"  Fold {fold_i+1}: {len(test_idx)} held-out subjects processed.", flush=True)
    
    # ---- Statistical testing ----
    print("\n  Computing per-edge Wilcoxon signed-rank tests...")
    p_values = np.zeros(num_edges)
    p_values_norm = np.zeros(num_edges)
    mean_delta_C = np.zeros(num_edges)
    median_delta_C = np.zeros(num_edges)
    mean_delta_C_norm = np.zeros(num_edges)
    
    for e in range(num_edges):
        dC = np.array(delta_C_all[e])
        dCn = np.array(delta_C_norm_all[e])
        mean_delta_C[e] = np.mean(dC)
        median_delta_C[e] = np.median(dC)
        mean_delta_C_norm[e] = np.mean(dCn)
    
    mean_delta_x = np.zeros(num_edges)
    for e in range(num_edges):
        mean_delta_x[e] = np.mean(delta_x_all[e])
        
    overall_corr = np.corrcoef(mean_delta_x, mean_delta_C)[0, 1]
    print(f"\n  [Exploratory] Correlation between mean(delta_x) and mean(delta_C) across edges: r={overall_corr:.4f}")
    
    # We need to compute p_values for each edge again!
    for e in range(num_edges):
        dC = np.array(delta_C_all[e])
        dCn = np.array(delta_C_norm_all[e])
        if np.all(dC == 0):
            p_values[e] = 1.0
        else:
            try:
                _, p = wilcoxon(dC, alternative='two-sided')
                p_values[e] = p
            except ValueError:
                p_values[e] = 1.0
        
        if np.all(dCn == 0):
            p_values_norm[e] = 1.0
        else:
            try:
                _, p = wilcoxon(dCn, alternative='two-sided')
                p_values_norm[e] = p
            except ValueError:
                p_values_norm[e] = 1.0
    
    # FDR correction (primary)
    rejected, p_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    n_sig = np.sum(rejected)
    
    # FDR correction (normalized - sensitivity)
    rejected_norm, p_fdr_norm, _, _ = multipletests(p_values_norm, alpha=0.05, method='fdr_bh')
    n_sig_norm = np.sum(rejected_norm)
    
    print(f"\n  Primary |f''|: {n_sig}/{num_edges} edges significant (FDR<0.05)")
    print(f"  Normalized |f''|/range(f): {n_sig_norm}/{num_edges} edges significant (FDR<0.05)")
    
    # Sort by significance
    sig_edges = np.where(rejected)[0]
    sig_edges_sorted = sig_edges[np.argsort(p_fdr[sig_edges])]
    
    print(f"\n  Top significant edges (primary metric, by FDR p-value):")
    print(f"  {'Edge':>5s}  {'ROI':>10s}  {'mean dC':>8s}  {'med dC':>7s}  {'mean dCn':>9s}  {'p_FDR':>8s}  {'p_FDR_n':>8s}")
    for e in sig_edges_sorted[:20]:
        src = base_edge_index[0, e].item()
        dst = base_edge_index[1, e].item()
        print(f"  {e:>5d}  {src:>3d}->{dst:<3d}  {mean_delta_C[e]:>+8.4f}  {median_delta_C[e]:>+7.4f}  "
              f"{mean_delta_C_norm[e]:>+9.4f}  {p_fdr[e]:>8.4f}  {p_fdr_norm[e]:>8.4f}")
    
    # ---- Fold-level robustness for significant edges ----
    print(f"\n  Fold-level robustness (direction consistency for top edges):")
    print(f"  {'Edge':>5s}  {'Fold1':>6s}  {'Fold2':>6s}  {'Fold3':>6s}  {'Fold4':>6s}  {'Fold5':>6s}  {'Consist':>7s}")
    for e in sig_edges_sorted[:10]:
        fold_means = [np.mean(fold_dC_all[e][fi]) for fi in range(N_SPLITS)]
        
        overall_positive = mean_delta_C[e] > 0
        n_consistent = sum((fm > 0) == overall_positive for fm in fold_means if fm != 0)
        vals = '  '.join([f"{fm:>+.3f}" for fm in fold_means])
        print(f"  {e:>5d}  {vals}  {n_consistent}/5")
    
    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Panel 1: Volcano plot (mean dC vs -log10 p_FDR)
    ax = axes[0, 0]
    neg_log_p = -np.log10(p_fdr + 1e-16)
    colors = np.where(rejected, 'crimson', 'gray')
    ax.scatter(mean_delta_C, neg_log_p, c=colors, alpha=0.6, edgecolors='k', linewidth=0.3, s=30)
    ax.axhline(-np.log10(0.05), color='red', ls='--', alpha=0.5, label='FDR=0.05')
    ax.axvline(0, color='black', ls='-', alpha=0.3)
    ax.set_xlabel("Mean delta_C (2BK - 0BK)", fontsize=12)
    ax.set_ylabel("-log10(p_FDR)", fontsize=12)
    ax.set_title(f"Q3A Volcano (primary |f''|)\n{n_sig}/{num_edges} significant", fontsize=13)
    ax.legend()
    
    # Panel 2: Volcano for normalized curvature (sensitivity)
    ax = axes[0, 1]
    neg_log_p_n = -np.log10(p_fdr_norm + 1e-16)
    colors_n = np.where(rejected_norm, 'crimson', 'gray')
    ax.scatter(mean_delta_C_norm, neg_log_p_n, c=colors_n, alpha=0.6, edgecolors='k', linewidth=0.3, s=30)
    ax.axhline(-np.log10(0.05), color='red', ls='--', alpha=0.5, label='FDR=0.05')
    ax.axvline(0, color='black', ls='-', alpha=0.3)
    ax.set_xlabel("Mean delta_C_norm (2BK - 0BK)", fontsize=12)
    ax.set_ylabel("-log10(p_FDR)", fontsize=12)
    ax.set_title(f"Q3A Volcano (normalized |f''|/range)\n{n_sig_norm}/{num_edges} significant", fontsize=13)
    ax.legend()
    
    # Panel 3: Operating point distributions for top edge
    if len(sig_edges_sorted) > 0:
        top_e = sig_edges_sorted[0]
    else:
        top_e = np.argmin(p_fdr)
    
    ax = axes[1, 0]
    ax.hist(op_points_0bk[top_e], bins=25, alpha=0.6, color='dodgerblue', label='0BK', density=True)
    ax.hist(op_points_2bk[top_e], bins=25, alpha=0.6, color='orangered', label='2BK', density=True)
    src_t = base_edge_index[0, top_e].item()
    dst_t = base_edge_index[1, top_e].item()
    ax.set_title(f"Operating Points: Edge {top_e} (ROI {src_t}->{dst_t})\np_FDR={p_fdr[top_e]:.4f}", fontsize=11)
    ax.set_xlabel("Input value (x_j)")
    ax.set_ylabel("Density")
    ax.legend()
    
    # Panel 4: 48x48 heatmap of mean dC
    ax = axes[1, 1]
    dc_matrix = np.full((48, 48), np.nan)
    for e in range(num_edges):
        src = base_edge_index[0, e].item()
        dst = base_edge_index[1, e].item()
        dc_matrix[dst, src] = mean_delta_C[e]
    
    vmax = np.nanmax(np.abs(dc_matrix))
    im = ax.imshow(dc_matrix, cmap='RdBu_r', aspect='equal', interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
    ax.set_title("Mean delta_C (2BK-0BK)\nRed = higher local second-derivative magnitude in 2BK", fontsize=11)
    ax.set_xlabel("Source ROI")
    ax.set_ylabel("Destination ROI")
    plt.colorbar(im, ax=ax, label="Mean delta_C")
    
    for e in sig_edges:
        src = base_edge_index[0, e].item()
        dst = base_edge_index[1, e].item()
        ax.plot(src, dst, 'k*', markersize=4)
    
    plt.suptitle("Q3A: Operating Regime Shift (0BK vs 2BK, Layer 1)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(ARTIFACTS, "q3a_regime_shift.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  [Q3A] Regime shift plot saved: {path}")
    
    return mean_delta_C, p_fdr, rejected, sig_edges_sorted


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("BrainKAN Core Neuroscience Analysis: Q1 -> Q2 -> Q3A")
    print("=" * 70)
    
    dataset_pairs, base_edge_index, num_nodes = load_real_hcp_data()
    t_values = torch.linspace(T_RANGE[0], T_RANGE[1], N_PROBE)
    
    # Train 5-fold models (reusing the same protocol as all previous experiments)
    print("\n--- Training 5-fold BrainKAN models (seed=42) ---")
    models = []
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(dataset_pairs)):
        train_data = [d for i in train_idx for d in dataset_pairs[i]]
        test_data = [d for i in test_idx for d in dataset_pairs[i]]
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        torch.manual_seed(SEED + fold_i)
        np.random.seed(SEED + fold_i)
        model = BrainKAN(base_edge_index, num_nodes, 1, 16, 2, grid_range=T_RANGE)
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        
        model.train()
        for epoch in range(15):
            for batch in train_loader:
                opt.zero_grad()
                out = model(batch.x, batch.edge_index, batch=batch.batch)
                loss = F.cross_entropy(out, batch.y)
                loss.backward()
                opt.step()
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.x, batch.edge_index, batch=batch.batch)
                correct += (out.argmax(1) == batch.y).sum().item()
        acc = correct / len(test_data)
        
        # Verify grid support
        grid_min = model.conv1.grid.min().item()
        grid_max = model.conv1.grid.max().item()
        
        print(f"  Fold {fold_i+1}: Acc={acc*100:.1f}%, Spline Grid=[{grid_min:.2f}, {grid_max:.2f}]")
        models.append(model)
    
    # ---- Q1 ----
    all_curves, all_sigs, avg_sigs = run_q1(models, base_edge_index, t_values)
    
    # ---- Q2 ----
    labels, best_k, sil_scores, ari_scores = run_q2(all_curves, all_sigs, avg_sigs, base_edge_index, t_values)
    
    # ---- Q3A ----
    mean_dC, p_fdr, rejected, sig_edges = run_q3a(models, dataset_pairs, base_edge_index, t_values)
    
    # ---- Final Summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nQ1: 240 edges characterized across {len(avg_sigs['nl'])} function signatures")
    print(f"    NL range: [{np.min(avg_sigs['nl']):.4f}, {np.max(avg_sigs['nl']):.4f}]")
    print(f"    Mean slope: {np.mean(avg_sigs['slope']):.4f}")
    
    print(f"\nQ2: {best_k} function clusters identified")
    print(f"    Silhouette: {sil_scores[best_k]:.4f}")
    print(f"    Cross-fold ARI: {np.mean(ari_scores):.4f} +/- {np.std(ari_scores):.4f}")
    
    n_sig = np.sum(rejected)
    n_pos = np.sum(mean_dC[rejected] > 0) if n_sig > 0 else 0
    n_neg = n_sig - n_pos
    print(f"\nQ3A: {n_sig}/{240} edges with significant operating-regime shift (FDR<0.05)")
    print(f"    {n_pos} edges: 2BK operates in higher-curvature regime")
    print(f"    {n_neg} edges: 0BK operates in higher-curvature regime")
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
