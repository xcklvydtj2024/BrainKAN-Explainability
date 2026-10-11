import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Ensure output directory exists
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Set high-quality styling
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# ==============================================================================
# Figure 1: Pipeline Overview
# ==============================================================================
def make_figure_1():
    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    ax.axis('off')
    
    stages = [
        {"title": "1. HCP fMRI Data", "desc": "100 Subjects (2-Back vs 0-Back)\nPair-matched fMRI matrices\n240 ROIs / Graph Nodes", "color": "#E3F2FD", "border": "#1976D2"},
        {"title": "2. BrainKAN Network", "desc": "Edge-specific B-spline Convolutions\nLearnable $\\Phi_{ij}(x) = W_{ij}^{base}\\sigma(x) + \\sum w_k B_k(x)$\nGraph message-passing", "color": "#EDE7F6", "border": "#512DA8"},
        {"title": "3. Spline Functions", "desc": "240 Edge Functions $\\Phi_{ij}(t)$\nEvaluated on controlled grid\nBounded probe range", "color": "#E8F5E9", "border": "#388E3C"},
        {"title": "4. DFL Measurement", "desc": "Deviation-From-Linearity\n$\\mathrm{DFL} = 1 - R^2$\nResidual curvature metric", "color": "#FFF3E0", "border": "#F57C00"},
        {"title": "5. Statistical Inference", "desc": "Seed-Marginalized Permutation (SMPT)\n$\\Delta_{\\mathrm{marginal}} = \\mathbb{E}_s[T(Y) - T(\\pi(Y))]$\nCommon-support domain control", "color": "#FCE4EC", "border": "#C2185B"}
    ]
    
    box_w = 0.16
    box_h = 0.65
    y_start = 0.18
    gap = 0.04
    
    for i, s in enumerate(stages):
        x = 0.02 + i * (box_w + gap)
        rect = patches.FancyBboxPatch(
            (x, y_start), box_w, box_h,
            boxstyle="round,pad=0.03,rounding_size=0.02",
            facecolor=s["color"], edgecolor=s["border"], linewidth=2
        )
        ax.add_patch(rect)
        
        ax.text(x + box_w/2, y_start + box_h - 0.10, s["title"],
                ha='center', va='center', weight='bold', color=s["border"], fontsize=12)
        ax.text(x + box_w/2, y_start + box_h/2 - 0.05, s["desc"],
                ha='center', va='center', color="#263238", fontsize=9.5, linespacing=1.4)
        
        # Arrow to next
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + box_w + gap - 0.005, y_start + box_h/2),
                        xytext=(x + box_w + 0.005, y_start + box_h/2),
                        arrowprops=dict(arrowstyle="-|>", color="#546E7A", lw=2.5, mutation_scale=15))
            
    fig.suptitle("Figure 1: End-to-End BrainKAN Interpretability & Statistical Diagnostic Pipeline",
                 fontsize=14, weight='bold', y=0.95, color="#1A237E")
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig1_pipeline.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 2: Reference Domain Effect (Exp 2 - The Killer Fig)
# ==============================================================================
def make_figure_2():
    fig = plt.figure(figsize=(14, 5), dpi=300)
    gs = GridSpec(1, 3, width_ratios=[1.2, 1.2, 0.9], wspace=0.3)
    
    # Mathematical ground truth function
    x_dense = np.linspace(-3.5, 3.5, 500)
    def true_f(x):
        return 1.2 * np.tanh(1.2 * x) + 0.15 * x
    
    y_dense = true_f(x_dense)
    
    # Domain A (Centered: [-1.2, 1.2])
    dom_a = np.linspace(-1.2, 1.2, 100)
    y_a = true_f(dom_a)
    p_a = np.polyfit(dom_a, y_a, 1)
    lin_a = np.polyval(p_a, dom_a)
    r2_a = 1.0 - np.sum((y_a - lin_a)**2) / np.sum((y_a - np.mean(y_a))**2)
    dfl_a = 1.0 - r2_a
    
    # Domain B (Shifted & Extended: [-0.5, 3.0])
    dom_b = np.linspace(-0.5, 3.0, 100)
    y_b = true_f(dom_b)
    p_b = np.polyfit(dom_b, y_b, 1)
    lin_b = np.polyval(p_b, dom_b)
    r2_b = 1.0 - np.sum((y_b - lin_b)**2) / np.sum((y_b - np.mean(y_b))**2)
    dfl_b = 1.0 - r2_b
    
    # Panel A: Centered Domain
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x_dense, y_dense, color='#90A4AE', lw=1.5, ls='--', label='Underlying function')
    ax1.plot(dom_a, y_a, color='#1E88E5', lw=3, label='Evaluated span')
    ax1.plot(dom_a, lin_a, color='#E53935', lw=2, ls=':', label=f'Linear fit (DFL={dfl_a:.4f})')
    ax1.axvspan(-1.2, 1.2, color='#E3F2FD', alpha=0.5)
    ax1.set_title("A. Reference Domain 1\n(Central Linear-Dominant Regime)", fontsize=11, weight='bold')
    ax1.set_xlabel("Input feature $x$")
    ax1.set_ylabel("Spline output $y$")
    ax1.set_ylim(-1.8, 1.8)
    ax1.legend(loc='upper left', frameon=True, fontsize=9)
    ax1.text(0, -1.5, f"Measured DFL: {dfl_a:.4f}", ha='center', weight='bold', color='#1E88E5',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1E88E5'))
    
    # Panel B: Shifted Domain
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x_dense, y_dense, color='#90A4AE', lw=1.5, ls='--', label='Underlying function')
    ax2.plot(dom_b, y_b, color='#7B1FA2', lw=3, label='Evaluated span')
    ax2.plot(dom_b, lin_b, color='#E53935', lw=2, ls=':', label=f'Linear fit (DFL={dfl_b:.4f})')
    ax2.axvspan(-0.5, 3.0, color='#F3E5F5', alpha=0.5)
    ax2.set_title("B. Reference Domain 2\n(Shifted into Saturation Regime)", fontsize=11, weight='bold')
    ax2.set_xlabel("Input feature $x$")
    ax2.set_ylabel("Spline output $y$")
    ax2.set_ylim(-1.8, 1.8)
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    ax2.text(1.2, -1.5, f"Measured DFL: {dfl_b:.4f}", ha='center', weight='bold', color='#7B1FA2',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5', edgecolor='#7B1FA2'))
    
    # Panel C: Exp 2 Calibration Results
    ax3 = fig.add_subplot(gs[2])
    conditions = ["Exp 2-A\nFixed Domain\n+ Shift", "Exp 2-B\nDomain Shift\nAlone", "Exp 2-C\nDensity Grid\nBias"]
    deltas = [0.0002, 0.1195, 0.0190]
    errors = [0.0049, 0.0106, 0.0031]
    colors = ['#43A047', '#D32F2F', '#FB8C00']
    
    bars = ax3.bar(conditions, deltas, yerr=errors, capsize=5, color=colors, width=0.55, edgecolor='#263238', lw=1.2)
    ax3.axhline(0, color='black', lw=1)
    ax3.set_ylabel("Apparent $\\Delta_{\\mathrm{DFL}}$ (Artifactual Effect)")
    ax3.set_title("C. Exp 2 Synthetic Calibration\n(Zero Ground-Truth Change)", fontsize=11, weight='bold')
    ax3.set_ylim(-0.01, 0.14)
    
    for bar, val in zip(bars, deltas):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.012, f"+{val:.4f}",
                 ha='center', va='bottom', weight='bold', fontsize=9.5)
        
    ax3.annotate("Massive Artifact!\n(100x task effect)", xy=(1, 0.12), xytext=(1.2, 0.08),
                 arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5),
                 color="#D32F2F", weight='bold', fontsize=9.5)
    
    fig.suptitle("Figure 2: Reference Domain Effect — Demonstrating That Nonlinearity Is Domain-Dependent",
                 fontsize=14, weight='bold', y=0.98, color="#1A237E")
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig2_reference_domain_effect.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 3: Seed Effect (Exp 5 vs Exp 1)
# ==============================================================================
def make_figure_3():
    csv_path = r"D:\BrainKNN\results\exp5_seed_stability\seed_stability_final.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback accurate data from Exp 5
        df = pd.DataFrame({
            'Seed': [42, 100, 1234, 1337, 2023, 2048, 5678, 777, 888, 9999],
            'T_real': [0.17895, 0.18108, 0.17787, 0.17836, 0.17832, 0.17795, 0.17964, 0.18041, 0.18028, 0.18107],
            'T_null_mean': [0.17723, 0.18138, 0.17807, 0.17923, 0.17939, 0.17993, 0.18070, 0.18086, 0.18172, 0.18126],
            'Delta_task': [0.00172, -0.00029, -0.00019, -0.00088, -0.00107, -0.00198, -0.00105, -0.00045, -0.00144, -0.00019]
        })
        
    df = df.sort_values(by='Seed').reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300, gridspec_kw={'width_ratios': [2.2, 1]})
    
    x = np.arange(len(df))
    colors = ['#D32F2F' if s == 42 else '#1976D2' for s in df['Seed']]
    
    # Left: Seed-by-seed Delta
    bars = ax1.bar(x, df['Delta_task'], color=colors, width=0.55, edgecolor='#263238', lw=1.2)
    ax1.axhline(0, color='black', lw=1.2, ls='--')
    
    mean_delta = df['Delta_task'].mean()
    sd_delta = df['Delta_task'].std()
    ci_lower = mean_delta - 1.96 * (sd_delta / np.sqrt(len(df)))
    ci_upper = mean_delta + 1.96 * (sd_delta / np.sqrt(len(df)))
    
    ax1.axhline(mean_delta, color='#00897B', lw=2, ls='-', label=f'Marginal Mean $\\bar{{\\Delta}} = {mean_delta:.5f}$')
    ax1.axhspan(ci_lower, ci_upper, color='#80CBC4', alpha=0.3, label='95% Confidence Interval')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Seed {s}" for s in df['Seed']], rotation=30, ha='right')
    ax1.set_ylabel("Excess Nonlinearity $\\Delta_{\\mathrm{task}} = T_{\\mathrm{real}} - T_{\\mathrm{null}}$")
    ax1.set_title("A. Excess Nonlinearity $\\Delta$ Across 10 Independent Initialization Seeds", fontsize=11, weight='bold')
    ax1.legend(loc='lower left', frameon=True)
    ax1.set_ylim(-0.0025, 0.0025)
    
    # Highlight Seed 42
    s42_idx = list(df['Seed']).index(42)
    ax1.annotate("Seed 42 (Only positive seed):\n$p=0.006$ conditional significance\nDoes NOT generalize to population!",
                 xy=(s42_idx, df.loc[s42_idx, 'Delta_task']),
                 xytext=(s42_idx + 0.8, 0.0019),
                 arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5),
                 color="#D32F2F", weight='bold', fontsize=9)
    
    # Right: Summary Box / Violin
    ax2.boxplot(df['Delta_task'], patch_artist=True,
                boxprops=dict(facecolor='#E0F2F1', color='#00695C', lw=1.5),
                medianprops=dict(color='#D81B60', lw=2),
                whiskerprops=dict(color='#00695C', lw=1.5),
                capprops=dict(color='#00695C', lw=1.5))
    ax2.scatter(np.ones(len(df)) + np.random.normal(0, 0.04, len(df)), df['Delta_task'],
                color=colors, s=60, edgecolors='black', zorder=4)
    ax2.axhline(0, color='black', lw=1, ls='--')
    ax2.set_xticks([1])
    ax2.set_xticklabels(["10-Seed Distribution"])
    ax2.set_title("B. Seed Marginalization Summary", fontsize=11, weight='bold')
    ax2.set_ylim(-0.0025, 0.0025)
    
    summary_text = (
        f"Mean: {mean_delta:.5f}\n"
        f"95% CI: [{ci_lower:.5f}, {ci_upper:.5f}]\n"
        "Positives: 1/10 (Seed 42)\n"
        r"$p_{\mathrm{marginal}} = 0.9513$"
    )
    ax2.text(1, -0.0022, summary_text,
             ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFDE7', edgecolor='#FBC02D'))
    
    fig.suptitle("Figure 3: Optimization Variability — Single-Seed Inference Fails Under Seed Marginalization",
                 fontsize=14, weight='bold', y=0.98, color="#1A237E")
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig3_seed_variability.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==============================================================================
# Figure 4: Final Methodological Framework (Bad vs Good Practice)
# ==============================================================================
def make_figure_4():
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    ax.axis('off')
    
    # Title
    fig.suptitle("Figure 4: Methodological Framework — Pitfalls vs. Recommended Practice for KAN Explainability",
                 fontsize=14, weight='bold', y=0.95, color="#1A237E")
    
    # Left Card: Bad Practice
    card_bad = patches.FancyBboxPatch(
        (0.04, 0.10), 0.43, 0.78,
        boxstyle="round,pad=0.03,rounding_size=0.03",
        facecolor="#FFEBEE", edgecolor="#C62828", linewidth=2.5
    )
    ax.add_patch(card_bad)
    
    ax.text(0.255, 0.82, "PITFALL: Conventional Practice\n(Prone to Evaluation Artifacts)",
            ha='center', va='center', weight='bold', color="#C62828", fontsize=13)
    
    bad_steps = [
        "1. Single Random Seed (e.g. Seed 42)\n   Optimizes into a specific loss basin;\n   cannot separate task effect from seed bias.",
        "2. Empirical / Variable Domain\n   Evaluates splines on shifted data quantiles;\n   causes massive DFL inflation (Exp 2-B: +0.1195).",
        "3. Uncalibrated Point Estimates\n   Treats raw DFL as biological curvature;\n   ignores intrinsic network capacity (~0.179)."
    ]
    
    for i, step in enumerate(bad_steps):
        y = 0.68 - i * 0.16
        rect = patches.FancyBboxPatch((0.07, y - 0.05), 0.37, 0.12,
                                     boxstyle="round,pad=0.02,rounding_size=0.015",
                                     facecolor="#FFFFFF", edgecolor="#EF9A9A", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(0.09, y, step, ha='left', va='center', color="#B71C1C", fontsize=9.5, linespacing=1.3)
        
    ax.text(0.255, 0.16, "CONSEQUENCE:\nFalse Biological Interpretability Claim\n($p=0.006$ condition does not generalize)",
            ha='center', va='center', weight='bold', color="#B71C1C", fontsize=10.5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFCDD2', edgecolor='#C62828', lw=1.5))
    
    # Right Card: Good Practice
    card_good = patches.FancyBboxPatch(
        (0.53, 0.10), 0.43, 0.78,
        boxstyle="round,pad=0.03,rounding_size=0.03",
        facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2.5
    )
    ax.add_patch(card_good)
    
    ax.text(0.745, 0.82, "RECOMMENDED: Practical Evaluation Protocol\n(Robust & Invariant Measurement)",
            ha='center', va='center', weight='bold', color="#2E7D32", fontsize=13)
    
    good_steps = [
        "1. Seed Marginalization (SMPT)\n   Averages across S seeds: $\\bar{T} = \\frac{1}{S}\\sum T_s$;\n   marginalizes out initialization stochasticity.",
        "2. Common Support Domain Bounding\n   Restricts evaluation strictly to overlapping support;\n   prevents boundary extrapolation and domain shifts.",
        "3. Paired Label-Permutation Null\n   Controls for network curvature budget (~0.179);\n   tests true marginal excess: $\\mathbb{E}_s[T(Y) - T(\\pi(Y))] > 0$."
    ]
    
    for i, step in enumerate(good_steps):
        y = 0.68 - i * 0.16
        rect = patches.FancyBboxPatch((0.56, y - 0.05), 0.37, 0.12,
                                     boxstyle="round,pad=0.02,rounding_size=0.015",
                                     facecolor="#FFFFFF", edgecolor="#A5D6A7", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(0.58, y, step, ha='left', va='center', color="#1B5E20", fontsize=9.5, linespacing=1.3)
        
    ax.text(0.745, 0.16, "GUARANTEE:\nReproducible, Statistically Honest Benchmark\n(Reveals True Invariants vs Optimization Nuances)",
            ha='center', va='center', weight='bold', color="#1B5E20", fontsize=10.5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#C8E6C9', edgecolor='#2E7D32', lw=1.5))
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig4_methodological_framework.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    print("Generating Figure 1...")
    make_figure_1()
    print("Generating Figure 2...")
    make_figure_2()
    print("Generating Figure 3...")
    make_figure_3()
    print("Generating Figure 4...")
    make_figure_4()
    print("All 4 core figures successfully generated in figures/ directory!")
