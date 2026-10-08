# BrainKAN Explainability: An Identifiability-Aware Framework for Evaluating Edge-Level Nonlinear Functions in Task-fMRI Graph Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **🔔 IMPORTANT:** 
> **For rigorous mathematical formulations, null-model statistical testing, and code mappings, please refer to the [Methodology Whitepaper (PDF)](./BrainKAN_Mathematical_Methodology.pdf).**

This repository presents an identifiability-aware framework for probing edge-level nonlinear functions in task-fMRI graph models. Rather than treating learned nonlinearities as direct evidence of biological computation, we explicitly test when such interpretations are supported—and when they are confounded by condition-dependent input regimes.

By developing a common-reference and null-calibrated framework, this research codebase characterizes what aspects of edge-level computation can—and cannot—be identified under the tested task-fMRI setting.

## 📌 Main Question
**Does a task-related change in regional activation imply a change in edge-level computation?**

## 🎯 Main Findings

| Question | Result | Interpretation |
|----------|--------|----------------|
| **Q1** | 240 edge functions characterized | learned nonlinear functions are heterogeneous |
| **Q2** | near-zero fold ARI (~0.0029) | no stable function taxonomy was observed across folds |
| **Q3A** | widespread significant ΔC | naive curvature is highly task-sensitive |
| **Q3B** | 0/240 FDR | no excess common-reference geometry detected |
| **Q3C** | 11/240 FDR | subset shows detectable effective-gain differences |
| **Synthetic** | confusability demonstrated | input regime shift can mimic computation shift |

## 🌟 The Core Scientific Narrative

```text
                  BrainKAN
                     │
          Explicit edge function Φij(x)
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
         Q1         Q2         Q3A
      Functions   Taxonomy   Curvature shift
                                │
                                ▼
                     Common-reference Q3B
                                │
                                ▼
                          0 / 240 FDR
                                │
                                ▼
                    No detectable excess
                    structural geometry
                                │
                                ▼
                      Effective Gain Q3C
                                │
                                ▼
                    11 / 240 FDR Significant
                                │
                                ▼
                     Synthetic Calibration
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
       What is identifiable?           What is confounded?
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                 IDENTIFIABILITY BOUNDARY
```

## 🧠 Model: EdgeSpecificKANConv
The core architecture is BrainKAN, built on `EdgeSpecificKANConv` which allocates independent B-Spline parameters exclusively for the existing edges of the base graph, preventing parameter explosion.

## 🔬 Experiments & Results

- **Q1**: Functions are highly heterogeneous across edges. We characterize this via Deviation-From-Linearity (DFL).
- **Q2**: No stable function taxonomy was observed across folds (Pairwise ARI ~0.0029). The apparent function taxonomy was not reproducible across folds, arguing against interpreting the learned clusters as stable computational types.
- **Q3A**: Naive evaluation shows massive apparent curvature remodeling, but this is highly sensitive to input shifts.
- **Q3B (Exploratory Calibration)**: Evaluates structural geometry using a common reference grid to isolate the model's learned structure from the input distribution. **0/240 edges survive FDR correction.**
- **Q3C (Held-out Calibrated Comparison)**: Measures effective-gain differences using permutation testing on 5 folds of held-out subjects. **11/240 edges survive FDR correction.** *(Note: Results were qualitatively stable across reasonable numerical differentiation resolutions).*
- **Synthetic**: Demonstrates that simple input shifts can create false computational nonlinearities under a naive estimator.

## 📊 Predictive Adequacy Check

Before interpreting edge functions, we must ensure BrainKAN successfully captures task-related information. It demonstrates stable predictive signal under subject-level cross-validation (N=100 subjects × 2 conditions). We benchmarked BrainKAN against standard graph neural networks simply to ensure it is learning effectively. Baselines are included only as predictive adequacy references rather than capacity-matched comparisons.

| Model | Mean 5-Fold Accuracy | Parameters | Edge-level Interpretability |
|-------|---------------------|------------|-----------------------------|
| **GCN** | ~85.5% | ~300 | No (Node-level only) |
| **GAT** | ~88.0% | ~400 | Attention weights only |
| **BrainKAN** | **~93.0%** | **~590k** | **Explicit edge-level function** $\Phi_{ij}(x)$ |

*Note: The primary goal here is **not predictive supremacy**, but establishing predictive adequacy. The capacity gap makes direct superiority claims unfair.*

## 🚀 Reproducibility

Full reproduction requires substantial compute. You can run all scripts automatically using the provided Makefile.

```bash
make tests      # Run unit tests (requires pytest)
make baseline   # Run baseline GCN/GAT adequacy check
make results    # Re-run all analyses
```

For the exact environment and git commit that generated the currently committed CSV results, see `results/manifest.json`.

## 📂 Data Provenance & Usage

Experiments were conducted using a processed lightweight subset of **HCP-YA working memory task fMRI data** for reproducibility. 

- **Dataset**: Human Connectome Project Young Adult (HCP-YA) Working Memory Task.
- **Subjects**: 100 subjects.
- **Conditions**: 0-back vs 2-back sequential pairing (1 subject = 2 graphs).
- **Atlas / Nodes**: HCP-MMP1 parcellation, reduced to 48 regions.
- **Edges**: 240 directed edges, thresholded by group-level correlation consistency.
- **Node Features**: The features represent average regional BOLD time-series metrics.
- **Preprocessing**: Processed via the minimal preprocessing pipeline, with standard bandpass filtering and confound regression.

Please refer to the HCP Data Use Terms before distributing or re-using the derived tensors in `data/`.

## ⚖️ Limitations & Future Work

- **Null Sample Size**: Q3B currently operates on $R_{null}=100$ due to computational constraints.
- **Factorial Synthetic Benchmark**: Future work should implement a fully factorial mechanism × input-shift benchmark to disentangle interaction effects perfectly.
- **Function-level stability**: Future work should move beyond cluster-level reproducibility (Q2) towards evaluating function-level similarity ($\Phi_e(x)$) across folds using integrated absolute deviation or derivative correlation.
- **Deviation-from-Linearity**: Our defined Nonlinearity (NL) metric ($1 - R^2$) acts as a *deviation-from-linearity (DFL)* index. Since the baseline function includes a `SiLU` activation, an edge with a zero spline component is still technically nonlinear.

## 📚 Citation

```bibtex
@software{qi2026brainkan,
  title = {BrainKAN Explainability: An Identifiability-Aware Framework for Evaluating Edge-Level Nonlinear Functions in Task-fMRI Graph Models},
  author = {Qi, Xin},
  year = {2026},
  url = {https://github.com/xcklvydtj2024/BrainKAN-Explainability}
}
```
