# BrainKAN Explainability: Unmasking the Identifiability Boundary of Edge-Level Nonlinear Functions in task-fMRI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository presents an identifiability-aware framework for probing edge-level nonlinear functions in task-fMRI graph models. Rather than treating learned nonlinearities as direct evidence of biological computation, we explicitly test when such interpretations are supported—and when they are confounded by condition-dependent input regimes.

By developing a common-reference and null-calibrated framework, this research codebase characterizes what aspects of edge-level computation can—and cannot—be identified under the tested task-fMRI setting.

---

## 🌟 The Core Scientific Narrative

```text
Scientific Question
────────────────────────────────────
Does a task-related change in regional activation imply a change in edge-level computation?
                  BrainKAN
                   │
                   ▼
        Explicit edge function Φij(x)
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
       Q1         Q2         Q3A
    Functions   Taxonomy   Curvature shift
                               │
                               ▼
                     Shuffled-label null
                               │
                               ▼
                Same effect without task labels
                               │
                               ▼
             NON-IDENTIFIABILITY OF NAIVE EFFECT
                               │
                               ▼
                  Common-reference Q3B
                               │
                               ▼
                         0 / 240 FDR
                               │
                               ▼
                  No detectable excess
                  task-related geometry
                               │
                               ▼
                     Effective Gain Q3C
                               │
                               ▼
                Input shift ≈ Gain modulation
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

## 📂 Repository Code Structure

The code strictly implements our **progressive falsification pipeline**:

- **`01_q1_q2_q3a_naive_analysis.py`** 
  Establishes function existence (Q1) and tests for clustering/taxonomy (Q2, which prevents biological over-interpretation). Crucially, it demonstrates the initial discovery of "remodeling" via naive $\Delta C$ (Q3A) and implements the **Shuffled-Label Null** that exposes the identifiability limitation.
  
- **`02_q3b_common_reference_geometry.py`** 
  The **Authoritative Pipeline**. Implements the Common-Reference geometry to remove input-shift bias, training 100 independent Null Models with Subject-Level label swaps. Result: **0/240 edges survived FDR correction under the tested common-reference estimand.** No detectable task-associated excess geometry was found under this analysis.

- **`03_synthetic_identifiability_benchmark.py`** 
  A synthetic encoding environment (S0-S5) across varying SNRs. It provides the **empirical Confusability Matrix** used to characterize the tested estimator's identifiability limits.
  
- **`04_q3c_effective_gain.py`** 
  Implements the Effective Gain estimator using a stringent **Subject-level Paired Permutation Test**.

- **`models/`**
  Contains the `EdgeSpecificKANConv` and `BrainKAN` architectural code, utilizing perfectly unified `_b_spline_basis` evaluations to prevent numerical divergence across scales.

- **`utils_data_loading.py`** 
  Handles strict subject-level cross-validation and standardizes the common fMRI tensor pipeline.

## 🔬 Methodology & Statistical Rigor

We applied absolute methodological constraints to prevent false positives:
1. **Statistical Unit**: The atomic unit of inference is the **Subject**, protecting paired nested structures.
2. **Q3A Null Unification (Authoritative Null)**: We use **Within-Subject Label Swapping** (randomly swapping 0BK $\leftrightarrow$ 2BK per subject) to break task association while preserving topological covariance.
3. **Q3B Calibration Alignment**: Identical statistic $T = 1-R^2$ computed strictly using identically built cross-fitted pipelines for $G^{real}$ and $G^{null}$.
4. **Q3C Exchangeability**: Evaluated using a Paired Permutation test that flips the sign of $dG_i^{real} - dG_i^{null}$, which uses subject-level paired sign-flipping under the specified strong-null exchangeability assumption.
5. **Handling Out-of-Range Domains**: `np.interp` silent clamping is strictly blocked; out-of-range evaluations map directly to `NaN` and are transparently excluded, preventing artificial flattenings of curvature.

## ⚖️ Final Interpretational Constraints

Through the lens of this repository, we explicitly restrict the scientific conclusions:
- **We DO NOT claim** BrainKAN discovers new, true biological "working memory mechanistic motifs."
- **We DO NOT claim** effective gain differences ($\Delta G$) necessarily represent synaptic gain modulation, since our Synthetic Benchmark proved it is inherently confusable with input-regime kinematic shifts.
- **We DO claim** that this codebase establishes a critical **Identifiability Boundary**: demonstrating that, under the tested edge-level estimator and task-fMRI observation regime, apparent task-related nonlinear remodeling can be confounded by condition-dependent input operating regimes.

---
*This repository represents a methodological study in Neuro-AI, providing both the tools and the self-falsifying guardrails required to analyze complex, edge-specific neural computations.*
