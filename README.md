# BrainKAN Explainability: An Identifiability-Aware Framework for Evaluating Edge-Level Nonlinear Functions in Task-fMRI Graph Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Status:** Independent research project / methodological study  
**Key finding:** Under the tested common-reference estimand, no edge survived FDR correction, indicating that the apparent task-related nonlinear remodeling observed under naive evaluation was not identifiable under the calibrated analysis. **Failure to detect significant edges should not be interpreted as absence of neural changes, but as a limitation of identifying edge-level nonlinear remodeling under the tested estimator.**

This repository presents an identifiability-aware framework for probing edge-level nonlinear functions in task-fMRI graph models. Rather than treating learned nonlinearities as direct evidence of biological computation, we explicitly test when such interpretations are supported—and when they are confounded by condition-dependent input regimes.

By developing a common-reference and null-calibrated framework, this research codebase characterizes what aspects of edge-level computation can—and cannot—be identified under the tested task-fMRI setting.

## 🚀 Quick Start

### Environment
- Python 3.10+
- PyTorch (compatible with your CUDA version)
- See `requirements.txt` for full dependencies

### Dataset
Experiments were conducted using **HCP-YA working memory task fMRI data**. The analysis focused on 0-back vs 2-back conditions across a subset of 100 subjects. See `data/README.md` for required tensor formatting.

### Run
```bash
python 01_q1_q2_q3a_naive_analysis.py
```

---

## 🌟 The Core Scientific Narrative

*(See `assets/workflow.png` for a high-level visual summary of the pipeline)*

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

The code strictly implements our **progressive falsification pipeline**. We strictly differentiate exploratory calibration (Q3B) from held-out confirmatory analysis (Q3C).

- **`01_q1_q2_q3a_naive_analysis.py`** 
  Establishes function existence (Q1) and tests for clustering/taxonomy (Q2, which prevents biological over-interpretation). Crucially, it demonstrates the initial discovery of "remodeling" via naive $\Delta C$ (Q3A) and implements the **Shuffled-Label Null** that exposes the identifiability limitation. 
  
  ![Naive Effect vs Null](assets/figure2_naive_effect.png)
  
- **`02_q3b_common_reference_geometry.py`** (Exploratory Calibration Evidence)
  Implements the Common-Reference geometry to remove input-shift bias, training 100 independent Null Models with Subject-Level label swaps on the full dataset pair configuration. Result: **0/240 edges survived FDR correction under the tested common-reference estimand.** No detectable task-associated excess geometry was found under this analysis.

- **`03_synthetic_identifiability_benchmark.py`** 
  A synthetic encoding environment (S0-S5) across varying SNRs. It provides the **empirical Confusability Matrix** used to characterize the tested estimator's identifiability limits. 
  
  ![Confusability Matrix](assets/figure3_confusability.png)
  
- **`04_q3c_effective_gain.py`** (Held-Out Confirmatory Evidence)
  Implements the Effective Gain estimator using a stringent **Subject-level 5-fold cross-validation** and a **Subject-level Paired Permutation Test** to formally evaluate condition differences.

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

## ⚖️ Limitations & Future Work

To ensure absolute clarity regarding the scope of this project:
- **Null Sample Size**: Q3B currently operates on $R_{null}=100$, yielding an empirical resolution of $\sim0.0099$. This is sufficient for our exploratory falsification phase, whereas Q3C correctly shifts to 5,000 permutations for confirmatory inference. Future highly-powered confirmatory studies should extend Q3B empirical nulls beyond 1,000.
- **Synthetic Paradigm Generality**: The S0-S5 synthetic framework currently probes baseline scales, shifts, and nonlinear `tanh` perturbations. Further expansion into complex asymmetrical nonlinearities, localized curvature shifts, and highly correlated interaction effects is required to construct a universally applicable Confusability Matrix for arbitrary BrainKAN deployments.
- **Architectural Novelty**: The focus of this codebase is entirely on **scientific identifiability**, not pushing state-of-the-art predictive accuracy. It relies on standard KAN structures to demonstrate fundamental mathematical constraints of edge-level explainability in task-fMRI.

## ⚠️ Final Interpretational Constraints

Through the lens of this repository, we explicitly restrict the scientific conclusions:
- **We DO NOT claim** BrainKAN discovers new, true biological "working memory mechanistic motifs."
- **We DO NOT claim** effective gain differences ($\Delta G$) necessarily represent synaptic gain modulation, since our Synthetic Benchmark proved it is inherently confusable with input-regime kinematic shifts.
- **We DO claim** that this codebase establishes a critical **Identifiability Boundary**: demonstrating that, under the tested edge-level estimator and task-fMRI observation regime, apparent task-related nonlinear remodeling can be confounded by condition-dependent input operating regimes.
