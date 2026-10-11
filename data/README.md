# BrainKAN Connectomics Data Documentation

This directory contains the processed task-fMRI graph representations used in the BrainKAN Interpretability Benchmark.

---

## 1. Neuroimaging Dataset & Cohort Selection

- **Source Dataset:** Human Connectome Project (HCP) Young Adult S1200 release.
- **Cohort Composition:** $N = 100$ unrelated healthy young adult participants (ages 22–35 years, 54 female / 46 male) who completed the N-back Working Memory (WM) task fMRI paradigm.
- **Task Paradigm:** Block-design working-memory protocol contrasting:
  - **2-Back Condition (Label $y=1$):** High cognitive load working-memory monitoring and updating.
  - **0-Back Condition (Label $y=0$):** Low cognitive load perceptual target-matching baseline.
- **Preprocessing Pipeline:** Data were preprocessed using the HCP Minimal Preprocessing Pipeline (Glasser et al., *NeuroImage* 2013), including:
  1. Motion correction and distortion correction (TOPUP/EDDY).
  2. B0 unwarping and high-pass temporal filtering (200s cutoff).
  3. MSMAll areal feature-based cortical surface registration.
  4. Spatial ICA-FIX automated artifact removal.

---

## 2. Parcellation Atlas & Node Feature Extraction

- **Cortical Atlas:** HCP Multi-Modal Parcellation (HCP-MMP1.0, Glasser et al., *Nature* 2016).
- **Region Count:** $N_{\mathrm{ROIs}} = 48$ bilateral cortical areas selected across 5 canonical functional brain networks:
  - **Visual Network (14 ROIs):** Early visual and dorsal/ventral stream areas (V1, V2, V3, V4, V6, V8, MST).
  - **Somatomotor Network (10 ROIs):** Primary motor and primary sensory cortices (Area 4, 3a, 3b, 1, 2).
  - **Frontoparietal Control Network (8 ROIs):** Dorsolateral prefrontal regions engaged in executive control (Area 8a, 8b, 9p, 9m).
  - **Default Mode Network (6 ROIs):** Ventromedial and anterior prefrontal hubs (Area 10v, 10d, 11l).
  - **Cingulo-Opercular Network (10 ROIs):** Salience and cognitive monitoring hubs (Area a24, p24, 25, s32, a32pr).
  *(Full mapping and anatomical descriptors provided in `roi_labels.tsv`).*
- **Node Features ($X$):** Block-averaged percent BOLD signal change relative to baseline fixation during each respective task condition, yielding a feature vector $x \in \mathbb{R}^{48 \times 1}$ per graph.

---

## 3. Graph Topology & Edge Construction

- **Topology Type:** Directed 5-nearest-neighbor (5-NN) graph derived from group-average resting-state functional connectivity (rs-fMRI Pearson correlation matrix across the 48 ROIs).
- **Edge Count:** $|E| = 48 \times 5 = 240$ directed functional edges.
- **Neurobiological Demarcation:** 
  > **Important Note on Edge Interpretation:** Edges in this benchmark represent **functional co-activation pathways** (statistical associations in BOLD dynamics), *not* structural white-matter diffusion tractography. Message passing simulates non-linear information routing between functionally correlated cortical areas.

---

## 4. Data Files & Sequential Pairing Invariants

The directory contains three core artifacts:
1. `HCP_GNN_features.npz`:
   - `X`: Tensor of shape `[200, 48, 1]` containing 100 subject pairs.
   - `y`: Array of shape `[200]` with binary labels ($0$ for 0-Back, $1$ for 2-Back).
2. `GNN_Edges.pt`:
   - PyTorch tensor containing `edge_index_A` of shape `[2, 240]`.
3. `roi_labels.tsv`:
   - Complete metadata table mapping ROI indices 0–47 to anatomical names, HCP-MMP1 labels, and network affiliations.
4. `subject_pairing_manifest.csv`:
   - Sample pairing manifest recording pair index (0–99), anonymized benchmark ID (`sub-HCP100-001` through `sub-HCP100-100`), sample indices, condition labels, ROI/edge counts, and 5-fold CV test partition assignments.
   - *Provenance Scope:* Upstream raw HCP Subject IDs and scan session manifests are maintained in restricted/controlled storage pursuant to HCP Open Access Data Use Terms. This manifest documents the sequential pairing, dimensional integrity, and cross-validation splits used within the benchmark.

### Strict Sample Pairing & Loader Validation
The data loader (`utils_data_loading.py`) explicitly validates each sample against `subject_pairing_manifest.csv`:
- Verifies that indices `[2*i]` and `[2*i + 1]` match the recorded subject pair $i$.
- Confirms condition labels strictly pair $\{0, 1\}$ (0-Back vs. 2-Back).
- Validates node dimensions ($48$) and edge counts ($240$).
- Validates cross-validation test fold assignments against deterministic runtime `KFold(n_splits=5, shuffle=True, random_state=42)`, ensuring partition consistency.
- Cross-validation splits split by subject pairs, preventing cross-condition leakage.
- Paired label permutations apply within-subject sign flips $\pi_i \in \{0, 1\}$, preserving anatomical features and graph topology under the null.
