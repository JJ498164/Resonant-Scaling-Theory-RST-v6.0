# Resonant Scaling Theory (RST) v5.1
**Author:** JJ Botha  
**Technical Field:** Computational Neuroscience / Network Dynamics  
**Status:** Hypothesis Generation & Preliminary Empirical Testing

---

## 🔬 Framework Abstract
Resonant Scaling Theory (RST) v5.1 is an exploratory mathematical framework designed to model neural recovery following diffuse axonal injury. By applying **Spectral Graph Theory** to EEG data, RST seeks to correlate recovery timescales with the Laplacian spectral properties of neural networks.

## 📊 Preliminary Observations (Feb 2026)

### 1. 8.75 Hz Alpha-Band Suppression (Exploratory N=1)
Initial comparison between a trauma subject (TUH 0284) and a control subject suggests a significant discrepancy in the 8.75 Hz alpha band.
- **Trauma (0284):** 0.00097 NPD
- **Control:** 0.01338 NPD
- **Ratio:** ~13.7x reduction in power.
- **Hypothesis:** This deficit may indicate impaired thalamocortical gating, localized to central-temporal hubs (Channels 7 & 8). 

### 2. The 6.1s Transition Bottleneck
The framework mathematically predicts a 6.1-second "state-transition bottleneck" emerging from high-dimensional topological friction. 
- **Current Status:** Theoretical derivation is complete. Empirical validation across a larger cohort (N≥20) is required to establish statistical significance.

### 3. Re-evaluation of 39 Hz Gamma Activity
Following adversarial testing, a suspected 39 Hz resonance was identified as environmental artifact (Δ = 0.000). 
- **Methodological Note:** This demonstrates the framework's internal falsification protocol. Current efforts are focused on identifying biological gamma signatures in the 30-45 Hz range that are distinct from 60 Hz harmonics.

### Preliminary Local Observation – v6.1 Deep Lite on 0284 test.edf
- File: test.edf (coma recovery segment)
- Output: Signal Lock at 39.1 Hz, Resonance Power 17,381,636.00
- Grounding Check: Complete
- Note: This result on a real coma EEG segment challenges the current falsification of 39 Hz as purely environmental noise. Further testing on full 0284 segments and TUH cases is needed to determine if this is intrinsic gamma persistence or artifact.

## 🛠 Methodology & Metrics

### Spectral Analytics
| Metric | Definition | Clinical Intent |
| :--- | :--- | :--- |
| **λ₂ (Algebraic Connectivity)** | Second smallest eigenvalue of the Laplacian. | Quantifying the structural integrity of neural "bridges." |
| **Fiedler Vector** | Partitioning eigenvector. | Localizing "Bridge Nodes" (Hubs) for potential targeted therapy. |
| **$R_{\text{eff}}$ (Effective Resistance)** | Network-wide resistance metric. | Modeling the energy cost of information transfer post-injury. |

---

## 📂 Repository Contents
- `/src`: Analysis scripts for Laplacian spectral density.
- `/data`: Metadata logs (PhysioNet/TUH). *Note: Raw EEG data is not hosted here due to licensing.*
- `/docs`: Mathematical derivations of the 6.1s bottleneck.

## 🤝 Call for Collaboration
RST v5.1 is currently in the **Hypothesis Generation** phase. I am seeking collaborators to:
1. Validate these findings in larger, multi-subject cohorts.
2. Correlate λ₂ values with clinical outcome scores (GOS-E).
3. Refine the 39 Hz detection algorithm to ensure biological origin.

