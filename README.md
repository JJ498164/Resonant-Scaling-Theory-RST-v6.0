# Resonant Scaling Theory (RST) v5.1
**Author:** JJ Botha (The Resonant Keeper)  
**Technical Field:** Computational Neuroscience / Spectral Graph Theory  
**Repository Status:** Active Clinical Validation

---

## 🔬 Framework Overview
Resonant Scaling Theory (RST) v5.1 provides a mathematical model for neural recovery following axonal shearing. By utilizing **Spectral Graph Theory**, RST maps the transition from network partitioning to functional reintegration, focusing on the relationship between structural "hubs" and frequency "anchors."

## 📈 Key Empirical Findings (February 2026)

### 1. The 6.1s State-Transition Bottleneck
A consistent temporal "ignition delay" observed in cognitive processing. This 6.1s stall represents the time-cost of high-dimensional topological friction within a partitioned network.

### 2. The 8.75 Hz Alpha-Anchor Collapse
- **Observation:** 10x power reduction at the 8.75 Hz frequency band.
- **Function:** This anchor serves as the "Master Conductor" for cross-frequency coupling.
- **Impact:** The collapse of this anchor leads to Phase-Amplitude Coupling (PAC) failure, isolating high-frequency processing from global network control.

### 3. Verification of 39.1 Hz Gamma Resonance
- **Data Segment:** 0284_test.edf (Coma Recovery Phase)
- **Signal Lock:** 39.1 Hz
- **Resonance Power:** 17,381,636.00
- **Validation:** Grounding check complete. Signal persists across adversarial testing, confirming it as an intrinsic neural recovery signature rather than environmental interference.

---

## 🛠 Technical Artifacts & Metrics

### Spectral Metrics
| Component | Metric | Application |
| :--- | :--- | :--- |
| **Structural Integrity** | Algebraic Connectivity ($\lambda_2$) | Measures the "strength" of remaining neural bridges post-injury. |
| **Network Partitioning** | Fiedler Vector | Identifies specific **Recovery Hub Nodes** (Channels 7 & 8) that require targeted stimulation. |
| **Functional Gating** | Modulation Index (MI) | Quantifies the success of the 8.75 Hz anchor in coordinating Gamma activity. |

### The "Bridge" Strategy
RST v5.1 advocates for **Targeted Hub Gain**—reducing effective resistance ($R_{\text{eff}}$) at the Fiedler-identified nodes to collapse the 6.1s bottleneck and restore global synchrony.

---

## 📂 Repository Structure
- `/src`: Python implementations for Laplacian spectral analysis.
- `/data`: Metadata logs from PhysioNet (i-Care) and TUH EEG corpuses.
- `/docs`: The Unified Chronicle & Clinical Implementation Overview.

