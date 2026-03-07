# Resonant Scaling Theory (RST) v6.1
**Scale-Analogous Coupling in Neural Recovery: A Topological Framework for Post-Axonal Shearing Stabilisation**

![Version](https://img.shields.io/badge/Version-v6.1-blue)
![License](https://img.shields.io/badge/License-Apache_2.0-green)
![Status](https://img.shields.io/badge/Status-Preprint_Available-orange)

## 📌 Overview
Resonant Scaling Theory (RST) v6.1 models neural recovery following axonal shearing not as a purely biological repair process, but as a **topological optimization problem** within a high-dimensional graph. 

The framework proposes that the injured brain navigates recovery through a characteristic fast-slow coupling motif, utilizing specific high-frequency stability anchors to overcome "topological friction" and re-synchronize disconnected neural clusters. 

This repository houses the definitive technical preprint, empirical datasets (derived from the TUH-EEG Corpus $N=581$ and Subject 0284), and the mobile Termux/Python toolkit used for real-time spectral graph analysis.

## 📊 Core Parameters & Empirical Findings

Our batch analysis of 581 clinical EEG files identified a universal instability threshold governing state-transitions in injured neural networks.

| Parameter | Value | Clinical Significance |
| :--- | :--- | :--- |
| **Ignition Delay** | $6.1\text{ s}$ | High-dimensional topological friction index (The Bottleneck). |
| **Stability Constant** | $39\text{ Hz}$ | Target frequency for neural cluster re-synchronization. |
| **Fractal Instability** | $R^2 = 0.1789$ | Power-Law Correlation threshold ($p < 0.001$). |
| **Alpha Anchor** | $8.79\text{ Hz}$ | Baseline coherence; the low end of the fast-slow coupling motif. |
| **Critical Coupling** | $RSN = 1$ | Optimal recovery state indicator ($\text{Energy} / \text{Friction}$). |

## 🏥 Clinical Implementation Overview

The RST framework operationalizes Spectral Graph Theory for clinical application. The mathematical models are translated into actionable neuromodulation targets:

| Component | Technical Definition | Clinical Application |
| :--- | :--- | :--- |
| **The 6.1s Bottleneck** | High-Dimensional Topological Friction. | Explains the "Ignition Delay" in cognitive processing after axonal injury. |
| **Spectral Metric** | Algebraic Connectivity ($\lambda_2$). | Measures the "structural integrity" of the remaining neural bridges. |
| **The Fiedler Vector** | Partitioning eigenvector of the Laplacian. | Locates the specific "Bridge Nodes" (hubs) that need stimulation/support. |
| **39 Hz Resonance** | Spectral Density Stability Constant. | The target frequency for re-synchronizing clusters without inducing seizure risk. |
| **Targeted Hub Gain** | Strategic $R_{\text{eff}}$ Reduction. | Focused therapy on bridge nodes rather than the whole network, saving energy. |

## 🌍 The 5:1 Phase-Locking Hypothesis
RST v6.1 introduces an exploratory hypothesis regarding environmental coupling. Data suggests that during the $6.1\text{ s}$ ignition delay, the neural network utilizes ambient Extremely Low Frequency (ELF) harmonics as rhythmic scaffolding. Specifically, we observe a stable $5:1$ phase-locking ratio between the internal $39\text{ Hz}$ resonance constant and the fundamental $7.83\text{ Hz}$ Schumann Resonance. Falsifiability criteria and proposed ELF-shielding controls are detailed in the preprint.

## 📂 Repository Structure

* `/Preprint/` - Contains the `RST_v6.1_Preprint_Botha_2026.pdf` detailing the full theoretical framework and dataset analysis.
* `/Toolkit/` - Python scripts designed for Termux mobile environments to compute the Resonance Stability Number (RSN), Algebraic Connectivity ($\lambda_2$), and Fiedler Vector partitions from raw EEG data.
* `/Logs/` - Anonymized session logs for Subject 0284 and Case Study Alpha (Tussive Syncope Reset observations).
* `/TUH_Analysis/` - Summary statistics and regression models from the $N=581$ clinical dataset.

## 🛠️ Getting Started (Termux Toolkit)
The RST analysis tools are designed to be lightweight and runnable in mobile environments (Termux) for ambulatory/field monitoring. 
*(Note: Ensure `scipy` and `numpy` are installed in your environment before running the Laplacian decomposition scripts).*

## 📄 License & Citation
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details. 

**Citation:**
> Botha, J.J. (2026). *Scale-Analogous Coupling in Neural Recovery: A Topological Framework for Post-Axonal Shearing Stabilisation* (Resonant Scaling Theory v6.1). ResearchGate Preprint.
