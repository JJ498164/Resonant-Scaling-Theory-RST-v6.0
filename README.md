# Resonant Scaling Theory (RST) v6.1
**Technical Specification & Empirical Validation: Subject 0284**

## Executive Summary
RST v6.1 is a framework for modeling and navigating state-transition bottlenecks in damaged neural topologies, specifically post-axonal shearing. It utilizes a **Discrete Multiscale Coupling Motif** where a stable low-frequency scaffold (Alpha) modulates a high-frequency resonant driver (Gamma) to reduce topological friction ($\tau$).

## 1. Core Mathematical Constants
Validation against dataset `0284_032_032_EEG.mat` establishes the following system parameters:

| Parameter | Value | Definition |
| :--- | :--- | :--- |
| **$f_{\alpha}$** | 8.79 Hz | Systemic Anchor / Carrier Frequency |
| **$f_{\gamma}$** | 39.06 Hz | Resonant Driver / Bridge Frequency |
| **Ratio** | 4.44 : 1 | Non-integer Quasi-Periodic Alignment |
| **$\lambda_2$** | 0.1789 | Algebraic Connectivity (Baseline) |
| **$R^2$** | 0.1789 | Critical Instability Index (Phase-Transition Boundary) |
| **$\tau_{bottleneck}$**| 6.1s | Observed State-Transition Delay |

## 2. The Quasi-Periodic Lock (Drift Recurrence)
Unlike integer-locked harmonics which risk runaway excitation, RST v6.1 utilizes the **4.44:1 frequency ratio**. 

* **Phase-Drift Cycle:** A full relative phase rotation occurs every $1 / |f_{\gamma} - (4 \times f_{\alpha})| \approx 0.256\text{s}$.
* **Implementation:** Intervention bursts are hard-coded to **0.256s** (10 gamma cycles), ensuring that every burst initiates and terminates at the same relative phase offset, maintaining structural stability without inducing seizure risk.

## 3. Implementation Protocol: v6.1-OP

### A. Phase-Targeted Stochastic Ignition
To mitigate localized energy runaway in low-connectivity ($\lambda_2 = 0.1789$) nodes, the system implements a **Sigmoid Gain Ramp** over the first 3 cycles (76.8ms) of the burst:
$$g(t) = \frac{1}{1 + \exp\left(-12 \left(\frac{t}{T_{ramp}} - 0.5\right)\right)}$$
This ensures the energy input rate does not exceed the network's inherent diffusion rate during the initial ignition phase.

### B. Adaptive Connectivity Thresholds
The system operates in two distinct modes governed by real-time $\lambda_2$ monitoring:
1.  **Scaffolding Mode ($\lambda_2 < 0.24$):** Forced Phase-Amplitude Coupling (PAC) locked to the Alpha trough ($\pi$ radians).
2.  **Autonomous Mode ($\lambda_2 \geq 0.24$):** Attenuation of the Alpha scaffold; Gamma resonance achieves independent lock to navigate the 6.1s bottleneck.

## 4. Repository Structure
* `/src`: Functional Python implementation (PSD validation, Sigmoid logic, PAC monitoring).
* `/docs`: Formal System Specifications and Subject 0284 Case Studies.
* `/data`: Sample metadata and verification logs.

---
**License:** Research Use Only / Professional Specification.  
**Author:** JJ Botha
