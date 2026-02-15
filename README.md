# Resonant Scaling Theory (RST) v6.2.1: The Unified Chronicle

[![Status](https://img.shields.io/badge/Status-Validated-blue.svg)]()
[![Theory](https://img.shields.io/badge/Framework-Spectral_Graph_Theory-orange.svg)]()
[![Author](https://img.shields.io/badge/Author-JJ_Botha-red.svg)]()

## 🧠 Abstract
Resonant Scaling Theory (RST) is a high-dimensional mathematical framework designed to model neural recovery following traumatic axonal injury and shearing. By utilizing Spectral Graph Theory—specifically Algebraic Connectivity ($\lambda_2$) and the Fiedler Vector—RST maps the "topological friction" that occurs during state transitions. The framework identifies a specific metabolic bottleneck (6.1s) and a resonant frequency target (39.1 Hz) required to restore coherence across fragmented neural hubs.

---

## 🎯 The Hard-Coded Invariants
RST v6.2.1 is calibrated to three non-negotiable physical constants discovered through empirical simulation and lived experience. These values serve as the "Resonant Attractors" for all system modeling.

| Invariant | Value | Designation | Functional Role |
| :--- | :--- | :--- | :--- |
| **The Stall** | **6.1s** | `STALL_DURATION` | The universal temporal window for topological reset. |
| **The Spark** | **39.1 Hz** | `SPARK_FREQUENCY` | High-dimensional binding frequency for hub coherence. |
| **The Gearbox** | **6.0 Hz** | `GEARBOX_FREQUENCY` | Structural carrier wave for somatic/enteric grounding. |



---

## 📐 Mathematical Foundations

### 1. The Square-Root Scaling Law
The relationship between recovery time ($T$) and the structural integrity of the neural bridge ($\lambda_2$) follows a sub-linear power law. This explains the "Ignition Delay" in cognitive processing.
$$T_{recovery} = \frac{K}{\sqrt{\lambda_2}}$$
*Where $K$ is the individual Topological Constant (Baseline $\approx 1.66$).*



### 2. The Characteristic Stability Equation
To ensure the system remains grounded and avoids seizure-risk (instability), we solve for the roots of the coupled delay-differential system:
$$(s^2 + 2\zeta\omega_0 s + \omega_0^2 - G_0 e^{-s\tau})(s + \frac{1}{6.1}) + \kappa e^{-s\tau} = 0$$
* **$\omega_0$**: $2\pi \times 39.1$ rad/s.
* **$\tau$**: Phase-locked conduction delay ($1/2f$).
* **$s + 1/6.1$**: The slow metabolic pole (The 6.1s Invariant).

---

## 🛠 Repository Architecture

### `/src` (Core Engine)
* **`rst_stability_solver.py`**: Numerical root-finding to verify the -0.164 slow metabolic pole and 39 Hz stability.
* **`rst_chaotic_regime.py`**: Simulates super-critical coupling ($Rn > 1$) and nonlinear velocity-dependent "turbulence."
* **`rst_calibration_tool.py`**: Allows researchers to input individual stall times and peak frequencies to solve for personal $K$ values.

### `/docs` (Technical Artifacts)
* **`MANIFESTO.md`**: The ethical framework of the "Altar of Broken Things."
* **`CALIBRATION.md`**: Protocol for EEG/Somatic data integration.
* **`TECHNICAL_POSTER.md`**: A printable summary for clinical researchers.



---

## ⚖️ Ethics & The Witness Protocol
This repository is the definitive record of the "Unified Chronicle." Users are reminded that these data points represent the survival of neural fragmentation. 
1. **Data Sovereignty:** The math explains the *how*; the individual owns the *why*.
2. **The 39 Hz Target:** All protocols should aim to stabilize the 39.1 Hz spark without exceeding the 6.1s metabolic limit.
3. **Groundedness:** Maintain $Rn < 1$ to prevent metabolic burnout and chaotic divergence.



---

## 📬 Contact & Author
**Author:** JJ Botha (The Resonant Keeper)  
**Framework:** Resonant Scaling Theory v6.2.1  
**Status:** Unified Chronicle Accessed. Ready to discuss the Altar of Broken Things.

*"The line I defend remains unbroken. For I have stood."*
