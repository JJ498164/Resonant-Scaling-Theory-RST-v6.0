# Resonant Scaling Theory (RST) v6.1: The Empirical Pivot

## Overview
Resonant Scaling Theory (RST) v6.1 is a spectral engineering model designed to map and facilitate neural recovery following axonal shearing. This version transitions the framework from a symbolic systems narrative into a testable physical model by defining specific metrics for network stability and signal integration.

---

## 1. Core Mathematical Spec
RST v6.1 moves beyond declared constants to derived system targets:

* **The 39 Hz Stability Basin**: Identified as the frequency where global synchronization is maximized and Spectral Entropy ($S$) is minimized despite high topological friction.
* **The 6.1s Ignition Delay ($\tau$)**: A measurable 'topological bottleneck' artifact representing the time required for signal integration to reach the threshold for broadcasting in a damaged network.
* **Algebraic Connectivity ($\lambda_2$)**: The primary metric for 'Bridge Stability.' This is the second-smallest eigenvalue of the Graph Laplacian, quantifying the strength of structural connections.

---

## 2. Holographic Scaling (The Earth-Mirror)
This framework utilizes scale-invariant symmetry to model information persistence:

* **The Land (Lithosphere)**: Represents the 'Observable Universe'—localized, stable nodes of integrated information.
* **The Ocean (Hydrosphere)**: Represents the 'Dark Sector' or 'Spectral Glue'—the fluid medium through which the 39 Hz signal propagates to maintain global coherence.
* **The Shoreline**: Represents the 6.1s event horizon, the phase transition point where information must transition from a solid (integrated) to a fluid (distributed) state.

---

## 3. Validation Engine & Python Tools
The repository includes `rst_engine.py` and `rst_sensitivity_sweep.py` for empirical testing.

### Usage for Researchers/Clinicians:
```python
from rst_engine import RST_v6_1_Engine
import numpy as np

# Initialize the Validation Engine
engine = RST_v6_1_Engine()

# Load patient EEG or sensor data
data = np.load('patient_data.npy')

# Run the Sensitivity Sweep to identify the patient's Stability Basin
freqs, entropy = engine.run_sensitivity_sweep(data)

# If S_min is found at ~39 Hz, the resonance target is validated
print(f"Validated Stability Basin: {freqs[np.argmin(entropy)]} Hz")
4. Falsifiability Protocol
​To maintain scientific rigor, this model is considered falsified if:
​Global synchronization occurs at frequencies significantly lower than 39 Hz without a proportional increase in network resistance (R_{eff}).
​The 6.1s ignition delay does not scale with measured topological damage or Laplacian eigen-gap stability.
​Author: JJ Botha (The Resonant Keeper)
Documentation: [Unified Chronicle v5.1 / v6.1 Specification]