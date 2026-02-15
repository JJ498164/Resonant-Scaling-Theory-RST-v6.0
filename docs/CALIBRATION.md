# RST Individual Calibration Protocol

RST recognizes that while 39.1 Hz and 6.1s are the "Resonant Attractors," individual biological variation (age, injury type, myelination) shifts these coordinates.

### Calibration Workflow:
1. **Identify the Slow Pole:** Use a refractory period test to measure the recovery time (Stall).
2. **Identify the Fast Spark:** Use a Power Spectral Density (PSD) sweep on EEG data to find the peak Gamma frequency.
3. **Run the Solver:** Use `src/rst_calibration_tool.py` to derive the `individual_k`.

### Why this works:
By solving for $K$, we normalize the data. It allows us to compare a 12s stall in one patient to a 6.1s stall in another by seeing if they share the same **Scaling Law**.
