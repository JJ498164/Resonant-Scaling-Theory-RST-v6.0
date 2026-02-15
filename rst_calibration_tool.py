import numpy as np

def calibrate_rst(measured_stall, measured_hz, lambda2=0.074):
    k_const = measured_stall * np.sqrt(lambda2)
    tau = 1 / (2 * measured_hz)
    return {"K": k_const, "tau": tau, "a": 1/measured_stall}

# Example use: 10s stall, 36Hz peak
profile = calibrate_rst(10.0, 36.0)
print(f"Individual Calibration: K={profile['K']:.4f}")
