import numpy as np

def calculate_topological_friction(signal, sampling_rate):
    """
    Measures Critical Slowing and spectral peaks to derive 
    system-specific resonance and bottlenecks.
    """
    # Perform Spectral Analysis (FFT)
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/sampling_rate)
    
    # Identify the Eigenfrequency (System Stability Peak)
    target_resonance = 39.0  # The Resonant Keeper's Baseline
    observed_peak = freqs[np.argmax(fft_vals)]
    
    # Calculate 'Friction' (Drift from stability target)
    spectral_drift = abs(observed_peak - target_resonance)
    
    return {
        "observed_peak": f"{observed_peak} Hz",
        "stability_target": f"{target_resonance} Hz",
        "topological_friction": spectral_drift
    }
