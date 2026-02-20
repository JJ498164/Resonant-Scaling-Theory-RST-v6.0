import numpy as np
import mne
from scipy.signal import welch

def calculate_normalized_power(file_path, fmin=8.5, fmax=9.0):
    """
    Calculates the normalized power within a specific frequency band.
    Standardizes data to account for global amplitude differences.
    """
    try:
        # Load EEG data (Standard 10-20 Montage expected)
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.filter(1, 50, fir_design='firwin', verbose=False)
        
        # Extract data and sampling frequency
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        # Calculate Power Spectral Density (PSD) using Welch's Method
        # Using a 2-second window for 0.5Hz resolution
        freqs, psd = welch(data, sfreq, nperseg=int(sfreq * 2))
        
        # Normalize PSD (Relative Power)
        psd_norm = psd / np.sum(psd, axis=-1, keepdims=True)
        
        # Average across all channels and find the band of interest
        avg_psd = np.mean(psd_norm, axis=0)
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        
        band_power = np.trapz(avg_psd[idx_band], freqs[idx_band])
        return band_power

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- RST v5.1 Empirical Verification ---
# Replace with your actual file paths
trauma_file = "subject_0284_trauma.edf"
control_file = "subject_control_clean.edf"

p_trauma = calculate_normalized_power(trauma_file)
p_control = calculate_normalized_power(control_file)

if p_trauma and p_control:
    ratio = p_control / p_trauma
    print("\n" + "="*40)
    print("      RST v5.1 ALPHA-ANCHOR REPORT")
    print("="*40)
    print(f"Trauma (8.75Hz Power):  {p_trauma:.6f}")
    print(f"Control (8.75Hz Power): {p_control:.6f}")
    print(f"Power Ratio (C/T):      {ratio:.2f}x")
    print("-" * 40)
    
    if ratio > 10:
        print("STATUS: Significant Alpha-Anchor Collapse detected.")
        print("CLINICAL IMPLICATION: High probability of 6.1s bottleneck.")
    else:
        print("STATUS: Nominal Alpha-Anchor persistence.")
    print("="*40 + "\n")
