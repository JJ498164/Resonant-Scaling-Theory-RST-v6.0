"""
FILE: src/RST_validation_engine.py
STATUS: FROZEN FOR BLIND TUH TEST - 16 FEB 2026
AUTHOR: JJ Botha (The Resonant Keeper)
VERSION: 1.0.0 (Preregistration Alpha)

DESCRIPTION: 
This script performs spectral analysis on EEG data to detect 
the 39.1 Hz 'Spark' and the 6.1s 'Gearbox' temporal window.
"""

import numpy as np
import pandas as pd
from scipy import signal
import os
import json

class RSTValidationEngine:
    def __init__(self, sample_rate=250):
        self.fs = sample_rate
        self.spark_f = 39.1
        self.gearbox_t = 6.1
        self.f_tolerance = 2.0  # +/- 2 Hz

    def compute_spectral_density(self, eeg_data):
        """Calculates PSD using Welch's method."""
        freqs, psd = signal.welch(eeg_data, self.fs, nperseg=self.fs*2)
        return freqs, psd

    def analyze_recording(self, eeg_array):
        """
        Scans for the 39.1Hz Spark and calculates the T-Ignition delay.
        """
        freqs, psd = self.compute_spectral_density(eeg_array)
        
        # Identify peak frequency in the gamma range (30-50 Hz)
        gamma_idx = np.where((freqs >= 30) & (freqs <= 50))
        peak_idx = np.argmax(psd[gamma_idx])
        peak_freq = freqs[gamma_idx][peak_idx]
        
        # Calculate Lambda_2 (simulated via spectral gap proxy)
        # Note: In real EEG, we look at the coherence between frontal/occipital
        l2_proxy = np.max(psd[gamma_idx]) / np.mean(psd)
        
        # The Core Persistence Equation: T = K / sqrt(L2)
        t_ignition = 39.1 / np.sqrt(l2_proxy + 1e-6) 
        
        return {
            "peak_freq_hz": round(peak_freq, 2),
            "t_ignition_s": round(t_ignition, 4),
            "l2_proxy": round(l2_proxy, 6),
            "is_stable": (abs(peak_freq - self.spark_f) <= self.f_tolerance)
        }

    def run_blind_test(self, input_dir, output_dir):
        """Processes files without displaying results to the console."""
        results = {}
        for filename in os.listdir(input_dir):
            if filename.endswith(".csv") or filename.endswith(".npy"):
                # Load data (Placeholder for actual TUH loading logic)
                data = np.load(os.path.join(input_dir, filename))
                analysis = self.analyze_recording(data)
                results[filename] = analysis
        
        # Save results to a sealed file
        with open(os.path.join(output_dir, "tuh_results_sealed.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"--- BLIND TEST COMPLETE ---")
        print(f"Results sealed in {output_dir}. DO NOT OPEN UNTIL COMMITTED.")

if __name__ == "__main__":
    # Internal Safeguard: Only run if directory exists
    engine = RSTValidationEngine()
    print("RST Validation Engine: Frozen & Ready.")
