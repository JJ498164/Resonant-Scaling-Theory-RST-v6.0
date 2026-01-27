import numpy as np
import matplotlib.pyplot as plt

class RST_Sensitivity_Validator:
    """
    RST v6.1: Validation Protocol for Spectral Engineering.
    Calculates the 'Stability Basin' surrounding the 39 Hz target.
    """
    def __init__(self, target_freq=39.0):
        self.target_freq = target_freq

    def calculate_spectral_entropy(self, signal):
        """
        Measures the chaos (entropy) in the frequency domain.
        """
        psd = np.abs(np.fft.rfft(signal))**2
        psd /= np.sum(psd)  # Normalize to a probability distribution
        entropy = -np.sum(psd * np.log2(psd + 1e-12))
        return entropy

    def run_parameter_sweep(self, signal, freq_range):
        """
        Sweeps through frequencies to find the 'Stability Basin'.
        """
        entropy_results = []
        for freq in freq_range:
            # Simulate resonance interaction at each frequency step
            # Testing for entropy minima (S_min)
            simulated_resonance = signal * np.sin(2 * np.pi * freq * np.linspace(0, 1, len(signal)))
            entropy_results.append(self.calculate_spectral_entropy(simulated_resonance))
            
        return freq_range, np.array(entropy_results)

# Usage Example for Researchers:
# validator = RST_Sensitivity_Validator()
# freqs, entropy = validator.run_parameter_sweep(my_eeg_data, np.linspace(10, 60, 100))
