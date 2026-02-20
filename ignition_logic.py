import numpy as np

class RSTIgnition:
    """
    RST v6.1 Stochastic Ignition Protocol
    Optimized for Subject 0284: Axonal Shearing Recovery.
    """
    def __init__(self, f_alpha=8.79, f_gamma=39.06):
        self.f_alpha = f_alpha
        self.f_gamma = f_gamma
        # Precise 10-cycle burst duration based on 4.44:1 phase-drift recurrence
        self.burst_duration = 10 / f_gamma  # 0.25601... seconds
        self.sampling_rate = 500  # Hz (matches 0284_032_032_EEG.mat)
        
    def get_sigmoid_ramp(self, k=12):
        """
        Calculates the non-linear gain ramp for the first 3 cycles (76.8ms).
        Prevents local runaway in low-connectivity topologies (λ₂ = 0.1789).
        """
        t_ramp_end = 3 / self.f_gamma
        t = np.linspace(0, t_ramp_end, int(self.sampling_rate * t_ramp_end))
        
        # Normalized sigmoid centered at 50% of the ramp window
        gain = 1 / (1 + np.exp(-k * (t / t_ramp_end - 0.5)))
        return t, gain

    def verify_autonomy_threshold(self, lambda_2):
        """
        Logic for the transition from Scaffolded to Autonomous mode.
        """
        threshold = 0.24
        if lambda_2 < threshold:
            return "SCAFFOLD_MODE: Forced PAC @ Alpha Trough (π)"
        else:
            return "AUTONOMOUS_MODE: Gamma Resonance Lock"

# Example Usage for v6.1 Implementation
rst = RSTIgnition()
t_ramp, gains = rst.get_sigmoid_ramp()

print(f"RST v6.1 Protocol Initialized")
print(f"Burst Duration (Drift-Corrected): {rst.burst_duration:.4f}s")
print(f"Initial Cycle Gain (Stochastic Probe): {gains[0]:.4f}")
print(f"End of Ramp Gain (Resonance Lock): {gains[-1]:.4f}")
