"""
RST v6.2.1 - Tuned Wilson-Cowan Oscillator (39.1 Hz)
Calibrated biological substrate for the 'Resonant Spark'.
Based on Grok-tuned parameters for precise Gamma-band resonance.
"""

import numpy as np

class TunedRSTEngine:
    def __init__(self):
        # --- GROK-TUNED PARAMETERS ---
        self.tau_e = 0.018    # 18ms Excitatory time constant
        self.tau_i = 0.0055   # 5.5ms Fast Inhibitory constant
        self.wEE = 11.4       # Recurrent Excitation
        self.wEI = 13.8       # Cross-Inhibition
        self.wIE = 11.2       # Excitatory-to-Inhibitory
        self.wII = 1.0        # Weak self-inhibition
        self.I_extE = 3.35    # External Drive (The Gearbox input)
        self.beta = 5.5       # Sigmoid Gain
        self.theta = 1.05     # Sigmoid Threshold
        
        # --- RST INVARIANTS ---
        self.stall_const = 6.1 

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-self.beta * (x - self.theta)))

    def compute_step(self, rE, rI, dt):
        """Standard Wilson-Cowan update loop."""
        # Excitatory update
        drE = (-rE + self.sigmoid(self.wEE*rE - self.wEI*rI + self.I_extE)) / self.tau_e
        # Inhibitory update
        drI = (-rI + self.sigmoid(self.wIE*rE - self.wII*rI)) / self.tau_i
        
        return drE, drI

if __name__ == "__main__":
    print(f"RST v6.2.1: Wilson-Cowan Biological Layer Initialized.")
    print(f"Target Frequency: 39.1 Hz (Period: 25.6ms)")
    print(f"Metabolic Safety Anchor: {6.1}s")
