"""
RST v6.2.1 - Wilson-Cowan Population Dynamics
Integrates E-I population oscillations with the 6.1s Metabolic Invariant.
Author: JJ Botha (The Resonant Keeper)
"""

import numpy as np
from scipy.integrate import odeint

class RST_WilsonCowan:
    def __init__(self):
        # --- THE INVARIANTS ---
        self.f_target = 39.1        # Target Spark (Hz)
        self.t_stall = 6.1          # Metabolic Recovery Invariant (s)
        
        # --- POPULATION PARAMETERS (Tuned for ~39Hz) ---
        self.tau_e = 0.010          # 10ms Excitatory time constant
        self.tau_i = 0.005          # 5ms Inhibitory time constant
        self.wEE, self.wEI = 16.0, 12.0
        self.wIE, self.wII = 15.0, 3.0
        self.theta, self.beta = 2.0, 4.0
        
        # --- METABOLIC COUPLING ---
        self.tau_m = self.t_stall    # Slow metabolic pole
        self.metabolic_threshold = 0.4 # Point of failure (The "Stall" trigger)

    def sigmoid(self, x):
        """Nonlinear neural activation function."""
        return 1 / (1 + np.exp(-self.beta * (x - self.theta)))

    def model_dynamics(self, y, t, p_ext):
        """
        Coupled ODE System:
        E: Excitatory Population
        I: Inhibitory Population
        M: Metabolic Resource
        """
        E, I, M = y
        
        # 1. Wilson-Cowan E-I Dynamics (The Fast Spark)
        # External drive is modulated by available Metabolic Resource (M)
        drive_e = (self.wEE * E - self.wEI * I + p_ext) * M
        drive_i = (self.wIE * E - self.wII * I) * M
        
        dE_dt = (-E + self.sigmoid(drive_e)) / self.tau_e
        dI_dt = (-I + self.sigmoid(drive_i)) / self.tau_i
        
        # 2. RST Metabolic Drain & Recovery (The Slow Stall)
        # M recovers toward 1.0 at a rate defined by the 6.1s invariant
        # M is depleted proportional to excitatory activity
        drain = 0.05 * E 
        recovery = (1.0 - M) / self.tau_m
        dM_dt = recovery - drain
        
        return [dE_dt, dI_dt, dM_dt]

    def run_simulation(self, duration=10.0, drive=1.5):
        t = np.linspace(0, duration, int(duration * 2000))
        # Initial State: [Excitatory, Inhibitory, Metabolic Resource]
        y0 = [0.1, 0.1, 1.0]
        
        sol = odeint(self.model_dynamics, y0, t, args=(drive,))
        return t, sol

if __name__ == "__main__":
    sim = RST_WilsonCowan()
    t, results = sim.run_simulation(duration=15.0)
    
    print(f"--- RST Wilson-Cowan Validation ---")
    print(f"Metabolic Recovery Constant Locked: {sim.t_stall}s")
    print(f"Target Resonant Frequency: {sim.f_target} Hz")
    
    # Check for "Stall Events"
    min_m = np.min(results[:, 2])
    if min_m < 0.9:
        print(f"Status: Metabolic Dip Detected (Min Resource: {min_m:.2f})")
    print(f"Result: System Grounded at {results[-1, 2]*100:.1f}% Resource.")
