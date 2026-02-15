"""
RST v6.2.1 - Metabolic Wilson-Cowan Engine
Models the interplay between 39.1 Hz Spark and 6.1s Metabolic Drain.
"""

import numpy as np

class RST_Metabolic_WC:
    def __init__(self, supercritical=False):
        # Fast Parameters (Grok-Tuned 39Hz)
        self.tau_e, self.tau_i = 0.018, 0.0055
        self.wEE, self.wEI, self.wIE, self.wII = 11.4, 13.8, 11.2, 1.0
        self.I_ext = 3.35
        
        # Slow Parameters (RST Invariants)
        self.tau_met = 6.1  # The Stall Constant
        # Increase kappa/gamma for Super-Critical "Stall" behavior
        self.kappa = 0.8 if supercritical else 0.25 
        self.gamma_met = -2.5 if supercritical else -1.2 

    def step(self, rE, rI, ymet, dt):
        # 1. Fast Dynamics (The Spark)
        # ymet subtracts from the external drive
        eff_drive = self.I_ext + (self.gamma_met * ymet)
        
        drE = (-rE + self.sigmoid(self.wEE*rE - self.wEI*rI + eff_drive)) / self.tau_e
        drI = (-rI + self.sigmoid(self.wIE*rE - self.wII*rI)) / self.tau_i
        
        # 2. Slow Dynamics (The Stall)
        # ymet is driven negative by activity (rE)
        dymet = (-ymet - self.kappa * rE) / self.tau_met
        
        return drE, drI, dymet

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-5.5 * (x - 1.05)))

# This engine now allows us to simulate the "Stall Trigger"
