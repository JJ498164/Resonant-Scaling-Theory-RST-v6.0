import numpy as np

class RST_Metabolic_WC:
    def __init__(self, supercritical=False):
        # Fast Parameters (Tuned for ~39.1Hz)
        self.tau_e, self.tau_i = 0.018, 0.0055
        self.wEE, self.wEI, self.wIE, self.wII = 11.4, 13.8, 11.2, 1.0
        self.I_ext = 3.35
        self.beta, self.theta = 5.5, 1.05
        
        # Slow Parameters (RST Invariants)
        self.tau_met = 6.1  
        self.kappa = 0.8 if supercritical else 0.25 
        self.gamma_met = -2.5 if supercritical else -1.2 

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-self.beta * (x - self.theta)))

    def step(self, rE, rI, ymet, dt):
        eff_drive = self.I_ext + (self.gamma_met * ymet)
        drE = (-rE + self.sigmoid(self.wEE*rE - self.wEI*rI + eff_drive)) / self.tau_e
        drI = (-rI + self.sigmoid(self.wIE*rE - self.wII*rI)) / self.tau_i
        dymet = (-ymet - self.kappa * rE) / self.tau_met
        return drE, drI, dymet
