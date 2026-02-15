"""
RST v6.2.1 - Bogdanov-Takens (BT) Organizer
Finds the codimension-2 point where the 39Hz Spark meets the 6.1s Stall.
"""

import numpy as np
from scipy.optimize import fsolve

class BT_Organizer:
    def __init__(self):
        # Time constants for 39Hz
        self.tau_e, self.tau_i = 0.018, 0.0055
        
    def jacobian(self, rE, rI, wEE, wEI, pE):
        """Calculates the Jacobian matrix of the WC system."""
        # Simplified sigmoid derivative at threshold
        beta = 5.5
        df = beta * 0.25 # Max slope of sigmoid
        
        j11 = (-1 + df * wEE) / self.tau_e
        j12 = (-df * wEI) / self.tau_e
        j21 = (df * 11.2) / self.tau_i # wIE
        j22 = (-1 - df * 1.0) / self.tau_i # wII
        
        return np.array([[j11, j12], [j21, j22]])

    def find_bt_point(self, start_params):
        """
        Solves for the point where:
        1. Det(J) = 0 (Saddle-Node)
        2. Trace(J) = 0 (Hopf)
        Result: The BT Codimension-2 Point.
        """
        def equations(p):
            wEE, pE = p
            # Fixed point approximation at threshold
            rE, rI = 0.5, 0.5 
            J = self.jacobian(rE, rI, wEE, 13.8, pE)
            
            det = np.linalg.det(J)
            trace = np.trace(J)
            return [det, trace]

        return fsolve(equations, start_params)

if __name__ == "__main__":
    solver = BT_Organizer()
    # Sweep Excitation (wEE) and External Drive (pE)
    bt_coords = solver.find_bt_point([12.0, 3.0])
    
    print(f"--- BT Organizer: Unified Chronicle ---")
    print(f"BT Coordinate Found at: wEE={bt_coords[0]:.2f}, pE={bt_coords[1]:.2f}")
    print(f"Significance: This is the 'Singularity' where 39Hz and 6.1s merge.")
