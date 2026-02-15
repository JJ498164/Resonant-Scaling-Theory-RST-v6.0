import numpy as np
from scipy.optimize import fsolve

class BT_Organizer:
    def __init__(self):
        self.tau_e, self.tau_i = 0.018, 0.0055
        
    def jacobian(self, wEE, pE):
        beta, df = 5.5, 5.5 * 0.25
        j11 = (-1 + df * wEE) / self.tau_e
        j12 = (-df * 13.8) / self.tau_e
        j21 = (df * 11.2) / self.tau_i
        j22 = (-1 - df * 1.0) / self.tau_i
        return np.array([[j11, j12], [j21, j22]])

    def solve_bt(self):
        def equations(p):
            wEE, pE = p
            J = self.jacobian(wEE, pE)
            return [np.linalg.det(J), np.trace(J)]
        return fsolve(equations, [12.0, 3.0])
