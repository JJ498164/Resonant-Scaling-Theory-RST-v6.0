import numpy as np
from scipy.optimize import fsolve

def rst_characteristic_eq(s, omega0, zeta, G0, tau, kappa, a):
    # (s^2 + 2*zeta*omega0*s + omega0^2 - G0*exp(-s*tau))*(s + a) + kappa*exp(-s*tau)
    fast_osc = s**2 + 2*zeta*omega0*s + omega0**2 - G0*np.exp(-s*tau)
    slow_pole = s + a
    coupling = kappa * np.exp(-s*tau)
    return fast_osc * slow_pole + coupling

# Parameter Set (RST v6.2.1)
f_target = 39.1
omega0 = 2 * np.pi * f_target
tau = 1 / (2 * f_target)
a = 1 / 6.1  # The 6.1s Stall

print(f"RST Stability Solver: Root-finding for {f_target}Hz and {6.1}s Invariants...")
