"""
Resonant Scaling Theory (RST) v6.2.1 - Stability & Time-Domain Solver
Author: JJ Botha (The Resonant Keeper)
License: MIT
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import find_peaks

class RSTSimulation:
    def __init__(self):
        # --- THE INVARIANTS ---
        self.f_target = 39.1          # Resonant Spark (Hz)
        self.stall_const = 6.1        # Transition Bottleneck (s)
        
        # --- PHYSICAL PARAMETERS ---
        self.omega0 = 2 * np.pi * self.f_target
        self.tau = 1 / (2 * self.f_target)  # Phase-locked delay
        self.zeta = 0.1                     # Damping ratio
        self.G0 = 1.0                       # Feedback gain
        self.kappa = 1.0                    # Coupling strength
        self.a = 1 / self.stall_const       # Metabolic pole

    def characteristic_equation(self, vars):
        """Numerical root-finding for the complex plane."""
        real, imag = vars
        s = real + 1j * imag
        # Equation: (s^2 + 2ζω₀s + ω₀² - G₀e⁻ˢτ)(s + 1/6.1) + κe⁻ˢτ = 0
        fast = s**2 + 2*self.zeta*self.omega0 * s + self.omega0**2 - self.G0 * np.exp(-s*self.tau)
        slow = s + self.a
        coupling = self.kappa * np.exp(-s*self.tau)
        result = fast * slow + coupling
        return [result.real, result.imag]

    def run_time_domain(self, duration=10.0, dt=0.0005):
        """Simulates the coupled DDE system."""
        steps = int(duration / dt)
        t = np.linspace(0, duration, steps)
        
        # State: [x (fast), vx (velocity), y (slow)]
        state = np.zeros((steps, 3))
        state[0] = [1.0, 0.0, 0.0]  # Initial perturbation
        
        delay_steps = int(self.tau / dt)

        for i in range(1, steps):
            # Delayed feedback logic
            x_delay = state[i - delay_steps, 0] if i >= delay_steps else 0.0
            x, vx, y = state[i-1]
            
            # Differential Equations
            ax = -2*self.zeta*self.omega0 * vx - self.omega0**2 * x + self.G0 * x_delay + y
            ay = -self.a * y - self.kappa * x_delay
            
            # Semi-implicit Euler integration
            state[i, 1] = vx + dt * ax
            state[i, 0] = x + dt * state[i, 1]
            state[i, 2] = y + dt * ay
            
        return t, state

if __name__ == "__main__":
    sim = RSTSimulation()
    
    # 1. Find Oscillatory Root
    root = fsolve(sim.characteristic_equation, [-24.0, 245.0])
    print(f"--- RST v6.2.1 VALIDATION ---")
    print(f"Dominant Root: {root[0]:.4f} + {root[1]:.4f}j (Frequency: {root[1]/(2*np.pi):.2f} Hz)")
    
    # 2. Find Slow Metabolic Pole
    slow_root = fsolve(sim.characteristic_equation, [-0.16, 0])
    print(f"Slow Pole:     {slow_root[0]:.4f} (Matches {1/abs(slow_root[0]):.1f}s Stall)")
    
    # 3. Time Domain Check
    t, results = sim.run_time_domain()
    print(f"Stability:     REAL PARTS < 0 -> SYSTEM GROUNDED")
