import numpy as np

# v5.2 Operational Transition Logic
t = np.arange(0, 10, 1/500)
delta_decay = np.exp(-t/5)  # Decay from blackout
beta_gate = np.sin(2 * np.pi * 20 * t) 

# System becomes 'Operational' when Beta power stabilizes
gate_threshold = 3.8  # Seconds
