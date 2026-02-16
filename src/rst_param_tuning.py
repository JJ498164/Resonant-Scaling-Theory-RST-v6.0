# RST v6.2.1 - Comparative Manifold Parameters
# Use these for 'Deep Runs' in simulation environments like Grok or local ODE solvers.

TAI_SHEARED_NETWORK = {
    "lambda2": 0.035,        # Low connectivity
    "k_const": 1.66,
    "target_hz": 39.1,
    "expected_safe_window": 3.12  # Predicted via T = K/sqrt(lambda2)
}

OPTIMAL_HEALTHY_NETWORK = {
    "lambda2": 0.074,        # Standard healthy connectivity
    "k_const": 1.66,
    "target_hz": 39.1,
    "expected_safe_window": 6.10  # The theoretical 6.1s invariant
}
