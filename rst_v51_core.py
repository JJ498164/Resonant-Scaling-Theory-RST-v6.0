import numpy as np
import scipy.linalg as la

def analyze_rst_stability(adj_matrix):
    """
    Analyzes neural network stability based on RST v5.1.
    
    Args:
        adj_matrix: A square matrix representing neural connectivity strength.
    """
    # 1. Generate the Laplacian Matrix (L = D - A)
    degrees = np.diag(np.sum(adj_matrix, axis=1))
    laplacian = degrees - adj_matrix
    
    # 2. Extract Eigenvalues and Eigenvectors
    evals, evecs = la.eigh(laplacian)
    
    # 3. Identify Algebraic Connectivity (lambda_2)
    # This is the "Spectral Gap" - the thickness of the bridge.
    lambda_2 = evals[1]
    
    # 4. Identify the Fiedler Vector
    # Used for targeted hub gain to find the critical bridge nodes.
    fiedler_vector = evecs[:, 1]
    
    # 5. Calculate Quadratic Friction (1 / lambda_2^2)
    # The metabolic cost of signal integration.
    friction = 1 / (lambda_2**2)
    
    # 6. Determine Stall Risk (The 6.1s Limit)
    # If friction exceeds a normalized threshold, the Ignition Delay triggers.
    stall_risk = "CRITICAL" if friction > 5.0 else "STABLE"
    
    return {
        "Spectral Gap (lambda_2)": round(lambda_2, 4),
        "Metabolic Friction": round(friction, 4),
        "Fiedler Vector": fiedler_vector,
        "System Status": stall_risk
    }

# Example usage with a 4-node network
network = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
])

results = analyze_rst_stability(network)
print(f"RST v5.1 Analysis: {results['System Status']}")
print(f"Friction Level: {results['Metabolic Friction']}")
