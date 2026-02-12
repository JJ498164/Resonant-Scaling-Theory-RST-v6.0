import numpy as np
import networkx as nx

def calculate_rst_metrics(adjacency_matrix):
    """
    Calculates the actual Algebraic Connectivity (λ2) 
    and predicts the bottleneck delay.
    """
    # Create Laplacian
    L = nx.laplacian_matrix(nx.from_numpy_array(adjacency_matrix)).toarray()
    
    # Get Eigenvalues
    eigenvalues = np.sort(np.linalg.eigvals(L))
    lambda_2 = eigenvalues[1] # The Spectral Gap
    
    # RST Formula: τ ≈ 6.1 / λ2 (Normalized)
    predicted_delay = 6.18 / (lambda_2 + 1e-6)
    
    return {
        "algebraic_connectivity": lambda_2,
        "predicted_bottleneck": predicted_delay,
        "status": "LOCKED" if lambda_2 > 1.0 else "SCATTERED"
    }

# This script moves RST from 'signal analysis' to 'topological analysis'.
