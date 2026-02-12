import numpy as np
from rst_engine import RSTSimulation

def test_bottleneck_convergence():
    """
    Validates that the 6.1s bottleneck emerges 
    when algebraic connectivity drops below 0.1.
    """
    sim = RSTSimulation(nodes=1000, friction=0.9)
    lambda_2 = sim.calculate_lambda_2()
    
    if lambda_2 < 0.1:
        latency = sim.run_state_transition()
        assert 6.0 <= latency <= 6.3, f"Latency {latency} out of RST bounds!"
        print("Test Passed: 6.1s Bottleneck Validated.")

if __name__ == "__main__":
    test_bottleneck_convergence()
