"""
Utility for computing LQR controllers.
"""


from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_discrete_are

@dataclass
class LqrSolution:
    """Solution to the linear quadratic regulator problem. 
    """

    """Solution to the DARE"""
    P: np.ndarray
    """Optimal linear control gain"""
    K: np.ndarray

    def __str__(self) -> str:
        return f"LQR\n---\nP = {self.P},\nK = {self.K}"


def dlqr(A, B, Q, R) -> LqrSolution: 
    """Solve the discrete time, infinite-horizon LQR problem.
    
    Args: 
        A: Discrete-time system matrix
        B: Discrete-time input-to-state mapping
        Q: Weight matrix on the states (PSD)
        R: Weight matrix on the controls (PD)
    """
    P = solve_discrete_are(A,B,Q,R)  # Solve DARE for P 
    K = -np.linalg.solve(R + B.T@P@B, B.T@P@A)
    return LqrSolution(P, K)