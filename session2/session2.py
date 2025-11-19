# Question 5
import cvxpy as cp
from dataclasses import dataclass
import numpy as np


@dataclass
class Problem:
    """Convenience class representing the problem data for session 2."""

    x0 = np.array([10,10])
    Ts: float = 0.3
    Q: np.ndarray = np.diag([10, 1])
    R: np.ndarray = np.diag([0.01])
    p_min: float = -150 # Minimal position
    p_max: float = 1.0  # Maximal position 
    v_min: float = -20  # Minimal velocity
    v_max: float = 25.0  # Maximal velocity
    u_min: float = -20.0
    u_max: float = 10.0
    N: int = 5

    A: np.ndarray = None
    B: np.ndarray = None

    def __post_init__(self):
        self.A = np.array([[1.0, self.Ts], [0, 1.0]])
        self.B = np.array([[0], [self.Ts]])

    @property
    def n_state(self):
        return self.A.shape[0]

    @property
    def n_input(self):
        return self.B.shape[1]
    

def question5():     
    problem = Problem()
    Q = problem.Q
    R = problem.R
    x0 = problem.x0
    N = 5
    
    u = cp.Variable((1,N+1))
    x = cp.Variable((2,N))
    cost = x.T @ Q @ x + u.T @ R @ u 
    constraints = [] # Set the constraints
    optimizer = cp.Problem(cp.Minimize(cost), constraints) # build optimizer
    solution = optimizer.solve()
    print(solution.x)
if __name__ == "__main__":
    question5()