from typing import Callable, List
from rcracers.utils.geometry import Polyhedron, Rectangle, Ellipsoid, plot_polytope, plot_ellipsoid
from rcracers.utils.lqr import LqrSolution, dlqr
from rcracers.simulator import simulate
import cvxpy as cp

import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt


import argparse
from given.problem import Problem


Matrix = np.ndarray



@dataclass
class InvSetResults:
    iterates: List[Polyhedron] = field(
        default_factory=list
    )  # Iterates of the algorithm
    n_its: int = 0  # Number of iterations performed
    success: bool = False

    @property
    def solution(self) -> Polyhedron:
        """Get the final solution, i.e., the last computed iterate."""
        return self.iterates[-1]


def build_state_set(problem: Problem) -> Polyhedron:
    """Build the state set (exercise 2).

    Args:
        problem (Problem): Problem instance

    Returns:
        Polyhedron: Set of states that are within the provided bounds. This set is a box.
    """

    H = np.vstack([np.eye(2), -np.eye(2)])
    h = np.array([problem.p_max, problem.v_max, -problem.p_min, -problem.v_min])

    return Polyhedron.from_inequalities(H, h)

def build_input_set_u(problem: Problem) -> Polyhedron:
    """Build a polyhedral set that describes all states that yield a feasible control action
    when applying the controller ``u = Kx``.  That is,

    {x | K x ∈ U }

    If U is described by H u ≤ h, then this is

    {x | HK x ≤ h}

    Args:
        problem (Problem): Problem description
        K (Matrix): State feedback gain

    Returns:
        Polyhedron: Set of states such that the controller is feasible
    """
    U = Rectangle(np.array([problem.u_min]), np.array([problem.u_max]))

    # Equivalent construction:
    # H = np.array([[1], [-1]])
    # h = np.array([10, 20])
    # U = Polyhedron.from_inequalities(H, h)

    H_in = U.H
    h_in = U.h
    return Polyhedron.from_inequalities(H_in, h_in)

def build_input_set(problem: Problem, K: Matrix) -> Polyhedron:
    """Build a polyhedral set that describes all states that yield a feasible control action
    when applying the controller ``u = Kx``.  That is,

    {x | K x ∈ U }

    If U is described by H u ≤ h, then this is

    {x | HK x ≤ h}

    Args:
        problem (Problem): Problem description
        K (Matrix): State feedback gain

    Returns:
        Polyhedron: Set of states such that the controller is feasible
    """
    U = Rectangle(np.array([problem.u_min]), np.array([problem.u_max]))

    # Equivalent construction:
    # H = np.array([[1], [-1]])
    # h = np.array([10, 20])
    # U = Polyhedron.from_inequalities(H, h)

    H_in = U.H @ K
    h_in = U.h
    return Polyhedron.from_inequalities(H_in, h_in)



def find_largest_ellipsoid(P, poly_set):
    """
        Computes the largest ellipsoid given P and a polyhedral set
    """
    ineq = poly_set.inequalities()
    H = ineq[:,:2]
    g = ineq[:,2]
    Pinv = np.linalg.inv(P)
    alpha = np.min([g[i]**2 / (H[i,:] @ Pinv @ H[i,:]) for i in range(len(g))])
    ellips = Ellipsoid(P/alpha)
    return ellips

def question2():
    """
        Plot the ellipsoidal set together with the polyhedral set
    """
    problem = Problem()
    print("--Compute LQR solution")
    lqr_solution = dlqr(problem.A, problem.B, problem.Q, problem.R)

    state_set = build_state_set(problem)
    input_set = build_input_set(problem, lqr_solution.K)
    state_kappa_set = state_set.intersect(input_set)
    # plot_polytope(state_set, label="X") # Don't plot because otherwise ellipsoid not visible
    fig = plt.figure()
    plot_polytope(
        state_set.intersect(input_set), label=r"${x \in X: Kx \in U}$", color="red"
    )
    plt.title("Comparison ellipsoid and polyhedral set $X_\kappa$")

    # Finding the largest ellipsoid
    ellips = find_largest_ellipsoid(lqr_solution.P, state_kappa_set)

    plot_ellipsoid(ellips, label=f"maximal ellipsoid centered at 0")
    plt.legend()
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.show()
    if args.figs:
        name = "ellipsoid_q2.png"
        fig.savefig(folder + name)
    return 0

def question3():
    """
    Plot the ellipsoidal set together with the polyhedral set using LMI optimization
    """
    problem = Problem()
    print("--Compute LQR solution")
    lqr_solution = dlqr(problem.A, problem.B, problem.Q, problem.R)
    K_lqr = lqr_solution.K

    state_set = build_state_set(problem)  # returns H_x, h_x
    input_set = build_input_set_u(problem)  # returns H_u, h_u

    # Intersected polyhedral set X_kappa
    input_x_set = build_input_set(problem, K_lqr)
    state_kappa_set = state_set.intersect(input_x_set)

    n = problem.A.shape[0]
    m_u = problem.B.shape[1]

    # Decision variables: S = P^{-1} and F = K S
    S = cp.Variable((n, n), symmetric=True)
    F = cp.Variable((m_u, n))

    # constraints = [S >> 1e-6*np.eye(n)]  # S > 0 for PSD
    constraints = []
    # State constraints: H_xi^T S H_xi <= h_xi^2
    H_x, h_x = state_set.H, state_set.h  # assume rows: H_x[i,:], h_x[i]
    for i in range(H_x.shape[0]):
            Hi = H_x[i,:].reshape(1, -1)
            # LMI equivalent to Hi @ P^{-1} @ Hi.T <= h_i**2
            constraints.append(Hi @ S @ Hi.T <= h_x[i]**2)

    A = problem.A
    B = problem.B
    M = cp.bmat([[S, (A @ S + B @ F).T],
                [(A @ S + B @ F), S]])
    constraints.append(M >> 0)

    # Input constraints: LMI using Schur complement
    H_u, h_u = input_set.H, input_set.h
    n = S.shape[0] # Assuming S is n x n, so n is the dimension of F
    
    for i in range(H_u.shape[0]):
        Hi = H_u[i, :].reshape(1, -1)  # 1 x n
    
        top_right = Hi @ F                  # 1 x n
        bottom_left = top_right.T           # n x 1
        
        scalar_block = cp.reshape(h_u[i]**2, (1, 1)) # 1 x 1

        # Stack blocks manually like NumPy
        top = cp.hstack([scalar_block, top_right]) # 1 x (1+n)
        bottom = cp.hstack([bottom_left, S])       # n x (1+n)
        LMI = cp.vstack([top, bottom])             # (1+n) x (1+n)

        constraints.append(LMI >> 0)
       
    # Objective: maximize trace(S) or logdet(S) to maximize ellipsoid volume
    obj = cp.Maximize(cp.log_det(S))

    # Solve SDP
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS)  # or cp.SCS, cp.CVXOPT

    print(f"Optimal log-det(S): {prob.value}")

    S_val = S.value
    F_val = F.value
    print(S_val)
    print(F_val)

    K_opt = F_val @ np.linalg.inv(S_val)

    # Plotting
    fig = plt.figure()
    plot_polytope(state_kappa_set, label=r"$X_\kappa$", color="red")

    # Construct ellipsoid: {x | x^T P x <= 1}, P = S^{-1}
    P_opt = np.linalg.inv(S_val)
    ellips = Ellipsoid(P_opt)  # Adapt this to your plotting function
    plot_ellipsoid(ellips, label=f"maximal ellipsoid centered at 0")

    plt.title("Maximal ellipsoid inside $X_\kappa$")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.legend()
    plt.show()
    if args.figs:
        name = "ellipsoid_q3_lmi.png"
        fig.savefig(folder + name)

    return 0

# GLOBAL VARIABLES
folder = "images/assignment3/"

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--figs", 
    action="store_true", 
    help="Show figures if this flag is provided"
)
args = parser.parse_args()
if args.figs:
    print("-----------------------")
    print("Saving figures enabled!")
    print("-----------------------")

if __name__ == "__main__":
    question3()