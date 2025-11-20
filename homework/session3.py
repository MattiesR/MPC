from typing import Callable, List
from rcracers.utils.geometry import Polyhedron, Rectangle, Ellipsoid, plot_polytope, plot_ellipsoid
from rcracers.utils.lqr import LqrSolution, dlqr
from rcracers.simulator import simulate

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
    fig = plot_polytope(
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
        plt.savefig(folder + name)



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
    question2()