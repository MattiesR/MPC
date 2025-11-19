"""Quadratic programming utilities 
"""

from collections import namedtuple
from dataclasses import dataclass
import numpy as np 

QuadraticProgram = namedtuple(
    "QuadraticProgram",
    ("H", "f", "A", "b", "Aeq", "beq", "lbx", "ubx")
)

@dataclass
class QuadProgSolution:
    """Solution for a QP."""

    """Minimizer"""
    x_opt: np.ndarray
    """Optimal cost"""
    value: float
    """Flag whether the solver converged or not"""
    solver_success: bool


def solve(qp: QuadraticProgram) -> QuadProgSolution:
    """Wrapper for the cvxopt quadratic programming interface.
    
    Solves 
    
    .. math::

        \\begin{aligned}
        &\operatorname{minimize}&& \\frac{1}{2} x^\\top H x + f^\\top x \\
        &\operatorname{s.t.}&& A x \leq b \\
        &&& A_{\\textrm{eq}} x = b_{\\textrm{eq}}\\
        &&& l_{\\textrm{b}} \leq x \leq u_{\\textrm{b}}
    """
    import cvxopt 
    cvxopt.solvers.options["show_progress"] = False 

    H = .5 * (qp.H + qp.H.T)  # make sure P is symmetric
    args = [cvxopt.matrix(H), cvxopt.matrix(qp.f.reshape((-1, 1)))]
    args.extend([cvxopt.matrix(np.array(qp.A)), cvxopt.matrix(np.atleast_1d(qp.b))])
    if qp.Aeq is not None:
        args.extend([cvxopt.matrix(qp.Aeq), cvxopt.matrix(qp.beq)])
    sol = cvxopt.solvers.qp(*args)
    success = 'optimal' in sol['status']
    
    if not success:
        f = None
        x = None 
    else: 
        f = sol["primal objective"]
        x = sol["x"]
    return QuadProgSolution(x, f, success)


if __name__ == "__main__":

    problem = QuadraticProgram(np.eye(2), np.zeros(2), np.ones((1,2)), -10., None, None, -10*np.ones(2), 10*np.ones(10))
    solve(problem)