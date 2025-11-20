from typing import Optional
import casadi as cs 
from given.problem import Problem
from rcracers.utils.geometry import Polyhedron
from rcracers.utils.lqr import LqrSolution
from rcracers.utils import quadprog
import cvxpy as cp 

import numpy as np

# -----------------------------------------------------------
# Helper functions (See also solution code of session 2.)
# -----------------------------------------------------------

def get_states(sol: quadprog.QuadProgSolution, problem: Problem) -> np.ndarray:
    """Given a solution of a QP solver, return the predicted state sequence.

    Args:
        sol (quadprog.QuadProgSolution): QP solution
        problem (Problem): problem

    Returns:
        np.ndarray: state sequence shape: (N, nx)
    """
    ns = problem.n_state
    N = problem.N
    return sol.x_opt[: ns * (N + 1)].reshape((-1, ns))


def get_inputs(sol: quadprog.QuadProgSolution, problem: Problem) -> np.ndarray:
    """Given a solution of a QP solver, return the predicted input sequence.

    Args:
        sol (qp_solver.QuadProgSolution): QP solution
        problem (Problem): problem

    Returns:
        np.ndarray: state sequence shape: (N, nu)
    """
    ns = problem.n_state
    N = problem.N
    nu = problem.n_input
    return sol.x_opt[ns * (N + 1) :].reshape((-1, nu))


class MPC:
    """Abstract baseclass for an MPC controller. 
    """
    def __init__(self, problem: Problem, Xf: Optional[Polyhedron] = None, lqr_solution: Optional[LqrSolution] = None):
        self.problem = problem
        self.Xf = Xf
        self.terminal_controller = lqr_solution
        print(" Building MPC problem")

        self.qp = self._build()
    
    def _build(self):
        """Build the optimization problem."""
        ...
    
    def solve(self, x) -> quadprog.QuadProgSolution:
        """Call the optimization problem for a given initial state"""
        ...

    def __call__(self, y, log) -> np.ndarray: 
        """Call the controller for a given measurement. 

        The controller assumes perfect state measurements.
        Solve the optimal control problem, write some stats to the log and return the control action. 
        """

        # If the state is nan, something already went wrong. 
        # There is no point in calling the solver in this case. 
        if np.isnan(y).any():
            log("solver_success", False)
            log("state_prediction", np.nan)
            log("input_prediction", np.nan)
            return np.nan * np.ones(self.problem.n_input)

        # Solve the MPC problem for the given state 
        solution = self.solve(y)
        
        log("solver_success", solution.solver_success)
        log("state_prediction", get_states(solution, self.problem))
        log("input_prediction", get_inputs(solution, self.problem))
        
        return get_inputs(solution, self.problem)[0]


class MPCCvxpy(MPC):
    name:str="cvxpy"

    def _build(self) -> cp.Problem:
        
        # Make symbolic variables for the states and inputs 
        x = [cp.Variable((self.problem.n_state,), name=f"x_{i}") for i in range(self.problem.N+1)]
        u = [cp.Variable((self.problem.n_input,), name=f"u_{i}") for i in range(self.problem.N)]
        
        # Symbolic variable for the parameter (initial state)
        x_init = cp.Parameter((self.problem.n_state,), name="x_init")

        # Equality constraints 
        # -- dynamics
        A = self.problem.A
        B = self.problem.B
        
        # Inequality constraints -- simple bounds on the variables 
        #  -- state constraints 
        xmax = np.array([self.problem.p_max, self.problem.v_max])
        xmin = np.array([self.problem.p_min, self.problem.v_min])

        #  -- Input constraints
        umax = np.array([self.problem.u_max])
        umin = np.array([self.problem.u_min])

        # Cost 
        Q, R = self.problem.Q, self.problem.R 
        
        # Sum of stage costs 
        cost = cp.sum([cp.quad_form(xt, Q) + cp.quad_form(ut, R) for (xt, ut) in zip(x,u)])

        constraints = [ uk <= umax for uk in u ] + \
                      [ uk >= umin for uk in u ] + \
                      [ xk <= xmax for xk in x ] + \
                      [ xk >= xmin for xk in x ] + \
                      [ x[0] == x_init] + \
                      [ xk1 == A@xk + B@uk for xk1, xk, uk in zip(x[1:], x, u)]

        #-----------------------------------------------------------
        # NEW -- terminal ingredients! 
        #-----------------------------------------------------------
        if self.terminal_controller is not None: 
            print(" Adding terminal cost!")
            Pf = self.terminal_controller.P
            
        else: 
            print(" Using x'Qx as terminal cost")
            Pf = self.problem.Q
        cost = cost + cp.quad_form(x[-1], Pf)  # Add terminal cost

        if self.Xf is not None:
            print(" Adding terminal set!")
            Xf = self.Xf # Terminal set 
            terminal_constraint = [ Xf.H @ x[-1] <= Xf.h ]
            constraints += terminal_constraint  # Append to list 
        
        solver = cp.Problem(cp.Minimize(cost), constraints)

        return solver

    def solve(self, x) -> quadprog.QuadProgSolution: 
        solver: cp.Problem = self.qp
        
        # Get the symbolic parameter for the initial state 
        solver.param_dict["x_init"].value = x 
        
        # Call the solver 
        method = cp.ECOS
        optimal_cost = solver.solve(solver=method)

        if solver.status == "unbounded":
            raise RuntimeError("The optimal control problem was detected to be unbounded. This should not occur and signifies an error in your formulation.")

        if solver.status == "infeasible":
            print("  The problem is infeasible!")
            success = False 
            optimizer = np.nan * np.ones(sum(v.size for v in solver.variables())) # Dummy input. 
            value = np.inf  # Infeasible => Infinite cost. 

        else: 
            success = True # Everything went well. 
            # Extract the first control action
            optimizer = np.concatenate([solver.var_dict[f"x_{i}"].value for i in range(self.problem.N + 1)]\
                                       + [solver.var_dict[f"u_{i}"].value for i in range(self.problem.N)])
    
            # Get the optimal cost 
            value = float(optimal_cost)
        
        return quadprog.QuadProgSolution(optimizer, value, success)

