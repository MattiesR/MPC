import numpy as np
from typing import Tuple , Callable
import matplotlib.pyplot as plt
import argparse
from given.problem import Problem

import sys
import os
sys.path.append(os.path.split(__file__)[0])  # Allow relative imports
from rcracers.utils import quadprog
import cvxpy as cp


# -----------------------------------------------------------
# Helper functions
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
    def __init__(self, problem: Problem):
        self.problem = problem 
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
        cost = cost + cp.quad_form(x[-1], Q)  # Add terminal cost

        constraints = [ uk <= umax for uk in u ] + \
                      [ uk >= umin for uk in u ] + \
                      [ xk <= xmax for xk in x ] + \
                      [ xk >= xmin for xk in x ] + \
                      [ x[0] == x_init] + \
                      [ xk1 == A@xk + B@uk for xk1, xk, uk in zip(x[1:], x, u)] 

        solver = cp.Problem(cp.Minimize(cost), constraints)

        return solver

    def solve(self, x) -> quadprog.QuadProgSolution: 
        solver: cp.Problem = self.qp
        
        # Get the symbolic parameter for the initial state 
        solver.param_dict["x_init"].value = x 
        
        # Call the solver 
        optimal_cost = solver.solve()

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
    


def check_init_feasibility(N, grid_size = 10):
    """
        Checks initial feasibility of the MPC problem
    """
    from rcracers.simulator import simulate 
    from given.log import ControllerLog

    # Get the problem data
    problem = Problem()

    # Define the simulator dynamics (we assume no model mismatch)
    def f(x, u):
        return problem.A @ x + problem.B @ u


    problem.N = N
    # Define the control policy
    policy = MPCCvxpy(problem)

    # Initial state 
    p_vals = np.linspace(-10,1,grid_size)
    v_vals = np.linspace(0,25,grid_size)
    feasible_grid = np.zeros((len(p_vals), len(v_vals)), dtype=bool)
    for i, v in enumerate(v_vals):
        for j, p in enumerate(p_vals):
            x0 = np.array([p, v])

            # Run the simulation
            logs = ControllerLog()
            _ = simulate(x0, f, n_steps=1, policy=policy, log=logs) # Only one step (initial feasibility)
            feasible_grid[i,j] = logs.solver_success[0]
    # Convert to integer for plotting
    Z = feasible_grid.astype(int)  # 1 = True (green), 0 = False (red)
    plt.imshow(Z, origin="lower",
    extent=[p_vals.min(), p_vals.max(), v_vals.min(), v_vals.max()],
    cmap=plt.cm.RdYlGn,  # red = 0, green = 1
    alpha=0.7,
    interpolation="nearest")
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.colorbar(label="Feasible (1) / Infeasible (0)")



def braking_distance(v0, Ts, umin):
    """
        Calculates the braking distance for a car in discrete time given initial velocity

        output: T: timesteps until stationarity
                d: braking distance
    """

    T = np.ceil(-v0/(Ts*umin))
    d = T*Ts*(v0 + umin*Ts *(T+1)/2)
    return T, d

def question4():
    # get problem situation
    problem = Problem()

    # Plot constraints


    # Plot breaking constraint (point of no return)
    v0_vals = np.linspace(1,20,50)
    T, d_vals = braking_distance(v0_vals, problem.Ts, problem.u_min)
    print(d_vals)
    print(T)



    for N in [2,5,10]:
        fig = plt.figure()
        const_style = dict(color="black", linestyle="--")
        plt.axvline(problem.p_max, **const_style)
        plt.axhline(problem.v_max, **const_style)
        plt.axhline(problem.v_min, **const_style)
        plt.xlabel("Position")
        plt.ylabel("Velocity")
        plt.plot(problem.p_max-d_vals, v0_vals)
        check_init_feasibility(N, grid_size = 20)
        plt.title(f"Initial feasibility N= {N}")
        plt.show()
        if args.figs:
            name = f"init_feas_N{N}.png"
            fig.savefig(folder + name)


# GLOBAL VARIABLES
folder = "images/assignment2/"

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
    question4()
