from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_discrete_are
from typing import Tuple , Callable
import matplotlib.pyplot as plt
import argparse

# GLOBAL VARIABLES
folder = "images/assignment1/"
@dataclass
class Problem:
    """Convenience class representing the problem data for session 1."""

    Ts: float = 0.5
    C = np.array([[1, -2./3]])
    Q = C.T@C + 1e-3 * np.eye(2)
    R = np.array([[0.1]])

    A: np.ndarray = None
    B: np.ndarray = None
    


    def __post_init__(self):
        self.A = np.array([[1.0, self.Ts], [0, 1.0]])
        self.B = np.array([[0], [-self.Ts]])

    @property
    def n_state(self):
        return self.A.shape[0]

    @property
    def n_input(self):
        return self.B.shape[1]







def calculate_Kinf(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """
        Calculates the infinite horizon controller Kinf by solving the Algebraic Ricatti Equation
    """
    Pinf = solve_discrete_are(A, B, Q, R)
    Kinf = -np.linalg.solve(R + B.T @ Pinf @ B, B.T @ Pinf @ A)
    return Kinf, Pinf 


def riccati_recursion(A: np.ndarray, B: np.ndarray, R: np.ndarray, Q: np.ndarray, Pf: np.ndarray, N: int): 
    """Solve the finite-horizon LQR problem through recursion

    Args:
        A: System A-matrix 
        B: System B-matrix 
        R: weights on the input (positive definite)
        Q: weights on the states (positive semidefinite)
        Pf: Initial value for the Hessian of the cost-to-go (positive definite)
        N: Control horizon  
    """
    import numpy.linalg as la  # Import linear algebra to solve linear system

    P = [Pf] 
    K = [] 
    for _ in range(N):
        Kk = -la.solve(R + B.T@P[-1]@B, B.T@P[-1]@A)
        K.append(Kk)
        Pk = Q + A.T@P[-1]@(A + B@K[-1])
        P.append(Pk)

    return P[::-1], K[::-1]  # Reverse the order for easier indexing later. 


def plot_cl_trajectory_inf_N(A:np.ndarray,B:np.ndarray,Q:np.ndarray,R:np.ndarray, Kinf: np.ndarray, sim_time=10):
    """
        Plot the simulated closed-loop trajectory for infinite horizon and finte horizon N cases
    """

    # Define the simulator dynamics (we assume no model mismatch) (linear dynamics)
    def f(x, u):
        """Dynamics"""
        return A @ x + B @ u
    
    def control_policy(gains,x,t):
        """Control policy (receding horizon) given gains"""
        return gains[0]@x
    


    x0 = np.array([10,10])

    fig = plt.figure()
    

    # Finite horizon control policy
    Pf = Q
    for N in [4, 5, 6, 8,10,15]:
        _, gains = riccati_recursion(A, B, R, Q, Pf, N)
        kappa = lambda x,t : control_policy(gains,x,t)
        x_closed_loop, cl_unstable = simulate(x0, f, kappa, sim_time)
        plt.plot(x_closed_loop[:,0], x_closed_loop[:,1], marker=".",label=f"Horizon: {N}")


    # Simulatino of infinite horizon closed-loop trajectory
    kappa_inv = lambda x,t : Kinf @ x   # control policy infinite horizon 
    x_closed_loop, cl_unstable = simulate(x0, f, kappa_inv, sim_time)
    plt.plot(x_closed_loop[:,0], x_closed_loop[:,1], marker=".", color='k',label=f"Horizon: $\infty$")
    

    plt.annotate("$x_0$", x0)
    plt.legend()
    plt.xlabel("Position [m]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Closed-loop trajectories for MPC controllers")
    plt.show()
    return fig

def simulate(x0: np.ndarray, f: Callable, policy: Callable, steps: int) -> Tuple[np.ndarray, bool]:
    """Generic simulation loop.
    
    Simulate the discrete-time dynamics f: (x, u) -> x
    using policy `policy`: (x, t) -> u 
    for `steps` steps ahead and return the sequence of states.

    Returns 
        x: sequence of states 
        instability_occurred: whether or not the state grew to a large norm, indicating instability 
    """
    instability_occured = False  # Keep a flag that indicates whenever we detected instability. 
    x = [x0]
    for t in range(steps):
        xt = x[-1]
        ut = policy(xt, t)
        xnext = f(xt, ut)
        x.append(xnext)
        if np.linalg.norm(xnext) > 100 and not instability_occured:  
            # If the state become very large, we flag instability. 
            # (This is a heuristic of course, but for this example, it suffices.)
            instability_occured = True 
    
    return np.array(x), instability_occured


def approximate_infinite_horizon_cost(x0: np.ndarray, A: np.ndarray, B: np.ndarray, K: np.ndarray, Q: np.ndarray, R: np.ndarray, sim_time: int=1000):
    """Approximate the infinite horizon cost of the closed-loop system.
    
    Args: 
        x0: Initial state 
        A: A-matrix of the system
        B: B-matrix of the system
        K: State-feedback gain
        R: weights on the input (positive definite)
        Q: weights on the states (positive semidefinite)
        sim_time: Number of terms to use for the approximation (default = 1000)
    Returns
        cost: The approximate infinite horizon cost
    """

    def f(x,u):
        """Dynamics"""
        return A@x + B@u

    def κ(x,t):
        """Control policy (receding horizon)"""
        return K@x
    
    x_closed_loop, _ = simulate(x0, f, κ, sim_time)

    # x_k' * Q * x_k + x_k' * K' * R * K * x_k = x_k' * (Q + K' * R * K) * x_k
    M = Q + K.T @ R @ K

    ## Slow way
    #cost = 0
    #for k in range(sim_time):
    #    xk = x_closed_loop[k,:].T
    #    cost += xk.T @ M @ xk
    
    ## Fast way
    cost = np.trace(M @ (x_closed_loop.T @ x_closed_loop))
    return cost


def question1():
    """
        Compares the finite horizon controllers K_N with the infinite horizon LQR controller
    """

    # Get problem setting
    print("Get the problem setup ...")
    problem = Problem()

    # 1) Compute the infinite horizon LQR controller u = K_inf x with inital state x0 = [10,10]
    print("Computing the infitie horizon LQR controller ...")
    A = problem.A
    B = problem.B
    Q = problem.Q
    R = problem.R
    Kinf, _ = calculate_Kinf(A=A, B=B, Q=Q, R=R)
    print(f"LQR gain= {Kinf}")

    # 2) Plot the simulated closed-loop trajectory of this controller with trajectories of the finite-horizon controller
    fig = plot_cl_trajectory_inf_N(A,B,Q,R,Kinf,sim_time=10)
    if args.figs:
        name = "cl_traj_comparison.png"
        fig.savefig(folder+name)


def question2():
    """
        Numerically compares the quality of the finite-horizon LQR controller to the infinite horizon controller. For the same fixed x_0 as before and for N ranging from 1 to 10.
    """
    # Getting problem setup
    problem = Problem()
    
    # Plot V_N = x0^T P x0 verses N
    A = problem.A
    B = problem.B
    R = problem.R
    Q = problem.Q

    Pf = Q
    x0 = np.array([10,10])
    VN_values = np.zeros(10)
    
    for N in range(1,10):
        Ps, _ = riccati_recursion(A=A, B=B, R=R, Q=Q, Pf=Pf, N=N)
        PN = Ps[0]      # First element of list
        VN_values[N] = x0.T @ PN @ x0
    
    fig = plt.figure()
    plt.plot(VN_values, label="$V_N$")
    
    # Plot Pinf
    Kinf, Pinf = calculate_Kinf(A=A, B=B, Q=Q, R=R)
    Vinf = x0.T @ Pinf @ x0
    print(f"Infinite horizon cost: {Vinf}")
    plt.axhline(y=Vinf, color='r', linestyle='--', label="$V_\infty$")

    # Approximate the infinite horizon cost for the infinite-horizon controller using a long state and input trajectory

    approx_cost_Kinf = approximate_infinite_horizon_cost(x0, A, B, Kinf, Q, R, 100)
    approx_cost_N_values = np.zeros(10)
    for N in range(1,10):
        _, Ks = riccati_recursion(A=A, B=B, R=R, Q=Q, Pf=Pf, N=N)
        # We apply Ks
        approx_cost_N_values[N] = approximate_infinite_horizon_cost(x0, A, B, Ks[0], Q, R, 100)
    plt.plot(approx_cost_N_values, label="$\hat{V}_N$")
    plt.axhline(y=approx_cost_Kinf, color='r', linestyle='--', label="$\hat{V}_\infty$")
    plt.xlabel("$N$")
    plt.ylabel("$\hat{V}_N$")  
    plt.legend()   
    plt.title("Approximate infinite horizon cost")
    plt.ylim([0,2000])
    plt.show()
    if args.figs:
        name = "plot_VNvsN.png"
        fig.savefig(folder + name) 

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
    question1()
    question2()