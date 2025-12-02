from typing import Callable, Tuple, Union
import sys
import casadi as cs
import os
WORKING_DIR = os.path.split(__file__)[0]
sys.path.append(os.path.join(WORKING_DIR, os.pardir))
from given.parameters import VehicleParameters
from given.animation import AnimateParking
from given.plotting import plot_state_trajectory, plot_input_sequence
from matplotlib.patches import Rectangle
from rcracers.simulator.dynamics import KinematicBicycle
import numpy as np
import matplotlib.pyplot as plt
from rcracers.simulator import simulate

PARK_DIMS = np.array((0.25, 0.12)) # w x h of the parking spot. Just for visualization purposes.

#-----------------------------------------------------------
# INTEGRATION
#-----------------------------------------------------------

def forward_euler(f, ts) -> Callable:
    def fw_eul(x,u):
        return x + f(x,u) * ts
    return fw_eul

def runge_kutta4(f, ts) -> Callable:
    def rk4_dyn(x,u):
        s1 = f(x,u)
        s2 = f(x + 0.5*ts*s1, u)
        s3 = f(x + 0.5*ts*s2, u)
        s4 = f(x + ts * s3, u)
        return x + ts/6. * (s1 + 2 * s2 + 2* s3 + s4)
    return rk4_dyn


def exact_integration(f, ts) -> Callable:
    """Ground truth for the integration

    Integrate the given dynamics using scipy.integrate.odeint, which is very accurate in
    comparison to the methods we implement in this settings, allowing it to serve as a
    reference to compare against.

    Args:
        f (dynamics): The dynamics to integrate (x,u) -> xdot
        ts (_type_): Sampling time

    Returns:
        Callable: Discrete-time dynamics (x, u) -> x+
    """
    from scipy.integrate import odeint  # Load scipy integrator as a ground truth
    def dt_dyn(x, u):
        f_wrap = lambda x, t: np.array(f(x, u)).reshape([x.size])
        y = odeint(f_wrap, x.reshape([x.size]), [0, ts])
        return y[-1].reshape((x.size,))
    return dt_dyn

def build_test_policy():
    # Define a policy to test the system
    acceleration = 1 # Fix a constant longitudinal acceleration
    policy = lambda y, t: np.array([acceleration, 0.1 * np.sin(t)])
    return policy


def compare_open_loop(ts: float, x0: np.ndarray, steps: int):
    """Compare the open-loop predictions using different discretization schemes.

    Args:
        ts (float): Sampling time (s)
        x0 (np.ndarray): Initial state
        steps (int): Number of steps to predict
    """
    params = VehicleParameters()
    kin_bicycle = KinematicBicycle(params)
    rk4_discrete_time = runge_kutta4(kin_bicycle, ts)
    fe = forward_euler(kin_bicycle, ts)
    gt_discrete_time = exact_integration(kin_bicycle, ts)

    test_policy = build_test_policy()

    # Plot the results
    _, axes = plt.subplots(constrained_layout = True)
    axes.set_xlabel("$p_{x}$")
    axes.set_ylabel("$p_{y}$")
    axes.set_title(f"Position trajectories Ts = {ts}")
    results = dict()
    for name, dynamics in {"Forward Euler": fe, "RK 4": rk4_discrete_time, "Ground truth": gt_discrete_time}.items():
        states = simulate(x0, dynamics, steps, policy=test_policy)
        axes.plot(states[:,0], states[:,1], label=name, linestyle="--")
        results[name] = states

    axes.legend()

    # Plot the errors
    plt.figure()
    plt.xlabel("Time step $k$")
    plt.ylabel("$\\| x_k - \\hat{x}_k\\|$")
    for name, dynamics in {"Forward Euler": fe, "RK 4": rk4_discrete_time}.items():
        error = np.linalg.norm(results["Ground truth"] - results[name], axis=1)
        plt.semilogy(error, label=name)

    plt.legend()
    plt.title(f"Open loop prediction errors ($T_s = {ts}s$)")
    plt.show()



#-----------------------------------------------------------
# MPC CONTROLLER
#-----------------------------------------------------------


class MPCController:

    def __init__(self, N: int, ts: float, *, params: VehicleParameters):
        """Constructor.

        Args:
            N (int): Prediction horizon
            ts (float): sampling time [s]
        """
        self.N = N
        self.ts = ts
        nlp_dict, self.bounds = self.build_ocp(params)

        opts = {"ipopt": {"print_level": 1}, "print_time": False}
        self.ipopt_solver = cs.nlpsol("solver", "ipopt", nlp_dict, opts)

    def solve(self, x) -> dict:
        return self.ipopt_solver(p=x, **self.bounds)

    def build_ocp(self, params: VehicleParameters) -> Tuple[dict, dict]:
        """
        Build a nonlinear program that represents the parametric optimization problem described above, with the initial state x as a parameter. Use a single shooting formulation, i.e., do not define a new decision variable for the states, but rather write them as functions of the initial state and the control variables. Also return the lower bound and upper bound on the decision variables and constraint functions:

        Args:
            VehicleParameters [params]: vehicle parameters
        Returns:
            solver [dict]: the nonlinear program as a dictionary:
                {"f": [cs.SX] cost (as a function of the decision variables, built as an expression, e.g., x + y, where x and y are CasADi SX.sym objects),
                "g": [cs.Expression] nonlinear constraint function as
                an expression of the variables and the parameters.
                These constraints represent the bounds on the state.
                "x": [cs.SX] decision_vars (all control actions over the prediction horizon (concatenated into a long vector)),
                "p": [cs.SX] parameters (initial state vector)}
            bounds [dict]: the bounds on the constraints
                {"lbx": [np.ndarray] Lower bounds on the decision variables,
                "ubx": [np.ndarray] Upper bounds on the decision variables,
                "lbg": [np.ndarray] Lower bounds on the nonlinear constraint g,
                "ubg": [np.ndarray] Upper bounds on the nonlinear constraint g
                }
        """

        # Create a parameter for the initial state.
        x0 = cs.SX.sym("x0", (4,1))
        x = x0

        # Create a decision variable for the controls
        u = [cs.SX.sym(f"u_{t}", (2,1)) for t in range(self.N)]
        self.u = u

        # Create the weights for the states
        Q = cs.diagcat(1, 3, 0.1, 0.01)
        QT = 5 * Q
        # controls weights matrix
        R = cs.diagcat(1., 1e-2)

        # Initialize the cost
        cost = 0

        # -- State bounds
        states_lb = np.array([params.min_pos_x, params.min_pos_y, params.min_heading, params.min_vel])
        states_ub = np.array([params.max_pos_x, params.max_pos_y, params.max_heading, params.max_vel])

        # -- Input bounds
        inputs_lb = np.array([params.min_drive, -params.max_steer])
        inputs_ub = np.array([params.max_drive, params.max_steer])

        lbx = []
        ubx = []
        g = []
        lbg = []
        ubg = []

        ode = KinematicBicycle(params, symbolic=True)
        f = forward_euler(ode, self.ts)

        # Build the cost function
        for i in range(self.N):
            cost += x.T @ Q @ x + u[i].T @ R @ u[i]
            x = f(x, u[i])
            lbx.append(inputs_lb)
            ubx.append(inputs_ub)
            g.append(x)   # Bound the state
            lbg.append(states_lb)
            ubg.append(states_ub)

        cost += x.T @ QT @ x

        variables = cs.vertcat(*u)
        nlp = {"f": cost,
            "x": variables,
            "g": cs.vertcat(*g),
            "p": x0}
        bounds = {"lbx": cs.vertcat(*lbx),
                "ubx": cs.vertcat(*ubx),
                "lbg": cs.vertcat(*lbg),
                "ubg": cs.vertcat(*ubg),
                }

        return nlp, bounds

    def reshape_input(self, sol):
        return np.reshape(sol["x"], ((-1, 2)))

    def __call__(self, y):
        """Solve the OCP for initial state y.

        Args:
            y (np.ndarray): Measured state
        """
        solution = self.solve(y)
        u = self.reshape_input(solution)
        return u[0]




class Bumper():
    def __init__(self, vehicle : VehicleParameters, position: np.ndarray, angle: float, n_c: int):
        self.vehicle = vehicle
        # Initial numerical position and angle (used by init_bumper)
        self.position = position 
        self.angle = angle
        self.n_c = n_c
        
        # self.local_centers: fixed in the vehicle frame (shape 2, n_c)
        self.local_centers = None 
        # self.global_centers: dynamic, holds the result of update_bumper (cs.SX or np.ndarray)
        self.global_centers = None
        self.r = None

    def init_bumper(self):
        """Calculates the radius and the static local centers."""
        d = self.vehicle.length/(2*self.n_c)
        self.r = np.sqrt((self.vehicle.width/2)**2 + d**2)
        
        # Calculate local centers (fixed in vehicle frame)
        local_centers_list = [
            [-self.vehicle.length / 2 + d * (1 + 2 * i), 0.0] for i in range(self.n_c)
        ]
        # Store as CasADi DM for use in both numerical and symbolic paths
        self.local_centers = cs.DM(np.array(local_centers_list).T) # Shape (2, n_c)
        
        # Apply initial rotation and translation (triggers numerical path)
        self.update_bumper(self.position, self.angle)

    def update_bumper(self, position: Union[np.ndarray, cs.SX], psi: Union[float, cs.SX]):
        """
        Updates the global bumper centers. Handles both numerical (NumPy) 
        and symbolic (CasADi) inputs.
        
        NOTE: This must always rotate/translate the original self.local_centers.
        """
        
        # --- NUMERICAL / SIMULATION CASE (np.ndarray) ---
        if isinstance(position, np.ndarray):
            # 1. Update the stored numerical state
            self.position = position
            self.angle = psi
            
            # 2. FIX: Reshape position from (2,) to (2, 1) for broadcasting
            position_col_vector = position.reshape(2, 1)
            
            R_psi = np.array([
                        [np.cos(psi), -np.sin(psi)],
                        [np.sin(psi), np.cos(psi)]
                    ])
            
            # Convert CasADi DM local centers to NumPy for the operation
            local_centers_np = np.array(self.local_centers)
            
            # Rotation (R_psi @ local_centers_np) + Translation (position_col_vector)
            self.global_centers = position_col_vector + R_psi @ local_centers_np

        # --- SYMBOLIC / OCP BUILD CASE (cs.SX) ---
        elif isinstance(position, cs.SX):
            # No reshape needed, as x[0:2] is already a (2, 1) column vector
            
            R_psi = cs.vertcat(
                cs.horzcat(cs.cos(psi), -cs.sin(psi)),
                cs.horzcat(cs.sin(psi), cs.cos(psi))
            )
            
            # CasADi handles the vector addition (translation) of the (2, 1) position
            self.global_centers = position + R_psi @ self.local_centers
            
        else:
            raise TypeError("Position input must be a NumPy array or a CasADi SX vector.")
        
        # Alias self.centers to global_centers for compatibility with OCP
        self.centers = self.global_centers 


class MPCControllerConstrained:
    def __init__(self, N: int, ts: float, bumper_controlled : Bumper, bumper_obstacle :Bumper, *, params: VehicleParameters):
        self.N = N
        self.ts = ts
        self.bumper_controlled = bumper_controlled
        self.bumper_obstacle = bumper_obstacle
        nlp_dict, self.bounds = self.build_ocp(params)

        opts = {"ipopt": {"print_level": 1}, "print_time": False}
        self.ipopt_solver = cs.nlpsol("solver", "ipopt", nlp_dict, opts)

    def solve(self, x) -> dict:
        return self.ipopt_solver(p=x, **self.bounds)

    def get_constraints(self):
            constraints = []
            
            # Calculate R^2 once outside the loop (R = r + r')
            R_sq = (self.bumper_controlled.r + self.bumper_obstacle.r)**2
            
            # Get the full CasADi center matrices
            C_controlled = self.bumper_controlled.centers
            C_obstacle = self.bumper_obstacle.centers

            for i in range(self.bumper_controlled.n_c):
                for j in range(self.bumper_obstacle.n_c):
                    
                    # 1. Index the centers to get single (2, 1) column vectors
                    c_i = C_controlled[:, i]          # CasADi SX vector (2, 1)
                    c_j_prime = C_obstacle[:, j]      # CasADi SX/DM vector (2, 1)
                    
                    # 2. Calculate the difference vector (z_ij)
                    diff = c_i - c_j_prime            # CasADi SX vector (2, 1)

                    # 3. Calculate the squared Euclidean distance: z_ij.T @ z_ij
                    dist_sq = diff.T @ diff
                    
                    # 4. Formulate the smooth constraint: R^2 - ||diff||^2 <= 0
                    constr = R_sq - dist_sq
                    
                    constraints.append(constr)
                    
            return constraints

    def build_ocp(self, params: VehicleParameters) -> Tuple[dict, dict]:
        """Builds the OCP using single shooting."""

        # Create a parameter for the initial state.
        x0 = cs.SX.sym("x0", (4,1))
        x = x0

        # Create a decision variable for the controls
        u = [cs.SX.sym(f"u_{t}", (2,1)) for t in range(self.N)]
        self.u = u

        # Create the weights for the states
        Q = cs.diagcat(1, 15, 0.4, 0.001)
        QT = 5 * Q + cs.diagcat(750,1000,100,0)
        # controls weights matrix
        R = cs.diagcat(1., 1e-4)

        # Initialize the cost
        cost = 0

        # -- State bounds
        states_lb = np.array([params.min_pos_x, params.min_pos_y, params.min_heading, params.min_vel])
        states_ub = np.array([params.max_pos_x, params.max_pos_y, params.max_heading, params.max_vel])

        # -- Input bounds
        inputs_lb = np.array([params.min_drive, -params.max_steer])
        inputs_ub = np.array([params.max_drive, params.max_steer])

        lbx = []
        ubx = []
        g = []
        lbg = []
        ubg = []

        ode = KinematicBicycle(params, symbolic=True)
        f = forward_euler(ode, self.ts)

        # Initial symbolic update (required for the first set of constraints at t=0)
        self.bumper_controlled.update_bumper(x[0:2], x[2])

        # Build the cost function and constraints
        for i in range(self.N):
            cost += x.T @ Q @ x + u[i].T @ R @ u[i]
            x = f(x, u[i])
            
            # Symbolic update inside the loop
            self.bumper_controlled.update_bumper(x[0:2], x[2])
            
            lbx.append(inputs_lb)
            ubx.append(inputs_ub)
            
            # State Bounds
            g.append(x)
            lbg.append(states_lb)
            ubg.append(states_ub)
            
            # Collision Constraints
            constraints_list = self.get_constraints() 
            g_collision = cs.vertcat(*constraints_list)
            g.append(g_collision)   

            num_constraints = len(constraints_list)
            lbg.append(cs.DM([-cs.inf] * num_constraints)) 
            ubg.append(cs.DM([0.0] * num_constraints))
            
        cost += x.T @ QT @ x    # Terminal cost

        variables = cs.vertcat(*u)
        nlp = {"f": cost,
            "x": variables,
            "g": cs.vertcat(*g),
            "p": x0}
            
        bounds = {"lbx": cs.vertcat(*lbx),
                "ubx": cs.vertcat(*ubx),
                "lbg": cs.vertcat(*lbg),
                "ubg": cs.vertcat(*ubg),
                }

        return nlp, bounds

    def reshape_input(self, sol):
        return np.reshape(sol["x"], ((-1, 2)))

    def __call__(self, y):
        solution = self.solve(y)
        u = self.reshape_input(solution)
        return u[0]




#-----------------------------------------------------------
# UTILITIES
#-----------------------------------------------------------


def plot_input_sequence(u_sequence, params: VehicleParameters):
    plt.subplot(2,2, (1,3))
    plt.title("Control actions")
    plt.plot(u_sequence[:,0], u_sequence[:,1], marker=".")
    bounds = Rectangle(np.array((params.min_drive, -params.max_steer)), params.max_drive-params.min_drive, 2*params.max_steer, fill=False)
    plt.gca().add_patch(bounds)
    plt.xlabel("$a$")
    plt.ylabel("$\\delta$");
    plt.subplot(2,2,2)
    plt.title("Steering angle")
    plt.plot(u_sequence[:,1].squeeze(), marker=".")
    style=dict(linestyle="--", color="black")
    plt.axhline(params.max_steer, **style)
    plt.axhline(-params.max_steer, **style)
    plt.ylabel("$\\delta$");
    plt.subplot(2,2,4)
    plt.title("Acceleration")
    plt.plot(u_sequence[:,0].squeeze(), marker=".")
    plt.axhline(params.min_drive, **style)
    plt.axhline(-params.max_drive, **style)
    plt.ylabel("$a$");
    plt.xlabel("$t$")
    plt.tight_layout()


def plot_state_trajectory(x_sequence, title: str = "Trajectory", ax = None, color="tab:blue", label: str=""):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    car_params = VehicleParameters()
    parking_area = Rectangle(-0.5*PARK_DIMS, *PARK_DIMS, ec="tab:green", fill=False)
    ax.add_patch(parking_area)
    extra_arg = dict()
    for i, xt in enumerate(x_sequence):
        if i==len(x_sequence)-1:
            extra_arg["label"]= label
        if i%2 == 0:  # Only plot a subset
            alpha = min(0.1 + i / len(x_sequence), 1)
            anchor = xt[:2] - 0.5 * np.array([car_params.length, car_params.width])
            car = Rectangle(anchor, car_params.length, car_params.width,
                angle=xt[2]/np.pi*180.,
                rotation_point="center",
                alpha=alpha,
                ec="black",
                fc=color,
                **extra_arg
            )
            ax.add_patch(car)
    plt.legend()
    ax.plot(x_sequence[:,0], x_sequence[:,1], marker=".", color="black")
    ax.set_xlabel("$p_x$ [m]")
    ax.set_ylabel("$p_y$ [m]")
    ax.set_aspect("equal")

def plot_states_separately(x_sequence):
    plt.subplot(4,1,1)
    plt.title("Position x")
    plt.plot(x_sequence[:,0].squeeze(), marker=".")
    plt.ylabel("$p_x$");
    plt.subplot(4,1,2)
    plt.title("Position y")
    plt.plot(x_sequence[:,1].squeeze(), marker=".")
    plt.ylabel("$y$")
    plt.subplot(4,1,3)
    plt.title("Angle")
    plt.plot(x_sequence[:,2].squeeze(), marker=".")
    plt.ylabel("$\\psi$")
    plt.subplot(4,1,4)
    plt.title("Velocity")
    plt.plot(x_sequence[:,3].squeeze(), marker=".")
    plt.ylabel("$v$")
    plt.xlabel("$t$")
    plt.tight_layout()


#-----------------------------------------------------------
# EXERCISES
#-----------------------------------------------------------

def question4():
    print("Assignment 4.4")
    x0 = np.array([0.3,-0.1,0,0])
    N = 30
    ts = 0.08
    obstacle_pos = np.array([0.25,0])

    n_c = 3
    bumper_controlled = Bumper(VehicleParameters(),x0[0:2], x0[2], n_c)
    bumper_obstacle = Bumper(VehicleParameters(), obstacle_pos, 0, n_c)
    bumper_controlled.init_bumper()
    bumper_obstacle.init_bumper()

    nstep = 100

    # Build the assumed model
    print("--Set up the MPC controller")
    controller = MPCControllerConstrained(N=N, ts=ts, bumper_controlled=bumper_controlled, bumper_obstacle=bumper_obstacle, params=VehicleParameters())

    bicycle_true = KinematicBicycle(VehicleParameters())
    dynamics_accurate = exact_integration(bicycle_true, ts)
    x_closed_loop_exact = simulate(x0, dynamics_accurate, n_steps=nstep, policy=controller)

    plt.figure()
    print("Using the accurate model")
    plot_state_trajectory(x_closed_loop_exact, color="tab:red", label="Real")
    
    plt.title("Trajectory (parameter error)")
    plt.show()

    print("---Extra: run an animation")
    anim = AnimateParking()
    anim.setup(x_closed_loop_exact, ts, obstacle_positions=[obstacle_pos])
    anim.add_car_trajectory(x_closed_loop_exact, color=(150, 10, 50))
    anim.trace(x_closed_loop_exact)
    anim.run()
    return 0

if __name__ == "__main__":
    question4()
    # exercise1()
    # exercise2()
    # exercise3()
    # exercise4()
    # exercise5()
