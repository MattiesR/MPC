
from typing import Callable
from given.homework import problem
import numpy as np 
import matplotlib.pyplot as plt

import os 
WORKING_DIR = os.path.split(__file__)[0]

def lqr_factor_step(N: int, nl: problem.NewtonLagrangeQP) -> problem.NewtonLagrangeFactors:
    ns = nl.Ak[0].shape[0]
    nu = nl.Bk[0].shape[1]

    K = np.zeros((N, nu, ns))
    e = np.zeros((N, nu, 1))
    s = np.zeros((N+1, ns, 1))
    P = np.zeros((N+1, ns, ns))

    s[N] = nl.qN
    P[N] = nl.QN

    for k in range(N-1, -1, -1):
        Rbar = nl.Rk[k] + nl.Bk[k].T @ P[k+1] @ nl.Bk[k]        # (nu, nu)

        Sbar = nl.Sk[k] + nl.Bk[k].T @ P[k+1] @ nl.Ak[k]        # (nu, ns)

        y = P[k+1] @ nl.ck[k].reshape(-1,1) + s[k+1]           # (ns,1)

        K[k] = -np.linalg.solve(Rbar, Sbar)                     # (nu, ns)
        e[k] = -np.linalg.solve(Rbar, nl.Bk[k].T @ y + nl.rk[k].reshape(-1,1))  # (nu,1)

        s[k] = Sbar.T @ e[k] + nl.Ak[k].T @ y + nl.qk[k].reshape(-1,1)          # (ns,1)
        P[k] = symmetric(nl.Qk[k] + nl.Ak[k].T @ P[k+1] @ nl.Ak[k] + Sbar.T @ K[k])

    return problem.NewtonLagrangeFactors(K, s, P, e)


def symmetric(P):
    return 0.5 * (P.T + P)

def lqr_solve_step(
    prob: problem.Problem,
    nl: problem.NewtonLagrangeQP,
    fac: problem.NewtonLagrangeFactors
) -> problem.NewtonLagrangeUpdate: 
    ns = prob.ns
    nu = prob.nu
    N = prob.N

    dx = np.zeros((N+1, ns)) # dx[0] = 0
    du = np.zeros((N, nu))
    p = np.zeros((N+1, ns))
    
    dx[0] = np.zeros((ns,))  # initial state is fixed
    for k in range(N):
        du[k] = fac.K[k] @ dx[k] + fac.e[k].ravel()
        dx[k+1] = nl.Ak[k] @ dx[k] + nl.Bk[k] @ du[k] + nl.ck[k].ravel()
        p[k+1] = fac.P[k+1] @ dx[k+1] + fac.s[k+1].ravel()
    return problem.NewtonLagrangeUpdate(dx, du, p)



def armijo_condition(merit: problem.FullCostFunction, x_plus, u_plus, x, u, dx, du, c, σ, α):
    φ, g, dJdx, dJdu = merit.phi, merit.h, merit.dJdx, merit.dJdu
    return φ(c, x_plus, u_plus) <= φ(c, x, u) + σ * α * (dJdx(x, u) @ dx + dJdu(x,u)@du - c * g(x,u))

def armijo_linesearch(zk: problem.NLIterate, update: problem.NewtonLagrangeUpdate, merit: problem.FullCostFunction, *, σ=1e-4) -> problem.NLIterate:
    alpha = 1.0
    beta = 0.5
    max_iter = 30

    x = zk.x
    u = zk.u
    

    dx = update.dx
    du = update.du

    p_plus = update.p

    c = np.linalg.norm(p_plus.ravel(), np.inf)*1.1
    for _ in range(max_iter):
        # Trial iterate
        x_plus = x + alpha * dx
        u_plus = u + alpha * du
        # Check Armijo condition
        if armijo_condition(merit, x_plus.ravel(), u_plus.ravel(), x.ravel(), u.ravel(), dx.ravel(), du.ravel(), c, σ, alpha):
            return problem.NLIterate(x_plus, u_plus, p_plus)

        # Reduce step size
        alpha *= beta

    # If line search fails, return the last trial iterate
    x_plus = x + alpha * dx
    u_plus = u + alpha * du

    return problem.NLIterate(x_plus, u_plus, p_plus)

def update_iterate(zk: problem.NLIterate, update: problem.NewtonLagrangeUpdate, *, linesearch: bool, merit_function: problem.FullCostFunction=None) -> problem.NLIterate:
    """Take the current iterate zk and the Newton-Lagrange update and return a new iterate. 

    If linesearch is True, then also perform a linesearch procedure. 

    Args:
        zk (problem.NLIterate): Current iterate 
        update (problem.NewtonLagrangeUpdate): Newton-Lagrange step 
        linesearch (bool): Perform line search or not? 
        merit_function (problem.FullCostFunction, optional): The merit function used for linesearch. Defaults to None.

    Raises:
        ValueError: If no merit function was passed, but linesearch was requested. 

    Returns:
        problem.NLIterate: Next Newton-Lagrange iterate.
    """
    
    if linesearch:
        if merit_function is None:
            raise ValueError("No merit function was passed but line search was requested")
        return armijo_linesearch(zk, update, merit_function)
    
    # Hint: The initial state must remain fixed. Only update from time index 1!
    xnext = zk.x.copy() 
    unext = zk.u.copy()
    pnext = update.p.copy()

    # Update controls
    unext += update.du

    # Update states [x0, fixed]
    xnext[1:] += update.dx[1:]
    return problem.NLIterate(
        x = xnext,
        u = unext, 
        p = pnext 
    )


def is_posdef(M):
    return np.min(np.linalg.eigvalsh(M)) > 0

def regularize(qp: problem.NewtonLagrangeQP):
    """Regularize the problem.

    If the given QP (obtained as a linearization of the problem) is nonconvex, 
    add an increasing multiple of the identity to the Hessian 
    until it is positive definite. 

    Side effects: the passed qp is modified by the regularization!

    Args:
        qp (problem.NewtonLagrangeQP): Linearization of the optimal control problem
    """
    for k in range(len(qp.Qk)):
        reg_factor = 1e-6
        
        H_block = np.block([[qp.Qk[k], qp.Sk[k].T], [qp.Sk[k], qp.Rk[k]]])
        while not is_posdef(H_block):
            qp.Qk[k] += np.eye(qp.Qk[k].shape[0]) * reg_factor
            qp.Rk[k] += np.eye(qp.Rk[k].shape[0]) * reg_factor
            H_block = np.block([[qp.Qk[k], qp.Sk[k].T], [qp.Sk[k], qp.Rk[k]]])
            reg_factor *= 2

def newton_lagrange(p: problem.Problem,
         initial_guess = problem.NLIterate, cfg: problem.NewtonLagrangeCfg = None, *,
         log_callback: Callable = lambda *args, **kwargs: ...
) -> problem.NewtonLagrangeStats:
    """Newton Lagrange method for nonlinear OCPs
    Args:
        p (problem.Problem): The problem description 
        initial_guess (NLIterate, optional): Initial guess. Defaults to problem.NewtonLagrangeIterate.
        cfg (problem.NewtonLagrangeCfg, optional): Settings. Defaults to None.
        log_callback (Callable): A function that takes the iteration count and the current iterate. Useful for logging purposes. 

    Returns:
        Solver stats  
    """
    stats = problem.NewtonLagrangeStats(0, initial_guess)
    # Set the default config if None was passed 
    if cfg is None:
        cfg = problem.NewtonLagrangeCfg()

    # Get the merit function ingredients in case line search was requested 
    if cfg.linesearch:
        full_cost = problem.build_cost_and_constraint(p)
    else: 
        full_cost = None # We don't need it in this case 
    
    QP_sym = problem.construct_newton_lagrange_qp(p)
    zk = initial_guess

    min_eigvals_iterations = []
    for it in range(cfg.max_iter):
        qp_it = QP_sym(zk)
        
        if cfg.regularize:
            regularize(qp_it)

        # obtain min eigenvalue of the Hessian of the QP
        min_eigvals = []
        for k in range(p.N):
            # Calculate eigenvalues of the Block Hessians
            Qk, Sk, Rk = qp_it.Qk[k], qp_it.Sk[k], qp_it.Rk[k]
            Hk = np.block([[Qk, Sk.T], [Sk, Rk]])
            min_eig = np.min(np.linalg.eigvalsh(Hk))
            min_eigvals.append(min_eig)
        overall_min_eig = min(min_eigvals)
        min_eigvals_iterations.append(overall_min_eig)

        factor = lqr_factor_step(p.N, qp_it)
        update = lqr_solve_step(p, qp_it, factor)

        zk = update_iterate(zk, update, linesearch=cfg.linesearch, merit_function=full_cost)

        stats.n_its = it 
        stats.solution = zk 
        # Call the logger. 
        log_callback(stats)

        # Sloppy heuristics as termination criteria.
        # In a real application, it's better to check the violation of the KKT conditions.
        # e.g., terminate based on the norm of the gradients of the Lagrangian.
        if np.linalg.norm(update.du.squeeze(), ord=np.inf)/np.linalg.norm(zk.u) < 1e-4:
            stats.exit_message = "Converged"
            stats.success = True 
            plot_min_eigvals(min_eigvals_iterations)
            return stats

        elif np.any(np.linalg.norm(update.du) > 1e4): 
            stats.exit_message = "Diverged"
            plot_min_eigvals(min_eigvals_iterations)
            return stats
    

    stats.exit_message = "Maximum number of iterations exceeded"
    return stats

def plot_min_eigvals(min_eigvals_iterations):
        plt.figure()
        plt.plot(min_eigvals_iterations)
        plt.title("Minimum eigenvalue of the QP Hessians over iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Minimum eigenvalue")
        plt.grid()
        plt.show()

def exercise1():
    print("Assignment 6.1.")
    p = problem.Problem()
    qp = problem.construct_newton_lagrange_qp(p)

def fw_euler(f, Ts):
    return lambda x,u,t: x + Ts*f(x,u)

def test_linear_system():
    p = problem.ToyProblem(cont_dynamics = problem.LinearSystem(), N=100)
    u0 = np.zeros((p.N, p.nu))
    x0 = np.zeros((p.N+1, p.ns))
    x0[0] = p.x0
    initial_guess = problem.NLIterate(x0, u0, np.zeros_like(x0))

    logger = problem.Logger(p, initial_guess)
    result = newton_lagrange(p, initial_guess, log_callback=logger)
    assert result.success, "Newton Lagrange did not converge on a linear system! Something is wrong!"
    assert result.n_its < 2, "Newton Lagrange took more than 2 iterations!"

def exercise2():
    print("Assignment 6.2.")
    from rcracers.simulator.core import simulate
    
    # Build the problem 
    p = problem.ToyProblem()


    N = p.N
    nx = p.ns
    nu = p.nu

    x = np.zeros((N+1, nx))
    u = np.zeros((N, nu))
    lam = np.zeros((N+1, nx))

    x[0] = p.x0

    initial_guess = problem.NLIterate(x=x, u=u, p=lam)
    logger = problem.Logger(p, initial_guess)
    
    cfg = problem.NewtonLagrangeCfg(linesearch=False, max_iter=100)

    stats = newton_lagrange(p, initial_guess, log_callback=logger, cfg=cfg)
    print(stats.exit_message)
    from given.homework import animate
    animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, "Assignment6-2"))


def exercise34(linesearch:bool):
    print("Assignment 6.3 and 6.4.")
    from rcracers.simulator.core import simulate
    f = problem.ToyDynamics(False)

    # Build the problem 
    p = problem.ToyProblem()

    # Select initial guess by running an open-loop simulation
    N = p.N
    nx = p.ns
    nu = p.nu

    # x = np.zeros((N+1, nx)) # run openloop simulation
    # # x[0] = p.x0
    u = np.zeros((N, nu)) # zeros as initial guess for input 
    lam = np.zeros((N+1, nx)) # zeros as initial guess for costates
    x = simulate(p.x0, fw_euler(f,p.Ts), p.N, policy=lambda y,t: u[t])
    t = np.linspace(0,1,N+1)
    fig = plt.figure()
    plt.plot(t,x)
    plt.title("Open loop simulation of states")
    plt.legend(["x_1","x_2","x_3"])
    plt.xlabel("Predicted time step k")
    plt.show()
    # fig.savefig("images/openloop.pdf")
    initial_guess = problem.NLIterate(x=x, u=u, p=lam) 
    
    # print(initial_guess)
    logger = problem.Logger(p, initial_guess)
    cfg = problem.NewtonLagrangeCfg(linesearch=linesearch, max_iter=200)
    final_iterate = newton_lagrange(p, initial_guess, log_callback=logger, cfg=cfg)
    print(final_iterate.exit_message)
    from given.homework import animate
    animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, "Assignment6-4"), ylim=(-10, 15))


def exercise56(regularize=False):
    # Build the problem 
    p = problem.ParkingProblem()
    
    # Select initial guess by running an open-loop simulation
    N = p.N
    ns = p.ns
    nu = p.nu

    x = np.zeros((N+1,ns))
    u = np.zeros((N,nu))
    lam = np.zeros((N+1,ns))
    x[0] = p.x0
    initial_guess = problem.NLIterate(x=x, u=u, p=lam) 
    logger = problem.Logger(p, initial_guess)
    cfg = problem.NewtonLagrangeCfg(linesearch=True, max_iter=200, regularize=regularize)
    final_iterate = newton_lagrange(p, initial_guess, log_callback=logger, cfg=cfg)
    print(final_iterate.exit_message)

    from given.homework import animate
    animate.animate_iterates(logger.iterates, os.path.join(WORKING_DIR, f"Assignment6-5-reg{regularize}"))
    animate.animate_positions(logger.iterates, os.path.join(WORKING_DIR, f"Assignment6-5-parking_regularize-{regularize}"))


if __name__ == "__main__":
    # test_linear_system()
    # exercise2()
    # exercise34(False)
    # exercise34(True)
    exercise56(regularize=False)
    # exercise56(regularize=True)