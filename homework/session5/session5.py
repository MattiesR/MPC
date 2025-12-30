import sys, os

sys.path.append(os.path.split(__file__)[0])  # Allow relative imports

import numpy as np
import numpy.random as npr
import casadi as cs

import matplotlib.pyplot as plt

from given.problem import (
    system_info,
    get_linear_dynamics,
    get_system_equations,
    build_mhe,
    default_config,
    simulate,
    Config,
    ObserverLog,
    LOGGER,
)


def parse_covariances(Q: np.ndarray, R: np.ndarray, P: np.ndarray):
    """Parse covariance arguments."""
    f, h = get_system_equations(symbolic=True, noise=True)
    nx, nw, nv = system_info(f, h)
    return (
        np.eye(nw) * Q if isinstance(Q, float) else Q,
        np.eye(nv) * R if isinstance(R, float) else R,
        np.eye(nx) * P if isinstance(P, float) else P,
    )


class EKF:
    def __init__(
        self,
        f: cs.Function,
        h: cs.Function,
        x0: np.ndarray,
        *,
        Q: np.ndarray = 0.002**2,
        R: np.ndarray = 0.25**2,
        P: np.ndarray = 0.5**2,
        clipping: bool = False,
    ):
        Q, R, P = parse_covariances(Q, R, P)
        self.dfdx, self.dfdw, self.dhdx = get_linear_dynamics(f, h)
        self.f, self.h = f, h
        self.Q, self.R, self.P = Q, R, P
        self.x = x0
        self.clipping = clipping

    def __call__(self, y: np.ndarray, log: LOGGER = None):
        P = self.P
        C: np.ndarray = cs.DM.full(self.dhdx(self.x))

        # measurement update
        L = np.linalg.solve(C @ P @ C.T + self.R, C @ P).T
        P = P - L @ C @ P
        _x = cs.DM.full(self.x + L @ (y - self.h(self.x)))
        if self.clipping:
            _x = np.maximum(_x, 0.0)

        # time update
        w = np.zeros(self.Q.shape[0])
        A: np.ndarray = cs.DM.full(self.dfdx(_x, w))
        # general expression: P = A @ P @ A.T + G @ Q @ G.T
        # but in this case G = np.eye(3), so we can omit it.
        self.P = A @ P @ A.T + self.Q
        self.x = np.squeeze(cs.DM.full(self.f(_x, w)))

        # log results
        if log is not None:
            log("y", y)
            log("x", np.squeeze(_x))


class MHE:
    def __init__(
        self,
        f: cs.Function,
        h: cs.Function,
        horizon: int,
        *,
        clipping : bool = False,
        use_prior: bool = False,
        x0,
        Q: float = 0.002**2,
        R: float = 0.25**2,
    ):
        Q, R, _ = parse_covariances(Q, R, 0.0)
        self.f, self.h = f, h
        self.horizon = horizon
        self.use_prior = use_prior

        self.loss = lambda w, v: (
            w.T @ np.linalg.inv(Q) @ w + v.T @ np.linalg.inv(R) @ v
        )
        self.lbx, self.ubx = 0.0, 10.0

        self.y = []
        self.solver = self.build(horizon)
        
        if use_prior:
            # Use prior
            self.x0 = x0
            self.ekf = EKF(self.f,self.h, self.x0, Q=Q, R=R, clipping=clipping)
            self.P_estimates = [np.eye(3)*0.5] # High confidence in initial prior
            self.x_estimates = [x0]
            self.measurement_buffer = []

    @property
    def nx(self):
        nx, _, _ = system_info(self.f, self.h)
        return nx

    @property
    def nw(self):
        _, nw, _ = system_info(self.f, self.h)
        return nw

    def build(self, horizon: int):
        if self.use_prior:
            return build_mhe(
                self.loss,
                self.f,
                self.h,
                horizon,
                lbx=self.lbx,
                ubx=self.ubx,
                use_prior = self.use_prior
            )

        else:
            return build_mhe(
                self.loss,
                self.f,
                self.h,
                horizon,
                lbx=self.lbx,
                ubx=self.ubx,
            )
    def __call__(self, y: np.ndarray, log: LOGGER):
        
        if self.use_prior:
            # store the new measurement
            self.y.append(y)
            if len(self.y) > self.horizon:
                self.y.pop(0)


            # get solver and bounds
            solver = self.solver
            if len(self.y) < self.horizon:
                solver = self.build(len(self.y))
            
            # 1. Solve MHE using previous arrival cost
            x, _ = solver(self.P_estimates[0], self.x_estimates[0], self.y)

            # 2. Extract arrival state for future use
            self.x_estimates.append(x[-1, :])
            if len(self.x_estimates) > self.horizon + 1:
                self.x_estimates.pop(0)

            # 3. NOW update EKF (used only for covariance)
            self.ekf(y)
            self.P_estimates.append(self.ekf.P)
            if len(self.P_estimates) > self.horizon + 1:
                self.P_estimates.pop(0)

        else:
            # store the new measurement
            self.y.append(y)
            if len(self.y) > self.horizon:
                self.y.pop(0)

            # get solver and bounds
            solver = self.solver
            if len(self.y) < self.horizon:
                solver = self.build(len(self.y))

            # update mhe
            x, _ = solver(self.y)

        # update log
        log("x", x[-1, :])
        log("y", y)
   


def show_result(t: np.ndarray, x: np.ndarray, x_):
    fig, ax = plt.subplots(1, 1)
    c = ["C0", "C1", "C2"]
    h = []
    for i, c in enumerate(c):
        h += ax.plot(t, x_[..., i], "--", color=c)
        h += ax.plot(t, x[..., i], "-", color=c)
    ax.set_xlim(t[0], t[-1])
    if np.max(x_) >= 10.0:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.legend(
        h,
        [
            "$A_{\mathrm{est}}$",
            "$A$",
            "$B_{\mathrm{est}}$",
            "B",
            "$C_{\mathrm{est}}$",
            "C",
        ],
        loc="lower left",
        mode="expand",
        ncol=6,
        bbox_to_anchor=(0, 0.88, 1, 0.2),
        borderaxespad=0,
    )
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    return fig



def part_1():
    """Implementation for Exercise 1."""
    print("\nExecuting Exercise 1\n" + "-" * 80)

    # problem setup
    cfg = default_config()
    n_steps = 400

    # gather dynamics
    fs, hs = get_system_equations(symbolic=True, noise=True, Ts=cfg.Ts)

    # setup the extended kalman filter
    ekf = EKF(fs, hs, x0=cfg.x0_est, clipping=False)

    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    x = simulate(cfg.x0, f, n_steps=n_steps, policy=ekf, measure=h, log=log)
    t = np.arange(0, n_steps + 1) * cfg.Ts

    # plot output in `x` and `log.x`
    show_result(t, x, log.x)
    plt.show()


def part_2():
    """Implementation for Exercise 2."""
    print("\nExecuting Exercise 2\n" + "-" * 80)
    
    for horizon in [15, 25]:
        # problem setup
        cfg = default_config()
        n_steps = 400

        # gather dynamics
        fs, hs = get_system_equations(symbolic=True, noise=True, Ts=cfg.Ts)

        # setup the moving horizon estimator
        mhe = MHE(fs, hs, horizon=horizon)

        # prepare log
        log = ObserverLog()
        log.append("x", cfg.x0_est)  # add initial estimate

        # simulate
        f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
        x = simulate(cfg.x0, f, n_steps=n_steps, policy=mhe, measure=h, log=log)
        t = np.arange(0, n_steps+1) * cfg.Ts

        # plot output in `x` and `log.x`
        show_result(t, x, log.x)
        plt.show()

def part_3():
    """ Implementation for homework assignment. """
    print("\nExecuting Homework assignment session 5.\n" + "-"*80)
    
    for clipping in [False, True]:
        for horizon in [25, 10]:
            # problem setup
            cfg = default_config()
            n_steps = 400

            # gather dynamics
            fs, hs = get_system_equations(symbolic=True, noise=True, Ts=cfg.Ts)

            # setup the moving horizon estimator
            mhe = MHE(fs, hs, horizon=horizon,clipping=clipping, use_prior=True, x0=cfg.x0_est)
            mhe_np = MHE(fs, hs, horizon=horizon,clipping=clipping, use_prior=False, x0=cfg.x0_est)

            # prepare log
            log = ObserverLog()
            log.append("x", cfg.x0_est)  # add initial estimate

            # simulate
            f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
            x = simulate(cfg.x0, f, n_steps=n_steps, policy=mhe, measure=h, log=log)
            t = np.arange(0, n_steps+1) * cfg.Ts
            
            # plot output in `x` and `log.x`
            title = f"MHE with prior filtering: N={horizon}, clipping={clipping}"
            fig1 = show_result(t, x, log.x)
            # plt.title(title)

            log_np = ObserverLog()
            log_np.append("x", cfg.x0_est)
            x_np = simulate(cfg.x0, f, n_steps=n_steps, policy=mhe_np, measure=h, log=log_np)
            fig2 = show_result(t,x_np, log_np.x)
            title2 = f"MHE without prior filtering: N={horizon}, clipping={clipping}"
            # plt.title(title2)
            if args.figs:
                name1 = f"MHE_prior_N_{horizon}_clip_{clipping}"
                fig1.savefig(folder + name1)

                name2 = f"MHE_np_N_{horizon}_clip_{clipping}"
                fig2.savefig(folder + name2)
    plt.show()

import argparse

# GLOBAL VARIABLES
folder = "images/assignment5/"

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
    # part_1()
    # part_2()
    part_3()
