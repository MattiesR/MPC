import numpy as np
import numpy.random as npr
import casadi as cs

import matplotlib.pyplot as plt

from problem import (
    get_system_equations,
    get_linear_dynamics,
    build_mhe,
    default_config,
    simulate,
    ObserverLog,
    LOGGER,
)


class EKF:
    def __init__(self, x0, Q, R, P, clipping:bool = False) -> None:
        """Create an instance of this `EKF`.
        TODO: Pass required arguments. You can use the output of
            `get_linear_dynamics`.
        """
        self.f, self.h = get_system_equations(symbolic=True, noise=True)
        self.dfdx, self.dfdw, self.dhdx = get_linear_dynamics(self.f, self.h)
        self.x = x0
        self.clipping = clipping

        self.Q = Q
        self.R = R
        self.P = P
        
        print("UNIMPLEMENTED: EKF.")
        print("Try using: `get_linear_dynamics`:")
        print(get_linear_dynamics.__doc__)

    def __call__(self, y: np.ndarray, log: LOGGER):
        """Process a measurement
            TODO: Implement EKF using the linearization produced by
                `get_linear_dynamics`.

        :param y: the measurement
        :param log: the logger for output
        """
        # log the state estimate and the measurement for plotting

        P = self.P
        C: np.ndarray = cs.DM.full(self.dhdx(self.x))

        # Measurement update
        L = np.linalg.solve(C @ P @ C.T + self.R, C @ P).T

        P = P - L @ C @ P
        _x = cs.DM.full(self.x + L @ (y - self.h(self.x)))

        if self.clipping:
            _x = np.maximum(_x, 0.0)

        w = np.zeros(self.Q.shape[0])
        A: np.ndarray = cs.DM.full(self.dfdx(_x, w))

        self.P = A @ P @ A.T + self.dfdw(_x, w) @ self.Q @ self.dfdw(_x,w).T
        self.x = np.squeeze(cs.DM.full(self.f(_x, w)))

        log("y", y)
        log("x", np.squeeze(_x))


class MHE:
    def __init__(self, x0, Q, R, P,sig_w, sig_v,N, clipping:bool = False, prior: bool = True) -> None:
        """Create an instance of this `MHE`.
        TODO: Pass required arguments and build the MHE problem using
            `build_mhe`. You can use the output of `get_system_equations`.
        """
        self.f, self.h = get_system_equations(symbolic=True, noise=True)
        self.dfdx, self.dfdw, self.dhdx = get_linear_dynamics(self.f, self.h)
        self.x = x0
        
        self.Q = Q
        self.R = R
        self.P = P

        self.sig_w = sig_w
        self.sig_v = sig_v
        self.N = N
        
        self.y = []

        self.clipping = clipping
        self.prior = prior
        print(build_mhe.__doc__)

    def build(self, horizon: int):
        return build_mhe(self.loss, self.f, self.h, horizon, lbx=self.lbx, ubx= self.ubx,self.prior)
    def __call__(self, y: np.ndarray, log: LOGGER):
        """Process a measurement
            TODO: Implement MHE using the solver produced by `build_mhe`.

        :param y: the measurement
        :param log: the logger for output
        """
        self.y.append(y)

        if len(self.y) > self.horizon:
            self.y.pop(0)
        
        solver = self.solver
        if len(self.y) < self.horizon:
            solver = self.build(len(self.y))
        loss = lambda w, v: w.T @ w / self.sig_w**2 + v.T @ v / self.sig_v**2
        solver = build_mhe(loss, self.f, self.h, self.N, lbx=0.0, ubx=10.0, use_prior=self.prior)
        _x, w = solver(P=np.eye(3), x0=self.x, y=np.zeros((10, 1)))
        self.x = _x[-1,:]

        # # log the state estimate and the measurement for plotting
        # P = self.P
        # C: np.ndarray = cs.DM.full(self.dhdx(self.x))

        # # Measurement update
        # L = np.linalg.solve(C @ P @ C.T + self.R, C @ P).T

        # P = P - L @ C @ P
        # _x = cs.DM.full(self.x + L @ (y - self.h(self.x)))

        # if self.clipping:
        #     _x = np.maximum(_x, 0.0)

        # w = np.zeros(self.Q.shape[0])
        # A: np.ndarray = cs.DM.full(self.dfdx(_x, w))

        # self.P = A @ P @ A.T + self.dfdw(_x, w) @ self.Q @ self.dfdw(_x,w).T
        # self.x = np.squeeze(cs.DM.full(self.f(_x, w)))

    
        # log the state estimate and the measurement for plotting
        log("y", y)
        log("x", _x[-1,:])

def show_result(t: np.ndarray, x: np.ndarray, x_: np.ndarray):
    _, ax = plt.subplots(1, 1)
    c = ["C0", "C1", "C2"]
    h = []
    for i, c in enumerate(c):
        h += ax.plot(t, x_[..., i], "--", color=c)
        h += ax.plot(t, x[..., i], "-", color=c)
    ax.set_xlim(t[0], t[-1])
    if np.max(x_) >= 10.0:
        ax.set_yscale('log')
    else:
        ax.set_ylim(-2.0, 2.0)
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
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        borderaxespad=0,
    )
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.show()

def part_1():
    """Implementation for Exercise 1."""
    print("\nExecuting Exercise 1\n" + "-" * 80)
    # problem setup
    cfg = default_config()
    n_steps = 400

    # setup the extended kalman filter
    x0 = np.array([1,0,4])
    Q = np.eye(3)*cfg.sig_w**2
    R = np.eye(1)*cfg.sig_v**2
    P = np.eye(3)*cfg.sig_p**2
    ekf = EKF(x0,Q,R,P,clipping=False)

    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    x = simulate(cfg.x0, f, n_steps=n_steps, policy=ekf, measure=h, log=log)
    t = np.arange(0, n_steps + 1) * cfg.Ts

    # plot output in `x` and `log.x`
    print(f'{x.shape=}')
    print(f'{log.x.shape=}')

    show_result(t, x, log.x)

# def parse_covariance(Q,R,P):
#     f,h = get_system_equations(symbolic=True, noise = True)
#     nx, nw,nv = system_info(f,h)

def part_2():
    """Implementation for Exercise 2."""
    print("\nExecuting Exercise 2\n" + "-" * 80)
    # problem setup
    cfg = default_config()

    # setup the extended kalman filter
    x0 = np.array([1,0,4])
    Q = np.eye(3)*cfg.sig_w**2
    R = np.eye(1)*cfg.sig_v**2
    P = np.eye(3)*cfg.sig_p**2

    N = 10
    # setup the extended kalman filter
    mhe = MHE(x0,Q,R,P,cfg.sig_w, cfg.sig_v, N, clipping=False, prior=True)
    
    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    n_steps = 400
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    x = simulate(cfg.x0, f, n_steps=n_steps, policy=mhe, measure=h, log=log)
    t = np.arange(0, n_steps + 1) * cfg.Ts

    # plot output in `x` and `log.x`
    print(f'{x.shape=}')
    print(f'{log.x.shape=}')

    show_result(t, x, log.x)


def part_3():
    """Implementation for Homework."""
    print("\nExecuting Homework\n" + "-" * 80)
    # problem setup
    cfg = default_config()

    # setup the extended kalman filter
    mhe = MHE()

    # prepare log
    log = ObserverLog()
    log.append("x", cfg.x0_est)  # add initial estimate

    # simulate
    f, h = get_system_equations(noise=(0.0, cfg.sig_v), Ts=cfg.Ts, rg=cfg.rg)
    x = simulate(cfg.x0, f, n_steps=400, policy=mhe, measure=h, log=log)

    # plot output in `x` and `log.x`
    print(f'{x.shape=}')
    print(f'{log.x.shape=}')


if __name__ == "__main__":
    # part_1()
    part_2()
    part_3()
