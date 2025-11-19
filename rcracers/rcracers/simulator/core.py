from abc import ABC
from dataclasses import field, is_dataclass
from dataclasses import asdict as _asdict
from typing import Tuple, Type

import numpy as np
import logging

from rich.progress import track
from functools import wraps

from ..control.signatures import t_policy, t_measurement, t_dynamics
from ..control.signatures import POLICY, MEASUREMENT, DYNAMICS, LOGGER
from ..control.sigexpand import expand


# shorthands
SIM_LOGGER = logging.getLogger(name=__name__)
list_field = lambda: field(default_factory=list, repr=False)


def asdict(obj):
    """Safe version of `asdict` for dataclasses.`

    Args:
        obj (Any): object to convert to dict

    Returns:
        dict: dictionary representation of dataclass (empty if not a dataclass).
    """
    if not is_dataclass(obj):
        return {}
    return _asdict(obj)


class BaseControllerLog(ABC):
    """Interface for controller logs."""

    def append(self, name: str, value):
        """Append a value to the given field of self.
        If the field does not exist, then a warning is raised and the value is
        discarded.

        Args:
            name (str): key of the list field to append to
            value: value to append

        Raises:
            AttributeError: no corresponding field
        """
        try:
            container = getattr(self, name)
            if not isinstance(container, list):
                raise AttributeError("Attribute is not a list.")
        except AttributeError:
            SIM_LOGGER.warning(
                f"ControllerLog does not keep track of field {name}. "
                "Discarding the given value. "
                f"Use one of the following fields: {[fld for fld in asdict(self).keys()]}"
            )
            return
        container.append(value)
        self._appended = True

    def merge(self, other: "BaseControllerLog"):
        """Merge the given ControllerLog object to self.
        This operation is performed in-place.

        Args:
            other: other control log to merge with.

        Raises:
            TypeError: other control log does not match this type.
        """
        """"""
        if other is None:
            return
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Can only append {type(self).__name__} object to {type(self).__name__}. Got {type(other)}"
            )

        for fieldname, container in asdict(self):
            container.extend(getattr(other, fieldname))

    def finish(self):
        """Called at the end of the simulation."""
        ...


def simulate(
    x0: np.ndarray,
    dynamics: DYNAMICS,
    n_steps: int,
    *,
    policy: POLICY = None,
    measure: MEASUREMENT = None,
    log: BaseControllerLog = None,
) -> Tuple[np.ndarray]:
    """Generic simulation loop.
    Simulate the discrete-time dynamics f: (x, u, t) -> x using policy 
    `policy`: (y, t) -> u for `n_steps` steps ahead starting from `x0` 
    and return the sequence of states. An optional logger can be provided 
    to be passed to the provided policy, measure and dynamics callables.

    Args:
        x0 (np.ndarray): (n,) Initial state
        dynamics (Callable): discrete-time dynamics (x, u, t, log) -> x
        n_steps (int): number of time steps to simulate
        policy (Callable): control policy (y, t, log) -> u, Defaults to u=None
        measure (Callable, optional): measurement function (y, t, log) -> y. Defaults to y=x
        log (Type, optional): Log instance. Defaults to None.

    Returns:
        x: sequence of states (n_steps, n)
    """
    # Expand signatures
    dynamics = expand(dynamics, template=t_dynamics)
    if policy is not None:
        policy = expand(policy, template=t_policy)
    if measure is not None:
        measure = expand(measure, template=t_measurement)

    # Create the log callback
    if log is None: 
        log_callback = lambda key, value: None
    else:
        log_callback = log.append

    # Main simulation loop
    x = [x0]
    for t in track(range(n_steps), description="Simulation ...", total=n_steps):
        xt = x[-1]  # Update current state
        if measure is None:
            yt = xt  # Default sensor
        else:
            yt = measure(xt, t, log_callback)  # Measure the output from the state

        if policy is None:
            ut = None
        else:
            ut = policy(yt, t, log_callback)  # Call the policy (with or without logger)
        x_next = dynamics(
            xt, ut, t, log_callback
        )  # Update the state using the dynamics
        x.append(x_next)  # Add the state to the simulation output
        if np.isnan(x_next).any():
            break # invalid state, returning early.

    if log is not None and getattr(log, "_appended", False):
        # the log was used
        log.finish()
    return np.array(x)


def fw_euler(f: DYNAMICS, /, *, Ts: float):
    """Discretize dynamics using forward Euler.

    Args:
        f (Callable): dynamics (x, u, t, log) -> x
        Ts (float): sample period
    """
    def wrap(f):
        f = expand(f, template=t_dynamics)
        
        def f_discrete(x, u, t):
            return x + Ts * f(x, u, t)

        return f_discrete

    if f is None:
        return wrap
    return wrap(f)


def rk4(f=None, /, *, Ts):
    """Discretize dynamics using RK4.

    Args:
        f (Callable): dynamics (x, u, t, log) -> x
        Ts (float): sample period
    """
    def wrap(f):
        fe = expand(f, template=t_dynamics)

        @wraps(f)
        def f_discrete(x=None, u=None, t=None, log=None):
            k1 = fe(x, u, t, log)
            k2 = fe(x + k1 * (Ts / 2), u, t, log)
            k3 = fe(x + k2 * (Ts / 2), u, t, log)
            k4 = fe(x + k3 * Ts, u, t, log)

            return x + (1 / 6) * Ts * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return f_discrete
    if f is None:
        return wrap
    return wrap(f)
