from typing import Callable, Tuple, Union
import numpy as np

# Define signatures for dynamics and policy. For type-hinting only. 
LOGGER = Callable[[str, any], None]
DYNAMICS = Callable[[np.ndarray, np.ndarray, int, LOGGER], np.ndarray]
MEASUREMENT = Callable[[np.ndarray, int, LOGGER], np.ndarray]
POLICY = Callable[[np.ndarray, int, LOGGER], np.ndarray]
REFERENCE = Union[np.ndarray, Callable[[np.ndarray, int, int, LOGGER], np.ndarray]]
OBSERVER = Callable[[np.ndarray, np.ndarray, int, LOGGER], np.ndarray]
TRACKER = Callable[[np.ndarray, np.ndarray, int, LOGGER], np.ndarray]


def t_dynamics(x: np.ndarray, u: np.ndarray, t: int, log: LOGGER) -> np.ndarray:
    """Discrete time dynamics callable. 

    Args:
        x (np.ndarray): state
        u (np.ndarray): input
        t (int): discrete time step
        log (LOGGER): logger callback

    Returns:
        np.ndarray: next state

    Example:
    >>> from rcracers.simulator import rk4
    >>> # discretize dynamics
    >>> @rk4(Ts=0.05)
    >>> def f(x, u):
    >>>     A = np.array([[0, 1], [0, 0]])
    >>>     B = np.array([[0], [1]])
    >>>     return A @ x + B @ u
    """
    return np.zeros(x.shape)


def t_measurement(x: np.ndarray, t: int, log: LOGGER) -> np.ndarray:
    """Measurement callable. 

    Args:
        x (np.ndarray): state
        t (int): discrete time step
        log (LOGGER): log callback

    Returns:
        np.ndarray: measurement

    Example:
    >>> def g(x):
    >>>     return x[:1] 
    """
    return x


def t_policy(y: np.ndarray, t: int, log: LOGGER) -> np.ndarray:
    """Policy callable.

    Args:
        y (np.ndarray): measurement
        t (int): discrete time step
        log (LOGGER): log callback

    Returns:
        np.ndarray: input
    """
    return np.zeros((0,))


def t_reference(x: np.ndarray, t: int, horizon: int, log: LOGGER) -> np.ndarray:
    """Reference callable.

    Args:
        x (np.ndarray): state
        t (int): discrete time step
        horizon (int): length of reference
        log (LOGGER): log callback

    Returns:
        np.ndarray: reference

    Example:
        # TODO update?
        see `rcracers.control.given.AdaptiveReference`.
    """
    return np.zeros((x.shape[0], horizon))

def t_observer(y: np.ndarray, u: np.ndarray, t: int, log: LOGGER) -> np.ndarray:
    """Observer callable

    Args:
        y (np.ndarray): measurement
        u (np.ndarray): input
        t (int): discrete time step
        log (LOGGER): log callback

    Returns:
        np.ndarray: state estimate

    Example:
    >>> class FiniteDifference:
    >>>     def __init__(self, Ts: float):
    >>>         self.Ts = Ts
    >>>         self.__prev = None       
    >>>     def __call__(self, y: np.ndarray):
    >>>         if self.__prev is None:
    >>>             self.__prev = y
    >>>             return np.array([y, 0.0])
    >>>         res = np.array([y, (y - self.__prev)/Ts])
    >>>         self.__prev = y
    >>>         return res
    """
    return y


def t_tracker(x: np.ndarray, r: np.ndarray, t: int, log: LOGGER) -> np.ndarray:
    """Tracker callable.

    Args:
        x (np.ndarray): state
        r (np.ndarray): reference
        t (int): discrete time step
        log (LOGGER): log callback

    Returns:
        np.ndarray: input
        
    Example:
    >>> def tracker(x, r, log):
    >>>     gain = np.array([[1, 0.5]])
    >>>     u = gain @ (r - x)
    >>>     log('u', u)
    """
    return np.zeros((0,))