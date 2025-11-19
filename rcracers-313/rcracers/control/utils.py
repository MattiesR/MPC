import numpy as np
from typing import List
from dataclasses import dataclass

from ..simulator.core import list_field, BaseControllerLog
from .sigexpand import expand
from .signatures import t_reference, t_tracker, t_observer
from .signatures import REFERENCE, TRACKER, OBSERVER, LOGGER


class AdaptiveReference:
    """Reference that adapts to the current state."""
    def __init__(self, reference: np.ndarray, *, continue_constant: int = 10):
        """Initialize an adaptive reference

        Args:
            reference (np.ndarray): The reference trajectory
            continue_constant (int, optional): Specifies how many time-steps it should take before the reference adapts. Defaults to 10.
        """
        self.reference = reference
        self.continue_constant = continue_constant
        self.__nb_constant = None
        self.__current_idx = None

    def __call__(self, x: np.ndarray, horizon: int = 1):
        """Construct reference starting (approximately) at x.

        Args:
            x (np.ndarray): current state
            horizon (int): length of reference

        Returns:
            np.ndarray: reference trajectory
        """
        if self.__current_idx is None or self.__nb_constant > self.continue_constant:
            dist = np.linalg.norm(
                self.reference[:, :2] - x[np.newaxis, :2], ord=2, axis=1
            )
            idx = np.argmin(dist)
            idx = np.arange(idx, idx + horizon)
            self.__nb_constant = 0
            self.__current_idx = idx

        self.__nb_constant += 1
        self.__current_idx = (self.__current_idx + 1) % self.reference.shape[0]
        return self.reference[self.__current_idx, :]

    @classmethod
    def from_positions(cls, positions: np.ndarray, v: float, Ts: float):
        """Resample reference from sequence of positions.

        Args:
            points (np.ndarray): position trajectory
            v (float): reference velocity
            Ts (float): sample period

        Returns:
            ReferenceGenerator: an instance of a reference generator.
        """
        # close the loop
        nb_points = positions.shape[0]
        positions = np.vstack([positions, positions[:1, :]])

        # construct parametric curves
        distance = np.zeros((nb_points + 1,))
        angles = np.empty((nb_points))
        for i in range(nb_points):
            dx = positions[i + 1, 0] - positions[i, 0]
            dy = positions[i + 1, 1] - positions[i, 1]
            distance[i + 1] = np.sqrt(dx**2 + dy**2)
            angles[i] = np.arctan2(dy, dx)

        # cumulative distance is the parameter
        s = np.cumsum(distance)

        # correct angles
        angles = np.unwrap(np.array([angles[-1], *angles, angles[0]]))
        angles = (angles[:-1] + angles[1:]) / 2

        # resample
        nb_samples = int(np.floor(s[-1] / (v * Ts)))
        s_ = [i * v * Ts for i in range(nb_samples)]

        # resample parametric curve
        ref = np.zeros((nb_samples, 4))
        ref[:, 0] = np.interp(s_, s, positions[:, 0])
        ref[:, 1] = np.interp(s_, s, positions[:, 1])
        ref[:, 2] = np.interp(s_, s, angles)
        ref[:, 3] = np.full((nb_samples,), v)

        return cls(ref)


@dataclass
class ControllerLog(BaseControllerLog):
    """Logger that supports some basic fields."""
    x: List[np.ndarray] = list_field()  # State estimate
    r: List[np.ndarray] = list_field()  # Reference
    u: List[np.ndarray] = list_field()  # Inputs
    y: List[np.ndarray] = list_field()  # Measurements

    def finish(self):
        self.x = np.array(self.x)
        self.r = np.array(self.r)
        self.u = np.array(self.u)



class StackedPolicy:
    """Combines an observer, (adaptive) reference and observer into one policy."""
    def __init__(
        self,
        tracker: TRACKER,
        observer: OBSERVER = None,
        reference: REFERENCE = None,
        *,
        horizon: int = 1
    ):
        self.tracker = expand(tracker, template=t_tracker)
        self.observer = expand(observer, template=t_observer)
        self.reference = expand(reference, template=t_reference)
        self.__last_ctrl = None
        self.horizon = horizon

    def __call__(self, y: np.ndarray, t: int, log: LOGGER) -> np.ndarray:
        """Compute the control action based on the provided measurement.

        Args:
            y (np.ndarray): Measurement
            t (int): Current time step
            log (LOGGER): Logger callback

        Returns:
            np.ndarray: The control action
        """
        # execute observer
        x = self.observer(y, self.__last_ctrl, t, log)

        # generate reference
        if hasattr(self.reference, "__call__"):
            r = self.reference(y, t, self.horizon, log)
        elif isinstance(self.reference, np.ndarray) and self.reference.ndim > 1:
            r = self.reference[t, ...]
        else:
            r = self.reference
        r = np.squeeze(r)

        # execute tracking policy
        u = self.tracker(x, r, t, log)

        # log results
        log("x", x)
        log("r", r)
        log("u", u)        
        log("y", y)


        # return result
        self.__last_ctrl = u
        return u
