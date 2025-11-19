from enum import Enum
from shutil import ReadError
from uuid import uuid4
import numpy as np
from typing import Protocol, Tuple


from rcracers.rendering.core import Camera, Window, Actor
from rcracers.rendering.environments import RacetrackEnvironment
from rcracers.rendering.objects import Trajectory, ColoredSprite, DEFAULT_LINE_WIDTH
from rcracers.rendering.ui import DataView

from pyglet import gl


class TrajectoryBundel(Trajectory):
    """Renderable for handling multiple trajectories at once."""
    def update(self, *coordinates):
        self.coordinates = np.array(coordinates)


class Animator(Protocol):
    """Protocol for Animator classes used for animating state-space trajectories."""

    def setup(self, states: np.ndarray, time_step: float = 0.02):
        """Setup this animator.

        Args:
            states (np.ndarray): The state trajectory to animate
            time_step (float, optional): The time-step of the trajectory. Defaults to 0.02.
        """
        ...

    def trace(self, positions: np.ndarray, *, color=(0, 0, 150), width=3):
        """Add a line that is updated as the animator progressed.

        Args:
            positions (np.ndarray): (T, d) The positions to include in the trajectory.
            color (tuple, optional): The color of the line. Defaults to (0, 0, 150).
            width (int, optional): The width of the line. Defaults to 3.
        """
        ...

    def bundle(self, positions: np.ndarray, *, color=(0, 0, 150), width=3):
        """Add a list of positions that the animator loops through while animating.

        Args:
            positions (np.ndarray): (T, N, d) The positions to include in the trajectory.
            color (tuple, optional): The color of the line. Defaults to (0, 0, 150).
            width (int, optional): The width of the line. Defaults to 3.
        """


class AnimateRacetrack(Window):
    """Animate a race car moving on a race track."""
    def setup(self, states: np.ndarray, time_step: float = 0.02):
        """Setup this animator.

        Args:
            states (np.ndarray): The state trajectory to animate
            time_step (float, optional): The time-step of the trajectory. Defaults to 0.02.
        """
        self.set_size(720, 520)
        self.camera.center = (0.15, 0.75)
        self.background_color = (20, 20, 30, 1)

        self.time_step = time_step

        self.scene = RacetrackEnvironment.load_default()
        self.register("scene", self.scene)

        self.vehicle = ColoredSprite("car", position=(0.2, 0))
        self.actor = Actor(self.vehicle, states[:, :3], time_step=time_step)
        self.register("actor", self.actor)

    def trace(self, positions: np.ndarray, *, width=3, color=(0, 0, 150)):
        """Add a line that is updated as the animator progressed.

        Args:
            positions (np.ndarray): (T, d) The positions to include in the trajectory.
            color (tuple, optional): The color of the line. Defaults to (0, 0, 150).
            width (int, optional): The width of the line. Defaults to 3.
        """
        if not positions.ndim == 2 or positions.shape[-1] < 2:
            raise ValueError(
                f"Invalid states array shape. Expected (horizon x nb states) and got {positions.shape}"
            )
        name = f"trace-{uuid4()}"
        trace = Trajectory(np.empty((0, 2)), color=color, width=width)
        self.register(name, Actor(trace, positions[:, :2]))
        return name

    def bundle(self, positions: np.ndarray, *, width=3, color=(0, 0, 150)):
        """Add a list of positions that the animator loops through while animating.

        Args:
            positions (np.ndarray): (T, N, d) The positions to include in the trajectory.
            color (tuple, optional): The color of the line. Defaults to (0, 0, 150).
            width (int, optional): The width of the line. Defaults to 3.
        """
        if not positions.ndim == 3 or positions.shape[-1] < 2:
            raise ValueError(
                f"Invalid states array shape. Expected (time steps x horizon x nb states) and got {positions.shape}"
            )
        name = f"bundle-{uuid4()}"
        bundle = TrajectoryBundel(positions[0, :, :2], width=width, color=color)
        self.register(name, Actor(bundle, positions[..., :2]))
        return name


class Scenario(Enum):
    """Scenario type."""
    RACETRACK = AnimateRacetrack


def animator(
    x: np.ndarray, *, Ts: float = 0.02, scenario: Scenario = Scenario.RACETRACK
):
    """Create an animator.

    Args:
        x (np.ndarray): The states to animate
        Ts (float, optional): The time period. Defaults to 0.02.
        scenario (Scenario, optional): The type of scenario. Defaults to Scenario.RACETRACK.

    Returns:
        Animator: The animator
    """
    animator: Animator = scenario.value(resizable=True)
    animator.setup(x, Ts)
    return animator
