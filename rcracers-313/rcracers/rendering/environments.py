from typing import Sequence

import numpy as np

import pyglet.gl as gl

from rcracers.resources.scene import Racetrack
from rcracers.rendering.objects import Trajectory, DEFAULT_LINE_WIDTH




class RacetrackEnvironment(Racetrack):
    """RaceTrack Renderable."""

    def __init__(
        self,
        inner: np.ndarray,
        outer: np.ndarray,
        center: np.ndarray,
        width: float = DEFAULT_LINE_WIDTH*3,
        color: Sequence[int] = (10, 10, 20),
        fill_color: Sequence[int] = (100, 100, 100),
    ):
        """Create an instance of this racetrack renderable.

        Args:
            inner (np.ndarray): The inner wall
            outer (np.ndarray): The outer wall
            center (np.ndarray): The center of the racetrack
            width (float, optional): The width of the lines. Defaults to DEFAULT_LINE_WIDTH*3.
            color (Sequence[int], optional): The color of the lines. Defaults to (10, 10, 20).
            fill_color (Sequence[int], optional): The background color. Defaults to (100, 100, 100).
        """
        super().__init__(inner, outer, center)

        # make sure arrays loop
        inner = np.vstack([inner, inner[:1, :]])
        center = np.vstack([center, center[:1, :]])
        outer = np.vstack([outer, outer[:1, :]])

        # create batch

        poly = np.zeros((inner.shape[0] + outer.shape[0], inner.shape[1]))
        poly[::2, :] = outer
        poly[1::2, :] = inner

        self.lines = [
            Trajectory(poly, width=width, color=fill_color, primitive=gl.GL_QUAD_STRIP),
            Trajectory(outer, width=width, color=color, primitive=gl.GL_LINE_LOOP),
            Trajectory(inner, width=width, color=color, primitive=gl.GL_LINE_LOOP),
            Trajectory(center, width=width, color=color, primitive=gl.GL_LINE_LOOP),
        ]

    def draw(self):
        """Draw this racetrack."""
        for line in self.lines:
            line.draw()