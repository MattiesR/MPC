from typing import List, Tuple
import numpy as np

import pyglet
import pyglet.gl as gl

from rcracers.resources import ResourceType, load

DEFAULT_LINE_WIDTH = 0.005


class Trajectory:
    """Renderable depicting a trajectory of positions."""

    def __init__(
        self,
        coordinates: np.ndarray,
        width=DEFAULT_LINE_WIDTH,
        color=(255, 255, 255),
        primitive=gl.GL_LINE_STRIP,
    ):
        """Initialize a trajectory

        Args:
            coordinates (np.ndarray): Coordinates
            width (float, optional): Width of the line. Defaults to DEFAULT_LINE_WIDTH.
            color (tuple, optional): Color of the line. Defaults to (255, 255, 255).
            primitive (_type_, optional): Type of line to draw. Defaults to gl.GL_LINE_STRIP.
        """
        self.batch = pyglet.graphics.Batch()
        self.v_list = None
        self.primitive = primitive
        self.width = width
        self.color = color
        self.coordinates = coordinates

    def draw(self):
        """Draw this line."""
        gl.glLineWidth(self.width)
        gl.glColor3f(*np.multiply(self.color, 1 / 255))
        self.batch.draw()

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value: np.ndarray):
        self._coordinates = value
        if self.v_list is not None:
            self.v_list.delete()

        self.v_list = self.batch.add(
            value.shape[0],
            self.primitive,
            None,
            ("v2f", value.flatten().tolist()),
        )

    def update(self, x: float, y: float, _: float = None):
        """Add a new point to this trajectory.

        Args:
            x (float): X coordinate
            y (float): Y coordinate
        """
        self.coordinates = np.append(self.coordinates, np.array([[x, y]]), axis=0)


class ColoredSprite:
    """Colored Sprite Renderable."""

    def __init__(
        self,
        name: str = "car",
        color: Tuple[int] = (230, 0, 0),
        scale: float = None,
        position: Tuple[int] = (0, 0),
    ):
        """Create an Colored Sprite. 
        
        The assets used to draw the sprite are
            - <name>-color.png (for the colored part)
            - <name>-frame.png (for the fixed part)
        The scale is loaded from `meta.json` provided with the assets if it is
        left unspecified.

        Args:
            name (str, optional): Name of the asset. Defaults to "car".
            color (Tuple[int], optional): Color of the sprite. Defaults to (230, 0, 0).
            scale (float, optional): Scale of the asset (meter/pixel). Defaults to None.
            position (Tuple[int], optional): Position of the sprite (in meters). Defaults to (0, 0).
        """
        self.name = name
        path, path_f = (
            ResourceType.ASSET.get_path(f"{name}-color.png"),
            ResourceType.ASSET.get_path(f"{name}-frame.png"),
        )

        if scale is None:
            # load scale from meta data
            meta: dict = load("meta.json", resource_type=ResourceType.ASSET).get(
                name, dict()
            )
            scale = meta.get("scale", 1.0)

        img: pyglet.image.AbstractImage = pyglet.image.load(path)
        img.anchor_x, img.anchor_y = img.width // 2, img.height // 2
        img_f: pyglet.image.AbstractImage = pyglet.image.load(path_f)
        img_f.anchor_x, img_f.anchor_y = img_f.width // 2, img_f.height // 2

        self.sprites: List[pyglet.sprite.Sprite] = [
            pyglet.sprite.Sprite(img, x=0, y=0, subpixel=True),
            pyglet.sprite.Sprite(img_f, x=0, y=0, subpixel=True),
        ]
        self.color: List[pyglet.sprite.Sprite] = [self.sprites[0]]
        self.color[0].color = color

        self.scale = scale
        self.position = position

    def draw(self):
        """Draw this colored sprite."""
        for s in self.sprites:
            s.draw()

    def update(self, x: float = None, y: float = None, rotation: float = None):
        """Update the position of this sprite

        Args:
            x (float): X coordinate
            y (float): Y coordinate
            rotation (float, optional): Rotation. Defaults to None.
        """
        for s in self.sprites:
            s.update(x, y, -np.rad2deg(rotation))

    @property
    def position(self):
        return self.sprites[0].position

    @position.setter
    def position(self, value):
        for s in self.sprites:
            s.position = value

    @property
    def scale(self):
        return self.sprites[0].position

    @scale.setter
    def scale(self, value):
        for s in self.sprites:
            s.scale = value
