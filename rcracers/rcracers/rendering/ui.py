from enum import Enum
from typing import Any, Sequence

from dataclasses import  fields, is_dataclass

import numpy as np

import pyglet


class Alignment(Enum):
    """Specifies alignment within a rectangle."""
    CENTER = (0.5, 0.5)
    CENTER_LEFT = (0.0, 0.5)
    CENTER_RIGHT = (1.0, 0.5)
    LOWER_CENTER = (0.5, 0.0)
    UPPER_CENTER = (0.5, 1.0)
    UPPER_LEFT = (0.0, 1.0)
    UPPER_RIGHT = (1.0, 1.0)
    LOWER_LEFT = (0.0, 0.0)
    LOWER_RIGHT = (1.0, 0.0)

    @classmethod
    def from_str(cls, label: str):
        """Get the alignment from a string."""
        return cls[str.upper(label).replace(" ", "_")]


class DataView:
    """Used to show a dataclass automatically on screen."""

    def __init__(
        self,
        info: Any,
        location: Alignment = Alignment.UPPER_LEFT,
        color: Sequence[int] = (50, 50, 55),
        text_color: Sequence[int] = (255, 255, 255, 255),
        size: Sequence[int] = (300, 200),
        scale: Sequence[float] = None,
        text_width: int = 30
    ):
        """Create an instance of a `DataView`.

        Args:
            info (Any): Dataclass to show.
            location (Alignment, optional): Alignment of the view. Defaults to Alignment.UPPER_LEFT.
            color (Sequence[int], optional): Background color. Defaults to (50, 50, 55).
            text_color (Sequence[int], optional): Text color. Defaults to (255, 255, 255, 255).
            size (Sequence[int], optional): Size of the view. Defaults to (300, 200) or computed using scale.
            scale (Sequence[float], optional): Compute the size based on the window size. Defaults to None.
            text_width (int, optional): Width of the text within the view. Defaults to 30.

        Raises:
            ValueError: The provided info is not a dataclass.
        """
        if not is_dataclass(info):
            raise ValueError("Expected dataclass for info.")
        if isinstance(location, str):
            location = Alignment.from_str(location)

        self.info = info
        self.background = pyglet.shapes.Rectangle(0, 0, size[0], size[1], color=color)
        self.location = location
        self.scale = scale
        self.text_color = text_color
        self.text_width = text_width

    @property
    def size(self):
        """Size of this view."""
        return self.background.width, self.background.height

    @size.setter
    def size(self, value: Sequence[int]):
        """Set the size of this view."""
        self.background.width, self.background.height = value

    @property
    def origin(self):
        """Origin of this view where the text is anchored."""
        return np.add(self.background.position, (18, -18))

    def draw(self, window: "Window"):
        """Draw the view on the provided window."""
        update = getattr(self.info, 'update', lambda: None)
        update()
        if self.scale is not None:
            self.size = np.multiply(self.scale, window.get_size())
        self.background.position = np.multiply(self.location.value, window.get_size())
        self.background.anchor_position = np.multiply(self.location.value, self.size)
        self.background.draw()

        msg = ''
        for f in fields(self.info):
            name, value = f.name, getattr(self.info, f.name)
            msg += f"{name}: {str(value)}" + '\n'*2

        document = pyglet.text.decode_text(msg)
        document.set_style(0, -1, dict(
            font_size=12,
            font_name="Consolas",
            color=self.text_color,
        ))
        label = pyglet.text.layout.TextLayout(
            document,
            width=self.background.width - self.origin[0],
            height=self.background.height - self.origin[1],
            wrap_lines=True,
            multiline=True
        )
        label.position = self.origin
        label.anchor_x = "left"
        label.anchor_y = "top"
        label.draw()