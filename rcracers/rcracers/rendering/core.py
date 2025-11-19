from typing import Dict, Protocol, Tuple
from inspect import signature

from dataclasses import dataclass

import numpy as np

import pyglet
import pyglet.gl as gl


DEFAULT_WINDOW_SIZE = (960, 640)
DEFAULT_CAMERA_CENTER = (-0.35, 0.65)
DEFAULT_CAMERA_MAGNIFY = 460


class Renderable(Protocol):
    """Renderable protocol that allows drawing objects to a window."""

    @property
    def position(self): 
        ...

    def draw(self, window: "Window"):
        ...


class Prop(Protocol):
    """Prop in a scene, the position of which can be updated."""
    def update(self, x: float = None, y: float = None, rotation: float = None):
        """Update the position of this Prop.

        Args:
            x (float, optional): X coordinate. Defaults to None.
            y (float, optional): Y coordinate. Defaults to None.
            rotation (float, optional): Rotation. Defaults to None.
        """
        ...

    def draw(self, window: "Window"):
        """Draw this prop on the provided window."""
        ...


@dataclass
class Camera:
    """Camera information for rendering."""

    width: int = DEFAULT_WINDOW_SIZE[0]
    height: int = DEFAULT_WINDOW_SIZE[1]
    magnify: float = DEFAULT_CAMERA_MAGNIFY
    center: Tuple[float] = DEFAULT_CAMERA_CENTER
    rotate: float = 90

    @property
    def height_ratio(self):
        return self.height / self.width

    @property
    def size(self):
        return (self.width, self.height)

    @size.setter
    def size(self, value):
        self.width, self.height = value

    @property
    def vertices(self):
        return [
            self.center[0] - self.width / self.magnify,
            self.center[0] + self.width / self.magnify,
            self.center[1] - self.height / self.magnify,
            self.center[1] + self.height / self.magnify,
        ]


class Actor:
    """Actor in a scene that supports animating Props."""

    def __init__(self, prop: Prop, coordinates: np.ndarray, time_step: float = 0.02, loop: bool = False):
        """Create an actor

        Args:
            prop (Prop): The prop to animate
            coordinates (np.ndarray): The trajectory of the prop
            time_step (float, optional): The time step of the animation. Defaults to 0.02.
            loop (bool, optional): Should the animation loop. Defaults to False.
        """
        self.prop = prop
        self.coordinates = coordinates
        self.time_step = time_step
        self.idx = 0
        self.loop = loop
        self.finished = False

    def update(self, _: float):
        """Update the animation."""
        if self.finished:
            return
        self.prop.update(*(self.coordinates[self.idx, ...]))
        if self.loop:
            idx = (self.idx + 1) % self.coordinates.shape[0]
            if idx == self.idx:
                self.finished = True
            self.idx = idx
        else:
            self.idx = min(self.idx + 1, self.coordinates.shape[0]-1)
    
    def draw(self):
        """Draw the actor on the screen."""
        self.prop.draw()


class Window(pyglet.window.Window):
    """Window of the application."""

    def __init__(
        self,
        background_color: Tuple[int] = (255, 255, 255, 255),
        camera: Camera = None,
        **kwargs,
    ):
        """Initialize the window.

        Args:
            background_color (Tuple[int], optional): The background color. Defaults to (255, 255, 255, 255).
            camera (Camera, optional): The camera configuration. Defaults to None.
        """
        if camera is None:
            camera = Camera()
        self.camera = camera
        super().__init__(width=camera.width, height=camera.height, **kwargs)
        self.objects: Dict[str, Renderable] = {}
        self.ui: Dict[str, Renderable] = {}
        self.background_color = background_color

    def run(self):
        """Run the application."""
        pyglet.app.run()

    def register(self, key: str, renderable: Renderable, *, ui: bool = False):
        """Register a renderable to this window.

        Args:
            key (str): Name of the renderable.
            renderable (Renderable): The renderable
            ui (bool, optional): Should the renderable be rendered on the UI Layer. Defaults to False.
        """
        if ui:
            self.ui[key] = renderable
        else:
            self.objects[key] = renderable

        if isinstance(renderable, Actor):
            pyglet.clock.schedule_interval(renderable.update, renderable.time_step)

    def pop(self, key: str, *, ui: bool = False):
        """Remove a renderable.

        Args:
            key (str): The name of the renderable
            ui (bool, optional): Does the renderable live on the UI Layer. Defaults to False.

        Returns:
            Renderable: The removed renderable
        """
        if ui:
            return self.ui.pop(key)
        return self.objects.pop(key)

    def on_resize(self, width, height):
        """Called when the size of the window changes.

        Args:
            width (float): The new width
            height (float): The new height
        """
        self.camera.size = (width, height)

    def set_size(self, width, height):
        """Set the size of the window directly.

        Args:
            width (float): The new width
            height (float): The new height
        """
        super().set_size(width, height)
        self.camera.size = (width, height)

    def apply_camera(self, *, ui: bool = False):
        """Apply the camera configuration

        Args:
            ui (bool, optional): Is the UI layer being updated. Defaults to False.
        """
        gl.glViewport(0, 0, self.width, self.height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        if ui:
            gl.glOrtho(0, self.camera.width, 0, self.camera.height, -1, 1)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
        else:
            gl.glOrtho(*self.camera.vertices, -1.0, 1.0)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()

    def draw_renderable(self, renderable: Renderable):
        """Draw a renderable

        Args:
            renderable (Renderable): The renderable to draw.
        """
        if len(signature(renderable.draw).parameters) == 0:
            renderable.draw()
        else:
            renderable.draw(self)

    def on_draw(self):
        """Called each update of the window."""
        gl.glClearColor(*np.multiply(1/255, self.background_color))
        self.clear()

        # render objects
        self.apply_camera(ui=False)
        for r in self.objects.values():
            self.draw_renderable(r)

        # render ui elements
        self.apply_camera(ui=True)
        for r in self.ui.values():
            self.draw_renderable(r)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Called when the mouse moves.

        Args:
            x (float): New x coordinate
            y (float): New y coordinate
            dx (float): Change in x coordinate
            dy (float): Change in y coordinate
            buttons: Button being pressed
            modifiers: Modifiers
        """
        self.camera.center = np.add(
            self.camera.center,
            (
                -2 * dx / self.camera.magnify,
                -2 * dy / self.camera.magnify,
            ),
        )

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Called when the scroll wheel is moved.

        Args:
            x (float): New x scroll
            y (float): New y scroll
            scroll_x (float): Change in x scroll
            scroll_y (float): Change in y scroll
        """
        self.camera.magnify *= 1.1**scroll_y
