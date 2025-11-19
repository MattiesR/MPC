from dataclasses import dataclass, field
from enum import Enum

from rcracers.simulator.core import BaseControllerLog
from .camera import CameraOptions, CameraParams, ImageStream
from .localization import PositioningOptions, Positioning
from .racer import Racer

from ..control.signatures import POLICY, LOGGER
import numpy as np


class MarkerColor(Enum):
    GREEN = 0


@dataclass
class ControllerConfiguration:
    """Configuration of the control loop"""

    port: str = "COM3"  # Port of the transmitter
    channel: int = 1  # Channel on which to communicate with the car
    sample_period: float = 1 / 50  # Sample period of the controller
    cam_opts: CameraOptions = field(
        default_factory=CameraOptions.load_default
    )  # Camera for the image stream
    cam_params: CameraParams = field(
        default_factory=CameraParams.load_default
    )  # Camera parameters for filtering
    pos_opts: PositioningOptions = field(
        default_factory=PositioningOptions.load_default
    )  # Positioning configuration

    def __post_init__(self):
        if self.cam_opts is None:
            self.cam_opts = CameraOptions.load_default()
            self.cam_opts.frame_rate = int(1 / self.sample_period)
        if self.cam_params is None:
            self.cam_params = CameraParams.load_default()
        if self.pos_opts is None:
            self.pos_opts = PositioningOptions.load_default()

    @classmethod
    def load_default(
        cls,
        port: str = "COM3",
        channel: int = 1,
        color: MarkerColor = MarkerColor.GREEN,
    ):
        """Load default controller configuration from disk

        Args:
            port (str, optional): Port of the transmitter. Defaults to "COM3".
            channel (int, optional): Channel on which to communicate with the car. Defaults to 1.
            color (MarkerColor, optional): Color of the car (for marker detection). Defaults to MarkerColor.GREEN.

        Raises:
            ValueError: When the color is not supported

        Returns:
            ControllerConfiguration: The default configuration
        """
        res = cls(port=port, channel=channel)
        if color is MarkerColor.GREEN:
            res.pos_opts = PositioningOptions.load_green()
        else:
            raise ValueError(f"Unsupported color {color.name}")
        return res


class Controller:
    """Controller interface."""

    def __init__(
        self, policy: POLICY, log: BaseControllerLog, *, config: ControllerConfiguration
    ):
        """Create an instance of this controller with the provided policy

        Args:
            policy (POLICY): The policy to run on the car (y, t, log) -> u
            log (BaseControllerLog): The logger to pass to the policy
            config (ControllerConfiguration): The controller configuration.
        """
        self.config = config
        self.policy = policy
        self.log = log
        self.t = 0
        self.running = False

    @property
    def cam_opts(self):
        self.config.cam_opts

    @property
    def cam_params(self):
        self.config.cam_params

    @property
    def pos_opts(self):
        self.config.pos_opts

    def start(self):
        """Start the controller."""
        with ImageStream(self.cam_opts) as stream:
            img = stream.read()
            imp = Positioning(img, params=self.cam_params, options=self.pos_opts)

            with Racer(self.config.port) as racer:
                self.t = 0
                self.running = True
                while self.running:
                    img = stream.read()
                    pos, angle = imp(img)
                    u = self.policy(np.array([*pos, angle]), self.t, self.log.append)
                    racer.apply(u[0], u[1], channel=self.config.channel)
                    self.t += 1

    def stop(self):
        """Stop the controller."""
        self.running = False
