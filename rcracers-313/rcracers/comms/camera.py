from dataclasses import dataclass

import time
import logging
import numpy as np
import cv2

PYSPIN_AVAILABLE = True
try:
    from simple_pyspin import Camera

    # http://softwareservices.flir.com/Spinnaker/latest/_programmer_guide.html
except ModuleNotFoundError:
    logging.warn(
        "Pyspin is not installed correctly. Some functionality may not be available!"
    )
    PYSPIN_AVAILABLE = False

from rcracers.resources import ResourceType, load


@dataclass
class CameraOptions:
    """Camera options for opening a camera stream."""

    color: bool = True  # Record colors
    frame_rate: float = 50  # If None, the camera default is used. Otherwise, a value between 1.0 and 170.59769242737315
    reset_device: bool = False  # Reset the camera before the run? Not recommended as this takes some time.
    shutter_time: int = 1200  # Exposure time in microseconds
    exposure_gain: float = (
        -1
    )  # Exposure gain in dB (if negative set gain to max). If None then gain is continuous
    output_dir: str = "camera_captures"  # Directory to save camera captures
    pixel_mode: str = "BayerRG8"  # Pixel format

    def apply(self, cam: Camera):
        if self.frame_rate is not None:
            cam.AcquisitionFrameRateEnable = True
            cam.AcquisitionFrameRate = self.frame_rate
        else:
            cam.AcquisitionFrameRateEnable = False

        if self.pixel_mode is not None:
            cam.PixelFormat = self.pixel_mode

        if self.exposure_gain is not None:
            # To control the exposure settings, we need to turn off auto
            cam.GainAuto = "Off"
            # Set the gain to provided or the maximum of the camera.
            if self.exposure_gain > 0:
                gain = min(self.exposure_gain, cam.get_info("Gain")["max"])
                logging.info(f"Setting gain to {gain} dB")
                cam.Gain = gain
            else:
                cam.Gain = cam.get_info("Gain")["max"]
        else:
            cam.GainAuto = "Continuous"  # Off / Once / Continuous
        if self.shutter_time is not None:
            cam.ExposureAuto = "Off"
            cam.ExposureTime = self.shutter_time
        else:
            cam.ExposureAuto = "Continuous"  # Off / Once / Continuous

    @classmethod
    def load_default(cls):
        return cls()


@dataclass
class CameraParams:
    """
    Camera parameters determined during callibration.
        see: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    """

    intrinsic: np.ndarray  # intrinsic camera matrix (focal length, optical center)
    extrinsic: np.ndarray  # [rotation | translation] applied in pixel space (TODO should be in world space)
    distortion: np.ndarray  # distortion coefficients
    height: float  # height of the camera

    # meta-data from fit
    error: np.ndarray
    sample_size: int
    img_size: "tuple[int]"

    @classmethod
    def load_default(cls):
        return load(
            "camera_parameters.json", cls=cls, resource_type=ResourceType.CONFIG
        )


class CameraModel:
    """Model used to represent the camera."""

    def __init__(self, params: CameraParams = None):
        """Initialize this camera model

        Args:
            params (CameraParams, optional): The parameters of the camera. Defaults to None.
        """
        if params is None:
            params = CameraParams.load_default()
        self.params = params

        # get rectification map
        h, w = self.params.img_size[:2]
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.params.intrinsic,
            self.params.distortion,
            np.eye(3),
            self.params.intrinsic,
            (w, h),
            cv2.CV_16SC2,
        )

        # get inverse rotation
        angle = self.params.extrinsic[-1]
        self.map3 = np.array([
            [np.cos(angle), np.sin(-angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        self.map4 = np.array([
            [np.cos(-angle), np.sin(angle)],
            [np.sin(-angle), np.cos(-angle)]
        ])
        self.angle = angle

    def undistort(self, img: np.ndarray):
        """Undistort an image.

        Args:
            img (np.ndarray): Image to process.

        Returns:
            np.ndarray: Undistorted image.
        """
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
    
    def screen_to_world(self, points: np.ndarray, angle: float = None):
        """Transform screen coordinates to world.
            see: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html

        Args:
            points (np.ndarray): Points to transform
            angle (float, optional): Angle to transform. Defaults to None.

        Returns:
            (np.ndarray, float): (position, angle) in meters and radians respectively.
        """
        mat = self.params.intrinsic
        z = self.params.height
        transform = self.params.extrinsic[:-1]

        points = points.reshape((-1, points.shape[-1]))
        f, c = np.diag(mat[:2, :2]), mat[:2, 2]
        points = (points - c[np.newaxis, :]) * z / f[np.newaxis, :]
        res = np.squeeze(points @ self.map3.T + transform[np.newaxis, :])
        if angle is None:
            return res
        return res, np.mod(angle + self.angle, 2*np.pi)

    def world_to_screen(self, points: np.ndarray, angle: float = None):
        """
        Transform world coordinates to screen.
            see: https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html

        Args:
            points (np.ndarray): Points to transform
            angle (float, optional): Angle to transform. Defaults to None.

        Returns:
            (np.ndarray, float): (position, angle) in pixels and radians respectively.
        """
        mat = self.params.intrinsic
        z = self.params.height
        transform = self.params.extrinsic[:-1]

        points = points.reshape((-1, points.shape[-1]))
        f, c = np.diag(mat[:2, :2]), mat[:2, 2]
        points = (points - transform[np.newaxis, :]) @ self.map4.T
        res= np.squeeze((points * f[np.newaxis, :] / z) + c[np.newaxis, :])
        if angle is None:
            return res
        return res, np.mod(angle - self.angle, 2*np.pi)


class ImageStream:
    """Stream images from a camera.
    
    Example:
    >>> options = CameraOptions(frame_rate=20)
    >>> imp = Localization()
    >>> display_size = (imp.img_size[0]//2, imp.img_size[1]//2)
    >>> with ImageStream(options=options) as stream:
    >>>    while True:
    >>>        img = stream.read()
    >>>        img = cv2.resize(img, display_size)
    >>>        cv2.imshow('Camera Stream', img)
    >>>        if cv2.waitKey(1) == 27:
    >>>            break
    """

    def __init__(self, options: CameraOptions = None):
        """Initialize this stream

        Args:
            options (CameraOptions, optional): The configuration of the camera. Defaults to None.
        """
        self.options = CameraOptions.load_default() if options is None else options
        self.__cam = None
        self.__overflow = 0
        if self.options.reset_device:
            self.reset()

    def reset(self):
        """Reset the camera."""
        
        logging.debug("Resetting device.")
        with Camera() as cam:
            cam.DeviceReset()
            time.sleep(5)

    def __enter__(self):
        self.__cam = Camera()
        self.__cam.init()
        self.options.apply(self.__cam)
        self.__cam.start()
        self.__overflow = self._overflow_count()
        return self

    def __exit__(self, *_):
        self.__cam.stop()
        self.__cam = None

    def missed_frames(self):
        """Get the number of unprocessed frames."""
        return self._buffer_count() + self._overflow_count()

    def _overflow_count(self):
        count = self.__cam.TransferQueueOverflowCount
        overflow = count - self.__overflow
        self.__overflow = count
        return overflow

    def _buffer_count(self):
        return self.__cam.TransferQueueCurrentBlockCount

    def clear_buffer(self):
        """Clear the buffer of the camera."""
        while self._buffer_count() > 0:
            self.__cam.get_array(wait=True)

    def read(self):
        """Read an image from the camera."""
        if self._buffer_count() > 1:
            logging.warn(
                f"Missed {self.missed_frames()} camera frames since last read."
            )
            self.clear_buffer()

        img = self.__cam.get_array(wait=True)
        return cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
