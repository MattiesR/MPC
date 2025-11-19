from dataclasses import dataclass, field

import logging
from typing import Tuple
import numpy as np
import cv2

import matplotlib.pyplot as plt


from .camera import CameraModel, CameraParams
from ..resources.scene import Racetrack
from ..resources import load, ResourceType


MISSING_MASK = '__default'


@dataclass
class PositioningOptions:
    """Configuration of the Positioning system."""

    lower: Tuple[int] = (125, 0.4, 0.7)  # hsv bound (0-360, 0-1, 0-1)
    upper: Tuple[int] = (30, 0.2, 0.25)  # hsv bound (0-360, 0-1, 0-1)
    marker_size: Tuple[int] = (10, 150)  # expected range for marker area
    blur: Tuple[float] = None  # gaussian blur
    undistort: bool = True  # remove camera distortion
    mask: np.ndarray = field(repr=False, default=MISSING_MASK)  # the camera mask
    margins: Tuple[int] = (150, 150)  # margins for cropping
    correct: Tuple[int] = (-0.02, 0, -0.012) # correction (px, py, angle)

    def __post_init__(self):
        if self.mask == MISSING_MASK:
            self.mask = get_mask(Racetrack.load_default())

    @classmethod
    def load_default(cls):
        """Load default positioning configuration (i.e. detect green markers)."""
        return cls.load_green()

    @classmethod
    def load_green(cls):
        """Load positioning configuration for the green markers."""
        # return cls(lower=(90, 0.2, 0.3), upper=(255, 0.7, 0.8), correct=(-0.02, 0, -0.012))
        return load('positioning_options_green.json', cls=cls, resource_type=ResourceType.CONFIG)

    @classmethod
    def load_white(cls):
        """Load positioning configuration for the white markers."""
        # return cls(lower=0.7, upper=0.8, correct=(-0.012, 0, -0.05))
        return load('positioning_options_white.json', cls=cls, resource_type=ResourceType.CONFIG)


def get_mask(
    track: Racetrack,
    *,
    loop: bool = True,
    model: CameraModel = None,
    margin: float = 0.05,
):
    """Generate a track mask based on inner and outer track boundaries.

    Args:
        track (Racetrack): The racetrack
        loop (bool, optional): Do the coordinates loop. Defaults to True.
        model (CameraModel, optional): The camera model. Defaults to CameraModel().
        margin (float, optional): Scale down the track. Defaults to 0.05.

    Returns:
        np.ndarray: The track mask
    """
    if model is None:
        model = CameraModel()
    shape = model.params.img_size

    def _draw_vertex(img: np.ndarray, x1, x2, x3):
        triangle_cnt = np.array([x1, x2, x3])
        cv2.drawContours(img, [triangle_cnt], 0, (255, 255, 255), -1)

    mask = np.zeros(shape[:2], dtype=np.uint8)
    inner, outer = track.inner, track.outer

    # scale towards center
    inner = (1 - margin) * inner + margin * track.center
    outer = (1 - margin) * outer + margin * track.center

    # convert to pixel coordinates
    nb_steps = inner.shape[0]
    for i in range(0 if loop else 1, nb_steps):
        y1, x1 = model.world_to_screen(np.array([inner[i - 1, 0], inner[i - 1, 1]]))
        y2, x2 = model.world_to_screen(np.array([inner[i, 0], inner[i, 1]]))
        y3, x3 = model.world_to_screen(np.array([outer[i - 1, 0], outer[i - 1, 1]]))
        y4, x4 = model.world_to_screen(np.array([outer[i, 0], outer[i, 1]]))

        xx1 = (int(x1), int(y1))
        xx2 = (int(x2), int(y2))
        xx3 = (int(x3), int(y3))
        xx4 = (int(x4), int(y4))

        _draw_vertex(mask, xx1, xx2, xx3)
        _draw_vertex(mask, xx2, xx3, xx4)

    return mask


class MarkerException(Exception):
    ...


class Localization:
    """Localization of markers."""

    def __init__(self, params: CameraParams = None, options: PositioningOptions = None):
        """Initialize the Localization process.

        Args:
            params (CameraParams, optional): The camera configuration. Defaults to None.
            options (PositioningOptions, optional): The positioning configuration. Defaults to None.
        """
        # process options
        self.params = CameraParams.load_default() if params is None else params
        self.model = CameraModel(self.params)
        self.options = PositioningOptions.load_default() if options is None else options

    @property
    def img_size(self):
        return self.model.params.img_size

    def correct_image(self, img: np.ndarray, crop: Tuple[int] = None):
        """Correct the image using the camera configuration.

        Args:
            img (np.ndarray): Image to correct
            crop (Tuple[int], optional): The range of pixels to crop (xmin, ymin, xmax, ymax). Defaults to None.

        Returns:
            np.ndarray: Corrected image
        """
        # undistort
        if self.options.undistort:
            img = self.model.undistort(img)

        # apply crop
        if crop is not None:
            img = img[crop[0] : crop[2], crop[1] : crop[3]]

        # apply mask
        mask = self.options.mask
        if not mask is None:
            if crop is not None:
                mask = mask[crop[0] : crop[2], crop[1] : crop[3]]
            img = cv2.bitwise_and(img, img, mask=mask)

        return img

    def prepare_image(self, img: np.ndarray):
        """Prepare a corrected image for marker detection.

        Args:
            img (np.ndarray): Image to prepare

        Returns:
            np.ndarray: Prepared image
        """
        # apply gaussian blur
        if self.options.blur is not None:
            img = cv2.GaussianBlur(img, self.options.blur, 0)
        return img

    def isolate_markers(self, img: np.ndarray):
        """Threshold the image to isolate the markers

        Args:
            img (np.ndarray): Image to threshold

        Returns:
            np.ndarray: Thresholded image.
        """
        # color detection
        lower, upper = self.options.lower, self.options.upper
        if isinstance(lower, float):
            thresholded = cv2.inRange(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                int(lower * 255),
                int(upper * 255),
            )
        else:
            lower = np.array((lower[0] / 2, lower[1] * 255, lower[2] * 255)).astype(
                np.uint8
            )
            upper = np.array((upper[0] / 2, upper[1] * 255, upper[2] * 255)).astype(
                np.uint8
            )
            thresholded = cv2.inRange(
                cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower, upper
            )
        return thresholded

    def find_markers(self, img: np.ndarray):
        """Find the positions of the markers in a thresholded image.

        Args:
            img (np.ndarray): Thresholded image

        Returns:
            np.ndarray: Marker positions
        """
        contours, _ = cv2.findContours(
            img.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )

        res = []
        min_size, max_size = self.options.marker_size
        for ctr in contours:
            size = cv2.contourArea(ctr)
            if size > min_size and size <= max_size:
                M = cv2.moments(ctr)
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]

                mean = np.array([cY, cX])
                res.append(mean)
        return np.array(res)

    def find_triangle(self, positions: "list[np.ndarray]"):
        """
        Maps the three car dots to the actual vehicles position/rotation

        Args:
            positions (list[np.ndarray]): A list of the three positions of the car dots

        Returns:
            (position, angle) - The position (in pixel coordinates) and angle of the vehicle
        """

        # determine configuration
        d0 = np.sqrt(
            (positions[1][0] - positions[2][0]) ** 2
            + (positions[1][1] - positions[2][1]) ** 2
        )
        d1 = np.sqrt(
            (positions[2][0] - positions[0][0]) ** 2
            + (positions[2][1] - positions[0][1]) ** 2
        )
        d2 = np.sqrt(
            (positions[0][0] - positions[1][0]) ** 2
            + (positions[0][1] - positions[1][1]) ** 2
        )

        min_dist_i = np.argmin([d0, d1, d2])
        if min_dist_i == 0:  # point 0 in front
            configuration = (0, 1, 2)
        elif min_dist_i == 1:  # point 1 in front
            configuration = (1, 2, 0)
        else:  # point 2 in front
            configuration = (2, 0, 1)

        # derive COM and rotation
        x0 = positions[configuration[0]]
        x1 = positions[configuration[1]]
        x2 = positions[configuration[2]]

        COM_weight = 0.435  # relative distance from center of mass to back dots

        # use weighted mean for asymmetric configuration
        xmean = (
            COM_weight * x0[0]
            + (1 - COM_weight) / 2 * x1[0]
            + (1 - COM_weight) / 2 * x2[0]
        )
        ymean = (
            COM_weight * x0[1]
            + (1 - COM_weight) / 2 * x1[1]
            + (1 - COM_weight) / 2 * x2[1]
        )
        xangle = (x1[0] + x2[0]) / 2
        yangle = (x1[1] + x2[1]) / 2

        a2 = np.array([-yangle + x0[1], -xangle + x0[0]])
        unit_vector_1 = np.array([0, 1])
        unit_vector_2 = a2 / np.linalg.norm(a2)
        angle = np.arctan2(unit_vector_2[0], unit_vector_2[1]) - np.arctan2(
            unit_vector_1[0], unit_vector_1[1]
        )

        # apply corrections
        xmean += self.options.correct[0]
        ymean += self.options.correct[1]
        angle += self.options.correct[2]

        # return result
        return np.array([xmean, ymean]), angle

    def process(
        self,
        img: np.ndarray,
        *,
        show_steps: bool = False,
        show_result: bool = False,
        crop: Tuple[int] = None,
    ):
        """Process the markers in an image

        Args:
            img (np.ndarray): The image to process
            show_steps (bool, optional): Show the steps during processing. Defaults to False.
            show_result (bool, optional): Show the resulting position and orientation. Defaults to False.
            crop (Tuple[int], optional): Crop the image (xmin, ymin, xmax, ymax). Defaults to None.

        Raises:
            MarkerException: Incorrect number of markers detected.

        Returns:
            (np.ndarray, float): (positions, angles) of the car in meters an radians respectively.
        """
        # remove camera distortions and prepare for thresholding
        img = self.correct_image(img, crop=crop)
        img = self.prepare_image(img)
        if show_steps:
            cv2.imshow("corrected", img)

        # threshold image
        dst = self.isolate_markers(img)
        if show_steps:
            cv2.imshow("thresholded", dst)

        # show steps
        if show_steps:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # find the markers in pixel coordinates
        points = self.find_markers(dst)

        # find the center and angle
        if len(points) == 3:
            center, angle = self.find_triangle(points)
        else:
            raise MarkerException("Incorrect number of markers.")

        # show result
        if show_result:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.scatter(points[:, 1], points[:, 0], marker="x", color="r", s=50)
            plt.scatter(center[1], center[0], marker=".", color="b", s=75)
            size = np.max(np.linalg.norm(points - center[np.newaxis, :], axis=1))
            arrow = np.array([np.cos(angle), np.sin(angle)]) * size * 1.5
            plt.arrow(
                center[1],
                center[0],
                arrow[1],
                arrow[0],
                width=1,
                color="b",
                head_width=10,
            )
            plt.show()

        if crop is None:
            return self.model.screen_to_world(center, angle)
        return self.model.screen_to_world(center + np.array(crop[:2]), angle)


class Positioning:
    """Positioning system for the car."""
    def __init__(
        self,
        initial: np.ndarray,
        params: CameraParams = None,
        options: PositioningOptions = None,
    ):
        """Initialize the positioning system for a car.

        Args:
            initial (np.ndarray): Initial position (or image in which to find a car).
            params (CameraParams, optional): The camera parameters. Defaults to None.
            options (PositioningOptions, optional): The configuration of the positioning system. Defaults to None.

        Raises:
            MarkerException: Incorrect number of markers in `initial` image.
        """
        self.localization = Localization(params, options)
        if initial.ndim > 1:  # initial is an image
            try:
                initial, _ = self.localization.process(initial)
            except MarkerException as e:
                print('Could not find car in initial image.')
                self.localization.process(initial, show_steps=True)
                raise e
        self.position = initial
        self.angle = 0.0

    @property
    def model(self):
        return self.localization.model

    def get_crop(self, position: np.ndarray):
        """Compute crop surrounding provided position

        Args:
            position (np.ndarray): Position of the car

        Returns:
            Tuple[int]: (xmin, ymin, xmax, ymax)
        """
        # load parameters
        shape = self.localization.img_size
        margin = np.array(self.localization.options.margins)

        # using fixed margins
        center = self.model.world_to_screen(position).astype(int)
        res = (*center - margin, *center + margin)
        return (
            max(0, min(res[0], shape[0])),
            max(0, min(res[1], shape[1])),
            max(0, min(res[2], shape[0])),
            max(0, min(res[3], shape[1])),
        )

    def __call__(
        self, img: np.ndarray, *, show_steps: bool = False, show_result: bool = False
    ):
        """Process the provided image.

        Args:
            img (np.ndarray): Image to process
            show_steps (bool, optional): Visualize steps (for debugging). Defaults to False.
            show_result (bool, optional): Visualize resulting position and angle (for debugging). Defaults to False.

        Returns:
            (np.ndarray, float): (position, angle) in meters and radians respectively.
        """
        # get the image crop
        crop = self.get_crop(self.position)

        # get the position measurement
        try:
            self.position, self.angle = self.localization.process(
                img, crop=crop, show_steps=show_steps, show_result=show_result
            )
        except MarkerException:
            logging.warn('Found incorrect number of markers. Repeating same state.')
        return self.position, self.angle


if __name__ == "__main__":
    from ..resources import load, ResourceType

    shape = (1024, 1280)
    track: Racetrack = load("racetrack.json", resource_type=ResourceType.SCENE)
    mask = get_mask(track, loop=True, shape=shape, model=CameraModel())
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
