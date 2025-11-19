import time
import numpy as np


class Timer:
    """A minimal class used for timing.

    Example:
    >>> import time
    >>> with Timer():
    >>>     time.sleep(0.1)
    """

    def __init__(self):
        self.__last = time.time()

    def __call__(self):
        current = time.time()
        res = current - self.__last
        self.__last = current
        return res

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print(self.__call__())


def angle_diff(a: float, b: float = 0.0):
    """Compute the difference between two angles

    Args:
        a (float): first angle (radians)
        b (float, optional): second angle (radians). Defaults to 0.0.

    Returns:
        float: The difference between both angles
    """

    return np.mod(a - b + np.pi, 2 * np.pi) - np.pi
