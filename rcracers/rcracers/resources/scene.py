from abc import ABC, abstractclassmethod
from enum import Enum
from typing import Dict
import numpy as np

from .core import ResourceType, serializable, load


class SceneType(Enum):
    """
    Enumerator to identify the scene type
    """

    RACETRACK = 0
    ROADINTERSECT = 1


SCENE_TYPE = "_scene__type"


class Scene(ABC):
    """
    General scene object, all possible scenes inherit this object

    Attributes:
        scene_type (SceneType): The scene type
        scene_data (dict): The scene data
    """

    def __init__(self, type: SceneType, data: Dict[str, np.ndarray]):
        self.type = type
        self.data = data

    @abstractclassmethod
    def from_data(cls, data: dict):
        ...

    def __serialize__(self):
        res = {SCENE_TYPE: self.type.name}
        for k, v in self.data.items():
            res[k] = v
        return res

    @classmethod
    def __deserialize__(cls, obj: dict):
        type = SceneType[obj.pop(SCENE_TYPE)]
        if type is SceneType.RACETRACK:
            return Racetrack.from_data(obj)
        else:
            raise RuntimeError(f"Unable to deserialize scene of type {type.name}.")


serializable("_scene", Scene)


DEFAULT_TRACK = 'racetrack.json'


class Racetrack(Scene):
    """
    Racetrack scene

    Attributes:
        inner (np.array Nx2): The inner racetrack points
        outer (np.array Nx2): The outer racetrack points
        center (np.array Nx2): The center racetrack points
    """

    def __init__(self, inner: np.ndarray, outer: np.ndarray, center: np.ndarray):
        """
        Initialize a racetrack scene from a file

        Args:
            filename (str): The filename of the racetrack file
        """
        self.inner, self.outer, self.center = inner, outer, center
        self.scene_data = {
            "inner": self.inner,
            "outer": self.outer,
            "center": self.center,
        }

        super().__init__(SceneType.RACETRACK, self.scene_data)

    @classmethod
    def from_data(cls, data: dict):
        """Generate Racetrack scene from data dict (for reading from disk).

        Args:
            data (dict): Data dictionary

        Returns:
            Racetrack: Instance of a racetrack.
        """
        inner, outer, center = data["inner"], data["outer"], data["center"]
        return cls(inner, outer, center)
    
    @classmethod
    def load_default(cls):
        """Load the default racetrack from disk."""
        res: Racetrack = load(DEFAULT_TRACK, resource_type=ResourceType.SCENE)
        return cls(res.inner, res.outer, res.center)
