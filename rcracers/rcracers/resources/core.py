from enum import Enum
import os

from dataclasses import asdict, is_dataclass
import json
import numpy as np
from typing import Type, Union


SER_TYPE = "_ser__type"
SER_CONTENT = "_ser__content"
SARRAY = "_num__sarray"

RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

_ser_types = {}


class ResourceType(Enum):
    """Resource types (for loading from disk)."""
    SCENE = 'scenes'
    ASSET = 'assets'
    CONFIG = 'configs'

    def get_path(self, name: str = None):
        """Get the path of a resource of this type with the provided name.

        Args:
            name (str, optional): Name of the resource. Defaults to None in which case the directory is returned.

        Returns:
            str: Path of the resource.
        """
        if name is None:
            return os.path.join(RESOURCE_DIR, self.value)
        return os.path.join(RESOURCE_DIR, self.value, name)

class sarray(np.ndarray):
    """Serializable view of an `np.ndarray`"""
    def __serialize__(self):
        """Serialize this array."""
        if self.size == 0:
            return {"shape": json.dumps(self.shape)}
        elif self.ndim == 1:
            return json.dumps(self.tolist())
        return [sarray.__serialize__(a) for a in self]

    @classmethod
    def __deserialize__(cls, obj: Union[list, str]):
        """Deserialize an array."""
        if isinstance(obj, str):
            return np.asarray(json.loads(obj)).view(cls)
        elif isinstance(obj, dict):
            return np.zeros(json.loads(obj.get("shape")))
        return np.asarray([sarray.__deserialize__(a) for a in obj]).view(cls)


class Encoder(json.JSONEncoder):
    """Encoder that can handle np.ndarray instances."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {SER_TYPE: SARRAY, SER_CONTENT: obj.view(sarray).__serialize__()}
        return json.JSONEncoder.default(self, obj)


def load(name: str, *, cls=None, resource_type: ResourceType = None):
    """Load a resource from disk.

    Args:
        name (str): Name of the resource
        cls (Any, optional): Hint for target type. Defaults to None.
        resource_type (ResourceType, optional): The type of the resource to load. Defaults to None.

    Returns:
        Any: Loaded object.
    """
    if resource_type is None:
        path = os.path.join(RESOURCE_DIR, name)
    else:
        path = resource_type.get_path(name)

    with open(path, 'r') as f:
        return deserialize(f.read(), cls=cls)


def save(name: str, obj, *, resource_type: ResourceType = None):
    """Save an object to disk

    Args:
        name (str): Name of the object
        obj (Any): Object to save
        resource_type (ResourceType, optional): Type of the object. Defaults to None.
    """
    if resource_type is None:
        path = os.path.join(RESOURCE_DIR, name)
    else:
        path = resource_type.get_path(name)

    with open(path, 'w') as f:
        return f.write(serialize(obj, indent=4))


def serializable(name: str, ser_type: Type):
    """Register a type as serializable.

    Args:
        name (str): Name of the type
        ser_type (Type): Type of the object
    """
    _ser_types[name] = ser_type
    

def serialize(obj, *, indent: int=None):
    """Serialize an object as a string using `json`.    

    Args:
        obj (Any): Object to serialize
        indent (int, optional): Indentation for string. Defaults to None.

    Returns:
        str: Serialized object as a string
    """
    for k, v in _ser_types.items():
        if isinstance(obj, v):
            obj = {SER_TYPE: k, SER_CONTENT: obj.__serialize__()}
    if is_dataclass(obj):
        obj = asdict(obj)
    return json.dumps(obj, indent=indent, cls=Encoder)


def deserialize(ser, *, cls=None):
    """Deserialize an object from a json string

    Args:
        ser (Any): String to deserialize.
        cls (Type, optional): Type of the object to return. Defaults to None.

    Returns:
        Any: Deserialized object
    """
    if not isinstance(ser, dict):
        ser = json.loads(ser)
    if isinstance(ser, dict):
        ser_cls = _ser_types.get(ser.get(SER_TYPE, None), None)
        if ser_cls is not None:
            content = ser.get(SER_CONTENT)
            content = deserialize(content)
            return ser_cls.__deserialize__(content)
        
        for k, v in ser.items():
            if isinstance(v, dict) and v.get(SER_TYPE, None) == SARRAY:                    
                ser[k] = sarray.__deserialize__(v.get(SER_CONTENT)).view(np.ndarray)

    if cls is None:
        return ser
    else:
        return cls(**ser)