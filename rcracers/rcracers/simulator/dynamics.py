import numpy as np
import casadi as cs
from dataclasses import dataclass

from ..control.signatures import LOGGER


@dataclass
class VehicleParameters:
    length: float = 0.17  # length of the car (meters)
    axis_front: float = 0.047  # distance cog and front axis (meters)
    axis_rear: float = 0.05  # distance cog and rear axis (meters)
    front: float = 0.08  # distance cog and front (meters)
    rear: float = 0.08  # distance cog and rear (meters)
    width: float = 0.08  # width of the car (meters)
    height: float = 0.055  # height of the car (meters)
    mass: float = 0.1735  # mass of the car (kg)
    inertia: float = 18.3e-5  # moment of inertia around vertical (kg*m^2)
    
    # input limits
    max_steer: float = 0.384  # max steering angle (radians)
    max_drive: float = 1.0  # maximum drive

    """Pacejka 'Magic Formula' parameters.
    Used for magic formula: `peak * sin(shape * arctan(stiffness * alpha))`
    as in Pacejka (2005) 'Tyre and Vehicle Dynamics', p. 161, Eq. (4.6)
    """
    # front
    bf: float = 3.1355  # front stiffness factor
    cf: float = 2.1767  # front shape factor
    df: float = 0.4399  # front peak factor

    # rear
    br: float = 2.8919  # rear stiffness factor
    cr: float = 2.4431  # rear shape factor
    dr: float = 0.6236  # rear peak factor

    # kinematic approximation
    friction: float = 1 # friction parameter
    acceleration: float = 2 # maximum acceleration

    # motor parameters
    cm1: float = 0.3697
    cm2: float = 0.001295
    cr1: float = 0.1629
    cr2: float = 0.02133


class KinematicBicycle:
    def __init__(
        self,
        params: VehicleParameters = None,
        *,
        symbolic: bool = False,
    ) -> None:
        """Initialize kinematic bicycle model.

        Args:
            params (VehicleParameters, optional): vehicle parameters. Defaults to VehicleParameters().
            symbolic (bool, optional): handle symbolic inputs. Defaults to False.
        """
        self.symbolic = symbolic
        self.params = VehicleParameters() if params is None else params
    
    def clip_inputs(self, d=None, δ=None):
        res = []
        if d is not None:
            res.append(np.clip(d, -self.params.max_drive, self.params.max_drive))
        if δ is not None:
            res.append(np.clip(δ, -self.params.max_steer, self.params.max_steer))
        if len(res) == 1:
            return res[0]
        return tuple(res)

    def __call__(self, x: np.ndarray, u: np.ndarray, t: int = None, log: LOGGER = None) -> np.ndarray:
        """Evaluate kinematic bicycle model.

        Args:
            x (np.ndarray): state [px, py, ɸ, v].
            u (np.ndarray): input [d, δ].

        Returns:
            np.ndarray: derivative of the state.
        """
        ops = cs if self.symbolic else np

        # unpack arguments
        d, δ = u[0], u[1]
        φ, v = x[2], x[3]

        # unpack parameters
        lf, lr = self.params.axis_front, self.params.axis_rear
        a, μ = self.params.acceleration, self.params.friction

        # prepare result
        if self.symbolic:
            res = cs.SX(4, 1)
            concat = cs.vertcat
        else:
            res = np.empty((4,))
            concat = lambda *x: x
            d, δ = self.clip_inputs(d, δ)   # also clip inputs if not symbolic

        # evaluate
        β = ops.arctan2(lf * ops.tan(δ), lf + lr)
        res[:] = concat(
            v * ops.cos(φ + β),  # px dot
            v * ops.sin(φ + β),  # py dot
            v * ops.sin(β) / lr,  # φ dot
            a * d - μ * v,  # v dot
        )
        
        return res