import numpy as np
from typing import Tuple, Callable
import numpy as np 
Matrix = np.ndarray 
Vector = np.ndarray 

class Ellipsoid: 
    def __init__(self, shape_matrix: Matrix, center: Vector = None): 
        if center is None:
            center = np.zeros(shape_matrix.shape[0])
        self.center = np.array(center)
        self._validate_shape(shape_matrix)
        self.shape_matrix = shape_matrix

    def _validate_shape(self, Q):
        if not all(q == self.dim for q in Q.shape):
            raise f"Dimension mismatch: expected {(self.dim,)*2} - got {Q.shape}."
        try:
            np.linalg.cholesky(Q+1e-10*np.eye(self.dim))
            return True
        except np.linalg.LinAlgError: 
            raise np.linalg.LinAlgError("Shape matrix of ellipsoid is not PSD!")

    def __contains__(self, point: Vector) -> bool:
        d = point - self.center 
        return d@(self.shape_matrix@d) <= 1

    @property 
    def dim(self) -> int: 
        return self.center.shape[0]

    @property 
    def empty(self) -> int:
        False
    
    def __rmul__(self, scale: float):
        return Ellipsoid(1/scale**2 * self.shape_matrix, scale * self.center)

    def as_inequality(self) -> Tuple[Callable, Vector, Vector]:
        """Return self as an inequality of the form 
        ```
        lower <= f(x) <= upper
        ```

        Returns:
            f: A function that takes a point that may or may not lie in the set 
            lower: the lower bound on f(x) such that x lies in the set
            upper: the upper bound on f(x) such that x lies in the set
        """
        f = lambda x: (x-self.center).T@self.shape_matrix@(x-self.center)
        lower = 0.
        upper = 1.
        return f, lower, upper
