from abc import ABC
import numpy as np
import cdd

from typing import Iterable, Sequence, Tuple, Union

from itertools import product
from scipy.optimize import brentq, linprog
from scipy.spatial import ConvexHull
from scipy.spatial._qhull import QhullError

ABS_TOL = 1e-5
ZERO_TOL = 1e-10

def prep_cdd(array: np.ndarray):
    array = np.copy(array)
    array[np.abs(array) < ZERO_TOL] = 0.0
    return np.round(array/ABS_TOL)*ABS_TOL


def reltol(v, tol):
    return tol * (ZERO_TOL + np.abs(v))

class LPRepresentation:
    def __init__(self, matrix: cdd.Matrix):
        self._matrix = matrix

    @property
    def matrix(self):
        return self.matrix

    def canonicalize(self):
        self.matrix.canonicalize()
        return self

    @property
    def size(self):
        return len(self.matrix[0]) - 1

    @property
    def nb_constraints(self):
        return len(self.matrix)

    @property
    def representation(self):
        return np.array(self.matrix[:])

    @property
    def equality_idx(self):
        return list(self.matrix.lin_set)

    @property
    def inequality_idx(self):
        return list(set(range(self.nb_constraints)) - self.matrix.lin_set)


class H_Representation(LPRepresentation):
    def __init__(self, matrix: cdd.Matrix):
        super().__init__(matrix)
        self._inequalities, self._equalities = None, None

    @classmethod
    def from_constraints(cls, inequalities: np.ndarray, equalities: np.ndarray = None):
        """
        Initialize a inequalities representation of a polytopic set
            x: A x + b <= 0, A_eq x  + b_eq = 0

        :param inequalities: [A, b]
        :param equality: [A_eq, b_eq]
        """
        if inequalities is None:
            inequalities = np.zeros((0, equalities.shape[1]))
        if equalities is None:
            equalities = np.zeros((0, inequalities.shape[1]))

        res = cls(None)
        res._inequalities = inequalities
        res._equalities = equalities
        return res

    @staticmethod
    def to_cdd(array: np.ndarray):
        return (-np.roll(prep_cdd(array), 1, axis=1)).tolist()

    @staticmethod
    def from_cdd(array: tuple):
        return np.roll(-np.array(array), -1, axis=1)

    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = cdd.Matrix(
                H_Representation.to_cdd(self._inequalities), number_type="float"
            )
            self._matrix.rep_type = cdd.RepType.INEQUALITY
            if np.size(self._equalities) > 0:
                self._matrix.extend(
                    H_Representation.to_cdd(self._equalities), linear=True
                )
        return self._matrix

    @property
    def size(self):
        return self.inequalities.shape[1] - 1

    @property
    def representation(self):
        return H_Representation.from_cdd(self.matrix[:])

    @property
    def inequalities(self):
        if self._matrix is None:
            return self._inequalities
        return np.take(self.representation, self.inequality_idx, axis=0)

    @property
    def equalities(self):
        if self._matrix is None:
            return self._equalities
        return np.take(self.representation, self.equality_idx, axis=0)


class V_Representation(LPRepresentation):
    def __init__(self, matrix: cdd.Matrix):
        super().__init__(matrix)
        self._vertices, self._nonnegspan, self._linspan = None, None, None

    @classmethod
    def degenerate(cls, size: int):
        return cls.from_generators(
            np.zeros((0, size)), np.zeros((0, size)), np.zeros((0, size))
        )

    @classmethod
    def from_generators(
        cls,
        vertices: np.ndarray,
        nonnegspan: np.ndarray = None,
        linspan: np.ndarray = None,
    ):
        size = [
            np.shape(v)[-1] for v in (vertices, nonnegspan, linspan) if v is not None
        ]
        if len(size) == 0:
            raise ValueError("At least one argument should be an array")
        size = size[0]

        if vertices is None:
            vertices = np.zeros((0, size))
        if nonnegspan is None:
            nonnegspan = np.zeros((0, size))
        if linspan is None:
            linspan = np.zeros((0, size))

        res = cls(None)
        res._vertices = vertices
        res._nonnegspan = nonnegspan
        res._linspan = linspan
        return res

    @staticmethod
    def to_cdd(array: np.ndarray):
        return prep_cdd(array)

    @property
    def matrix(self):
        if self._matrix is None:
            stack = np.hstack([np.ones((self._vertices.shape[0], 1)), self._vertices])
            if np.size(self._nonnegspan) > 0:
                stack = np.block(
                    [
                        [stack],
                        [np.zeros((self._nonnegspan.shape[0], 1)), self._nonnegspan],
                    ]
                )

            self._matrix = cdd.Matrix(self.to_cdd(stack), number_type="float")
            self._matrix.rep_type = cdd.RepType.GENERATOR
            if np.size(self._linspan) > 0:
                self._matrix.extend(
                    self.to_cdd(np.hstack([np.zeros((self._linspan.shape[0], 1)), self._linspan])),
                    linear=True,
                )
        return self._matrix

    @property
    def size(self):
        return self.vertices.shape[1]

    @property
    def vertices(self):
        if self._matrix is None:
            return self._vertices
        vertices = np.take(self.representation, self.inequality_idx, axis=0)
        if len(vertices) == 0:
            return vertices
        return vertices[vertices[:, 0] == 1, 1:]

    @property
    def nonnegspan(self):
        if self._matrix is None:
            return self._nonnegspan
        vertices = np.take(self.representation, self.inequality_idx, axis=0)
        if len(vertices) == 0:
            return vertices
        return vertices[vertices[:, 0] == 0, 1:]

    @property
    def linspan(self):
        if self._matrix is None:
            return self._linspan
        return np.take(self.representation, self.equality_idx, axis=0)[:, 1:]


class Polyhedron:
    def __init__(self, description: LPRepresentation, auto_compute: bool = False):
        self.description = description
        self._h_rep, self._v_rep = None, None
        self._backend = None
        if auto_compute:
            self._polyhedron()
        elif isinstance(description, H_Representation):
            self._h_rep = description
        elif isinstance(description, V_Representation):
            self._v_rep = description

    @classmethod
    def from_constraints(cls, inequalities: np.ndarray, equalities: np.ndarray = None):
        """
        Initialize a polyhedral set from an inequality representation
            x: A x + b <= 0, A_eq x  + b_eq = 0

        Args:
            inequalities: [A, b]
            equality: [A_eq, b_eq]
        """
        return cls(H_Representation.from_constraints(inequalities, equalities))

    @classmethod
    def from_generators(
        cls,
        vertices: np.ndarray,
        nonnegspan: np.ndarray = None,
        linspan: np.ndarray = None,
    ):
        """
        Initialize a polyhedral set from from vertices
            x: x = sum_i vi pi + sum_j nj qj + sum_k lk rk
                for some sum_i pi = 1, pi >= 0, qj >= 0 and any rk. 

        Args:
            vertices: [v1, v2, ..., vNv].T
            nonnegspan: [n1, n2, ..., nNn].T
            linspan: [l1, l2, ..., lNl].T
        """
        return cls(V_Representation.from_generators(vertices, nonnegspan, linspan))

    @classmethod
    def from_inequalities(cls, H: np.ndarray, h: np.ndarray) -> "Polyhedron":
        """Initialize a polyhedral set from inequalities
            x: Hx <= h
        """
        inequalities = np.hstack([H.copy(), -np.reshape(h.copy(), (-1, 1))])
        return cls.from_constraints(inequalities)

    def _polyhedron(self):
        if self._backend is None:
            self._backend = cdd.Polyhedron(self.description.matrix)
        return self._backend

    def constraints(self):
        if self._h_rep is None:
            self._h_rep = H_Representation(self._polyhedron().get_inequalities())
        return self._h_rep

    def generators(self):
        if self._v_rep is None:
            res = self._polyhedron().get_generators()
            if len(res) == 0:
                self._v_rep = V_Representation.degenerate(self.size)
            else:
                self._v_rep = V_Representation(res)
        return self._v_rep

    def get_vertex_adjacency(self):
        if isinstance(self.description, H_Representation):
            return self._polyhedron().get_adjacency()
        else:
            return self._polyhedron().get_input_adjacency()

    def get_face_adjacency(self):
        if isinstance(self.description, V_Representation):
            return self._polyhedron().get_adjacency()
        else:
            return self._polyhedron().get_input_adjacency()

    def get_vertex_incidence(self):
        if isinstance(self.description, H_Representation):
            return self._polyhedron().get_incidence()
        else:
            return self._polyhedron().get_input_incidence()

    def get_face_incidence(self):
        if isinstance(self.description, V_Representation):
            return self._polyhedron().get_incidence()
        else:
            return self._polyhedron().get_input_incidence()

    @property
    def size(self):
        return self.description.size

    @property
    def dim(self):
        return self.description.size

    @property
    def nb_constraints(self):
        return self.constraints().nb_constraints

    @property
    def nb_inequalities(self):
        return len(self.inequalities())

    @property
    def nb_equalities(self):
        return len(self.equalities())

    @property
    def equality_constrained(self):
        return self.nb_equalities > 0

    @property
    def nb_vertices(self):
        return len(self.vertices())

    def canonicalize(self, *, full: bool = False):
        """Remove any redundant constraints.
        
        Args:
            full (bool): Compute the V and H representation and canonicalize both, defaults to False.
        """
        if full or self._h_rep is not None:
            self.constraints().canonicalize()
        if full or self._v_rep is not None:
            self.generators().canonicalize()
        return self

    def vertices(self) -> np.ndarray:
        return self.generators().vertices

    def nonnegspan(self) -> np.ndarray:
        return self.generators().nonnegspan

    def linspan(self) -> np.ndarray:
        return self.generators().linspan

    @property
    def H(self) -> np.ndarray:
        return np.vstack([self.H_in, self.H_eq, -self.H_eq])
    
    @property
    def h(self) -> np.ndarray:
        return np.concatenate([self.h_in, self.h_eq, -self.h_eq])

    @property 
    def H_in(self) -> np.ndarray: 
        return self.inequalities()[:,:-1]
    
    @property 
    def h_in(self) -> np.ndarray:
        return -self.inequalities()[:,-1]

    @property 
    def H_eq(self) -> np.ndarray: 
        return self.equalities()[:,:-1]
    
    @property
    def h_eq(self) -> np.ndarray: 
        return -self.equalities()[:, -1]

    def is_compact(self):
        # using Tucker's theorem of alternatives
        #   proof is then based on using the alternative to construct an unbounded ray
        # see Mangasarian (1994) "Nonlinear Programming", Table 2.4.1 on p 34
        if self._v_rep is not None:
            return np.size(self.nonnegspan()) == 0 and np.size(self.linspan()) == 0

        normals = self.inequalities()[:, :-1]
        normals_eq = self.equalities()[:, :-1]
        if np.linalg.matrix_rank(np.vstack([normals, normals_eq])) < normals.shape[1]:
            return False

        normals_eq = self.equalities()[:, :-1]

        d = normals.shape[0]
        m = normals_eq.shape[0]
        c = np.ones((d + m))
        res = linprog(
            c,
            A_eq=np.hstack([normals.T, normals_eq.T]),
            b_eq=np.zeros(normals.shape[1]),
            bounds=[(1, None) for _ in range(d)] + [(None, None) for _ in range(m)],
            method="revised simplex",
        )
        return res.status == 0

    def inequalities(self) -> np.ndarray:
        return self.constraints().inequalities

    def equalities(self) -> np.ndarray:
        return self.constraints().equalities

    def intersect(self, other: "Polyhedron") -> "Polyhedron":
        """Compute the intersection between two Polyhedrons."""
        if self.size != other.size:
            raise ValueError("Can only intersect polyhedrons of the same dimension.")
        return Polyhedron.from_constraints(
            np.vstack([self.inequalities(), other.inequalities()]),
            np.vstack([self.equalities(), other.equalities()]),
        )

    def stride(self, spaces: int, offset: int = 0):
        raise NotImplementedError('Stride not implemented for Polyhedral set.')

    def product(self, other: "Polyhedron"):
        normals, offset = self.inequalities()[:, :-1], self.inequalities()[:, -1]
        normals_eq, offset_eq = self.equalities()[:, :-1], self.equalities()[:, -1]
        o_normals, o_offset = other.inequalities()[:, :-1], other.inequalities()[:, -1]
        o_normals_eq, o_offset_eq = (
            other.equalities()[:, :-1],
            other.equalities()[:, -1],
        )

        r_normals = np.zeros(
            (
                normals.shape[0] + o_normals.shape[0],
                normals.shape[1] + o_normals.shape[1],
            )
        )
        r_normals[: normals.shape[0], : self.size] = normals
        r_normals[normals.shape[0] :, self.size :] = o_normals
        r_offset = np.concatenate([offset, o_offset])[:, np.newaxis]

        r_normals_eq = np.zeros(
            (
                normals_eq.shape[0] + o_normals_eq.shape[0],
                normals_eq.shape[1] + o_normals_eq.shape[1],
            )
        )
        r_normals_eq[: normals_eq.shape[0], : self.size] = normals_eq
        r_normals_eq[normals_eq.shape[0] :, self.size :] = o_normals_eq
        r_offset_eq = np.concatenate([offset_eq, o_offset_eq])[:, np.newaxis]

        return Polyhedron.from_constraints(
            np.concatenate([r_normals, r_offset], axis=-1),
            np.concatenate([r_normals_eq, r_offset_eq], axis=-1),
        )

    def power(self, n: int):
        res = self
        for _ in range(n - 1):
            res = res.product(self)
        return res
    
    def nb_active(self, other: np.ndarray, axis: int = -1, tol: float = 1e-10):
        self.canonicalize()
        if other.ndim == 1:
            other = np.reshape(other, (1, -1))
        if len(self.equalities()) > 0:
            raise RuntimeError('Equality constraints not supported')

        normals, offset = self.inequalities()[:, :-1], self.inequalities()[:, -1]
        other = np.moveaxis(other, source=axis, destination=-1)
        i_check = np.abs(np.einsum("ij,...j->i...", normals, other) + offset[..., np.newaxis]) <= tol
        return np.sum(i_check)

    @property 
    def has_vrep(self):
        return self._v_rep is not None 

    def _contains_poly(self, other: "Polyhedron", tol: float):
        if other.has_vrep:
            # Quick check of vertices if they are cached
            return np.all([self.contains(vertex, tol=tol) for vertex in other.vertices()])
        if self.has_vrep:
            return not np.any([other.contains(vertex, tol=tol) for vertex in other.vertices()])


        # Have to use Hrep
        # Prop 3.31 (Blanchini et. al., 2015) adjusted with equality constraints
        m = other.inequalities().shape[0]
        q = self.inequalities().shape[0]
        n = other.equalities().shape[0]
        p = self.equalities().shape[0]
        r = q + 2 * p

        A_eq = np.kron(other.inequalities()[:, :-1].T, np.eye(r))
        A_ub = -np.kron(other.inequalities()[:, -1::].T, np.eye(r))
        if n > 0:
            A_eq = np.hstack(
                [A_eq, np.kron(other.equalities()[:, :-1].T, np.eye(r))]
            )
            A_ub = np.hstack(
                [A_ub, -np.kron(other.equalities()[:, -1::].T, np.eye(r))]
            )

        if p > 0:
            b_eq = np.reshape(
                np.vstack(
                    [
                        self.inequalities()[:, :-1],
                        self.equalities()[:, :-1],
                        -self.equalities()[:, :-1],
                    ]
                ),
                (-1),
                order="F",
            )
            b_ub = np.concatenate(
                [
                    -self.inequalities()[:, -1],
                    -self.equalities()[:, -1],
                    self.equalities()[:, -1],
                ]
            )
        else:
            b_eq = np.reshape(self.inequalities()[:, :-1], (-1), order="F")
            b_ub = -self.inequalities()[:, -1] 
        c = np.ones((r * m + r * n,)) 

        # Modify the linprog call to account for flexibility 
        A_ub = np.vstack([A_ub, A_eq, -A_eq])
        b_ub = np.concatenate([b_ub + reltol(b_ub,tol), b_eq + reltol(b_eq, tol), -b_eq + reltol(b_eq, tol)])
        A_eq = None
        b_eq = None 

        res = linprog(
            c,
            A_ub,
            b_ub,
            A_eq,
            b_eq,
            [(0, None)] * (r * m) + [(None, None)] * (r * n),
            # method="revised simplex",
        )
        return res.success


    def contains(
        self, other: Union[np.ndarray, "Polyhedron"], axis: int = -1, tol: float = 1e-10
    ):
        if isinstance(other, Polyhedron):
            return self._contains_poly(other, tol)
        
        # Other is a vector 
        if other.ndim == 1:
            other = np.reshape(other, (1, -1))


        normals, offset = self.inequalities()[:, :-1], self.inequalities()[:, -1]
        normals_eq, offset_eq = self.equalities()[:, :-1], self.equalities()[:, -1]

        other = np.moveaxis(other, source=axis, destination=-1)
        i_check = np.all(
            np.einsum("ij,...j->i...", normals, other) + offset[..., np.newaxis] <= reltol(offset, tol),
            axis=0,
        )
        e_check = np.all(
            np.abs(
                np.einsum("ij,...j->i...", normals_eq, other)
                + offset_eq[..., np.newaxis]
            )
            <= reltol(offset_eq, tol),
            axis=0,
        )
        return np.logical_and(i_check, e_check)

    def diameter(self, ord=2):
        if not self.is_compact():
            return np.inf

        if len(self.vertices()) == 0:
            return 0.0

        res = -np.inf
        for v, w in product(self.vertices(), self.vertices()):
            res = max(res, np.linalg.norm(v-w, ord=ord))
        return res

    def interior(self, *, return_radius: bool = False):
        halfspaces = self.inequalities()
        norm_vector = np.reshape(
            np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1)
        )
        c = np.zeros((halfspaces.shape[1],))
        c[-1] = -1
        A = np.hstack((halfspaces[:, :-1], norm_vector))
        b = -halfspaces[:, -1:]

        A_eq, b_eq = self.equalities()[:, :-1], -self.equalities()[:, -1]
        if len(A_eq) == 0:
            A_eq, b_eq = None, None
        else:
            A_eq = np.hstack((A_eq, np.zeros((A_eq.shape[0], 1))))

        res = linprog(
            c,
            A_ub=A,
            b_ub=b,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(None, None),
            method="revised simplex",
        )
        if return_radius:
            return res.x[:-1], res.x[-1]
        return res.x[:-1]

    def support(self, vector: np.ndarray):
        A, b = self.inequalities()[:, :-1], -self.inequalities()[:, -1:]
        A_eq, b_eq = self.equalities()[:, :-1], -self.equalities()[:, -1]
        if len(A_eq) == 0:
            A_eq, b_eq = None, None
        res = linprog(
            -vector,
            A_ub=A,
            b_ub=b,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=(None, None),
            method="revised simplex",
        )
        return -res.fun
    
    def gauge(self, vector: np.ndarray):
        if not self.is_compact():
            return np.nan
        if not self.contains(np.zeros(self.size)):
            return np.nan

        if self._h_rep is None:
            vertices = self.vertices().T
            normals = np.block([
                [vertices, np.zeros((vertices.shape[0], 1))],
                [np.ones((1, vertices.shape[1])), -np.eye(1)]
            ])
            offset = np.concatenate([vector, np.zeros(1)])
            bounds = [(0, None) for _ in range(normals.shape[1])]
            cost = np.concatenate([np.zeros(vertices.shape[1]), np.ones(1)])
            res = linprog(
                cost,
                A_eq=normals,
                b_eq=offset,
                bounds=bounds,
                method='revised simplex'
            )
            return res.fun

        else:
            A, b = self.inequalities()[:, :-1], self.inequalities()[:, -1:]
            A_eq, b_eq = self.equalities()[:, :-1], self.equalities()[:, -1:]
            if len(A_eq) == 0:
                A_eq, b_eq = None, None
            else:
                A_eq, b_eq = b_eq, - A_eq @ vector
        
            res = linprog(
                np.ones(1),
                A_ub=b, 
                b_ub=-A @ vector,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=(0, None),
            )
            return res.fun

    def volume(self, *, projected: bool = False):
        if not self.is_compact():
            return np.inf

        vertices = self.vertices()
        if projected:
            _, s, svd_v = np.linalg.svd(self.equalities()[:, :-1], full_matrices=True)
            vertices = vertices @ svd_v[len(s) :, :].T
        try:
            if vertices.shape[-1] == 1:
                return np.max(vertices) - np.min(vertices)

            return ConvexHull(vertices).volume
        except (QhullError, ValueError):
            return 0

    def scale(self, value: float):
        h_rep, v_rep = None, None
        if self._h_rep is not None:
            h_rep = H_Representation.from_constraints(
                np.hstack([
                    self.inequalities()[:, :-1], self.inequalities()[:, -1:]*value
                ]),
                np.hstack([
                    self.equalities()[:, :-1], self.equalities()[:, -1:]*value
                ])
            )
        if self._v_rep is not None:
            v_rep = V_Representation.from_generators(
                self.vertices()*value,
                self.nonnegspan()*value,
                self.linspan()*value
            )
        res = None
        if isinstance(self.description, H_Representation):
            res = Polyhedron(h_rep)
            res._v_rep = v_rep
        else:
            res = Polyhedron(v_rep)
            res._h_rep = h_rep
        return res
    
    def translate(self, value: np.ndarray):
        if len(value.shape) != 1 or value.shape[0] != self.size:
            raise ValueError('Translation value is not of the correct shape.')

        h_rep, v_rep = None, None
        if self._h_rep is not None:
            h_rep = H_Representation.from_constraints(
                np.hstack([
                    self.inequalities()[:, :-1], self.inequalities()[:, -1:] - self.inequalities()[:, :-1] @ value
                ]),
                np.hstack([
                    self.equalities()[:, :-1], self.equalities()[:, -1:] - self.equalities()[:, :-1] @ value
                ])
            )
        if self._v_rep is not None:
            v_rep = V_Representation.from_generators(
                self.vertices() + value[np.newaxis, :],
                self.nonnegspan(),
                self.linspan()
            )
        res: Polyhedron = None
        if isinstance(self.description, H_Representation):
            res = Polyhedron(h_rep)
            res._v_rep = v_rep
        else:
            res = Polyhedron(v_rep)
            res._h_rep = h_rep
        return res

    def coordinate_projection(self, coords: Sequence[int]) -> "Polyhedron":
        """Compute the projection of the polyhedron along the given axis.

        Args:
            coords (Sequence[int]): Sequence of indices along which to compute the projection.
        """
        self = self.canonicalize()
        vertices = self.vertices()
        vertices_lowdim = vertices.take(coords, axis=1)
        vertices_lowdim = prep_cdd(vertices_lowdim)
        return Polyhedron.from_generators(vertices=vertices_lowdim).canonicalize()

    def image(self, transformation: np.ndarray, inverse: bool = False):
        if transformation.ndim != 2:
            raise ValueError('Transformation should be a matrix.')
        
        if inverse:
            if transformation.shape[0] != self.size + 1:
                transformation = np.block([
                    [transformation, np.zeros((transformation.shape[0], 1))],
                    [np.zeros((1, transformation.shape[1])), np.eye(1)]    
                ])

            if transformation.shape[0] != self.size + 1:
                raise ValueError('Transformation does not have a valid shape.')

            return Polyhedron.from_constraints(
                np.hstack([
                    self.inequalities()[:, :-1] @ transformation[:-1, :-1], self.inequalities()[:, :-1] @ transformation[:-1, -1:] + self.inequalities()[:, -1:]
                ]),
                np.hstack([
                    self.equalities()[:, :-1] @ transformation[:-1, :-1], self.equalities()[:, :-1] @ transformation[:-1, -1:] + self.equalities()[:, -1:]
                ]),
            )
        else:
            if transformation.shape[1] != self.size + 1:
                transformation = np.block([
                    [transformation, np.zeros((transformation.shape[0], 1))],
                    [np.zeros((1, transformation.shape[1])), np.eye(1)]    
                ])

            if transformation.shape[1] != self.size + 1:
                raise ValueError('Transformation does not have a valid shape.')

            return Polyhedron.from_generators(
                self.vertices() @ transformation[:-1, :-1].T + transformation[:-1, -1:].T, 
                self.nonnegspan() @ transformation[:-1, :-1].T, 
                self.linspan() @ transformation[:-1, :-1].T
            )

    def __eq__(self, other: "Polyhedron"):
        return self.contains(other, tol=1e-6) and other.contains(self, tol=1e-6)

    def __str__(self):
        res = f"{type(self).__name__} with \nHin=\n{self.H_in}\nhin=\n{self.h_in}"
        if self.equality_constrained:
            res += f"\nHeq=\n{self.H_eq}\nheq=\n{self.h_eq}"
        return res 

    def __repr__(self):
        return str(self)  # Dummy implementation


class Rectangle(Polyhedron):
    @classmethod
    def __get_box_halfspace(cls, xmin: np.ndarray, xmax: np.ndarray): 
        dim = len(xmin)
        H = np.vstack([np.eye(dim), -np.eye(dim)])
        xmin = np.atleast_1d(xmin) 
        xmax = np.atleast_1d(xmax) 
        h = np.concatenate([xmax, -xmin])
        return H, h

    def __new__(cls, xmin: np.ndarray, xmax: np.ndarray):
        H, h = cls.__get_box_halfspace(xmin, xmax)
        return Polyhedron.from_inequalities(H, h)
    
    def __init__(axis: np.ndarray):
        ...


class Singleton(Polyhedron):
    def __new__(cls, point: np.ndarray):
        if isinstance(point, list):
            point = np.array(point)
        return Polyhedron.from_generators(np.reshape(point, (1, -1)))


