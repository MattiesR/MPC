"""Utilities for visualizing geometric objects."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .polyhedron import Polyhedron
from .ellipsoid import Ellipsoid
from scipy.spatial import ConvexHull
from itertools import combinations, product


def pairs(*lists):
    for t in combinations(lists, 2):
        for pair in product(*t):
            if pair[0] != pair[1]:
                yield pair


def plot_polytope(p: Polyhedron, *, ax: plt.Axes = None, **style):
    """Plot a Polyhedron set in 2D or 3D.

    Args:
        p (Polyhedron): The polyhedron to plot. 
        ax (plt.Axes, optional): matplotlib axes to plot to. Defaults to None.

    Raises:
        TypeError: If ``p`` is not a Polyhedron. 

    Returns:
        plt.Axes: The axes the plot was drawn to. 
    """
    if not isinstance(p, Polyhedron): 
        raise TypeError(f"Expected an Polyhedron, got {type(p)}")
    if p.dim == 2:
        return _plot_polytope2d(p, ax=ax, **style)
    if p.dim == 3:
        if not isinstance(ax, Axes3D):
            ax = plt.subplot(projection="3d")
        return _plot_polytope3d(p, ax=ax, **style)


def _plot_polytope2d(polyhedron: Polyhedron, *, ax=None, **style):
    if ax is None: 
        ax = plt.subplot()
    if not isinstance(polyhedron, Polyhedron):
        raise TypeError(f"Expected a Polyhedron to plot. Got a {type(polyhedron)}")

    default_style = dict(ec="tab:blue", linestyle='solid', fc='tab:blue', alpha=0.5)
    default_style.update(style)
    style = default_style

    if polyhedron.size != 2:
        raise ValueError(f'Can only plot polyhedrons of dimension 2, got {polyhedron.size}.')
    if not polyhedron.is_compact():
        raise ValueError(f'Can only plot bounded polyhedron.')
    
    verts = polyhedron.vertices()
    if len(verts) == 0:
        return

    faces = polyhedron.get_face_incidence()
    neighbors = polyhedron.get_vertex_adjacency()

    previous, current = -1, 0
    # for current in range(len(verts)):
    #     if len(neighbors[current]) > 0:
    #         break
    # else:
    #     raise RuntimeError('No vertices with neighbors.')

    order = []
    for _ in range(len(verts)-1):
        order.append(current)
        (current, *_), previous = neighbors[current] - set([previous]), current
    order.append(current)

    ax.fill(verts[order, 0], verts[order, 1], **style)
    return ax 


def _plot_polytope3d(polyhedron: Polyhedron, *, ax: Axes3D=None, **style):    
    if ax is None:
        ax = plt.subplot(projection="3d")
    
    default_style = dict(
        edge = dict(color=style.get('color', 'r'), linestyle='solid'),
        face = dict(fc=style.get('color', 'r'), alpha=style.get('alpha', 0.5))
    )
    default_style.update(style)
    style=default_style

    if polyhedron.size != 3:
        raise ValueError(f'Can only plot polyhedrons of dimension 3, got {polyhedron.size}.')
    if not polyhedron.is_compact():
        raise ValueError(f'Can only plot bounded polyhedron.')

    verts = polyhedron.vertices()
    if len(verts) == 0:
        return
        
    hull = ConvexHull(verts)
    for s in hull.simplices:
        tri = Poly3DCollection([verts[s]], **style['face'])
        tri.set_edgecolor('none')
        ax.add_collection3d(tri)

        for v0, v1 in pairs(s, s):
            # check if the center point is on an edge (i.e. at least two active inequality constraints)
            nb_active = polyhedron.nb_active(0.5*(verts[v0] + verts[v1]))
            if nb_active >= 2:
                ax.plot(xs=verts[[v0, v1], 0], ys=verts[[v0, v1], 1], zs=verts[[v0, v1], 2], **style['edge'])
    
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=0)
    return ax 


def plot_ellipsoid(e: Ellipsoid, ax = None, **style):
    """Plot an Ellipsoid set in 2D or 3D."""
    if not isinstance(e, Ellipsoid): 
        raise TypeError(f"Expected an ellipsoid, got {type(e)}")
    if e.dim == 2:
        return _plot_ellipsoid2d(e, ax, **style)
    if e.dim == 3:
        if not isinstance(ax, Axes3D):
            ax = plt.subplot(projection="3d")
        return _plot_ellipsoid3d(e, ax, **style)
    

def _plot_ellipsoid2d(ellipsoid: Ellipsoid, ax=None, **style):
    if ellipsoid.dim != 2: 
        raise ValueError(f"Expected 2D ellipse, got dimension {ellipsoid.dim}")
    
    from matplotlib.patches import Ellipse
    ax = plt.gca() if ax is None else ax
    eigval, eigvec = np.linalg.eigh(ellipsoid.shape_matrix)
    angle = np.arctan2(*np.flip(eigvec[0,:])) / np.pi * 180.
    lenx = 2./np.sqrt(eigval[0])
    leny = 2./np.sqrt(eigval[1])
    ellipse_patch = Ellipse(xy=ellipsoid.center, width=lenx, height=leny, angle=angle, **style)
    ax.add_patch(ellipse_patch)
    ax.autoscale_view()
    return ax 

def _plot_ellipsoid3d(ellipsoid: Ellipsoid, ax: Axes3D=None, **style): 
    from scipy.linalg import sqrtm  # Matrix square-root 

    if ax is None:
        ax = plt.subplot(projection="3d")
    
    # Sample points on the unit ball 
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Inverse transform (suboptimal implementation, but it's just for plotting.)
    Qhinv = np.linalg.inv(sqrtm(ellipsoid.shape_matrix))
    transformed = np.dot(Qhinv, np.stack([x,y,z]).transpose((1,0,2)),)

    transformed = transformed + ellipsoid.center[:, np.newaxis, np.newaxis]
    ax.plot_surface(*transformed)

    return ax