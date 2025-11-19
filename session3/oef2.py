from rcracers.utils.geometry import plot_polytope
import matplotlib.pyplot as plt
from rcracers.utils.geometry import Polyhedron, Rectangle
import numpy as np
from scipy.linalg import solve_discrete_are
from given.problem import Problem

problem = Problem()

N = 5


H = np.vstack([np.eye(2), -np.eye(2)])
h = np.array([1.,25., 120.,50])
poly = Polyhedron.from_inequalities(H, h)



def exercise3():
    print("Exercise 3.")
    A = problem.A
    B = problem.B
    Q = problem.Q
    R = problem.R

    # from scipy.linalg import solve_discrete_are
    Pinf = solve_discrete_are(A, B, Q, R)
    Kinf = -np.linalg.solve(R + B.T @ Pinf @ B, B.T @ Pinf @ A) # Calculate the LQR K

    Hx = np.vstack([np.eye(2), -np.eye(2)])
    hx = np.array([1.,25., 120.,50]).reshape(-1,1)

    Hu = np.array([[1], [-1]]) @ Kinf
    hu = np.array([[problem.u_max], [-problem.u_min]])
    H = np.vstack([Hx, Hu])
    h = np.vstack([hx, hu])
    poly = Polyhedron.from_inequalities(H, h)
    rect = Rectangle([-120,-50],[1,25])
    plot_polytope(poly, color="b")
    plot_polytope(rect, color="g")
    plt.title("Feasible set input constraints")    
    plt.show()

    Acl = A + B @ Kinf
    Hk = np.vstack([Acl, -Acl])
    H = np.vstack([Hx, Hu, Hk])
    h = np.vstack([hx, hu, hx])
    poly = Polyhedron.from_inequalities(H, h)
    rect = Rectangle([-120,-50],[1,25])
    plot_polytope(poly, color="b")
    plot_polytope(rect, color="g")
    plt.title("Feasible set after 1 iteration")    
    plt.show()
    
    


if __name__ == "__main__":
    exercise3()