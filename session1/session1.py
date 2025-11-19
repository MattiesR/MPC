import numpy as np
import matplotlib.pyplot as plt


Ts = 0.5  # Sampling time

A = (np.array([[0,1],[0,0]]))*Ts + np.eye(2)  # State matrix
B = np.array([[0],[-1]])*Ts
C = np.array([[1, -2/3]])  # Output matrix
Q = C.T @ C + 0.001*np.eye(2) # State weighting matrix
R = np.array([[0.1]])  # Input weighting matrix
Pf = Q



def backward_recursion(Pf, A, B, Q, R,N):
    K = -np.linalg.inv(R+B.T @ Pf @ B) @ (B.T @ Pf @ A)
    K_list = [K]
    for _ in range(N):
        Pn = Q + A.T @ Pf @ A - A.T @ Pf @ B @ np.linalg.inv(R + B.T @ Pf @ B) @ B.T @ Pf @ A
        Pf = Pn
        K = -np.linalg.inv(R + B.T @ Pf @ B) @ (B.T @ Pf @ A)
        K_list.append(K)
    K_list.reverse()
    return K_list

def simulate_system(A, B, K, x0, N):
    x = x0
    trajectory = [x0]
    for i in range(N):
        u = K[i] @ x
        x = A @ x + B @ u
        trajectory.append(x)
    return np.array(trajectory)


if __name__ == "__main__":
    N = 10  # Time horizion
    timesteps = 10
    K = backward_recursion(Pf, A, B, Q, R,N)
    x0 = np.array(([[10], [10]]))
    plt.ion()
    for _ in range(timesteps):
        prediction = simulate_system(A, B, K, x0, N)
        x0 = prediction[1]  # Update initial state for next simulation
        plt.plot(prediction[:,0], prediction[:,1])
        #plt.plot(trajectories[:,0], trajectories[:,1], "r")  # All starting points
        plt.xlim(-20,20)
        plt.ylim(-20,20)
        plt.grid()
        plt.pause(0.5)
        plt.clf()

    plt.xlabel('State x1')
    plt.ylabel('State x2')


    plt.title('State Trajectories over Time')