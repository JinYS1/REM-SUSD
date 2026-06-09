import numpy as np
import control

class LQR_Model:
    def __init__(self, dt=1):
        self.dt = dt
        A, B, Q, R = self.system()
        C = np.eye(A.shape[0])
        D = np.zeros((A.shape[0], B.shape[1]))

        sys = control.ss(A, B, C, D)
        sysd = sys.sample(dt)
        A, B, C, D = control.ssdata(sysd)
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def predict(self, x, u):
        if np.isnan(x).any():
            print("Warning: x contains NaN values")
        if np.isinf(x).any():
            print("Warning: x contains inf values")
        if np.isnan(u).any() or np.isinf(u).any():
            u = np.array([0])

        return np.asarray(np.matmul(self.A, x) + np.matmul(self.B, u))

    def compute_cost(self,x,u):
        # cost = x @ self.Q @ x.T + u @ self.R @ u.T
        # return cost
        return np.sum(x*np.matmul(self.Q, x), axis=0) + np.sum(u*np.matmul(self.R, u), axis=0)

    def system(self):
        A = np.array([[1, 1],
                      [0, 1]])
        B = np.array([[1],
                      [1]])
        Q = np.array([[1,0],
                     [0,0]])
        R = np.array([0.3])

        return A, B, Q, R

    def dsystem(self):
        return self.A, self.B, self.Q, self.R