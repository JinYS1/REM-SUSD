import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,N_Units,N_STATES,N_ACTIONS):
        super(Net, self).__init__()
        self.units = N_Units

        self.fc1 = nn.Linear(N_STATES, self.units)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(self.units, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        actions_value = self.out(x)
        return actions_value

class SUSD_network():
    def __init__(self,k,optimize_method,N_Units,N_STATES,N_ACTIONS,Low_dimension):
        # Hyperparameter optimization and network structure initialization
        self.k = k
        self.optimize_mehod = optimize_method
        self.target_net = Net(N_Units,N_STATES,N_ACTIONS)
        self.Low_dimension = Low_dimension
        self.N_Sample = (N_STATES+1) * N_Units + (N_Units+1) * N_ACTIONS
        self.sample_net = []
        for i in range(self.Low_dimension):
            self.sample_net.append(Net(N_Units,N_STATES,N_ACTIONS))

    def Create_REM_marix(self):
        A = np.random.normal(loc=0, scale=0.1, size=(self.N_Sample, self.Low_dimension))
        return A

    def Create_LD_solutions(self):
        LD_solutions = np.random.normal(loc=0, scale=np.sqrt(0.1), size=(self.Low_dimension, self.Low_dimension))
        return LD_solutions

    def SUSD_Cov_Operation(self,ad_cov_matrix):
        cov_matrix = np.cov(ad_cov_matrix)
        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            print("cov_matrix contains NaN or inf values")

        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        real_eigenverctors = np.real(eigenvectors)
        location = np.where(eigenvalues == np.min(eigenvalues))[0][0]
        SUSD_direction = real_eigenverctors[:, location]

        return SUSD_direction

    # Return the evaluated value
    def Simulation(self):
        raise NotImplementedError

    # Updated Solutions
    def Update_para(self):
        raise NotImplementedError
