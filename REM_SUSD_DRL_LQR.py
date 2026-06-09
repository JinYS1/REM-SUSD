import numpy as np
import control
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import matplotlib.pyplot as plt
import scipy.linalg
from LQR_Model import LQR_Model
from REM_SUSD_framework import SUSD_network

# Define the sampling time.
dt = 1
length = 20

N_ACTIONS = 1
N_STATES = 2

# Number of neurons in the hidden layer
N_Units = 100

N_Sample = (N_STATES+1) * N_Units + (N_Units+1) * N_ACTIONS

Low_dimiension = 10

class LQR_REM_SUSD_network(SUSD_network):
    def __init__(self,lqr_model):
        super(LQR_REM_SUSD_network, self).__init__(k=0.1,optimize_method="minimize",N_Units=N_Units,N_STATES=N_STATES,N_ACTIONS=N_ACTIONS,Low_dimension=Low_dimiension)
        self.lqr_model = lqr_model

    def simulate(self,xt,index,horizen_length):
        cost = 0
        u_plot=[]
        x1_plot = []
        x2_plot = []
        for i in range(horizen_length):
            x1_plot.append(xt[0])
            x2_plot.append(xt[1])
            xt_1 = torch.unsqueeze(torch.FloatTensor(xt), 0)
            with torch.no_grad():
                u_1 = self.sample_net[index].forward(xt_1)
                u = u_1[0].data.numpy()
            u_plot.append(u)
            cost += self.lqr_model.compute_cost(xt, u)
            xt = self.lqr_model.predict(xt, u)

        return u_plot, cost, x1_plot, x2_plot

    def update_para(self,x0,max_iter=100,horizen_length=2000):
        best_theta = None
        best_LD_solution = None
        min_cost = np.inf
        u_plot=None

        Initialize = False
        mini_cost_list = []
        init_times = 5  # re‑initialization

        it_cost = []
        nhold = None

        for it in range(max_iter):

            # Initial low-dimensional solution
            LD_solutions = self.Create_LD_solutions()

            # Random embedding matrix
            REM_matrix = self.Create_REM_marix()

            train_step = 100
            for k in range(train_step):
                if np.isnan(LD_solutions).any() or np.isinf(LD_solutions).any():
                    raise ValueError("LD_solutions contains NaN or Inf values.")

                # Ensure consistency in the direction of updates
                SUSD_direction = self.SUSD_Cov_Operation(LD_solutions)
                if k > 0 and np.dot(SUSD_direction, nhold) < 0:
                    SUSD_direction = -SUSD_direction
                nhold = SUSD_direction

                # Transitioning from a low-dimensional solution to a high-dimensional solution
                HD_solutions = np.matmul(REM_matrix, LD_solutions)

                cost = []

                # The high-dimensional solution assignment is fed into the DNN to obtain the evaluation function value
                with torch.no_grad():
                    for i in range(len(self.sample_net)):
                        for name, param in self.sample_net[i].named_parameters():
                            if 'fc1.weight' in name:
                                fc1_weight = torch.from_numpy(HD_solutions[:, i][0:N_STATES * N_Units].reshape(N_STATES, N_Units).T)
                                param.copy_(fc1_weight)
                            if 'fc1.bias' in name:
                                fc1_bias = torch.from_numpy(HD_solutions[:, i][N_STATES * N_Units:N_STATES * N_Units + N_Units])
                                param.copy_(fc1_bias)
                            if 'out.weight' in name:
                                out_weight = torch.from_numpy(HD_solutions[:, i][N_STATES * N_Units + N_Units:(N_STATES * N_Units + N_Units) + N_Units * N_ACTIONS].reshape(N_Units, N_ACTIONS).T)
                                param.copy_(out_weight)
                            if 'out.bias' in name:
                                out_bias = torch.from_numpy(HD_solutions[:, i][(N_STATES * N_Units + N_Units) + N_Units * N_ACTIONS:])
                                param.copy_(out_bias)

                        xt = copy.deepcopy(x0)
                        sample_u_plot ,sample_cost ,sample_x1_plot , sample_x2_plot  = self.simulate(xt,i,horizen_length)
                        cost.append(sample_cost)

                        if min_cost > sample_cost:
                            min_cost = sample_cost
                            u_plot = sample_u_plot
                            x1_plot = sample_x1_plot
                            x2_plot = sample_x2_plot
                            best_theta = HD_solutions[:, i]
                            best_LD_solution = LD_solutions[:, i]

                for i in range(len(self.sample_net)):
                    z = 1 - math.exp(min(cost) - cost[i])
                    if (len(mini_cost_list) == init_times) and abs(min(cost) - (sum(mini_cost_list) / len(mini_cost_list))) < 1e-2 and z != 0:
                        LD_solutions[:, i] = best_LD_solution + np.random.normal(0, 0.1, SUSD_direction.shape)
                        Initialize = True
                        continue
                    else:
                        q = self.k * z * SUSD_direction
                        LD_solutions[:, i] = LD_solutions[:, i] + q

                if Initialize:
                    mini_cost_list = []
                    Initialize = False

                if len(mini_cost_list) < init_times:
                    mini_cost_list.append(min(cost))
                else:
                    mini_cost_list[it % init_times] = min(cost)

            it_cost.append(min_cost)

            with torch.no_grad():
                for name, param in self.target_net.named_parameters():
                    if 'fc1.weight' in name:
                        fc1_weight = torch.from_numpy(best_theta[0:N_STATES * N_Units].reshape(N_STATES, N_Units).T)
                        param.copy_(fc1_weight)
                    if 'fc1.bias' in name:
                        fc1_bias = torch.from_numpy(best_theta[N_STATES * N_Units:N_STATES * N_Units + N_Units])
                        param.copy_(fc1_bias)
                    if 'out.weight' in name:
                        out_weight = torch.from_numpy(best_theta[N_STATES * N_Units + N_Units:(N_STATES * N_Units + N_Units) + N_Units * N_ACTIONS].reshape(N_Units, N_ACTIONS).T)
                        param.copy_(out_weight)
                    if 'out.bias' in name:
                        out_bias = torch.from_numpy(best_theta[(N_STATES * N_Units + N_Units) + N_Units * N_ACTIONS:])
                        param.copy_(out_bias)

            print('iteration:{},min_cost:{}'.format(it, min_cost))
        return u_plot,x1_plot,x2_plot,it_cost


x0 = np.array([1, -1])
x = x0.copy()
lqr_model = LQR_Model(dt)
network = LQR_REM_SUSD_network(lqr_model)

import psutil, os,time
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024**2  # MB
start = time.time()

u_plot,x1_plot,x2_plot, cost_plot = network.update_para(x0,max_iter=30,horizen_length=length)
print(f"Training Time: {time.time() - start:.2f} 秒")
mem_after = process.memory_info().rss / 1024**2  # MB
print(f"Memory usage increases: {mem_after-mem_before:.2f} MB")
print(f"Total memory at the end of training: {mem_after:.2f} MB")



