

import gym
import numpy as np
# import torch

from matplotlib import pyplot as plt

# from augment.dim_reduction.autoencoders import VAE
from augment.rl.algs.td3 import TD3
from my_gym.envs.my_env import MyEnv


class MeetUpEnv(MyEnv):
    def __init__(self, delta=0.025, sparse=1, rbf_n=None, d_fourier=None, neural=False, d=1, dist_features=False):

        self.n = 2
        self.action_space = gym.spaces.Box(low=np.zeros(4), high=np.array([1, 2*np.pi, 1, 2*np.pi]), shape=(2*self.n,))
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.dist_features = dist_features
        if self.dist_features:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.n,))
        else:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2*self.n,))

        self.step_num = 0
        self.delta = delta

        self.sparse = sparse
        self.d = d
        self.x_norm = None
        super().__init__(rbf_n=rbf_n, d_fourier=d_fourier, neural=neural)

    def step(self, a):
        self.step_num += 1

        ux1 = a[0] * np.cos(a[1])
        uy1 = a[0] * np.sin(a[1])
        u1 = np.array([ux1, uy1])

        ux2 = a[2] * np.cos(a[3])
        uy2 = a[2] * np.sin(a[3])
        u2 = np.array([ux2, uy2])

        self.x1 += u1 * self.delta
        self.x2 += u2 * self.delta

        dist = np.linalg.norm(self.x1 - self.x2)
        done = dist < 0.05

        if self.sparse:
            reward = +1.0 if done else -0.1
        else:
            reward = -dist

        info = {}
        if self.dist_features:
            self.obs = self.x1 - self.x2
        else:
            self.obs = np.concatenate((self.x1, self.x2))
        return self._get_obs(), reward, done, info

    def reset(self):
        self.step_num = 0

        self.x1 = np.random.uniform(low=-self.d, high=self.d, size=(self.n,))
        self.x2 = np.random.uniform(low=-self.d, high=self.d, size=(self.n,))

        if self.dist_features:
            # Don't use norm. Norm cannot distinguish between rotations.
            self.obs = self.x1 - self.x2
        else:
            self.obs = np.concatenate((self.x1, self.x2))
        return self._get_obs()

    # def set_state(self, pos, goal):
    #     self.x = pos
    #     self.goal = goal
    #     if self.shape == 'disk':
    #         self.x_norm = np.linalg.norm(self.x)

# class PredatorPreyBoxEnv(PredatorPreyEnv):
#     def __init__(self, d=1, shape='box', rbf_n=None, d_fourier=None, neural=False, obstacles=False):
#         super().__init__(delta=0.025, sparse=1, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, shape=shape, obstacles=obstacles)
#
# class PredatorPreyDenseEnv(PredatorPreyEnv):
#     def __init__(self, d=1, shape='disk', rbf_n=None, d_fourier=None, neural=False, obstacles=False):
#         super().__init__(delta=0.025, sparse=0, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, shape=shape, obstacles=obstacles)
#
# class PredatorPreyBoxDenseEnv(PredatorPreyEnv):
#     def __init__(self, d=1, shape='box', rbf_n=None, d_fourier=None, neural=False, obstacles=False):
#         super().__init__(delta=0.025, sparse=0, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, shape=shape, obstacles=obstacles)