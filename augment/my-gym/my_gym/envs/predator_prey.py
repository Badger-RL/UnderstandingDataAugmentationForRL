

import gym
import numpy as np
# import torch

from matplotlib import pyplot as plt

# from augment.dim_reduction.autoencoders import VAE
from augment.rl.algs.td3 import TD3
from my_gym.envs.my_env import MyEnv


class PredatorPreyEnv(MyEnv):
    def __init__(self, delta=0.025, sparse=1, rbf_n=None, d_fourier=None, neural=False, d=1, shape='disk'):

        self.n = 2
        self.action_space = gym.spaces.Box(low=np.zeros(2), high=np.array([1, 2 * np.pi]), shape=(self.n,))
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

        self.boundary = 1.05
        self.observation_space = gym.spaces.Box(-self.boundary, +self.boundary, shape=(2 * self.n,))

        self.step_num = 0
        self.delta = delta

        self.sparse = sparse
        self.d = d
        self.shape = shape
        self.x_norm = None
        super().__init__(rbf_n=rbf_n, d_fourier=d_fourier, neural=neural)

        # vae = VAE(2, 1)
        # state_dict = torch.load(
        #     '/Users/nicholascorrado/code/augment/augment/dim_reduction/autoencoders/PredatorPreyBox-v0/VAE/1/autoencoder.pt')
        # vae.load_state_dict(state_dict)
        # vae.decoder.requires_grad_(False)
        # vae.eval()
        # self.vae = vae

    def _clip_position(self):
        # Note: clipping makes dynamics nonlinear
        if self.shape == 'disk':
            self.x_norm = np.linalg.norm(self.x)
            if self.x_norm > self.boundary:
                self.x /= self.x_norm
                self.x_norm = 1
        elif self.shape == 'box':
            self.x = np.clip(self.x, -self.boundary, +self.boundary)

    def step(self, a):
        #
        # a = torch.from_numpy(a).float()
        # a, _ = self.vae.decode(a)
        # # features = self.neural_features(obs)
        # # print(features.detach().numpy())
        # a = a.detach().numpy()
        # print(a)

        self.step_num += 1
        ux = a[0] * np.cos(a[1])
        uy = a[0] * np.sin(a[1])
        u = np.array([ux, uy])

        self.x += u * self.delta
        self._clip_position()

        dist = np.linalg.norm(self.x - self.goal)
        done = dist < 0.05

        if self.sparse:
            reward = +1.0 if done else -0.1
        else:
            reward = -dist

        info = {}
        self.obs = np.concatenate((self.x, self.goal))
        return self._get_obs(), reward, done, info

    def reset(self):
        self.step_num = 0

        if self.shape == 'disk':
            theta = np.random.uniform(-np.pi, +np.pi)
            r = np.random.uniform(0, self.d)
            self.goal = np.array([r * np.cos(theta), r * np.sin(theta)])

            theta = np.random.uniform(-np.pi, +np.pi)
            r = np.random.uniform(0, self.d)
            self.x = np.array([r * np.cos(theta), r * np.sin(theta)])
            self.x_norm = np.linalg.norm(self.x)

        if self.shape == 'box':
            self.goal = np.random.uniform(low=-self.d, high=self.d, size=(self.n,))
            self.x = np.random.uniform(-1, 1, size=(self.n,))

        self.obs = np.concatenate((self.x, self.goal))
        return self._get_obs()

    def set_state(self, pos, goal):
        self.x = pos
        self.goal = goal
        if self.shape == 'disk':
            self.x_norm = np.linalg.norm(self.x)

class PredatorPreyBoxEnv(PredatorPreyEnv):
    def __init__(self, d=1, shape='box', rbf_n=None, d_fourier=None, neural=False):
        super().__init__(delta=0.025, sparse=1, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, shape=shape)

class PredatorPreyDenseEnv(PredatorPreyEnv):
    def __init__(self, d=1, shape='disk', rbf_n=None, d_fourier=None, neural=False):
        super().__init__(delta=0.025, sparse=0, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, shape=shape)

class PredatorPreyBoxDenseEnv(PredatorPreyEnv):
    def __init__(self, d=1, shape='box', rbf_n=None, d_fourier=None, neural=False):
        super().__init__(delta=0.025, sparse=0, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, shape=shape)