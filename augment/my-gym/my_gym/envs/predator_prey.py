import gym
import numpy as np

from matplotlib import pyplot as plt

from augment.rl.algs.td3 import TD3
from my_gym.envs.my_env import MyEnv

class PredatorPreyEnv(MyEnv):
    def __init__(self, delta=0.025, sparse=1, rbf_n=None, neural=False, r=1, init_dist='disk'):

        self.n = 2
        # self.action_space = gym.spaces.Box(-1, +1, shape=(n,))
        self.action_space = gym.spaces.Box(low=np.zeros(2), high=np.array([1, 2 * np.pi]), shape=(self.n,))
        self.boundary = 1.05
        self.observation_space = gym.spaces.Box(-self.boundary, +self.boundary, shape=(2 * self.n,))
        # self.observation_space = gym.spaces.Box(-np.inf, +np.inf, shape=(1,))

        self.step_num = 0
        self.delta = delta

        self.sparse = sparse
        self.r = r
        self.init_dist = init_dist
        self.position_norm = None
        super().__init__(rbf_n=rbf_n, neural=neural)



    def step(self, a):
        self.step_num += 1
        ux = a[0] * np.cos(a[1])
        uy = a[0] * np.sin(a[1])
        u = np.array([ux, uy])

        self.x += u*self.delta
        # self.x = np.clip(self.x, -self.boundary, +self.boundary) # clipping makes dynamics nonlinear
        self.x_norm = np.linalg.norm(self.x)
        if self.x_norm > self.boundary:
            self.x /= self.x_norm
            self.x_norm = 1

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

        if self.init_dist == 'square':
            self.goal = np.random.uniform(low=-self.r, high=self.r, size=(self.n,))
            self.x = np.random.uniform(-1, 1, size=(self.n,))
        elif self.init_dist == 'disk':
            theta = np.random.uniform(-np.pi, +np.pi)
            r = np.random.uniform(0, self.r)
            self.goal = np.array([r * np.cos(theta), r * np.sin(theta)])

            theta = np.random.uniform(-np.pi, +np.pi)
            r = np.random.uniform(0, self.r)
            self.x = np.array([r * np.cos(theta), r * np.sin(theta)])

        self.obs = np.concatenate((self.x, self.goal))
        self.x_norm = np.linalg.norm(self.x)
        return self._get_obs()

    def set_state(self, pos, goal):
        self.x = pos
        self.x_norm = np.linalg.norm(self.x)
        self.goal = goal

class PredatorPreyEasyEnv(PredatorPreyEnv):
    def __init__(self, rbf_n=None):
        super().__init__(delta=0.25, rbf_n=None)

class PredatorPreyDenseEnv(PredatorPreyEnv):
    def __init__(self, r=1, init_dist='disk', rbf_n=None, neural=False):
        super().__init__(delta=0.025, sparse=0, rbf_n=rbf_n, neural=neural, r=r, init_dist=init_dist)