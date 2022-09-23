import gym
import numpy as np

from matplotlib import pyplot as plt

from augment.rl.algs.td3 import TD3
from my_gym.envs.my_env import MyEnv

class PredatorPreyEnv(MyEnv):
    def __init__(self, delta=0.05, sparse=1, rbf_n=None, neural=False, r=None):

        self.n = 2
        # self.action_space = gym.spaces.Box(-1, +1, shape=(n,))
        self.action_space = gym.spaces.Box(low=np.zeros(2), high=np.array([1, 2 * np.pi]), shape=(self.n,))
        self.observation_space = gym.spaces.Box(-1, +1, shape=(2 * self.n,))
        # self.observation_space = gym.spaces.Box(-np.inf, +np.inf, shape=(1,))

        self.step_num = 0
        self.delta = delta

        self.sparse = sparse
        self.r = r
        super().__init__(rbf_n=rbf_n, neural=neural)



    def step(self, u):
        self.step_num += 1
        ux = u[0] * np.cos(u[1])
        uy = u[0] * np.sin(u[1])
        u = np.array([ux, uy])

        self.x += u*self.delta
        self.x = np.clip(self.x, -1, +1) # clipping makes dynamics nonlinear

        done = False

        if self.sparse:
            reward = -0.1
            if np.linalg.norm(self.x - self.goal) < 0.05:
                reward = +1.0
                done = True
        else:
            reward = -np.linalg.norm(self.x - self.goal)
            if np.linalg.norm(self.x - self.goal) < 0.05:
                reward = 0
                done = True

        # if self.step_num == self.horizon:
        #     done = True

        info = {}
        self.obs = np.concatenate((self.x, self.goal))
        # dist = self.x-self.goal
        # theta = np.arctan2(dist[1], dist[0])
        # self.obs = np.concatenate(([theta],))
        return self._get_obs(), reward, done, info

    def reset(self):
        self.step_num = 0
        # self.x = np.zeros(self.n)
        self.x = np.random.uniform(-1, 1, size=(self.n,))
        # self.x = np.zeros(2)



        if self.r:
            # theta = np.random.uniform(-np.pi, +np.pi)
            # r = np.random.uniform(*self.r)
            # self.goal = np.array([r * np.cos(theta), r * np.sin(theta)])
            self.goal = np.random.uniform(low=-self.r, high=self.r, size=(self.n,))
        else:
            self.goal = np.random.uniform(-1, 1, size=(self.n,))
        # self.goal = np.ones(2)*0.5

        self.obs = np.concatenate((self.x, self.goal))

        # dist = self.x-self.goal
        # theta = np.arctan2(dist[1], dist[0])
        #
        # self.obs = np.concatenate(([theta],))

        return self._get_obs()

    def set_state(self, pos, goal):
        self.x = pos
        self.goal = goal

class PredatorPreyEasyEnv(PredatorPreyEnv):
    def __init__(self, rbf_n=None):
        super().__init__(delta=0.1, rbf_n=None)