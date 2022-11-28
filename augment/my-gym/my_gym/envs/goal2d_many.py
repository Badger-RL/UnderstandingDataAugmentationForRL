from typing import Optional, Tuple

import gym
import numpy as np
from gym.core import ObsType

from my_gym.envs import PredatorPreyEnv


class Goal2DManyEnv(PredatorPreyEnv):
    def __init__(self, delta=0.025, sparse=1, rbf_n=None, d_fourier=None, neural=False, d=1, shape='disk',
                 quadrant=False, center=False):

        super().__init__(delta, sparse, rbf_n, d_fourier, neural, d, shape, quadrant, center)
        self.observation_space = gym.spaces.Box(-self.boundary, +self.boundary, shape=(3 * self.n + 1,), dtype="float64")


    def step(self, a):

        self.step_num += 1
        ux = a[0] * np.cos(a[1])
        uy = a[0] * np.sin(a[1])
        u = np.array([ux, uy])

        self.x += u * self.delta
        self._clip_position()

        dist = np.linalg.norm(self.x - self.curr_goal)
        at_goal = dist < 0.05
        goal_num = self.obs[-1]

        terminated = False
        if at_goal:
            if goal_num == 1:
                terminated = True
            else:
                self.obs[-1] = 1
                self.curr_goal = self.goals[2:]

        truncated = False

        if self.sparse:
            reward = +1.0 if terminated else -0.1
        else:
            reward = -dist

        info = {}
        self.obs[:2] = self.x
        return self._get_obs(), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:

        self.step_num = 0

        self.goals = np.random.uniform(low=-0.5, high=0.5, size=(2*self.n,))
        self.curr_goal = self.goals[:2]
        self.x = np.random.uniform(-0.5, 0.5, size=(self.n,))

        self.obs = np.concatenate((self.x, self.goals, np.zeros(1)))
        return self._get_obs(), {}

    def set_state(self, pos, goal):
        self.x = pos
        self.goal = goal
        if self.shape == 'disk':
            self.x_norm = np.linalg.norm(self.x)

class PredatorPreyBoxEnv(PredatorPreyEnv):
    def __init__(self, d=1, shape='box', rbf_n=None, d_fourier=None, neural=False, center=False):
        super().__init__(delta=0.025, sparse=1, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, shape=shape, center=center)

class PredatorPreyDenseEnv(PredatorPreyEnv):
    def __init__(self, d=1, shape='disk', rbf_n=None, d_fourier=None, neural=False):
        super().__init__(delta=0.025, sparse=0, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, shape=shape)

class PredatorPreyBoxDenseEnv(PredatorPreyEnv):
    def __init__(self, d=1, shape='box', rbf_n=None, d_fourier=None, neural=False):
        super().__init__(delta=0.025, sparse=0, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, shape=shape)


class PredatorPreyBoxQuadrantEnv(PredatorPreyEnv):
    def __init__(self, d=1, shape='box', rbf_n=None, d_fourier=None, neural=False):
        super().__init__(delta=0.025, sparse=1, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, shape=shape, quadrant=1)