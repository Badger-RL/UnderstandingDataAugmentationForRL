from typing import Optional, Tuple

import gym
import numpy as np
from gym.core import ObsType

from my_gym.envs import Goal2DKeyEnv


class Goal2DKeyEnv(Goal2DKeyEnv):
    def __init__(self, delta=0.025, sparse=1, rbf_n=None, d_fourier=None, neural=False, d=1,
                 quadrant=False, center=False):

        super().__init__(delta, sparse, rbf_n, d_fourier, neural, d, quadrant, center)
        self.observation_space = gym.spaces.Box(-self.boundary, +self.boundary, shape=(3 * self.n + 1,), dtype="float64")


    def step(self, a):

        self.step_num += 1
        ux = a[0] * np.cos(a[1])
        uy = a[0] * np.sin(a[1])
        u = np.array([ux, uy])

        self.x += u * self.delta
        self._clip_position()

        dist_goal = np.linalg.norm(self.x - self.goal)
        dist_key = np.linalg.norm(self.x - self.key)

        at_goal = dist_goal < 0.05
        self.has_key = (dist_key < 0.05) or self.has_key

        self.obs[:2] = self.x
        self.obs[-1] = self.has_key

        terminated = at_goal and self.has_key
        truncated = False

        if self.sparse:
            reward = +1.0 if terminated else -0.1
        else:
            reward = -dist_goal

        info = {}
        return self._get_obs(), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:

        self.step_num = 0

        self.goal = np.random.uniform(low=-0.5, high=0.5, size=(self.n,))
        self.key = np.random.uniform(low=-0.5, high=0.5, size=(self.n,))
        self.x = np.random.uniform(-0.5, 0.5, size=(self.n,))
        self.has_key = False

        self.obs = np.zeros(self.observation_space.shape)
        self.obs[:2] = self.x
        self.obs[2:4] = self.goal
        self.obs[4:6] = self.key
        self.obs[-1] = self.has_key

        return self._get_obs(), {}

    def set_state(self, pos, goal):
        self.x = pos
        self.goal = goal