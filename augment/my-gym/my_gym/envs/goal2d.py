from typing import Optional, Tuple

import gym
import numpy as np
from gym.core import ObsType

from my_gym.envs.my_env import MyEnv


class Goal2DEnv(MyEnv):
    def __init__(self, delta=0.025, sparse=1, rbf_n=None, d_fourier=None, neural=False, d=1, quadrant=False, center=False):

        self.n = 2
        self.action_space = gym.spaces.Box(low=np.zeros(2), high=np.array([1, 2 * np.pi]), shape=(self.n,))

        self.boundary = 1.05
        self.observation_space = gym.spaces.Box(-self.boundary, +self.boundary, shape=(2 * self.n,), dtype="float64")

        self.step_num = 0
        self.delta = delta

        self.sparse = sparse
        self.d = d
        self.x_norm = None
        self.quadrant = quadrant
        self.center = center
        super().__init__(rbf_n=rbf_n, d_fourier=d_fourier, neural=neural)

    def _clip_position(self):
        # Note: clipping makes dynamics nonlinear
        self.x = np.clip(self.x, -self.boundary, +self.boundary)

    def step(self, a):

        self.step_num += 1
        ux = a[0] * np.cos(a[1])
        uy = a[0] * np.sin(a[1])
        u = np.array([ux, uy])

        self.x += u * self.delta
        self._clip_position()

        dist = np.linalg.norm(self.x - self.goal)
        terminated = dist < 0.05
        truncated = False

        if self.sparse:
            reward = +1.0 if terminated else -0.1
        else:
            reward = -dist

        info = {}
        self.obs = np.concatenate((self.x, self.goal))
        return self._get_obs(), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:

        self.step_num = 0

        self.goal = np.random.uniform(low=-self.d, high=self.d, size=(self.n,))
        if self.quadrant:
            self.goal = np.random.uniform(low=0, high=1, size=(self.n,))
        self.x = np.random.uniform(-1, 1, size=(self.n,))
        if self.center:
            self.x = np.zeros(self.n)

        self.obs = np.concatenate((self.x, self.goal))
        return self._get_obs(), {}

    def set_state(self, pos, goal):
        self.x = pos
        self.goal = goal


class Goal2DQuadrantEnv(Goal2DEnv):
    def __init__(self, d=1, rbf_n=None, d_fourier=None, neural=False):
        super().__init__(delta=0.025, sparse=1, rbf_n=rbf_n, d_fourier=d_fourier, neural=neural, d=d, quadrant=True)