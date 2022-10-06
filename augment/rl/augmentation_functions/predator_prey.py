from typing import Dict, List, Any

import numpy as np
import torch
from matplotlib import pyplot as plt

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
import gym, my_gym

class PredatorPreyTranslate(AugmentationFunction):

    def __init__(self, delta=0.05, d=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self.d = d
        print('d', d)

    def _translate(self, obs, next_obs, action):
        n = obs.shape[0]
        v = np.random.uniform(low=-self.d, high=+self.d, size=(n, 2))

        obs[:, :2] = v
        dx = action[:, 0] * np.cos(action[:, 1])
        dy = action[:, 0] * np.sin(action[:, 1])
        next_obs[:, 0] = v[:, 0] + dx * self.delta
        next_obs[:, 1] = v[:, 1] + dy * self.delta
        next_obs[:, :2] = np.clip(next_obs[:, :2], -1, +1)

    def _get_at_goal(self, next_obs):
        dist = np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:], axis=-1)
        at_goal = (dist < 0.05)
        return at_goal

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):
        self._translate(obs, next_obs, action)
        dist = np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:], axis=-1)
        at_goal = (dist < 0.05)
        done |= at_goal
        reward[at_goal] = +1
        reward[~at_goal] = -0.1

        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

        return obs, next_obs, action, reward, done, infos


class PredatorPreyRotate(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thetas = [np.pi / 2, np.pi, np.pi * 3 / 2]

    def _rotate_obs(self, obs, theta):
        # rotate agent position
        x = np.copy(obs[:, 0])
        y = np.copy(obs[:, 1])
        obs[:, 0] = x * np.cos(theta) - y * np.sin(theta)
        obs[:, 1] = x * np.sin(theta) + y * np.cos(theta)

        # rotate goal position
        x = np.copy(obs[:, 2])
        y = np.copy(obs[:, 3])
        obs[:, 2] = x * np.cos(theta) - y * np.sin(theta)
        obs[:, 3] = x * np.sin(theta) + y * np.cos(theta)

    def _rotate_action(self, action, theta):
        action[:, 1] += theta
        action[:, 1] %= (2 * np.pi)

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                p=None,
                ):
        n = obs.shape[0]
        theta = np.random.choice(self.thetas, replace=False, size=(n,))

        self._rotate_obs(obs, theta)
        self._rotate_obs(next_obs, theta)
        self._rotate_action(action, theta)

        return obs, next_obs, action, reward, done, infos


class PredatorPreyTranslateDense(PredatorPreyTranslate):

    def __init__(self, d=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = 0.01
        self.d = d
        print('d', d)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):
        self._translate(obs, next_obs, action)
        dist = np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:], axis=-1)
        reward = -dist

        return obs, next_obs, action, reward, done, infos

class PredatorPreyTranslate02(PredatorPreyTranslate):
    def __init__(self, **kwargs):
        super().__init__(d=0.2, **kwargs)

class PredatorPreyTranslateDense02(PredatorPreyTranslateDense):
    def __init__(self, **kwargs):
        super().__init__(d=0.2, **kwargs)

class PredatorPreyTranslateProximal(PredatorPreyTranslate):
    def __init__(self, p=0.5, **kwargs):
        super().__init__( **kwargs)
        self.p = p

    def _translate(self, obs, next_obs, action):
        n = obs.shape[0]
        if np.random.random() < self.p:
            # guaranteed success
            goal = next_obs[:,2:]
            r = np.random.uniform(0, self.delta, size=(n,))
            theta = np.random.uniform(low=-np.pi, high=+np.pi, size=(n,))
            disp = r*np.array([np.cos(theta), np.sin(theta)])
            v = np.clip(goal + disp.T, -1, +1)
            # dist = np.linalg.norm(v - goal, axis=-1)
            # assert np.all(dist < 0.05)

        else:
            # guaranteed failure
            v = np.random.uniform(low=-self.d, high=+self.d, size=(n, 2))
            dist = np.linalg.norm(v-next_obs[:, 2:], axis=-1)
            while np.any(dist < 0.05):
                v = np.random.uniform(low=-self.d, high=+self.d, size=(n, 2))
                dist = np.linalg.norm(v-next_obs[:, 2:])
            # assert np.all(dist > 0.05)
        next_obs[:, :2] = v
        dx = action[:, 0] * np.cos(action[:, 1])
        dy = action[:, 0] * np.sin(action[:, 1])
        obs[:, 0] = v[:, 0] - dx * self.delta
        obs[:, 1] = v[:, 1] - dy * self.delta

