from typing import Dict, List, Any

import numpy as np
import torch
from matplotlib import pyplot as plt

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
import gym, my_gym

class PredatorPreyAugmentationFunction(AugmentationFunction):
    def __init__(self, d=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = self.env.delta
        self.boundary = self.env.boundary
        self.d = d
        print('delta:', self.delta)
        print('boundary:', self.boundary)
        print('d:', self.d)

    def _get_at_goal(self, next_obs):
        dist = np.linalg.norm(next_obs[:, 2:] - next_obs[:, :2], axis=-1)
        at_goal = (dist < 0.05)
        return at_goal

    def _set_reward(self, **kwargs):
        raise NotImplementedError

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

class PredatorPreyTranslate(PredatorPreyAugmentationFunction):

    def __init__(self, d=1.0, **kwargs):
        super().__init__(d=d, **kwargs)

    def _translate(self, obs, next_obs, action):
        n = obs.shape[0]
        v = np.random.uniform(low=-self.d, high=+self.d, size=(n, 2))

        obs[:, :2] = v
        dx = action[:, 0] * np.cos(action[:, 1])
        dy = action[:, 0] * np.sin(action[:, 1])
        next_obs[:, 0] = v[:, 0] + dx * self.delta
        next_obs[:, 1] = v[:, 1] + dy * self.delta
        # next_obs[:, :2] = np.clip(next_obs[:, :2], -self.boundary, self.boundary)

        norm = np.linalg.norm(next_obs[:, :2])
        if norm > self.boundary:
            next_obs[:, :2] /= norm

    def _set_reward(self, reward, at_goal, **kwargs):
        reward[at_goal] = +1
        reward[~at_goal] = -0.1

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
        at_goal = self._get_at_goal(next_obs)
        self._set_reward(reward, at_goal)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos


class PredatorPreyRotate(AugmentationFunction):

    def __init__(self, restricted=False, **kwargs):
        super().__init__(**kwargs)
        self.restricted = restricted
        self.thetas = [np.pi / 2, np.pi, np.pi * 3 / 2]
        print('restricted:', restricted)
        print('thetas:', self.thetas)

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
        if self.restricted:
            theta = np.random.choice(self.thetas, replace=False, size=(n,))
        else:
            theta = np.random.uniform(-np.pi, np.pi, size=(n,))

        self._rotate_obs(obs, theta)
        self._rotate_obs(next_obs, theta)
        self._rotate_action(action, theta)

        return obs, next_obs, action, reward, done, infos

PredatorPreyDenseRotate = PredatorPreyRotate # reward not chaanged after rotation

class PredatorPreyDenseTranslate(PredatorPreyTranslate):

    def __init__(self, d=1.0, **kwargs):
        super().__init__(d=d, **kwargs)

    def _set_reward(self, reward, next_obs, **kwargs):
        dist = np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:], axis=-1)
        reward[:] = -dist

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
        at_goal = self._get_at_goal(next_obs)
        self._set_reward(reward, next_obs)
        self._set_done_and_info(done, infos, at_goal)
        return obs, next_obs, action, reward, done, infos

class PredatorPreyTranslateProximal(PredatorPreyTranslate):
    def __init__(self, p=1, d=1, **kwargs):
        super().__init__(d=d, **kwargs)
        self.p = p
        print('p:', self.p)

    def _translate_proximal(self, obs):
        n = obs.shape[0]
        goal = obs[:, 2:]
        r = np.random.uniform(0, self.delta, size=(n,))
        theta = np.random.uniform(low=-np.pi, high=+np.pi, size=(n,))
        disp = r * np.array([np.cos(theta), np.sin(theta)])
        v = goal + disp.T
        return v

    def _translate_uniform(self, obs):
        n = obs.shape[0]
        v = np.random.uniform(low=-self.d, high=+self.d, size=(n, 2))
        dist = np.linalg.norm(v - obs[:, 2:], axis=-1)
        while np.any(dist < 0.05):
            v = np.random.uniform(low=-self.d, high=+self.d, size=(n, 2))
            dist = np.linalg.norm(v - obs[:, 2:])
        # assert np.all(dist > 0.05)
        return v

    def _translate(self, obs, next_obs, action):
        if np.random.random() < self.p:
            v = self._translate_proximal(next_obs) # guaranteed success
        else:
            v = self._translate_uniform(next_obs) # guaranteed failure
        next_obs[:, :2] = v
        dx = action[:, 0] * np.cos(action[:, 1])
        dy = action[:, 0] * np.sin(action[:, 1])
        obs[:, 0] = v[:, 0] - dx * self.delta
        obs[:, 1] = v[:, 1] - dy * self.delta

class PredatorPreyDenseTranslateProximal(PredatorPreyTranslateProximal):
    def __init__(self, p=1, **kwargs):
        super().__init__( **kwargs)
        self.p = p
        print('p:', self.p)

    def _set_reward(self, reward, next_obs, **kwargs):
        dist = np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:], axis=-1)
        reward[:] = -dist

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
        at_goal = self._get_at_goal(next_obs)
        self._set_reward(reward, next_obs)
        self._set_done_and_info(done, infos, at_goal)
        return obs, next_obs, action, reward, done, infos
