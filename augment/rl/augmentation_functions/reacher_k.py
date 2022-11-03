import time
import gym, my_gym
from typing import Dict, List, Any

import numpy as np
import torch

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction


class ReacherRotate(AugmentationFunction):
    '''
    Rotate arm and goal.
    '''
    def __init__(self, k=2, sparse=True, **kwargs):
        super().__init__()
        self.k = k
        self.sparse = sparse
        print(self.k)
        print(self.sparse)
        if self.sparse:
            self._reward_function = self._set_sparse_reward
        else:
            self._reward_function = self._set_dense_reward

    def _set_reward(self, reward, fingertip_dist, action):
        self._reward_function(reward, fingertip_dist, action)

    def _set_sparse_reward(self, reward, fingertip_dist, action):
        reward[:] = np.linalg.norm(fingertip_dist) < 0.05

    def _set_dense_reward(self, reward, fingertip_dist, action):
        reward[:] = -np.linalg.norm(fingertip_dist) * self.k - np.square(action).sum()

    def _rotate_goal(self, obs, theta):
        x = np.copy(obs[:, 2*self.k])
        y = np.copy(obs[:, 2*self.k+1])
        obs[:, 2*self.k]   = x * np.cos(theta) - y * np.sin(theta)
        obs[:, 2*self.k+1] = x * np.sin(theta) + y * np.cos(theta)

    def _rotate_central_joint(self, obs, theta):
        theta_curr = np.arctan2(obs[:, self.k], obs[:, 0])
        obs[:, 0] = np.cos(theta_curr + theta)
        obs[:, self.k] = np.sin(theta_curr + theta)

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):

        delta = np.random.uniform(low=np.pi, high=+np.pi)

        M = np.array([[np.cos(delta), -np.sin(delta)],
                      [np.sin(delta), np.cos(delta)]])

        # rotate central joint
        self._rotate_central_joint(obs, delta)
        self._rotate_central_joint(next_obs, delta)

        # rotate targets
        self._rotate_goal(obs, delta)
        self._rotate_goal(next_obs, delta)

        # rotate fingertips (use original goal to compute fingertip)
        fingertip_dist = obs[:, -3:-1]
        obs[:, -3:-1] = M.dot(fingertip_dist[0])

        fingertip_dist_next = next_obs[:, -3:-1]
        next_obs[:, -3:-1] = M.dot(fingertip_dist_next[0])

        # # reward should be unchanged
        # aug_fingertip_dist = obs[:, -3:-1]
        # self._set_reward(reward, aug_fingertip_dist, action)

        return obs, next_obs, action, reward, done, infos


class ReacherReflect(AugmentationFunction):
    '''
    Rotate arm and goal.
    '''
    def __init__(self, sigma=0.1, k=2, **kwargs):
        super().__init__()
        self.k = k

    def _rotate_goal(self, obs, theta):
        x = np.copy(obs[:, 2*self.k])
        y = np.copy(obs[:, 2*self.k+1])
        obs[:, 2*self.k]   = x * np.cos(theta) - y * np.sin(theta)
        obs[:, 2*self.k+1] = x * np.sin(theta) + y * np.cos(theta)

    def _rotate_central_joint(self, obs, theta):
        theta_curr = np.arctan2(obs[:, self.k], obs[:, 0])
        obs[:, 0] = np.cos(theta_curr + theta)
        obs[:, self.k] = np.sin(theta_curr + theta)

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):

        delta = np.random.uniform(low=np.pi, high=+np.pi)

        M = np.array([[np.cos(delta), -np.sin(delta)],
                      [np.sin(delta), np.cos(delta)]])

        # rotate central joint
        theta = np.arctan2(obs[:, self.k], obs[:, 0])
        theta_next = np.arctan2(next_obs[:, self.k], next_obs[:, 0])
        delta_theta = theta_next-theta

        next_obs[:, 0] = np.cos(theta - delta_theta)
        next_obs[:, self.k] = np.sin(theta - delta_theta)


        # rotate fingertips (use original goal to compute fingertip)
        fingertip_dist = obs[:, -3:-1]
        obs[:, -3:-1] = M.dot(fingertip_dist[0])

        fingertip_dist_next = next_obs[:, -3:-1]
        next_obs[:, -3:-1] = M.dot(fingertip_dist_next[0])

        # reward should be unchanged
        aug_fingertip_dist = obs[:, -3:-1]
        reward[:] = -np.linalg.norm(aug_fingertip_dist)*self.k - np.square(action).sum()

        return obs, next_obs, action, reward, done, infos
