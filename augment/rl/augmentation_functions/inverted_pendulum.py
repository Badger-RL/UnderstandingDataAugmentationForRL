import time
from typing import Dict, List, Any

import gym#, my_gym
import numpy as np
import torch

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

class InvertedPendulumTranslate(AugmentationFunction):

    def __init__(self,  noise='uniform', **kwargs):
        super().__init__()

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
        delta = np.random.uniform(low=-0.9, high=+0.9, size=(n,))
        delta_x = next_obs[:,0] - obs[:,0]
        obs[:,0] = delta
        next_obs[:,0] = np.clip(delta_x + delta, -1, 1)

        return obs, next_obs, action, reward, done, infos

class InvertedPendulumReflect(AugmentationFunction):
    def __init__(self, **kwargs):
        super().__init__()

    def _augment(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            **kwargs,
    ):

        delta_x = next_obs[:, 0] - obs[:,0]

        obs[:,0:] *= -1
        next_obs[:,0:] *= -1
        # next_obs[:,0] -= 2*delta_x
        action = ~action

        return obs, next_obs, action, reward, done, infos