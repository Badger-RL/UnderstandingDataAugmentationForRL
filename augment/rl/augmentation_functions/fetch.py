from typing import Dict, List, Any

import numpy as np
import torch
from matplotlib import pyplot as plt

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
import gym, my_gym

class FetchReachHER(AugmentationFunction):

    def __init__(self, delta=0.05, d=1.0, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        print('d:', self.d)

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
        for i in range(n):
            obs[i]['desired_goal'] = obs[i]['achieved_goal']
            infos[i][0]['is_success'] = 1.0
            reward[i] = 0

        return obs, next_obs, action, reward, done, infos

class FetchReachReflect(AugmentationFunction):

    def __init__(self, delta=0.05, d=1.0, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        print('d:', self.d)

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
        for i in range(n):
            obs[i]['desired_goal'] = obs[i]['achieved_goal']
            infos[i][0]['is_success'] = 1.0
            reward[i] = 0

        return obs, next_obs, action, reward, done, infos