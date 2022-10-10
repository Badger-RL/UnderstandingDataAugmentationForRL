from copy import deepcopy
from typing import Dict, List, Any

import numpy as np

class AugmentationFunction:

    def __init__(self, env=None, **kwargs):
        self.env = env

    def _deepcopy_transition(
            self,
            augmentation_n: int,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ):
        aug_obs = np.repeat(deepcopy(obs), augmentation_n, axis=0)
        aug_next_obs = np.repeat(deepcopy(next_obs), augmentation_n, axis=0)
        aug_action = np.repeat(deepcopy(action), augmentation_n, axis=0)
        aug_reward = np.repeat(deepcopy(reward), augmentation_n, axis=0)
        aug_done = np.repeat(deepcopy(done), augmentation_n, axis=0)
        aug_infos = np.repeat(deepcopy([infos]), augmentation_n, axis=0)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

    def augment(self,
                 aug_n: int,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):

        return self._augment(*self._deepcopy_transition(aug_n, obs, next_obs, action, reward, done, infos), **kwargs)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):
        raise NotImplementedError("Augmentation function not implemented.")

