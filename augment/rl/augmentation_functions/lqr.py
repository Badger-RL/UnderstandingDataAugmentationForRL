from typing import Dict, List, Any

import numpy as np
import torch

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

class LQRTranslate(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__()

    def augment(self,
                augmentation_n: int,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                p=None,
                ):

        delta = torch.from_numpy(np.random.uniform(low=-1, high=+1, size=(augmentation_n, 2)))

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        aug_obs[:,:2] = delta
        # aug_next_obs[:,0] = delta

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

class LQRRotate(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__()

    def augment(self,
                augmentation_n: int,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                p=None,
                ):

        delta = torch.from_numpy(np.random.uniform(low=-np.pi, high=+np.pi, size=(augmentation_n,)))

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        # print('before:', aug_obs, aug_next_obs)

        x = aug_obs[0,0]
        y = aug_obs[0,1]
        aug_obs[0,0] = x*np.cos(delta) - y*np.sin(delta)
        aug_obs[0,1] = x*np.sin(delta) + y*np.cos(delta)

        x = aug_obs[0,2]
        y = aug_obs[0,3]
        aug_obs[0,2] = x*np.cos(delta) - y*np.sin(delta)
        aug_obs[0,3] = x*np.sin(delta) + y*np.cos(delta)

        x = aug_next_obs[0,0]
        y = aug_next_obs[0,1]
        aug_next_obs[0,0] = x*np.cos(delta) - y*np.sin(delta)
        aug_next_obs[0,1] = x*np.sin(delta) + y*np.cos(delta)

        x = aug_next_obs[0,2]
        y = aug_next_obs[0,3]
        aug_next_obs[0,2] = x*np.cos(delta) - y*np.sin(delta)
        aug_next_obs[0,3] = x*np.sin(delta) + y*np.cos(delta)
        # aug_next_obs[:,0] = delta


        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos