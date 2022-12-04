from copy import deepcopy
from typing import Dict, List, Any

import numpy as np

class AugmentationFunction:

    def __init__(self, env=None, **kwargs):
        self.env = env
        self.is_her = False

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
        aug_obs = np.repeat(obs, augmentation_n, axis=0)
        aug_next_obs = np.repeat(next_obs, augmentation_n, axis=0)
        aug_action = np.repeat(action, augmentation_n, axis=0)
        aug_reward = np.repeat(reward, augmentation_n, axis=0)
        aug_done = np.repeat(done, augmentation_n, axis=0).astype(np.bool)
        aug_infos = np.repeat(infos, augmentation_n, axis=0)

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

        self.num_obs = obs.shape[0]
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


class HERAugmentationFunction(AugmentationFunction):

    def __init__(self, env=None, **kwargs):
        super().__init__(env, **kwargs)
        self.is_her = True
        self.aug_n = None

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
        aug_obs = np.tile(obs, (augmentation_n,1,1))
        aug_next_obs = np.tile(next_obs, (augmentation_n,1,1))
        aug_action = np.tile(action, (augmentation_n,1,1))
        aug_reward = np.tile(reward, (augmentation_n,1,1))
        aug_done = np.tile(done, (augmentation_n,1,1)).astype(np.bool)
        aug_infos = np.tile([infos], (augmentation_n,1,1))

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
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = \
            self._deepcopy_transition(aug_n, obs, next_obs, action, reward, done, infos)

        for i in range(aug_n):
            self._augment(aug_obs[i], aug_next_obs[i], aug_action[i], aug_reward[i][0], aug_done[i][0], aug_infos[i], **kwargs)

        aug_obs = aug_obs.reshape((-1, aug_obs.shape[-1]))
        aug_next_obs = aug_next_obs.reshape((-1, aug_next_obs.shape[-1]))
        aug_action = aug_action.reshape((-1, aug_action.shape[-1]))
        aug_reward = aug_reward.reshape(-1)
        aug_done = aug_done.reshape(-1)
        aug_infos = aug_infos.reshape((-1,1))

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):
        raise NotImplementedError("Augmentation function not implemented.")
