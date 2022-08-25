from copy import deepcopy
from typing import Dict, List, Any

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

class AugmentationFunction:

    def __init__(self, **kwargs):
        pass

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
        aug_obs = deepcopy(obs).repeat(augmentation_n, axis=0)
        aug_next_obs = deepcopy(next_obs).repeat(augmentation_n, axis=0)
        aug_action = deepcopy(action).repeat(augmentation_n, axis=0)
        aug_reward = deepcopy(reward).repeat(augmentation_n, axis=0)
        aug_done = deepcopy(done).repeat(augmentation_n, axis=0)
        aug_infos = [deepcopy(infos) for _ in range(augmentation_n)]

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

    def augment(self,
                aug_n: int,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                **kwargs,
                ):

        raise NotImplementedError("Augmentation function not implemented.")
