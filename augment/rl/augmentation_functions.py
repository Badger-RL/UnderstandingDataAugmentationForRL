import copy
from copy import deepcopy
from typing import Dict, List, Any

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class AugmentationFunction:

    def __init__(self):
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
                replay_buffer: ReplayBuffer,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]]
                ):

        raise NotImplementedError("Augmentation function not implemented.")

class HorizontalTranslation(AugmentationFunction):

    def __init__(self, sigma=0.1, clip=True, noise='uniform'):
        super().__init__()
        self.sigma = sigma
        self.noise_function = np.random.uniform

    def augment(self,
                augmentation_n: int,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]]
                ):

        # state_dim = obs.shape[-1]
        delta = self.noise_function(low=-self.sigma, high=+self.sigma, size=(augmentation_n,))

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        aug_obs[:,0] += delta
        aug_next_obs[:,0] += delta

        aug_obs[:,0].clip(-1, +1, aug_obs[:,0])
        aug_next_obs[:,0].clip(-1, +1, aug_next_obs[:,0])


        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

    def augment_on_policy(self,
                obs: np.ndarray,
                action: np.ndarray,
                ):


        aug_obs = deepcopy(obs).repeat(augmentation_n, 1)
        aug_action = deepcopy(action).repeat(augmentation_n, 1)

        delta = np.random.uniform(low=-self.sigma, high=+self.sigma, size=(len(aug_obs)))

        aug_obs[:, 0] += delta

        return aug_obs, aug_action




