import copy
from copy import deepcopy
from typing import Dict, List, Any

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class AugmentationFunction:

    def __init__(self, n=1):
        self.n = n

    def _deepcopy_transition(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ):
        aug_obs = deepcopy(obs).repeat(self.n, axis=0)
        aug_next_obs = deepcopy(next_obs).repeat(self.n, axis=0)
        aug_action = deepcopy(action).repeat(self.n, axis=0)
        aug_reward = deepcopy(reward).repeat(self.n, axis=0)
        aug_done = deepcopy(done).repeat(self.n, axis=0)
        aug_infos = [deepcopy(infos) for _ in range(self.n)]

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

    def _append_to_replay_buffer(
            self,
            replay_buffer: ReplayBuffer,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[List[Dict[str, Any]]]):

        for i in range(self.n):
            replay_buffer.add(obs[i], next_obs[i], action[i], reward[i], done[i], infos[i])


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

    def __init__(self, n=1, sigma=0.1):
        super().__init__(n=n)
        self.sigma = sigma
        # self.noise_function =

    def augment(self,
                replay_buffer: ReplayBuffer,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]]
                ):

        # state_dim = obs.shape[-1]
        delta = np.random.uniform(low=-self.sigma, high=+self.sigma, size=(self.n))

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(obs, next_obs, action, reward, done, infos)

        aug_obs[:,0] += delta
        aug_next_obs[:,0] += delta

        # aug_obs[:, 0] = np.clip(aug_obs[:, 0]+delta, -1, +1)
        # aug_next_obs[:, 0] = np.clip(aug_next_obs[:, 0]+delta, -1, +1)


        self._append_to_replay_buffer(
            replay_buffer, aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos)

        # return aug_obs, aug_next_obs, action, reward, done, infos


