from typing import Dict, List, Any

import numpy as np
import torch

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction


class InvertedPendulumTranslate(AugmentationFunction):

    def __init__(self, sigma=0.1, clip=True, noise='uniform', **kwargs):
        super().__init__()
        self.sigma = sigma
        self.clip = clip
        if noise == 'uniform':
            self.noise_function = np.random.uniform

    def augment(self,
                augmentation_n: int,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                **kwargs
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

class InvertedPendulumTranslateUniform(AugmentationFunction):

    def __init__(self, sigma=0.1, clip=True, noise='uniform', **kwargs):
        super().__init__()
        self.sigma = sigma
        self.clip = clip

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

        if p is not None:
            bin = torch.from_numpy(np.random.multinomial(augmentation_n, pvals=p))
            bin = np.argwhere(bin)[0]
            delta = -1 + bin*0.02
            delta +=  torch.from_numpy(np.random.uniform(low=0, high=+0.02, size=(augmentation_n,)))
            # print(p)
        else:
            delta = torch.from_numpy(np.random.uniform(low=-1, high=+1, size=(augmentation_n,)))

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        aug_obs[:,0] = delta
        # aug_next_obs[:,0] = delta

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

class InvertedPendulumReflect(AugmentationFunction):
    def __init__(self, **kwargs):
        super().__init__()

    def augment(
            self,
            aug_n: int,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            **kwargs,
    ):

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            aug_n, obs, next_obs, action, reward, done, infos)

        aug_obs[:,1:] *= -1
        aug_next_obs[:,1:] *= -1
        aug_action[:] *= -1

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

