from typing import Dict, List, Any

import numpy as np
import torch

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction


class TranslatePaddle(AugmentationFunction):

    def __init__(self, sigma=0.1, clip=True, noise='uniform', **kwargs):
        super().__init__()
        self.sigma = sigma
        self.clip = clip
        if noise == 'uniform':
            self.noise_function = np.random.uniform

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                **kwargs
                ):

        n = obs.shape[0]
        delta = self.noise_function(low=-self.sigma, high=+self.sigma, size=(n,))

        obs[:,0] += delta
        next_obs[:,0] += delta

        obs[:,0].clip(-1, +1, obs[:,0])
        next_obs[:,0].clip(-1, +1, next_obs[:,0])

        return obs, next_obs, action, reward, done, infos

class InvertedPendulumTranslateUniform(AugmentationFunction):

    def __init__(self, sigma=0.1, clip=True, noise='uniform', **kwargs):
        super().__init__()
        self.sigma = sigma
        self.clip = clip

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
        if p is not None:
            bin = torch.from_numpy(np.random.multinomial(n, pvals=p[0]))
            bin = np.argwhere(bin)[0]
            bin_width = 2/len(p[0])
            delta = -1 + bin*bin_width
            delta +=  torch.from_numpy(np.random.uniform(low=0, high=bin_width, size=(n,)))
            # print(p)
        else:
            delta = torch.from_numpy(np.random.uniform(low=-1, high=+1, size=(n,)))


        obs[:,0] = delta
        # next_obs[:,0] = delta

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

        obs[:,1:] *= -1
        next_obs[:,1:] *= -1
        action[:] *= -1

        return obs, next_obs, action, reward, done, infos

