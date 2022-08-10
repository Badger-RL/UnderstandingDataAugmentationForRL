from typing import Dict, List, Any

import numpy as np
import torch

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction


class Rotate(AugmentationFunction):

    def __init__(self, sigma=0.1, k=4, **kwargs):
        super().__init__()
        self.sigma = sigma
        self.k = k


    def augment(self,
                augmentation_n: int,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]]
                ):

        delta = np.random.uniform(low=-self.sigma, high=+self.sigma, size=(augmentation_n,))

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        cos_delta = np.cos(delta)
        sin_delta = np.sin(delta)

        # ONLY ROTATE THE FIRST JOINT.
        theta = np.arccos(aug_obs[:, 0])
        goal = aug_obs[:, 2*self.k:2*self.k+2]
        fingertip_x = aug_obs[:, -3] + goal[:,0]
        fingertip_y = aug_obs[:, -2] + goal[:,1]

        aug_obs[:, 0] = np.cos(theta + delta)
        aug_obs[:, self.k] = np.sin(theta + delta)
        aug_obs[:, -3] = (fingertip_x*cos_delta - fingertip_y*sin_delta) - goal[:,0]
        aug_obs[:, -2] = (fingertip_x*sin_delta + fingertip_y*cos_delta) - goal[:,1]

        theta_next = np.arccos(aug_next_obs[:, 0])
        goal_next = aug_next_obs[:, 2*self.k:2*self.k+2]

        fingertip_x_next = aug_next_obs[:, -3] + goal_next[:,0]
        fingertip_y_next = aug_next_obs[:, -2] + goal_next[:,1]

        aug_next_obs[:, 0] = np.cos(theta_next + delta)
        aug_next_obs[:, self.k] = np.sin(theta_next + delta)
        aug_next_obs[:, -3] = (fingertip_x_next*cos_delta - fingertip_y_next*sin_delta) - goal_next[:,0]
        aug_next_obs[:, -2] = (fingertip_x_next*sin_delta + fingertip_y_next*cos_delta) - goal_next[:,1]

        aug_reward[:] = -np.linalg.norm(aug_next_obs[:, -3:-1]-goal_next)*self.k - np.square(aug_action).sum()

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos
