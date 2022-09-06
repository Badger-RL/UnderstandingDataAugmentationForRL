import time
from typing import Dict, List, Any

import numpy as np
import torch

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction


class SwimmerReflect(AugmentationFunction):

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
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):

        k = obs.shape[-1]//2
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        aug_obs[:,:k] *= -1
        aug_next_obs[:,:k] *= -1
        aug_action *= -1

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos


import gym, my_gym

if __name__ == "__main__":

    k=3
    env = gym.make('Swimmer-v3')
    env.reset()

    x = 0 #np.pi/4
    qpos = np.array([1,1] + [0.5*(-1)**i for i in range(1,k+1)])
    qpos = np.array([1,1] + [0.2 for i in range(1,k+1)])

    qvel = np.zeros(k+2)
    env.set_state(qpos, qvel)
    obs = env.get_obs()
    obs = obs.reshape(1, -1)

    f = Reflect()

    for t in range(1000):
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, obs, obs, obs, obs, [{}])

        env.set_state(qpos, qvel)

        env.render()
        time.sleep(0.001)

        qpos_aug = np.copy(qpos)
        qpos_aug[2:] = aug_obs[0,:k]
        env.set_state(qpos_aug, qvel)

        env.render()
        time.sleep(0.001)


        # time.sleep(0.1)

    # self.set_state(qpos, qvel)
