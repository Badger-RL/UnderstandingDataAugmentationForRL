from typing import Dict, List, Any

import numpy as np
import torch
from matplotlib import pyplot as plt

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

        x = torch.from_numpy(aug_obs[:,0])
        y = torch.from_numpy(aug_obs[:,1])
        aug_obs[:,0] = x*torch.cos(delta) - y*torch.sin(delta)
        aug_obs[:,1] = x*torch.sin(delta) + y*torch.cos(delta)

        x = torch.from_numpy(aug_obs[:,2])
        y = torch.from_numpy(aug_obs[:,3])
        aug_obs[:,2] = x*torch.cos(delta) - y*torch.sin(delta)
        aug_obs[:,3] = x*torch.sin(delta) + y*torch.cos(delta)

        x = torch.from_numpy(aug_next_obs[:,0])
        y = torch.from_numpy(aug_next_obs[:,1])
        aug_next_obs[:,0] = x*torch.cos(delta) - y*torch.sin(delta)
        aug_next_obs[:,1] = x*torch.sin(delta) + y*torch.cos(delta)

        x = torch.from_numpy(aug_next_obs[:,2])
        y = torch.from_numpy(aug_next_obs[:,3])
        aug_next_obs[:,2] = x*torch.cos(delta) - y*torch.sin(delta)
        aug_next_obs[:,3] = x*torch.sin(delta) + y*torch.cos(delta)
        # aug_next_obs[:,0] = delta


        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

import gym, my_gym

if __name__ == "__main__":

    env = gym.make('LQRGoal-v0')
    env.x = np.array([0.5,0.5])

    obs = env.reset()
    obs = obs.reshape(1, -1)

    f = LQRRotate()

    action = np.zeros(2).reshape(1, -1)

    print(obs)

    x = 1
    y = 0
    gx, gy = [], []

    for delta in np.linspace(0, 2*np.pi, 100):

        gx.append(x*np.cos(delta) - y*np.sin(delta))
        gy.append(x*np.sin(delta) + y*np.cos(delta))

    plt.scatter(gx, gy)
    plt.show()