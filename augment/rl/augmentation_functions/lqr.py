from typing import Dict, List, Any

import numpy as np
import torch
from matplotlib import pyplot as plt

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

class LQRTranslate(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__()

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
        delta = torch.from_numpy(np.random.uniform(low=-1, high=+1, size=(n, 2)))

        obs[:,:2] = delta
        # next_obs[:,0] = delta

        return obs, next_obs, action, reward, done, infos

class LQRRotate(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__()

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
        delta = torch.from_numpy(np.random.uniform(low=-np.pi, high=+np.pi, size=(n,)))

        # print('before:', obs, next_obs)
    # TODO NEED DEEPCOPY
        x = torch.from_numpy(obs[:,0])
        y = torch.from_numpy(obs[:,1])
        obs[:,0] = x*torch.cos(delta) - y*torch.sin(delta)
        obs[:,1] = x*torch.sin(delta) + y*torch.cos(delta)

        x = torch.from_numpy(obs[:,2])
        y = torch.from_numpy(obs[:,3])
        obs[:,2] = x*torch.cos(delta) - y*torch.sin(delta)
        obs[:,3] = x*torch.sin(delta) + y*torch.cos(delta)

        x = torch.from_numpy(next_obs[:,0])
        y = torch.from_numpy(next_obs[:,1])
        next_obs[:,0] = x*torch.cos(delta) - y*torch.sin(delta)
        next_obs[:,1] = x*torch.sin(delta) + y*torch.cos(delta)

        x = torch.from_numpy(next_obs[:,2])
        y = torch.from_numpy(next_obs[:,3])
        next_obs[:,2] = x*torch.cos(delta) - y*torch.sin(delta)
        next_obs[:,3] = x*torch.sin(delta) + y*torch.cos(delta)
        # next_obs[:,0] = delta


        return obs, next_obs, action, reward, done, infos

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