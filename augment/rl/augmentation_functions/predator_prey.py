from typing import Dict, List, Any

import numpy as np
import torch
from matplotlib import pyplot as plt

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
import gym, my_gym

class PredatorPreyTranslate(AugmentationFunction):

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

        v = np.random.uniform(low=-1, high=+1, size=(augmentation_n,))

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        delta_v = aug_next_obs[:, :2] - aug_obs[:, :2]
        # delta_x = aug_next_obs[:,0] - aug_obs[:,0]
        aug_obs[:, :2] = v
        aug_obs[:, 2:] += v
        aug_next_obs[:, :2] += v
        aug_next_obs[:, 2:] += v

        aug_obs = np.clip(aug_obs, -1, +1)
        aug_next_obs = np.clip(aug_next_obs, -1, +1)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

class PredatorPreyRotate(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__()
        self.thetas = [np.pi/2, np.pi, np.pi*3/2]

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


        theta = np.random.choice(self.thetas, replace=False)
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        # R = np.array([[np.cos(theta), -np.sin(theta)],
        #               [np.sin(theta),  np.cos(theta)]])
        # R = np.concatenate((R,R), axis=1)

        # v0
        # aug_obs = R.dot(aug_obs.T)
        # aug_next_obs = R.dot(aug_next_obs)

        # obs0 = np.copy(aug_obs)
        # v0 = obs0[:, :2]
        # g0 = obs0[:, 2:]
        #
        # obs1 = np.copy(aug_obs)
        # v1 = obs0[:, :2]
        # g1 = obs0[:, 2:]

        # v0
        x = obs[:,0]
        y = obs[:,1]
        aug_obs[:,0] = x*np.cos(theta) - y*np.sin(theta)
        aug_obs[:,1] = x*np.sin(theta) + y*np.cos(theta)

        # g0
        x = obs[:,2]
        y = obs[:,3]
        aug_obs[:,2] = x*np.cos(theta) - y*np.sin(theta)
        aug_obs[:,3] = x*np.sin(theta) + y*np.cos(theta)

        # v1
        x = next_obs[:,0]
        y = next_obs[:,1]
        aug_next_obs[:,0] = x*np.cos(theta) - y*np.sin(theta)
        aug_next_obs[:,1] = x*np.sin(theta) + y*np.cos(theta)

        # g1
        x = next_obs[:,2]
        y = next_obs[:,3]
        aug_next_obs[:,2] = x*np.cos(theta) - y*np.sin(theta)
        aug_next_obs[:,3] = x*np.sin(theta) + y*np.cos(theta)

        aug_action[:, 1] += theta
        aug_action[:, 1] %= (2*np.pi)


        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos


if __name__ == "__main__":

    env = gym.make('PredatorPrey-v0')
    env.reset()

    # env.set_state(np.array([0.,0]), np.array([1,1]))
    # obs = np.concatenate((env.x, env.goal))
    # action = np.array([1,np.pi/2])
    # next_obs, reward, done, info = env.step(action)
    #
    # obs = obs.reshape(1, -1)
    # action = action.reshape(1, -1)
    # next_obs = next_obs.reshape(1, -1)
    # reward = np.array([[reward]])
    # done = np.array([[done]])
    # infos = [info]

    f = PredatorPreyRotate()

    for i in range(1000):
        obs = env.reset()
        action = np.random.uniform(low=np.zeros(2), high=np.array([1, 2*np.pi]))
        next_obs, reward, done, info = env.step(action)

        obs = obs.reshape(1, -1)
        action = action.reshape(1, -1)
        next_obs = next_obs.reshape(1, -1)
        reward = np.array([[reward]])
        done = np.array([[done]])
        infos = [info]

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, next_obs, action, reward, done, infos)

        env.set_state(aug_obs[0, :2], aug_obs[0, 2:])

        next_obs2, reward2, done2, info2 = env.step(aug_action[0])

        print(next_obs2)
        print(aug_next_obs)
        assert np.allclose(next_obs2, aug_next_obs)
        assert np.allclose(reward2, aug_reward)
        assert np.allclose(done, aug_done)

