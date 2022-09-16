from typing import Dict, List, Any

import numpy as np
import torch
from matplotlib import pyplot as plt

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
import gym, my_gym

class PredatorPreyTranslate(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        if self.rbf_n:
            obs = self.rbf_inverse(obs)
            next_obs = self.rbf_inverse(next_obs)

        v = np.random.uniform(low=-0.1, high=+0.1, size=(augmentation_n,2))
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        # delta_v = aug_next_obs[:, :2] - aug_obs[:, :2]
        # delta_x = aug_next_obs[:,0] - aug_obs[:,0]
        aug_obs[:, :2] = v
        # aug_next_obs[:, :2] = np.clip(v + delta_v, -1, +1)
        dx = aug_action[:,0]*np.cos(aug_action[:, 1])
        dy = aug_action[:,0]*np.sin(aug_action[:, 1])
        aug_next_obs[:, 0] = v[:,0] + dx*0.05
        aug_next_obs[:, 1] = v[:,1] + dy*0.05
        aug_next_obs[:, :2] = np.clip(aug_next_obs[:, :2], -1, +1)

        # print(delta_v)
        # if delta_v[0,1] > -1:
        #     stop = 0

        dist = np.linalg.norm(aug_next_obs[:, :2] - aug_next_obs[:, 2:], axis=-1)
        at_goal = (dist < 0.05)
        aug_done = at_goal | done

        aug_reward[at_goal] = +1
        aug_reward[~at_goal] = -0.1

        # aug_obs = np.clip(aug_obs, -1, +1)

        if self.rbf_n:
            aug_obs = self.rbf(aug_obs)
            aug_next_obs = self.rbf(aug_next_obs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

class PredatorPreyRotate(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

        if self.rbf_n:
            obs = self.rbf_inverse(obs)
            next_obs = self.rbf_inverse(next_obs)

        theta = np.random.choice(self.thetas, replace=False)
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

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

        if self.rbf_n:
            aug_obs = self.rbf(aug_obs)
            aug_next_obs = self.rbf(aug_next_obs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos


class PredatorPreyRotateRBF(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

        obs = self.rbf_inverse(obs)
        next_obs = self.rbf_inverse(next_obs)

        theta = np.random.choice(self.thetas, replace=False)
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

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

        aug_obs = self.rbf(aug_obs)
        aug_next_obs = self.rbf(aug_next_obs)

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos


if __name__ == "__main__":

    env = gym.make('PredatorPrey-v0', rbf_n=16)
    env.reset()

    # env.set_state(np.array([0.9,0.9]), np.array([0.58032204 -0.39712917]))
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

    f = PredatorPreyRotate(rbf_n=16)
    f = PredatorPreyTranslate(rbf_n=16)
    # f = PredatorPreyRotateRBF(rbf_n=16)


    for i in range(1000):
        obs = env.reset()
        action = np.random.uniform(low=np.zeros(2), high=np.array([1, 2*np.pi]))
        # action = np.array([1, np.pi*3/2])

        next_obs, reward, done, info = env.step(action)

        obs = obs.reshape(1, -1)
        action = action.reshape(1, -1)
        next_obs = next_obs.reshape(1, -1)
        reward = np.array([[reward]])
        done = np.array([[done]])
        infos = [info]

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, next_obs, action, reward, done, infos)

        if env.rbf_n:
            inv_aug_obs = f.rbf_inverse(aug_obs)
            env.set_state(np.copy(inv_aug_obs[0, :2]), np.copy(inv_aug_obs[0, 2:]))
        else:
            env.set_state(np.copy(aug_obs[0, :2]), np.copy(aug_obs[0, 2:]))

        next_obs2, reward2, done2, info2 = env.step(aug_action[0])

        # print('dv2', next_obs2[:2]-aug_obs[:,:2])
        print(next_obs2)
        print(aug_next_obs)
        print(reward2, aug_reward)

        if env.rbf_n:
            print()
            inv_aug_next_obs = f.rbf_inverse(aug_next_obs)
            inv_next_obs2 = f.rbf_inverse(next_obs2)
            print(inv_aug_obs)
            print(inv_aug_next_obs)
            print(inv_next_obs2)

        assert np.allclose(next_obs2, aug_next_obs)
        assert np.allclose(reward2, aug_reward)
        # assert np.allclose(done, aug_done)

