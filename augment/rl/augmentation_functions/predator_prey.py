from typing import Dict, List, Any

import numpy as np
import torch
from matplotlib import pyplot as plt

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
import gym, my_gym

from augment.simulate import simulate


class PredatorPreyTranslate(AugmentationFunction):

    def __init__(self, delta=0.05, d=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self.d = d
        print('d', d)

    def _translate(self, augmentation_n, obs, next_obs, action):
        n = obs.shape[0]
        v = np.random.uniform(low=-self.d, high=+self.d, size=(n,2))

        obs[:, :2] = v
        dx = action[:,0]*np.cos(action[:, 1])
        dy = action[:,0]*np.sin(action[:, 1])
        next_obs[:, 0] = v[:,0] + dx*self.delta
        next_obs[:, 1] = v[:,1] + dy*self.delta
        next_obs[:, :2] = np.clip(next_obs[:, :2], -1, +1)

    def _get_at_goal(self, next_obs):
        dist = np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:], axis=-1)
        at_goal = (dist < 0.05)
        return at_goal

    def _augment(self,
                augmentation_n: int,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                p=None,
                ):

        self._translate(augmentation_n, obs, next_obs, action)
        at_goal = self._get_at_goal(next_obs)
        done |= at_goal
        reward[at_goal] = +1
        reward[~at_goal] = -0.1

        return obs, next_obs, action, reward, done, infos

class PredatorPreyRotate(AugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thetas = [np.pi/2, np.pi, np.pi*3/2]

    def _rotate_obs(self, obs, theta):
        # rotate agent position
        x = np.copy(obs[:,0])
        y = np.copy(obs[:,1])
        obs[:,0] = x*np.cos(theta) - y*np.sin(theta)
        obs[:,1] = x*np.sin(theta) + y*np.cos(theta)

        # rotate goal position
        x = np.copy(obs[:,2])
        y = np.copy(obs[:,3])
        obs[:,2] = x*np.cos(theta) - y*np.sin(theta)
        obs[:,3] = x*np.sin(theta) + y*np.cos(theta)

    def _rotate_action(self, action, theta):
        action[:, 1] += theta
        action[:, 1] %= (2*np.pi)

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
        # v0
        self._rotate_obs(obs,theta)
        self._rotate_obs(next_obs,theta)
        self._rotate_action(action, theta)

        return obs, next_obs, action, reward, done, infos

class PredatorPreyTranslateDense(PredatorPreyTranslate):

    def __init__(self, d=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = 0.01
        self.d = d
        print('d', d)

    def _augment(self,
                augmentation_n: int,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                p=None,
                ):

        self._translate(augmentation_n, obs, next_obs, action)
        dist = np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:], axis=-1)
        reward = -dist

        return obs, next_obs, action, reward, done, infos

class PredatorPreyTranslate02(PredatorPreyTranslate):
    def __init__(self, **kwargs):
        super().__init__(d=0.2, **kwargs)

class PredatorPreyTranslateDense02(PredatorPreyTranslateDense):
    def __init__(self, **kwargs):
        super().__init__(d=0.2, **kwargs)

if __name__ == "__main__":

    env = gym.make('PredatorPreyDense-v0')
    env.reset()

    f = PredatorPreyRotate()
    # f = PredatorPreyTranslate()
    f = PredatorPreyTranslateDense()

    for ep in range(100):
        observations, next_observations, actions, rewards, dones, infos = simulate(model=None, env=env, num_episodes=1, seed=0, render=False, flatten=True, verbose=0)
        aug_observations, aug_next_observations, aug_actions, aug_rewards, aug_dones, aug_infos = f.augment(3, observations, next_observations, actions, rewards, dones, infos)

        for aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info in zip(aug_observations, aug_next_observations, aug_actions, aug_rewards, aug_dones, aug_infos):
            env.reset()
            env.set_state(np.copy(aug_obs[:2]), np.copy(aug_obs[2:]))
            next_obs_true, reward_true, done_true, info_true = env.step(aug_action)

            assert np.allclose(aug_next_obs, next_obs_true)
            assert np.allclose(aug_reward, reward_true)

            # aug_info != info_true in general.
            dist = np.linalg.norm(aug_next_obs[:2]-aug_next_obs[2:])
            assert aug_done == ((dist < 0.05) or (aug_info == {'TimeLimit.truncated': True}))
        # assert np.allclose(done, aug_done)

