import time
from typing import Dict, List, Any

import gym#, my_gym
import numpy as np
import torch

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction


class InvertedPendulumTranslate(AugmentationFunction):

    def __init__(self, sigma=0.1, clip=True, noise='uniform', **kwargs):
        super().__init__()
        self.sigma = sigma
        self.clip = clip
        self.delta = np.random.uniform(-1, 1)
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

        # state_dim = obs.shape[-1]
        # delta = self.noise_function(low=-self.sigma, high=+self.sigma, size=(augmentation_n,))
        if done[0]: self.delta = np.random.uniform(-1, 1)
        obs[:,0] += self.delta
        next_obs[:,0] += self.delta

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
            delta = np.random.uniform(low=-0.95, high=+0.95, size=(n,))

        delta_x = next_obs[:,0] - obs[:,0]
        # delta_x = torch.from_numpy(delta_x)
        obs[:,0] = delta
        # print(delta_x, delta)
        next_obs[:,0] = np.clip(delta_x + delta, -1, 1)

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

        delta_x = next_obs[:, 0] - obs[:,0]

        obs[:,0:] *= -1
        next_obs[:,0:] *= -1
        # next_obs[:,0] -= 2*delta_x
        action = ~action

        return obs, next_obs, action, reward, done, infos



if __name__ == "__main__":
    env = gym.make('InvertedPendulum-v2')

    f = InvertedPendulumReflect()
    f = InvertedPendulumTranslateUniform()

    for i in range(1000):
        env.reset()
        # set initial qpos, qvel
        # qpos = np.copy(env.sim.data.qpos)
        # qvel = np.copy(env.sim.data.qvel)

        qpos = np.array([0, 0])
        qvel = np.zeros(2)
        env.set_state(qpos, qvel)
        obs = env.get_obs()

        # get transition
        action = np.ones(1)*3
        next_obs, reward, done, info = env.step(action)
        obs = obs.reshape(1, -1)
        next_obs = next_obs.reshape(1, -1)
        reward = np.array([reward])
        action = action.reshape(1, -1)
        done = np.array([done]).reshape(1, -1)

        # print(obs)
        # print(next_obs)
        # print()
        # continue
        # env.render()
        # time.sleep(0.1)


        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, next_obs, action, reward, done, [{}])

        # Make sure aug transition matches simulation
        # aug_obs to qpos, qvel
        qpos2, qvel2 = env.obs_to_q(aug_obs[0])
        env.set_state(qpos2, qvel2)
        # env.render()
        # time.sleep(0.1)
        obs2 = env.get_obs()
        # qpos = np.array([0, 0])
        # qvel = np.zeros(2)
        # env.set_state(qpos, qvel)
        obs = env.get_obs()
        next_obs2, reward2, done2, info2 = env.step(aug_action[0])

        print(aug_next_obs)
        print(next_obs2)


        print(aug_obs - obs2)
        print(aug_next_obs - next_obs2)
        print(aug_reward - reward2, aug_reward, reward2)

        assert np.allclose(aug_obs, obs2)
        assert np.allclose(aug_next_obs, next_obs2)
        assert np.allclose(aug_reward, reward2)