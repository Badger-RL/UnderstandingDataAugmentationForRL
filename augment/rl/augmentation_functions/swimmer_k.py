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


    def _augment(self,
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

        k = (obs.shape[-1]-2)//2
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        aug_obs[:,:k] *= -1
        aug_obs[:,-k:] *= -1
        aug_obs[:,k+1] *= -1


        aug_next_obs[:,:k] *= -1
        aug_next_obs[:,-k:] *= -1
        aug_next_obs[:,k+1] *= -1


        aug_action *= -1

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos


import gym, my_gym

if __name__ == "__main__":
    k=10
    env = gym.make('Swimmer10-v3')

    f = SwimmerReflect()

    for i in range(1000):
        env.reset()
        # set initial qpos, qvel
        qpos = np.copy(env.sim.data.qpos)
        qvel = np.copy(env.sim.data.qvel)

        qpos[2:] = np.random.uniform(-np.pi, np.pi, k)
        # qpos = np.ones(5)*0.3
        # qvel = np.zeros(5)
        # +1 index since qpos includes x position
        env.set_state(qpos, qvel)
        obs = env.get_obs()

        # env.render()

        # get transition
        action = np.ones(k-1)
        next_obs, reward, done, info = env.step(action)
        obs = obs.reshape(1, -1)
        next_obs = next_obs.reshape(1, -1)
        action = action.reshape(1, -1)
        done = np.array([done]).reshape(1, -1)

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, next_obs, action, reward, done, [{}],)

        # Make sure aug transition matches simulation
        # aug_obs to qpos, qvel
        qpos2, qvel2 = env.obs_to_q(aug_obs[0])
        qpos2[:2] = qpos[:2]
        env.set_state(qpos2, qvel2)
        obs2 = env.get_obs()
        next_obs2, reward2, done2, info2 = env.step(aug_action[0])
        # env.render()

        # print(info, info2)

        # print(aug_obs)
        print(aug_next_obs)

        # print(obs2)
        print(next_obs2)


        # print(aug_obs - obs2)
        # print(aug_next_obs - next_obs2)
        print(aug_reward - reward2, aug_reward, reward2)

        assert np.allclose(aug_obs, obs2)
        assert np.allclose(aug_next_obs, next_obs2)
        assert np.allclose(aug_reward, reward2)


# if __name__ == "__main__":
#
#     k=3
#     env = gym.make('Swimmer-v3')
#     env.reset()
#
#     x = 0 #np.pi/4
#     qpos = np.array([1,1] + [0.5*(-1)**i for i in range(1,k+1)])
#     qpos = np.array([1,1] + [0.2 for i in range(1,k+1)])
#
#     qvel = np.zeros(k+2)
#     env.set_state(qpos, qvel)
#     obs = env.get_obs()
#     obs = obs.reshape(1, -1)
#
#     f = Reflect()
#
#     for t in range(1000):
#         aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, obs, obs, obs, obs, [{}])
#
#         env.set_state(qpos, qvel)
#
#         env.render()
#         time.sleep(0.001)
#
#         qpos_aug = np.copy(qpos)
#         qpos_aug[2:] = aug_obs[0,:k]
#         env.set_state(qpos_aug, qvel)
#
#         env.render()
#         time.sleep(0.001)
#
#
#         # time.sleep(0.1)
#
#     # self.set_state(qpos, qvel)
