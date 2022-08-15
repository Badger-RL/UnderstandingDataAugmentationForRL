import time
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

        # delta = np.pi/2
        cos_delta = np.cos(delta)
        sin_delta = np.sin(delta)

        # np.concatenate(
        #     [
        #         np.cos(theta),  # 4 cos(joint angles)
        #         np.sin(theta),  # 4 sin(joint angles
        #         self.sim.data.qpos.flat[self.num_links:],  # 2 target
        #         self.sim.data.qvel.flat[:self.num_links],  # 4 joint velocities
        #         self.get_body_com("fingertip") - self.get_body_com("target"),  # 1 distance
        #     ]

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


import gym, my_gym

if __name__ == "__main__":
    env = gym.make('Reacher4-v3')
    env.reset()

    x = 0 #np.pi/4
    qpos = np.array([x, 0,0,0, 0.4, 0.4])
    qvel = np.zeros(6)
    env.set_state(qpos, qvel)
    obs = env.get_obs()
    obs = obs.reshape(1, -1)

    f = Rotate(sigma=1)
    aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, obs, obs, obs, obs, [{}])

    print(obs[0,-3:])
    print(aug_obs[0,-3:])
    while True:
        env.set_state(qpos, qvel)
        env.render()
        time.sleep(0.1)

        qpos_aug = np.copy(qpos)
        qpos_aug[:4] = np.arcsin(aug_obs[0, 4:8])
        env.set_state(qpos_aug, qvel)
        env.render()
        time.sleep(0.1)

    # self.set_state(qpos, qvel)
