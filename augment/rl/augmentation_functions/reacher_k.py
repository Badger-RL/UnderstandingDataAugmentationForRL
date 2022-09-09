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
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):

        if delta is None:
            delta = np.random.uniform(low=-self.sigma, high=+self.sigma)



        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = self._deepcopy_transition(
            augmentation_n, obs, next_obs, action, reward, done, infos)

        M = np.array([[np.cos(delta), -np.sin(delta)],
                      [np.sin(delta), np.cos(delta)]])

        # rotate targets
        goal = aug_obs[:, 2*self.k:2*self.k+2]
        aug_goal = M.dot(goal[0])
        aug_obs[:, 2*self.k:2*self.k+2] = aug_goal

        goal_next = aug_next_obs[:, 2*self.k:2*self.k+2]
        aug_goal_next = M.dot(goal_next[0])
        aug_next_obs[:, 2*self.k:2*self.k+2] = aug_goal_next

        # rotate central joint
        theta = np.arctan2(aug_obs[:, self.k], aug_obs[:, 0])
        aug_obs[:, 0] = np.cos(theta + delta)
        aug_obs[:, self.k] = np.sin(theta + delta)

        theta_next = np.arctan2(aug_next_obs[:, self.k], aug_next_obs[:, 0])
        aug_next_obs[:, 0] = np.cos(theta_next + delta)
        aug_next_obs[:, self.k] = np.sin(theta_next + delta)

        # rotate fingertips (use original goal to compute fingertip)
        fingertip_dist = aug_obs[:, -3:-1]
        aug_obs[:, -3:-1] = M.dot(fingertip_dist[0])

        fingertip_dist_next = aug_next_obs[:, -3:-1]
        aug_next_obs[:, -3:-1] = M.dot(fingertip_dist_next[0])

        # reward should be unchanged
        aug_fingertip_dist = aug_obs[:, -3:-1]
        aug_reward[:] = -np.linalg.norm(aug_fingertip_dist)*self.k - np.square(action).sum()

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

import gym, my_gym

if __name__ == "__main__":
    env = gym.make('Reacher4-v3')
    env.reset()

    # set initial qpos, qvel
    qpos = np.array([0.2, 0.3, 0.5, 0.7] + [0, 1])
    qvel = np.zeros(6)
    env.set_state(qpos, qvel)
    obs = env.get_obs()

    # get transition
    action = np.ones(4)
    next_obs, reward, done, info = env.step(action)
    obs = obs.reshape(1,-1)
    next_obs = next_obs.reshape(1,-1)
    action = action.reshape(1,-1)
    done = np.array([done]).reshape(1, -1)

    f = Rotate(sigma=1)

    for i in range(1000):
        delta = np.random.uniform(-np.pi, np.pi)
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, next_obs, action, reward, done, [{}], delta=delta)

        # Make sure aug transition matches simulation
        # aug_obs to qpos, qvel
        qpos2, qvel2 = env.obs_to_q(aug_obs[0])
        env.set_state(qpos2, qvel2)
        obs2 = env.get_obs()
        next_obs2, reward2, done2, info2 = env.step(action)

        assert np.allclose(aug_obs, obs2)
        assert np.allclose(aug_next_obs, next_obs2)
        assert np.allclose(aug_reward, reward2)

        # print(aug_obs - obs2)
        # print(aug_next_obs - next_obs2)
        # print(aug_reward - reward2, aug_reward, reward2)

    # qpos[qpos < 0] += 2*np.pi

    #
    #
    # x = 0 #np.pi/4
    # qpos = np.array([x, 0.5, 0.5, 0.5, 0.4, 0.4])
    # qvel = np.zeros(6)
    # env.set_state(qpos, qvel)
    # obs = env.get_obs()
    # obs = obs.reshape(1, -1)
    #
    #
    # for delta in np.linspace(0, 4*np.pi, 100):
    #
    #     aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = f.augment(1, obs, obs, obs, obs, obs, [{}], delta=delta)
    #
    #     theta = np.arctan2(aug_obs[0, 4], aug_obs[0, 0])
    #     if theta < 0: theta += 2*np.pi
    #
    #     print(delta/np.pi, theta/np.pi, delta/np.pi - theta/np.pi)
    #     qpos_aug = np.copy(qpos)
    #     qpos_aug[0] = theta
    #
    #     env.set_state(qpos_aug, qvel)
    #     env.render()
    #     time.sleep(0.01)
    #

