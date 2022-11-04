import time
import gym, my_gym
from typing import Dict, List, Any

import numpy as np
import torch

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction


class ReacherRotate(AugmentationFunction):
    '''
    Rotate arm and goal.
    '''
    def __init__(self, k=2, sparse=True, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.sparse = self.env.sparse
        print(self.k)
        print(self.sparse)
        if self.sparse:
            self._reward_function = self._set_sparse_reward
        else:
            self._reward_function = self._set_dense_reward

    def _set_reward(self, reward, fingertip_dist, action):
        self._reward_function(reward, fingertip_dist, action)

    def _set_sparse_reward(self, reward, fingertip_dist, action):
        reward[:] = np.linalg.norm(fingertip_dist) < 0.05

    def _set_dense_reward(self, reward, fingertip_dist, action):
        reward[:] = -np.linalg.norm(fingertip_dist) * self.k - np.square(action).sum()

    def _rotate_goal(self, M, obs):
        obs[0,2*self.k:2*self.k+2] = M.dot(obs[0,2*self.k:2*self.k+2])

    def _rotate_central_joint(self, obs, theta, delta):
        obs[:, 0] = np.cos(theta[0] + delta)
        obs[:, self.k] = np.sin(theta[0] + delta)

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):

        delta = np.random.uniform(low=-np.pi, high=+np.pi)

        M = np.array([[np.cos(delta), -np.sin(delta)],
                      [np.sin(delta), np.cos(delta)]])


        theta = infos[0][0]['theta']
        theta_next = infos[0][0]['theta_next']

        # rotate central joint
        self._rotate_central_joint(obs, theta, delta)
        self._rotate_central_joint(next_obs, theta_next, delta)

        # rotate targets
        self._rotate_goal(M, obs)
        self._rotate_goal(M, next_obs)

        # rotate fingertips (use original goal to compute fingertip)
        fingertip_dist = obs[:, -3:-1]
        obs[:, -3:-1] = M.dot(fingertip_dist[0])

        fingertip_dist_next = next_obs[:, -3:-1]
        next_obs[:, -3:-1] = M.dot(fingertip_dist_next[0])

        # # reward should be unchanged
        # aug_fingertip_dist = obs[:, -3:-1]
        # self._set_reward(reward, aug_fingertip_dist, action)

        return obs, next_obs, action, reward, done, infos


class ReacherReflect(AugmentationFunction):
    '''
    Rotate arm and goal.
    '''
    def __init__(self, k=2, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.sparse = self.env.sparse
        print(self.k)
        print(self.sparse)
        if self.sparse:
            self._reward_function = self._set_sparse_reward
        else:
            self._reward_function = self._set_dense_reward

        self.mask = np.zeros(self.k*2+2+3).astype(bool)
        self.mask[:2*self.k] = True
        self.mask[2*self.k+2:-3] = True


    def _set_reward(self, reward, fingertip_dist, action):
        self._reward_function(reward, fingertip_dist, action)

    def _set_sparse_reward(self, reward, fingertip_dist, action):
        reward[:] = np.linalg.norm(fingertip_dist) < 0.05

    def _set_dense_reward(self, reward, fingertip_dist, action):
        reward[:] = -np.linalg.norm(fingertip_dist) * self.k - np.square(action).sum()

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):

        # delta_qpos = next_obs[:, :2*self.k] - obs[:, :2*self.k]
        # delta_qvel = next_obs[:, 2*self.k+2:-3] - obs[:, 2*self.k+2:-3]
        # sin(A+B) = sin A cos B + cos A sin B
        # cos(A+B) = cos A cos B − sin A sin B

        theta = infos[0][0]['theta']
        theta_next = infos[0][0]['theta_next']
        delta_theta = theta_next - theta

        next_obs[:, :self.k] = np.cos(theta-delta_theta)
        next_obs[:, self.k:self.k*2] = np.sin(theta-delta_theta)
        next_obs[:, 2*self.k+2:-3] *= -1
        next_obs[:, -3:] -= 2*(next_obs[:, -3:] - obs[:, -3:])

        action[:] *= -1

        # delta = next_obs - obs
        # next_obs[:, :2*self.k] -= 2*delta[:, :2*self.k]
        # next_obs[:, 2*self.k+2:-3] -= 2*delta[:, 2*self.k+2:-3]


        self._set_reward(reward, next_obs[:, -3:1], action)

        return obs, next_obs, action, reward, done, infos

if __name__ == "__main__":
    k = 2
    env = gym.make(f'Reacher{k}-v3')
    f = ReacherReflect(env=env, k=k)

    obs = env.reset()
    for i in range(1000):
        obs = env.reset()
        action = np.ones(2)
        # action[0] = 1
        action = np.random.uniform(size=(2,))

        # qpos = np.zeros(4)
        # qpos[0] = (i*0.1) %( 2*np.pi)
        # qvel = np.zeros(4)
        # env.set_state(qpos, qvel)

        next_obs, reward, done, info = env.step(action)
        print('normal', next_obs-obs)

        env.reset()


        obs = obs.reshape(1,-1)
        next_obs = next_obs.reshape(1,-1)
        action = action.reshape(1,-1)
        reward = np.array([reward])
        done = np.array([done])
        info = [[info]]

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info = f.augment(1, obs, next_obs, action, reward, done, info)
        qpos, qvel = env.obs_to_q(aug_obs[0])
        env.set_state(qpos, qvel)
        env.render()


        next_obs, reward, done, info = env.step(aug_action[0])
        print('aug', next_obs-obs)

        env.render()