import time
from typing import Dict, List, Any

import gym, my_gym
import numpy as np
import torch

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
from augment.rl.augmentation_functions.validate import validate_augmentation


class ReacherRotate(AugmentationFunction):
    '''
    Rotate arm and goal.
    '''
    def __init__(self, k=2, sparse=True, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        # self.sparse = self.env.sparse
        print(self.k)
        # print(self.sparse)
        self._reward_function = self._set_dense_reward

        # if self.sparse:
        #     self._reward_function = self._set_sparse_reward
        # else:
        #     self._reward_function = self._set_dense_reward

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
        # self.sparse = self.env.sparse
        print(self.k)
        # print(self.sparse)
        # if self.sparse:
        #     self._reward_function = self._set_sparse_reward
        # else:
        #     self._reward_function = self._set_dense_reward
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

    def reflect_action(self, action):
        action[:] *= -1

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
        next_obs[:, 6:7+1] *= -1
        # next_obs[:, -3:] -= 2*(next_obs[:, -3:] - obs[:, -3:])
        self.reflect_action(action)


        # delta = next_obs - obs
        # next_obs[:, :2*self.k] -= 2*delta[:, :2*self.k]
        # next_obs[:, 2*self.k+2:-3] -= 2*delta[:, 2*self.k+2:-3]


        self._set_reward(reward, next_obs[:, -3:1], action)

        return obs, next_obs, action, reward, done, infos

def check_valid(env, aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info):

    # set env to aug_obs
    # env = gym.make('Walker2d-v4', render_mode='human')

    # env.reset()
    qpos, qvel = np.concatenate([aug_info[0]['theta'], aug_info[0]['target']]), np.concatenate([aug_obs[6:7+1], np.zeros(2)])
    env.set_state(qpos, qvel)

    # determine ture next_obs, reward
    next_obs_true, reward_true, terminated_true, truncated_true, info_true = env.step(aug_action)
    print(f'\ttrue\t\taug\tis_close')
    is_close = np.isclose(next_obs_true, aug_next_obs)
    for i in range(11):
        print(f'{i}\t{next_obs_true[i]:.8f}\t{aug_next_obs[i]:.8f}\t{is_close[i]}')
    print(np.all(is_close))
    print('reward', aug_reward-reward_true)
    assert np.allclose(aug_next_obs, next_obs_true)
    assert np.allclose(aug_reward, reward_true)

def sanity_check():
    env = gym.make('Reacher-v4', render_mode=None)
    env.reset()

    f = ReacherReflect(env=env, k=2)

    action = np.zeros(2, dtype=np.float32).reshape(1, -1)
    action[:, 0] = 0.01
    # action[:, 3:6] = 1
    # action[:, 11:13] = 1

    env.reset()
    # f.reflect_action(action)
    print(action)
    for i in range(1):
        next_obs, reward, terminated, truncated, info = env.step(action[0])
    true = next_obs.copy()
    aug = next_obs.copy().reshape(1, -1)
    aug.reshape(-1)


    env.reset()
    # action[:, 0] = -0.01
    f.reflect_action(action)

    print(action)
    for i in range(1):
        next_obs, reward, terminated, truncated, info = env.step(action[0])
    true_reflect = next_obs.copy()

    print(f'\ttrue\t\ttrue_reflect\taug\t\tis_close')
    is_close = np.isclose(true_reflect, aug[0])
    for i in range(11):
        print(f'{i}\t{true[i]:.8f}\t{true_reflect[i]:.8f}\t{aug[0][i]:.8f}\t{is_close[i]}')
    print(np.all(is_close))

REACHER_AUG_FUNCTIONS = {
    'reflect': ReacherReflect,
    'rotate': ReacherRotate,
}



if __name__ == "__main__":
    sanity_check()
    '''

    '''
    # env = gym.make('Reacher-v4')
    # aug_func = ReacherReflect()
    # validate_augmentation(env, aug_func, check_valid)