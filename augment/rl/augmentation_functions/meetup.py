from typing import List, Dict, Any

import numpy as np

from augment.rl.augmentation_functions import AugmentationFunction


class MeetUpAugmentationFunction(AugmentationFunction):
    def __init__(self, aug_d=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = self.env.delta
        self.sparse = self.env.sparse
        print('delta:', self.delta)
        print('sparse:', self.sparse)

        if self.sparse:
            self._set_reward_function = self._set_sparse_reward
        else:
            self._set_reward_function = self._set_dense_reward

    def _set_dynamics(self, obs, next_obs, action):
        raise NotImplementedError

    def _get_at_goal(self, next_obs):
        dist = np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:], axis=-1)
        at_goal = (dist < 0.05)
        return at_goal

    def _set_dense_reward(self, reward, next_obs, at_goal):
        dist = np.linalg.norm(next_obs[:, :2] - next_obs[:, 2:], axis=-1)
        reward[:] = -dist

    def _set_sparse_reward(self, reward, next_obs, at_goal):
        reward[at_goal] = +1
        reward[~at_goal] = -0.1

    def _set_reward(self, reward, next_obs, at_goal):
        self._set_reward_function(reward, next_obs, at_goal)

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):
        self._set_dynamics(obs, next_obs, action)
        at_goal = self._get_at_goal(next_obs)
        self._set_reward(reward, next_obs, at_goal)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos


class MeetUpTranslate(MeetUpAugmentationFunction):

    def __init__(self, aug_d=1.0, **kwargs):
        super().__init__(aug_d=aug_d, **kwargs)
        self.aug_d = aug_d

    def _set_dynamics(self, obs, next_obs, action):
        n = 1
        v = np.random.uniform(low=-self.aug_d, high=+self.aug_d, size=(n, 2))
        obs[:, :2] += v
        obs[:, 2:] += v

        next_obs[:, :2] += v
        next_obs[:, 2:] += v

class MeetUpRotate(MeetUpAugmentationFunction):

    def __init__(self, restricted=False, **kwargs):
        super().__init__(**kwargs)
        self.restricted = restricted
        self.thetas = [np.pi / 2, np.pi, np.pi * 3 / 2]
        print('restricted:', restricted)
        print('thetas:', self.thetas)

    def _rotate_position(self, pos, theta):
        x = np.copy(pos[:, 0])
        y = np.copy(pos[:, 1])
        pos[:, 0] = x * np.cos(theta) - y * np.sin(theta)
        pos[:, 1] = x * np.sin(theta) + y * np.cos(theta)

    def _rotate_action(self, action, theta):
        action[:, 1] += theta
        action[:, 1] %= (2 * np.pi)
        action[:, 3] += theta
        action[:, 3] %= (2 * np.pi)

    def _set_dynamics(self, obs, next_obs, action):
        n = 1
        if self.restricted:
            theta = np.random.choice(self.thetas, replace=False, size=(n,))
        else:
            theta = np.random.uniform(-np.pi, np.pi, size=(n,))

        self._rotate_position(obs[:,:2], theta)
        self._rotate_position(obs[:,2:], theta)
        self._rotate_position(next_obs[:,:2], theta)
        self._rotate_position(next_obs[:,2:], theta)
        self._rotate_action(action, theta)


class MeetUpRotateTranslate(MeetUpAugmentationFunction):

    def __init__(self, restricted=False, **kwargs):
        super().__init__(**kwargs)
        self.restricted = restricted
        self.thetas = [np.pi / 2, np.pi, np.pi * 3 / 2]
        print('restricted:', restricted)
        print('thetas:', self.thetas)

    def _rotate_position(self, pos, theta):
        x = np.copy(pos[:, 0])
        y = np.copy(pos[:, 1])
        pos[:, 0] = x * np.cos(theta) - y * np.sin(theta)
        pos[:, 1] = x * np.sin(theta) + y * np.cos(theta)

    def _rotate_action(self, action, theta):
        action[:, 1] += theta
        action[:, 1] %= (2 * np.pi)
        action[:, 3] += theta
        action[:, 3] %= (2 * np.pi)

    def _set_dynamics(self, obs, next_obs, action):
        n = 1
        if self.restricted:
            theta = np.random.choice(self.thetas, replace=False, size=(n,))
        else:
            theta = np.random.uniform(-np.pi, np.pi, size=(n,))

        self._rotate_position(obs[:,:2], theta)
        self._rotate_position(obs[:,2:], theta)
        self._rotate_position(next_obs[:,:2], theta)
        self._rotate_position(next_obs[:,2:], theta)
        self._rotate_action(action, theta)

        n = 1
        v = np.random.uniform(low=-0.25, high=+0.25, size=(n, 2))
        obs[:, :2] += v
        obs[:, 2:] += v

        next_obs[:, :2] += v
        next_obs[:, 2:] += v