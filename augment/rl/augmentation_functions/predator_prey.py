import numpy as np
from typing import Dict, List, Any
from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

def random_sample_on_disk(d, n):
    r = np.random.uniform(0, d, size=(n,))
    theta = np.random.uniform(-np.pi, np.pi, size=(n,))
    return np.array([r*np.cos(theta), r*np.sin(theta)]).T

def random_sample_on_box(d, n):
    return np.random.uniform(-d, d, size=(n,2))

class PredatorPreyAugmentationFunction(AugmentationFunction):
    def __init__(self, aug_d=1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = self.env.delta
        self.boundary = self.env.boundary
        self.sparse = self.env.sparse
        self.shape = self.env.shape
        self.aug_d = aug_d
        print('delta:', self.delta)
        print('boundary:', self.boundary)
        print('sparse:', self.sparse)
        print('shape:', self.shape)
        print('aug_d:', self.aug_d)

        if self.sparse:
            self._set_reward_function = self._set_sparse_reward
        else:
            self._set_reward_function = self._set_dense_reward

        assert self.shape in ['disk', 'box']
        if self.shape == 'disk':
            self._sampling_function = random_sample_on_disk
            self._clipping_function = self._clip_to_disk
        elif self.shape == 'box':
            self._sampling_function = random_sample_on_box
            self._clipping_function = self._clip_to_box

    def _set_dynamics(self, obs, next_obs, action):
        raise NotImplementedError

    def _clip_to_disk(self, obs):
        x_norm = np.linalg.norm(obs[:, :2], axis=-1)
        mask = x_norm > 1
        obs[mask, 0] /= x_norm[mask]
        obs[mask, 1] /= x_norm[mask]

    def _clip_to_box(self, obs):
        obs[:, :2] = np.clip(obs[:, :2], -self.boundary, +self.boundary)

    def _clip_obs(self, obs):
        self._clipping_function(obs)

    def _get_at_goal(self, next_obs):
        dist = np.linalg.norm(next_obs[:, 2:] - next_obs[:, :2], axis=-1)
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
        self._clip_obs(next_obs)
        at_goal = self._get_at_goal(next_obs)
        self._set_reward(reward, next_obs, at_goal)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

class PredatorPreyTranslate(PredatorPreyAugmentationFunction):

    def __init__(self, aug_d=1.0, **kwargs):
        super().__init__(aug_d=aug_d, **kwargs)

    def _set_dynamics(self, obs, next_obs, action):
        n = 1
        v = np.random.uniform(low=-self.aug_d, high=+self.aug_d, size=(n, 2))

        obs[:, :2] = v
        dx = action[:, 0] * np.cos(action[:, 1])
        dy = action[:, 0] * np.sin(action[:, 1])
        next_obs[:, 0] = v[:, 0] + dx * self.delta
        next_obs[:, 1] = v[:, 1] + dy * self.delta
        next_obs[:, :2] = np.clip(next_obs[:, :2], -self.boundary, self.boundary)


class PredatorPreyRotate(PredatorPreyAugmentationFunction):

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

    def _set_dynamics(self, obs, next_obs, action):
        n = 1
        if self.restricted:
            theta = np.random.choice(self.thetas, replace=False, size=(n,))
        else:
            theta = np.random.uniform(-np.pi, np.pi, size=(n,))

        self._rotate_position(obs[:,:2], theta)
        self._rotate_position(obs[:,2:], theta)
        self._rotate_action(action, theta)

        # recompute next agent position, since clipping behavior differs near the boundaries for shape=='box'
        dx = action[:, 0] * np.cos(action[:, 1])
        dy = action[:, 0] * np.sin(action[:, 1])
        next_obs[:, 0] = obs[:, 0] + dx * self.delta
        next_obs[:, 1] = obs[:, 1] + dy * self.delta

        self._rotate_position(next_obs[:, 2:], theta)

class PredatorPreyRotateRestricted(PredatorPreyRotate):
    def __init__(self, **kwargs):
        super().__init__(restricted=True, **kwargs)

class PredatorPreyTranslateProximal(PredatorPreyTranslate):
    def __init__(self, p=0.5, aug_d=1, **kwargs):
        super().__init__(aug_d=aug_d, **kwargs)
        self.p = p
        print('p:', self.p)

    def _translate_proximal(self, obs):
        n = 1
        goal = obs[:, 2:]
        disp = self._sampling_function(self.delta, n)
        v = goal + disp
        return v

    def _translate_uniform(self, obs):
        n = 1
        goal = obs[:, 2:]
        v = self._sampling_function(self.aug_d, n)
        norm = np.linalg.norm(goal-v)
        while norm < 0.05:
            v = self._sampling_function(self.aug_d, n)
            norm = np.linalg.norm(goal-v)
        return v

    def _set_dynamics(self, obs, next_obs, action):
        if np.random.random() < self.p:
            v = self._translate_proximal(next_obs) # guaranteed success
        else:
            v = self._translate_uniform(next_obs) # guaranteed failure
        next_obs[:, :2] = v
        dx = action[:, 0] * np.cos(action[:, 1])
        dy = action[:, 0] * np.sin(action[:, 1])
        obs[:, 0] = v[:, 0] - dx * self.delta
        obs[:, 1] = v[:, 1] - dy * self.delta


class PredatorPreyHER(PredatorPreyAugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_dynamics(self, obs, next_obs, action):
        final_pos = next_obs[-1, :2].copy()
        obs[:, 2:] = final_pos
        next_obs[:, 2:] = final_pos

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
        self._clip_obs(next_obs)
        at_goal = self._get_at_goal(next_obs)
        final_step = np.argmax(at_goal)
        obs = obs[:final_step+1]
        next_obs = next_obs[:final_step+1]
        action = action[:final_step+1]
        reward = reward[:final_step+1]
        done = done[:final_step+1]
        infos = infos[:final_step+1]
        at_goal = at_goal[:final_step+1]

        self._set_reward(reward, next_obs, at_goal)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos