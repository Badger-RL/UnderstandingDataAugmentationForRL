import numpy as np
from typing import Dict, List, Any
from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

def random_sample_on_disk(d, n):
    r = np.random.uniform(0, d, size=(n,))
    theta = np.random.uniform(-np.pi, np.pi, size=(n,))
    return np.array([r*np.cos(theta), r*np.sin(theta)]).T

def random_sample_on_box(d, n):
    return np.random.uniform(-d, d, size=(n,2))


class Goal2DManyAugmentationFunction(AugmentationFunction):
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

    def _set_dynamics(self, obs, next_obs, action):
        raise NotImplementedError

    def _clip_obs(self, obs):
        obs[:, :2] = np.clip(obs[:, :2], -self.boundary, +self.boundary)
        
    def _set_last_dim(self, obs, next_obs, at_goal_1):
        idx = np.argmax(at_goal_1)
        obs[:idx+1, -1] = 0
        obs[idx+1:, -1] = 1
        next_obs[:idx, -1] = 0
        next_obs[idx:, -1] = 1
        
    def _get_at_goal_1(self, next_obs):
        dist = np.linalg.norm(next_obs[:, 2:4] - next_obs[:, :2], axis=-1)
        at_goal_1 = (dist < 0.05)
        return at_goal_1
    
    def _get_at_goal_2(self, next_obs):
        dist = np.linalg.norm(next_obs[:, 4:6] - next_obs[:, :2], axis=-1)
        at_goal_2 = (dist < 0.05)
        return at_goal_2

    def _set_reward(self, reward, at_goal_2):
        reward[at_goal_2] = +1
        reward[~at_goal_2] = -0.1

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
        at_goal_1 = self._get_at_goal_1(next_obs)
        self._set_last_dim(obs, next_obs, at_goal_1)
        
        at_goal_2 = self._get_at_goal_2(next_obs)
        self._set_reward(reward, at_goal_2)
        self._set_done_and_info(done, infos, at_goal_2)

        return obs, next_obs, action, reward, done, infos

class Goal2DManyTranslate(Goal2DManyAugmentationFunction):

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


class Goal2DManyRotate(Goal2DManyAugmentationFunction):

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
        self._rotate_position(obs[:,2:4], theta)
        self._rotate_position(obs[:,4:6], theta)
        self._rotate_action(action, theta)

        # recompute next agent position, since clipping behavior differs near the boundaries for shape=='box'
        dx = action[:, 0] * np.cos(action[:, 1])
        dy = action[:, 0] * np.sin(action[:, 1])
        next_obs[:, 0] = obs[:, 0] + dx * self.delta
        next_obs[:, 1] = obs[:, 1] + dy * self.delta

class Goal2DManyRotateRestricted(Goal2DManyRotate):
    def __init__(self, **kwargs):
        super().__init__(restricted=True, **kwargs)

class Goal2DManyTranslateProximal(Goal2DManyTranslate):
    def __init__(self, p=0.5, q=0.5, aug_d=0.5, **kwargs):
        super().__init__(aug_d=aug_d, **kwargs)
        self.p = p
        self.q = q
        print('p:', self.p)

    def _translate_proximal(self, goal):
        n = 1
        disp = random_sample_on_box(self.delta, n)
        v = goal + disp
        return v

    def _translate_uniform(self, goal):
        n = 1
        v = random_sample_on_box(self.aug_d, n)
        norm = np.linalg.norm(goal-v)
        while norm < 0.05:
            v = random_sample_on_box(self.aug_d, n)
            norm = np.linalg.norm(goal-v)
        return v

    def _set_dynamics(self, obs, next_obs, action):
        goal = next_obs[:, 2:4]
        if np.random.random() < self.q:
            goal = next_obs[:, 4:6]
        if np.random.random() < self.p:
            v = self._translate_proximal(goal) # guaranteed success
        else:
            v = self._translate_uniform(goal) # guaranteed failure
        next_obs[:, :2] = v
        dx = action[:, 0] * np.cos(action[:, 1])
        dy = action[:, 0] * np.sin(action[:, 1])
        obs[:, 0] = v[:, 0] - dx * self.delta
        obs[:, 1] = v[:, 1] - dy * self.delta


class Goal2DManyHER(Goal2DManyAugmentationFunction):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_dynamics(self, obs, next_obs, action):
        final_pos = next_obs[-1, :2].copy()

        n = obs.shape[0]
        idx = np.random.randint(n)
        random_pos = next_obs[idx, :2].copy()

        obs[:, 2:4] = random_pos
        next_obs[:, 2:4] = random_pos
        obs[:, 4:6] = final_pos
        next_obs[:, 4:6] = final_pos

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
        at_goal_1 = self._get_at_goal_1(next_obs)
        self._set_last_dim(obs, next_obs, at_goal_1)

        at_goal_2 = self._get_at_goal_2(next_obs)
        self._set_reward(reward, at_goal_2)
        self._set_done_and_info(done, infos, at_goal_2)

        final_step = np.argmax(at_goal_2)
        obs = obs[:final_step+1]
        next_obs = next_obs[:final_step+1]
        action = action[:final_step+1]
        reward = reward[:final_step+1]
        done = done[:final_step+1]
        infos = infos[:final_step+1]

        return obs, next_obs, action, reward, done, infos

class Goal2DManyRotateRestrictedHER(Goal2DManyRotateRestricted):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_goal(self, obs, next_obs, action):
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
        self._set_goal(obs, next_obs, action)
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