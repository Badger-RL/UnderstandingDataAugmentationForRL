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
        self.aug_d = aug_d
        print('delta:', self.delta)
        print('boundary:', self.boundary)
        print('sparse:', self.sparse)
        print('aug_d:', self.aug_d)

    def _set_dynamics(self, obs, next_obs, action):
        raise NotImplementedError

    def _clip_obs(self, obs):
        obs[:, :2] = np.clip(obs[:, :2], -self.boundary, +self.boundary)
        
    def _set_last_dim(self, obs, next_obs, has_key):
        idx = np.argmax(has_key)
        obs[:idx+1, -1] = 0
        obs[idx+1:, -1] = 1
        next_obs[:idx, -1] = 0
        next_obs[idx:, -1] = 1
        
    def _get_has_key(self, next_obs):
        dist = np.linalg.norm(next_obs[:, 4:6] - next_obs[:, :2], axis=-1)
        has_key = (dist < 0.05) | (next_obs[:, -1] == 1)
        idx_find_key = np.argmax(has_key)
        has_key[idx_find_key:] = True
        return has_key
    
    def _get_at_goal(self, next_obs):
        dist = np.linalg.norm(next_obs[:, 2:4] - next_obs[:, :2], axis=-1)
        at_goal = (dist < 0.05)
        return at_goal

    def _set_reward(self, reward, terminated):
        reward[:] = -0.1
        reward[terminated] = +1

    def _set_done_and_info(self, done, infos, terminated):
        done |= terminated
        infos[done & ~terminated] = [{'TimeLimit.truncated': True}]
        infos[done & terminated] = [{'TimeLimit.truncated': False}]

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
        has_key = self._get_has_key(next_obs)
        self._set_last_dim(obs, next_obs, has_key)

        at_goal = self._get_at_goal(next_obs)
        terminated = has_key & at_goal
        self._set_reward(reward, terminated)
        self._set_done_and_info(done, infos, terminated)

        return obs, next_obs, action, reward, done, infos

class Goal2DManyTranslate(Goal2DManyAugmentationFunction):

    def __init__(self, aug_d=1.0, **kwargs):
        super().__init__(aug_d=aug_d, **kwargs)

    def _set_dynamics(self, obs, next_obs, action):
        n = 1
        v = np.random.uniform(low=-0.5, high=+0.5, size=(n, 2))

        obs[:, :2] = v
        dx = action[:, 0] * np.cos(action[:, 1])
        dy = action[:, 0] * np.sin(action[:, 1])
        next_obs[:, 0] = v[:, 0] + dx * self.delta
        next_obs[:, 1] = v[:, 1] + dy * self.delta

class Goal2DManyTranslateGoal(Goal2DManyAugmentationFunction):

    def __init__(self, aug_d=1.0, **kwargs):
        super().__init__(aug_d=aug_d, **kwargs)

    def _set_dynamics(self, obs, next_obs, action):
        n = 1
        v = np.random.uniform(low=-0.5, high=+0.5, size=(n, 4))

        obs[:, 2:6] = v
        dx = action[:, 0] * np.cos(action[:, 1])
        dy = action[:, 0] * np.sin(action[:, 1])
        next_obs[:, 0] = v[:, 0] + dx * self.delta
        next_obs[:, 1] = v[:, 1] + dy * self.delta


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
        # n = 1
        n = obs.shape[0]
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
        disp = random_sample_on_disk(self.delta, n)
        v = goal + disp
        return v

    def _translate_uniform(self, goal):
        n = 1
        v = random_sample_on_box(0.5, n)
        norm = np.linalg.norm(goal-v)
        while norm < 0.05:
            v = random_sample_on_box(0.5, n)
            norm = np.linalg.norm(goal-v)
        return v

    def _set_dynamics(self, obs, next_obs, action):
        goal = next_obs[:, 2:4]
        if np.random.random() < self.q:
            obs[:, -1] = 1
            next_obs[:, -1] = 1
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

    def __init__(self,strategy='future', **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
        if self.strategy == 'future':
            self.sampler = self._sample_future
        else:
            self.sampler = self._sample_last

    def _sample_future(self, next_obs):
        n = next_obs.shape[0]
        low = np.arange(n)
        indices = np.random.randint(low=low, high=n)
        final_pos = next_obs[indices, :2].copy()
        return final_pos, indices

    def _sample_last(self, next_obs):
        n = next_obs.shape[0]
        final_pos = next_obs[-1, :2].copy()
        return final_pos, np.ones(n)*n

    def _sample_goals(self, next_obs):
        return self.sampler(next_obs)

    def _set_dynamics(self, obs, next_obs, action):
        new_goal, goal_indices = self._sample_goals(next_obs)

        idx = np.random.randint(low=0, high=goal_indices+1)
        new_key = next_obs[idx, :2].copy()

        obs[:, 4:6] = new_key
        next_obs[:, 4:6] = new_key
        obs[:, 2:4] = new_goal
        next_obs[:, 2:4] = new_goal

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
        has_key = self._get_has_key(next_obs)
        self._set_last_dim(obs, next_obs, has_key)

        at_goal = self._get_at_goal(next_obs)
        terminated = has_key & at_goal
        self._set_reward(reward, terminated)
        self._set_done_and_info(done, infos, terminated)

        final_step = np.argmax(at_goal)
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

    def _rotate_position(self, pos, theta):
        x = np.copy(pos[:, 0])
        y = np.copy(pos[:, 1])
        pos[:, 0] = x * np.cos(theta) - y * np.sin(theta)
        pos[:, 1] = x * np.sin(theta) + y * np.cos(theta)

    def _rotate_action(self, action, theta):
        action[:, 1] += theta
        action[:, 1] %= (2 * np.pi)

    def _rotate_dynamics(self, obs, next_obs, action):
        # n = 1
        n = obs.shape[0]
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

    def _set_dynamics(self, obs, next_obs, action):
        new_goal = next_obs[-1, :2].copy()

        n = obs.shape[0]
        idx = np.random.randint(n)
        new_key = next_obs[idx, :2].copy()

        obs[:, 4:6] = new_key
        next_obs[:, 4:6] = new_key
        obs[:, 2:4] = new_goal
        next_obs[:, 2:4] = new_goal

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
        has_key = self._get_has_key(next_obs)
        self._set_last_dim(obs, next_obs, has_key)

        at_goal = self._get_at_goal(next_obs)
        terminated = has_key & at_goal
        self._set_reward(reward, terminated)
        self._set_done_and_info(done, infos, terminated)

        final_step = np.argmax(at_goal)
        obs = obs[:final_step+1]
        next_obs = next_obs[:final_step+1]
        action = action[:final_step+1]
        reward = reward[:final_step+1]
        done = done[:final_step+1]
        infos = infos[:final_step+1]

GOAL2DKEY_AUG_FUNCTIONS = {
    'rotate': Goal2DManyRotateRestricted,
    'translate': Goal2DManyTranslate,
    'translate_goal': Goal2DManyTranslateGoal,
    'translate_proximal': Goal2DManyTranslateProximal,
    'her': Goal2DManyHER,
    # 'rotate_her': Goal2DRotateRestrictedHER,
}
