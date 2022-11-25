from typing import Dict, List, Any
import numpy as np

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

class FetchReachAugmentationFunction(AugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.env = env
        self.lo = env.initial_gripper_xpos[:3] - env.target_range
        self.hi = env.initial_gripper_xpos[:3] + env.target_range
        self.delta = 0.05
        self.desired_mask = env.desired_mask
        self.achieved_mask = env.achieved_mask

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

class FetchHER(FetchReachAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        obs[:, self.desired_mask] = obs[-1, self.achieved_mask]
        next_obs[:, self.desired_mask] = next_obs[-1, self.achieved_mask]
        at_goal = self.env.is_success(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask]).astype(bool)
        reward = self.env.compute_reward(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask], infos)

        self._set_done_and_info(done, infos, at_goal)

        end = np.argmax(done)+1
        obs = obs[:end]
        next_obs = next_obs[:end]
        action = action[:end]
        reward = reward[:end]
        done = done[:end]
        infos = infos[:end]


        return obs, next_obs, action, reward, done, infos

class FetchTranslateGoal(FetchReachAugmentationFunction):

    def __init__(self, env, use_z=False, **kwargs):
        super().__init__(env=env, **kwargs)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        new_goal = np.random.uniform(-self.lo, self.hi)

        obs[:, self.desired_mask] = new_goal
        next_obs[:, self.desired_mask] = new_goal

        at_goal = self.env.is_success(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask]).astype(bool)
        reward = self.env.compute_reward(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask], infos)

        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos


class FetchTranslateGoalProximal(FetchReachAugmentationFunction):

    def __init__(self, env, p=0.5, **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = p

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        if np.random.random() < self.p:
            r = np.random.uniform(0, self.delta)
            theta = np.random.uniform(-np.pi, np.pi)
            phi = np.random.uniform(-np.pi/2, np.pi/2)
            dx = r*np.sin(phi)*np.cos(theta)
            dy = r*np.sin(phi)*np.sin(theta)
            dz = 0 # r*np.cos(phi)
            new_goal = obs[:, self.desired_mask] + np.array([dx, dy, dz])
        else:
            new_goal = np.random.uniform(-self.lo, self.hi)

        obs[:, self.desired_mask] = new_goal
        next_obs[:, self.desired_mask] = new_goal

        at_goal = self.env.is_success(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask]).astype(bool)
        reward = self.env.compute_reward(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask], infos)

        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos