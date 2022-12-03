from typing import Dict, List, Any
import numpy as np

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction, HERAugmentationFunction


class TranslateGoal(AugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.env = env
        self.lo = env.task.goal_range_low
        self.hi = env.task.goal_range_high
        self.delta = 0.05


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

        n = obs.shape[0]
        new_goal = np.random.uniform(low=self.lo, high=self.hi, size=(n,3))
        obs[:, -3:] = new_goal
        next_obs[:, -3:] = new_goal

        achieved_goal = next_obs[:, self.env.achieved_idx]
        at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, new_goal, infos)

        self._set_done_and_info(done, infos, at_goal)

        end = np.argmax(done)+1
        obs = obs[:end]
        next_obs = next_obs[:end]
        action = action[:end]
        reward = reward[:end]
        done = done[:end]
        infos = infos[:end]


        return obs, next_obs, action, reward, done, infos


class HER(HERAugmentationFunction):
    def __init__(self, env, strategy='future', **kwargs):
        super().__init__(env=env, **kwargs)
        self.lo = env.task.goal_range_low
        self.hi = env.task.goal_range_high
        self.delta = 0.05
        self.strategy = strategy
        if self.strategy == 'future':
            self.sampler = self._sample_future
        else:
            self.sampler = self._sample_last

    def _sample_future(self, next_obs):
        n = next_obs.shape[0]
        low = np.arange(n)
        indices = np.random.randint(low=low, high=n)
        final_pos = next_obs[indices, -3].copy()
        return final_pos

    def _sample_last(self, next_obs):
        final_pos = next_obs[-1, -3:].copy()
        return final_pos

    def _sample_goals(self, next_obs):
        return self.sampler(next_obs)

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

        n = obs.shape[0]
        new_goal = next_obs[:, -3:].copy()
        obs[:, -3:] = new_goal
        next_obs[:, -3:] = new_goal

        achieved_goal = next_obs[:, self.env.achieved_idx]
        at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, new_goal, infos)

        self._set_done_and_info(done, infos, at_goal)

        if self.strategy != 'future':
            end = np.argmax(done)+1
            obs = obs[:end]
            next_obs = next_obs[:end]
            action = action[:end]
            reward = reward[:end]
            done = done[:end]
            infos = infos[:end]

        return obs, next_obs, action, reward, done, infos

PANDA_AUG_FUNCTIONS = {
    'her': HER,
    'translate_goal': TranslateGoal,
    # 'TranslateGoalProximal': TranslateGoalProximal,
}