from typing import Dict, List, Any
import numpy as np

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction, HERAugmentationFunction
from augment.rl.augmentation_functions.coda import CoDAPanda


class GoalAugmentationFunction(HERAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.delta = 0.05

        self.goal_length = self.env.goal_idx.shape[-1]

    def _sample_goals(self, next_obs, n):
        raise NotImplementedError()

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
        new_goal = self._sample_goals(next_obs, n)
        obs[:, self.env.goal_idx] = new_goal
        next_obs[:, self.env.goal_idx] = new_goal

        achieved_goal = next_obs[:, self.env.achieved_idx]
        at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, new_goal, infos)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

class TranslateGoal(GoalAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_goals(self, next_obs, n):
        return self.env.task._sample_n_goals(n)

class TranslateGoalProximal(GoalAugmentationFunction):

    def __init__(self, env, p=0.5,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = p

    def _sample_goals(self, next_obs, n):
        if np.random.random() < self.p:
            r = np.random.uniform(0, self.delta)
            theta = np.random.uniform(-np.pi, np.pi)
            phi = np.random.uniform(-np.pi/2, np.pi/2)
            dx = r*np.sin(phi)*np.cos(theta)
            dy = r*np.sin(phi)*np.sin(theta)
            dz = r*np.cos(phi)
            if self.env.task.goal_range_high[-1] == 0:
                dz = 0
            new_goal = next_obs[:, self.env.goal_idx] + np.array([dx, dy, dz])
        else:
            new_goal = self.env.task._sample_n_goals(n)

        return new_goal

class HER(GoalAugmentationFunction):
    def __init__(self, env, strategy='future', **kwargs):
        super().__init__(env=env, **kwargs)
        self.delta = 0.05
        self.strategy = strategy
        if self.strategy == 'future':
            self.goal_sampler = self._sample_future
        else:
            self.goal_sampler = self._sample_last

    def _sample_future(self, next_obs, n):
        low = np.arange(n)
        indices = np.random.randint(low=low, high=n)
        final_pos = next_obs[indices].copy()
        final_pos = final_pos[:, self.env.achieved_idx]
        return final_pos

    def _sample_last(self, next_obs, n):
        final_pos = next_obs[:, self.env.achieved_idx].copy()
        return final_pos

    def _sample_goals(self, next_obs, n):
        return self.goal_sampler(next_obs, n)

class HERMixed(GoalAugmentationFunction):
    def __init__(self, env, aug_function, strategy='future', prob=0.5, **kwargs):
        super().__init__(env, **kwargs)
        self.HER = HER(env, strategy, **kwargs)
        self.aug_function = aug_function(env, **kwargs)
        self.prob = prob

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
            return self.HER._augment(obs, next_obs, action, reward, done, infos, p)
        else:
            return self.aug_function._augment(obs, next_obs, action, reward, done, infos, p)

class HERTranslateGoal(HERMixed):
    def __init__(self, env, strategy='future', prob=0.5, **kwargs):
        super().__init__(env=env, aug_function=TranslateGoal, strategy=strategy, prob=prob, **kwargs)

class HERTranslateGoalProximal(HERMixed):
    def __init__(self, env, strategy='future', p=0.5, prob=0.5, **kwargs):
        super().__init__(env=env, aug_function=TranslateGoalProximal, strategy=strategy, p=p, prob=p,  **kwargs)

class HERReflect(HERMixed):
    def __init__(self, env, strategy='future', p=0.5, **kwargs):
        super().__init__(env=env, aug_function=Reflect, strategy=strategy, p=p, **kwargs)

class RobotAugmentationFunction(HERAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.delta = 0.05
        self.goal_length = self.env.goal_idx.shape[-1]

    def _sample_goals(self, next_obs, n):
        raise NotImplementedError()

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
        new_goal = self._sample_goals(next_obs, n)
        obs[:, self.env.goal_idx] = new_goal
        next_obs[:, self.env.goal_idx] = new_goal

        achieved_goal = next_obs[:, self.env.achieved_idx]
        at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, new_goal, infos)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

class Reflect(RobotAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.env = env
        self.delta = 0.05

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

    def _reflect_robot_obs(self, obs):
        # y reflection
        obs[:, 1] *= -1
        obs[:, 4] *= -1

    def _reflect_object_obs(self, obs):
        # y reflection
        obs[:, 7] *= -1
        obs[:, 9] *= -1
        obs[:, 13] *= -1

    def _reflect_goal_obs(self, obs):
        # y reflection
        obs[:, self.env.goal_idx[1]] *= -1

    def _reflect_obs(self, obs):
        self._reflect_robot_obs(obs)
        self._reflect_object_obs(obs)
        self._reflect_goal_obs(obs)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        self._reflect_obs(obs)
        self._reflect_obs(next_obs)

        action[:, 1] *= -1

        return obs, next_obs, action, reward, done, infos



# Reach, Push, Slide, PickAndPlace only
PANDA_AUG_FUNCTIONS = {
    'her': HER,
    'her_translate_goal': HERTranslateGoal,
    'her_translate_goal_proximal': HERTranslateGoalProximal,
    'translate_goal': TranslateGoal,
    'translate_goal_proximal': TranslateGoalProximal,
    'coda': CoDAPanda,
}

PANDA_PUSH_AUG_FUNCTIONS = {
    'reflect': Reflect,
    'her_reflect': HERReflect,

}
PANDA_PUSH_AUG_FUNCTIONS.update(PANDA_AUG_FUNCTIONS)

PANDA_SLIDE_AUG_FUNCTIONS = {
    'reflect': Reflect,
    'her_reflect': HERReflect,
}
PANDA_PUSH_AUG_FUNCTIONS.update(PANDA_AUG_FUNCTIONS)
PANDA_SLIDE_AUG_FUNCTIONS.update(PANDA_AUG_FUNCTIONS)