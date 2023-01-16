import copy
from typing import Any, List, Dict

import numpy as np

from augment.rl.augmentation_functions.panda.common import GoalAugmentationFunction, PANDA_AUG_FUNCTIONS, HER, HERMixed, \
    ObjectAugmentationFunction


class TranslateGoalProximal(GoalAugmentationFunction):

    def __init__(self, env, p=0.5,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = p

    def quaternion_multiply(self, q0, q1):
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def _sample_goals(self, next_obs):
        n = next_obs.shape[0]
        achieved_goal = next_obs[:, self.env.achieved_idx]

        if np.random.random() < self.p:
            a = np.arccos(achieved_goal[:, 0])
            theta = np.random.uniform(-0.927, +0.927, size=(n,)) # arccos(0.6) ~= +/-0.927
            q_rotation = np.array([
                np.cos(theta / 2),
                a * np.sin(theta / 2),
                a * np.sin(theta / 2),
                a * np.sin(theta / 2),
            ]).T
            new_goal = self.quaternion_multiply(achieved_goal.T, q_rotation.T).T
        else:
            # new goal results in no reward signal
            new_goal = self.env.task._sample_n_goals(n)
            at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)

            # resample if success (rejection sampling)
            while np.any(at_goal):
                new_goal[at_goal] = self.env.task._sample_n_goals(n)[at_goal]
                at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        return new_goal


class TranslateGoalProximal0(TranslateGoalProximal):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, p=0, **kwargs)


class HERTranslateGoalProximal(HERMixed):
    def __init__(self, env, strategy='future', q=0.5, p=0.5, **kwargs):
        super().__init__(env=env, aug_function=TranslateGoalProximal, strategy=strategy, q=q, p=p, **kwargs)

class HERTranslateGoalProximal0(GoalAugmentationFunction):
    def __init__(self, env, strategy='future', q=0.5, **kwargs):
        super().__init__(env, **kwargs)
        self.HER = HER(env, strategy, **kwargs)
        self.aug_function = TranslateGoalProximal(env, p=0, **kwargs)
        self.q = q

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        if np.random.random() < self.q:
            return self.HER._augment(obs, next_obs, action, reward, done, infos, p)
        else:
            return self.aug_function._augment(obs, next_obs, action, reward, done, infos)

class HERTranslateGoalProximal09(GoalAugmentationFunction):
    def __init__(self, env, strategy='future', q=0.5, **kwargs):
        super().__init__(env, **kwargs)
        self.HER = HER(env, strategy, **kwargs)
        self.aug_function = TranslateGoalProximal(env, p=0.09, **kwargs)
        self.q = q

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        if np.random.random() < self.q:
            return self.HER._augment(obs, next_obs, action, reward, done, infos, p)
        else:
            return self.aug_function._augment(obs, next_obs, action, reward, done, infos)


class TranslateObject(ObjectAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.obj_pos_mask = np.zeros_like(self.env.obj_idx, dtype=bool)
        obj_pos_idx = np.argmax(self.env.obj_idx)
        self.obj_pos_mask[obj_pos_idx:obj_pos_idx+3] = True

        self.obj_vel_mask = np.zeros_like(self.env.obj_idx, dtype=bool)
        self.obj_vel_mask[obj_pos_idx+3:-3] = True

    def _sample_object(self, next_obs):
        ep_length = next_obs.shape[0]
        new_obj, new_rot = self.env.task._sample_n_objects(ep_length) # new_rot = np.zeros(3)
        return new_obj

class HERTranslateObject(HERMixed):
    def __init__(self, env, strategy='future', q=0.5, **kwargs):
        super().__init__(env=env, aug_function=TranslateObject, strategy=strategy, q=q, **kwargs)

PANDA_FLIP_AUG_FUNCTIONS = copy.deepcopy(PANDA_AUG_FUNCTIONS)
PANDA_FLIP_AUG_FUNCTIONS.update(
    {
        'translate_goal_proximal': TranslateGoalProximal,
        'translate_goal_proximal_0': TranslateGoalProximal0,
        'her_translate_goal_proximal_0': TranslateGoalProximal,
        'translate_object': TranslateObject,
        'her_translate_object': HERTranslateObject,
    })
