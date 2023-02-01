import copy
from typing import Any, List, Dict

import numpy as np

from augment.rl.augmentation_functions.panda.common import GoalAugmentationFunction, PANDA_AUG_FUNCTIONS, HER, \
    ObjectAugmentationFunction, CoDA, HERMixed


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

    def _sample_goals(self, next_obs, **kwargs):
        ep_length = next_obs.shape[0]
        achieved_goal = next_obs[:, self.env.achieved_idx]
        if np.random.random() < self.p:
            a = np.arccos(achieved_goal[:, 0])
            theta = np.random.uniform(-0.927, +0.927, size=(ep_length,))  # arccos(0.6) ~= +/-0.927
            q_rotation = np.array([
                np.cos(theta / 2),
                a * np.sin(theta / 2),
                a * np.sin(theta / 2),
                a * np.sin(theta / 2),
            ]).T
            new_goal = self.quaternion_multiply(achieved_goal.T, q_rotation.T).T
        else:
            new_goal = self.env.task._sample_n_goals(ep_length)
            at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)

            # resample if success (rejection sampling)
            while np.any(at_goal):
                new_goal[at_goal] = self.env.task._sample_n_goals(ep_length)[at_goal]
                at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        return new_goal


class TranslateGoalProximal0(TranslateGoalProximal):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, p=0, **kwargs)


class TranslateGoalDynamic(GoalAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)

    def quaternion_multiply(self, q0, q1):
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def _sample_goals(self, next_obs, p=None, **kwargs):
        n = next_obs.shape[0]

        if np.random.random() < p:
            new_goal = next_obs[:, self.env.achieved_idx]
        else:
            new_goal = self.env.task._sample_n_goals(n)
        return new_goal


class HERFlip(HER):
    def __init__(self, env, strategy='future', **kwargs):
        super().__init__(env=env, **kwargs)

    def quaternion_multiply(self, q0, q1):
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def _sample_future(self, next_obs):
        ep_length = next_obs.shape[0]
        low = np.arange(ep_length)
        indices = np.random.randint(low=low, high=ep_length)
        final_pos = next_obs[indices]
        final_pos = final_pos[:, self.env.achieved_idx]

        a = np.arccos(final_pos[:, 0])
        theta = np.random.uniform(-0.927, +0.927, size=(ep_length,))  # arccos(0.6) ~= +/-0.927
        q_rotation = np.array([
            np.cos(theta / 2),
            a * np.sin(theta / 2),
            a * np.sin(theta / 2),
            a * np.sin(theta / 2),
        ]).T
        new_goal = self.quaternion_multiply(final_pos.T, q_rotation.T).T
        return new_goal


# class HERTranslateGoalProximal(HERMixed):
#     def __init__(self, env, strategy='future', q=0.5, p=0.5, **kwargs):
#         super().__init__(env=env, aug_function=TranslateGoalProximal, strategy=strategy, q=q, p=p, **kwargs)
#
# class HERTranslateGoalProximal0(GoalAugmentationFunction):
#     def __init__(self, env, strategy='future', q=0.5, **kwargs):
#         super().__init__(env, **kwargs)
#         self.HER = HER(env, strategy, **kwargs)
#         self.aug_function = TranslateGoalProximal(env, p=0, **kwargs)
#         self.q = q
#
#     def _augment(self,
#                  obs: np.ndarray,
#                  next_obs: np.ndarray,
#                  action: np.ndarray,
#                  reward: np.ndarray,
#                  done: np.ndarray,
#                  infos: List[Dict[str, Any]],
#                  p=None,
#                  **kwargs,
#                  ):
#
#         if np.random.random() < self.q:
#             return self.HER._augment(obs, next_obs, action, reward, done, infos, p)
#         else:
#             return self.aug_function._augment(obs, next_obs, action, reward, done, infos)
#
# class HERTranslateGoalProximal09(GoalAugmentationFunction):
#     def __init__(self, env, strategy='future', q=0.5, **kwargs):
#         super().__init__(env, **kwargs)
#         self.HER = HER(env, strategy, **kwargs)
#         self.aug_function = TranslateGoalProximal(env, p=0.09, **kwargs)
#         self.q = q
#
#     def _augment(self,
#                  obs: np.ndarray,
#                  next_obs: np.ndarray,
#                  action: np.ndarray,
#                  reward: np.ndarray,
#                  done: np.ndarray,
#                  infos: List[Dict[str, Any]],
#                  p=None,
#                  **kwargs
#                  ):
#
#         if np.random.random() < self.q:
#             return self.HER._augment(obs, next_obs, action, reward, done, infos, p)
#         else:
#             return self.aug_function._augment(obs, next_obs, action, reward, done, infos)

class TranslateObjectFlip(ObjectAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.obj_pos_mask = np.zeros_like(self.env.obj_idx, dtype=bool)
        obj_pos_idx = np.argmax(self.env.obj_idx)
        self.obj_pos_mask[obj_pos_idx:obj_pos_idx+3] = True

    def _sample_object(self, n):
        new_obj, new_rot = self.env.task._sample_n_objects(n) # new_rot = np.zeros(3)
        return new_obj

class TranslateObjectProximalFlip(TranslateObjectFlip):
    def __init__(self, env, p=0.5, **kwargs):
        super().__init__(env, **kwargs)
        self.TranslateObjectProximal0Flip = TranslateObjectFlip(env, **kwargs)
        self.TranslateGoalProximal1 = TranslateGoalProximal(env, p=1, **kwargs)
        self.q = p

    def _passes_checks(self, obs, next_obs, reward, **kwargs):
        diff = np.abs((obs[:, :3] - obs[:, self.obj_pos_mask]))
        next_diff = np.abs((next_obs[:, :3] - next_obs[:,self.obj_pos_mask]))
        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)
        observed_is_independent = is_independent & next_is_independent

        # No need to check if ee is too close to goal, since goal is an orientation.
        return np.all(observed_is_independent)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 **kwargs,
                 ):

        if np.random.random() < self.q:
            return self.TranslateGoalProximal1._augment(obs, next_obs, action, reward, done, infos, **kwargs,)
        else:
            return self.TranslateObjectProximal0Flip._augment(obs, next_obs, action, reward, done, infos, p, **kwargs)


class TranslateObjectProximal0Flip(TranslateObjectProximalFlip):

    def __init__(self, env, p=0.5, **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = p
        self.aug_threshold = np.array([0.06, 0.1, 0.06])  # largest distance from center to block edge = 0.02

    def _sample_object(self, n):
        new_obj, new_rot = self.env.task._sample_n_objects(n) # new_rot = np.zeros(3)
        return new_obj

    def _passes_checks(self, obs, next_obs, reward, **kwargs):

        assert reward.shape[0] == 1
        if reward[0] == 0:
            return False
        diff = np.abs((obs[:, :3] - obs[:, self.obj_pos_mask]))
        next_diff = np.abs((next_obs[:, :3] - next_obs[:,self.obj_pos_mask]))
        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)
        observed_is_independent = is_independent & next_is_independent

        # No need to check if ee is too close to goal, since goal is an orientation.
        return np.all(observed_is_independent)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 **kwargs
                 ):
        ep_length = obs.shape[0]

        if reward[0] == 0:
            return None, None, None, None, None, None

        mask = np.ones(ep_length, dtype=bool)
        independent_obj = np.empty(shape=(ep_length, self.obj_size))
        independent_next_obj = np.empty(shape=(ep_length, self.obj_size))

        sample_new_obj = True
        while sample_new_obj:
            new_obj, new_next_obj = self._sample_objects(obs, next_obs)
            independent_obj[mask] = new_obj[mask]
            independent_next_obj[mask] = new_next_obj[mask]

            is_independent, next_is_independent = self._check_independence(obs, next_obs, new_obj, new_next_obj, mask)
            mask[mask] = ~(is_independent & next_is_independent)
            sample_new_obj = np.any(mask)

        return self._make_transition(obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj)

class CoDAProximalFlip0(CoDA):

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def _passes_checks(self, obs, next_obs, reward, replay_buffer=None, **kwargs):
        if replay_buffer.size() < 1000:
            return False

        diff = np.abs((obs[:, :3] - obs[:, self.obj_pos_mask]))
        next_diff = np.abs((next_obs[:, :3] - next_obs[:,self.obj_pos_mask]))
        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)
        observed_is_independent = is_independent & next_is_independent
        return np.all(observed_is_independent)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 replay_buffer=None,
                 **kwargs
                 ):

        ep_length = obs.shape[0]
        mask = np.ones(ep_length, dtype=bool)
        independent_obj = np.empty(shape=(ep_length, self.env.obj_idx.shape[0]))
        independent_next_obj = np.empty(shape=(ep_length, self.env.obj_idx.shape[0]))
        desired_goal = next_obs[:, self.env.goal_idx]

        sample_new_obj = True
        while sample_new_obj:
            new_obs, _, new_next_obs, _, _, _ = replay_buffer.sample_array(batch_size=ep_length)
            independent_obj[mask] = new_obs[mask]
            independent_next_obj[mask] = new_next_obs[mask]

            diff = np.abs((obs[:, :3] - new_obs[:, self.obj_pos_mask])[mask])
            next_diff = np.abs((next_obs[:, :3] - new_next_obs[:, self.obj_pos_mask])[mask])

            is_independent = np.any(diff > self.aug_threshold, axis=-1)
            next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)

            at_goal = self.env.task.is_success(new_next_obs[mask, self.env.achieved_idx], desired_goal[mask]).astype(bool)

            mask[mask] = ~(is_independent & next_is_independent & ~at_goal)
            sample_new_obj = np.any(mask)

        obs[:] = independent_obj
        next_obs[:] = independent_next_obj

        achieved_goal = next_obs[:, self.env.achieved_idx]
        desired_goal = next_obs[:, self.env.goal_idx]
        at_goal = self.env.task.is_success(achieved_goal, desired_goal).astype(bool)
        reward[:] = self.env.task.compute_reward(achieved_goal, desired_goal, infos)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

class CoDAFlip(CoDA):

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.aug_threshold = np.array([0.06, 0.1, 0.06])  # largest distance from center to block edge = 0.02

class CoDAProximalFlip(CoDAFlip):
    def __init__(self, env, p=0.5, **kwargs):
        super().__init__(env, **kwargs)
        self.aug_threshold = np.array([0.06, 0.1, 0.06])  # largest distance from center to block edge = 0.02
        self.CoDA0 = CoDAProximalFlip0(env, **kwargs)
        self.TranslateGoalProximalFlip1 = TranslateGoalProximal(env, p=1, **kwargs)
        self.q = p

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 **kwargs,
                 ):

        if np.random.random() < self.q:
            return self.TranslateGoalProximalFlip1._augment(obs, next_obs, action, reward, done, infos, **kwargs,)
        else:
            return self.CoDA0._augment(obs, next_obs, action, reward, done, infos, p, **kwargs)



class HERTranslateObject(HERMixed):
    def __init__(self, env, strategy='future', q=0.5, **kwargs):
        super().__init__(env=env, aug_function=TranslateObjectFlip, strategy=strategy, q=q, **kwargs)

PANDA_FLIP_AUG_FUNCTIONS = copy.deepcopy(PANDA_AUG_FUNCTIONS)
PANDA_FLIP_AUG_FUNCTIONS.update(
    {
        'translate_goal_proximal': TranslateGoalProximal,
        'translate_goal_proximal_0': TranslateGoalProximal0,
        'translate_goal_dynamic': TranslateGoalDynamic,
        # 'her_translate_goal_proximal_0': HERTranslateGoalProximal0,
        'translate_object': TranslateObjectFlip,
        'translate_object_proximal': TranslateObjectProximalFlip, # not supported, can't generate additional reward signal by translation.
        'translate_object_proximal_0': TranslateObjectProximal0Flip,
        'her_translate_object': HERTranslateObject,
        'coda': CoDAFlip,
        'coda_proximal_0': CoDAProximalFlip0,
        'coda_proximal': CoDAProximalFlip,
        'her': HERFlip,

    })
