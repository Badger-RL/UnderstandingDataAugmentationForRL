from copy import deepcopy
from typing import Dict, List, Any

import numpy as np
# from stable_baselines3.common.vec_env import VecNormalize

class AugmentationFunction:

    def __init__(self, env=None, **kwargs):
        self.env = env
        self.is_her = True
        self.aug_n = None

    def _deepcopy_transition(
            self,
            augmentation_n: int,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ):
        aug_obs = np.tile(obs, (augmentation_n,1,1))
        aug_next_obs = np.tile(next_obs, (augmentation_n,1,1))
        aug_action = np.tile(action, (augmentation_n,1,1))
        aug_reward = np.tile(reward, (augmentation_n,1,1))
        aug_done = np.tile(done, (augmentation_n,1,1)).astype(np.bool)
        aug_infos = np.tile([infos], (augmentation_n,1,1))

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

    def _check_observed_constraints(self, obs, next_obs, reward, **kwargs):
        return True

    def augment(self,
                 aug_n: int,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):

        if not self._check_observed_constraints(obs, next_obs, reward):
            return None, None, None, None, None, None

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = \
            self._deepcopy_transition(aug_n, obs, next_obs, action, reward, done, infos)

        for i in range(aug_n):
            self._augment(aug_obs[i], aug_next_obs[i], aug_action[i], aug_reward[i][0], aug_done[i][0], aug_infos[i], **kwargs)

        aug_obs = aug_obs.reshape((-1, 1, aug_obs.shape[-1]))
        aug_next_obs = aug_next_obs.reshape((-1, 1, aug_next_obs.shape[-1]))
        aug_action = aug_action.reshape((-1, 1, aug_action.shape[-1]))
        aug_reward = aug_reward.reshape(-1, 1)
        aug_done = aug_done.reshape(-1, 1)
        aug_infos = aug_infos.reshape((-1,1))

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):
        raise NotImplementedError("Augmentation function not implemented.")


class GoalAugmentationFunction(AugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        # self.goal_length = self.env.goal_idx.shape[-1]
        self.desired_goal_mask = None
        self.achieved_goal_mask = None
        # self.robot_mask = None
        # self.object_mask = None

    def _sample_goals(self, next_obs, **kwargs):
        raise NotImplementedError()

    def _sample_goal_noise(self, n, **kwargs):
        raise NotImplementedError()

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

    def _is_at_goal(self, achieved_goal, desired_goal, **kwargs):
        raise NotImplementedError()

    def _compute_reward(self, achieved_goal, desired_goal, **kwargs):
        raise NotImplementedError()

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

        new_goal = self._sample_goals(next_obs, p=p, **kwargs)
        obs[:, self.desired_goal_mask] = new_goal
        next_obs[:, self.desired_goal_mask] = new_goal

        achieved_goal = next_obs[:, self.achieved_goal_mask]
        at_goal = self._is_at_goal(achieved_goal, new_goal)
        reward[:] = self._compute_reward(achieved_goal, new_goal)
        self._set_done_and_info(done, infos, at_goal)


class ObjectAugmentationFunction(AugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        # self.goal_length = self.env.goal_idx.shape[-1]
        self.desired_goal_mask = None
        self.achieved_goal_mask = None
        self.robot_mask = None
        self.object_mask = None

    def _sample_object(self, n, **kwargs):
        raise NotImplementedError()

    def _sample_objects(self, obs, next_obs, **kwargs):
        raise NotImplementedError()

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

    def _is_at_goal(self, achieved_goal, desired_goal, **kwargs):
        raise NotImplementedError()

    def _compute_reward(self, achieved_goal, desired_goal, **kwargs):
        raise NotImplementedError()

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

        new_obj, new_next_obj = self._sample_objects(obs, next_obs, p=p, **kwargs)
        obs[:, self.object_mask] = new_obj
        next_obs[:, self.object_mask] = new_next_obj

        achieved_goal = next_obs[:, self.achieved_goal_mask]
        desired_goal = next_obs[:, self.desired_goal_mask]

        at_goal = self._is_at_goal(achieved_goal, desired_goal)
        reward[:] = self._compute_reward(achieved_goal, desired_goal)
        self._set_done_and_info(done, infos, at_goal)