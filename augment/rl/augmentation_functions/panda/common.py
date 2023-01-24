from typing import Dict, List, Any
import numpy as np
import torch

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

#######################################################################################################################
#######################################################################################################################

class GoalAugmentationFunction(AugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.delta = 0.05

        self.goal_length = self.env.goal_idx.shape[-1]

    def _sample_goals(self, next_obs, **kwargs):
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
                 **kwargs,
                 ):

        new_goal = self._sample_goals(next_obs, p=p, **kwargs)
        obs[:, self.env.goal_idx] = new_goal
        next_obs[:, self.env.goal_idx] = new_goal

        achieved_goal = next_obs[:, self.env.achieved_idx]
        at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        reward[:] = self.env.task.compute_reward(achieved_goal, new_goal, infos)
        self._set_done_and_info(done, infos, at_goal)

class TranslateGoal(GoalAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_goals(self, next_obs, **kwargs):
        ep_length = next_obs.shape[0]
        return self.env.task._sample_n_goals(ep_length)

class TranslateGoalProximal(GoalAugmentationFunction):

    def __init__(self, env, p=0.5,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = p

    def _sample_goals(self, next_obs, **kwargs):
        ep_length = next_obs.shape[0]
        if np.random.random() < self.p:
            r = np.random.uniform(0, self.delta, size=ep_length)
            theta = np.random.uniform(-np.pi, np.pi, size=ep_length)
            phi = np.random.uniform(-np.pi/2, np.pi/2, size=ep_length)
            dx = r*np.sin(phi)*np.cos(theta)
            dy = r*np.sin(phi)*np.sin(theta)
            dz = r*np.cos(phi)
            dz[:] = 0
            noise = np.array([dx, dy, dz]).T
            new_goal = next_obs[:, self.env.goal_idx] + noise
        else:
            new_goal = self.env.task._sample_n_goals(ep_length)
            achieved_goal = next_obs[:, self.env.achieved_idx]
            at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)

            # resample if success (rejection sampling)
            while np.any(at_goal):
                new_goal[at_goal] = self.env.task._sample_n_goals(ep_length)[at_goal]
                at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        return new_goal

class TranslateGoalProximal0(TranslateGoalProximal):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, p=0, **kwargs)

#######################################################################################################################
#######################################################################################################################

class HER(GoalAugmentationFunction):
    def __init__(self, env, strategy='future', **kwargs):
        super().__init__(env=env, **kwargs)
        self.delta = 0.05
        self.strategy = strategy
        if self.strategy == 'future':
            self.goal_sampler = self._sample_future
        elif self.strategy == 'random':
            self.goal_sampler = self._sample_random
        else:
            self.goal_sampler = self._sample_last

    def _sample_future(self, next_obs):
        ep_length = next_obs.shape[0]
        low = np.arange(ep_length)
        indices = np.random.randint(low=low, high=ep_length)
        final_pos = next_obs[indices]
        final_pos = final_pos[:, self.env.achieved_idx]

        r = np.random.uniform(0, self.delta, size=ep_length)
        theta = np.random.uniform(-np.pi, np.pi, size=ep_length)
        phi = np.random.uniform(-np.pi / 2, np.pi / 2, size=ep_length)
        dx = r * np.sin(phi) * np.cos(theta)
        dy = r * np.sin(phi) * np.sin(theta)
        dz = r * np.cos(phi)
        dz = np.clip(dz, 0.0, 0.18)
        if self.env.task.goal_range_high[-1] == 0:
            dz[:] = 0
        noise = np.array([dx, dy, dz]).T
        new_goal = final_pos + noise
        return new_goal

    def _sample_last(self, next_obs):
        final_pos = next_obs[:, self.env.achieved_idx].copy()
        return final_pos

    def _sample_random(self, next_obs):
        ep_length = next_obs.shape[0]
        return self.env.task._sample_n_goals(ep_length)

    def _sample_goals(self, next_obs, **kwargs):
        return self.goal_sampler(next_obs)

class HERMixed(GoalAugmentationFunction):
    def __init__(self, env, aug_function, strategy='future', q=0.5, **kwargs):
        super().__init__(env, **kwargs)
        self.HER = HER(env, strategy, **kwargs)
        self.aug_function = aug_function(env, **kwargs)
        self.q = q


    def augment(self,
                 aug_n: int,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = \
            self._deepcopy_transition(aug_n, obs, next_obs, action, reward, done, infos)

        for i in range(aug_n):
            self._augment(aug_obs[i], aug_next_obs[i], aug_action[i], aug_reward[i][0], aug_done[i][0], aug_infos[i], **kwargs)

        aug_obs = aug_obs.reshape((-1, aug_obs.shape[-1]))
        aug_next_obs = aug_next_obs.reshape((-1, aug_next_obs.shape[-1]))
        aug_action = aug_action.reshape((-1, aug_action.shape[-1]))
        aug_reward = aug_reward.reshape(-1)
        aug_done = aug_done.reshape(-1)
        aug_infos = aug_infos.reshape((-1,1))

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

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
            return self.HER._augment(obs, next_obs, action, reward, done, infos, p, **kwargs)
        else:
            if not self._passes_checks(obs, next_obs, reward):
                return None, None, None, None, None, None
            return self.aug_function._augment(obs, next_obs, action, reward, done, infos, **kwargs,)

class HERTranslateGoal(HERMixed):
    def __init__(self, env, strategy='future', q=0.5, **kwargs):
        super().__init__(env=env, aug_function=TranslateGoal, strategy=strategy, q=q, **kwargs)

class HERTranslateObject(HERMixed):
    def __init__(self, env, strategy='future', q=0.5, **kwargs):
        super().__init__(env=env, aug_function=TranslateObject, strategy=strategy, q=q, **kwargs)

class HERCoDA(HERMixed):
    def __init__(self, env, strategy='future', q=0.5, **kwargs):
        super().__init__(env=env, aug_function=CoDA, strategy=strategy, q=q, **kwargs)

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
                 **kwargs,
                 ):

        if np.random.random() < self.q:
            return self.HER._augment(obs, next_obs, action, reward, done, infos, p)
        else:
            return self.aug_function._augment(obs, next_obs, action, reward, done, infos)

#######################################################################################################################
#######################################################################################################################

class ObjectAugmentationFunction(AugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.delta = 0.05
        self.aug_threshold = np.array([0.06, 0.06, 0.06])  # largest distance from center to block edge = 0.02

        self.obj_pos_mask = np.zeros_like(self.env.obj_idx, dtype=bool)
        self.obj_pos_idx = np.argmax(self.env.obj_idx)
        self.obj_pos_mask[self.obj_pos_idx:self.obj_pos_idx + 3] = True

        self.obj_size = 3

        # self.lo = np.array([-0.02, -0.02, 0])
        # self.hi = np.array([0.02, 0.02, 0])
        #
        # self.min = np.array([-0.15, -0.15, 0.2])
        # self.max = np.array([0.15, 0.15, 0.2])

    def _sample_object(self, n, **kwargs):
        raise NotImplementedError()

    def _sample_objects(self, obs, next_obs):
        n = obs.shape[0]
        new_obj = self._sample_object(n)
        # obj_pos_diff = next_obs[:, self.obj_pos_mask] - obs[:, self.obj_pos_mask]
        new_next_obj = new_obj #+ obj_pos_diff
        return new_obj, new_next_obj

    def _check_independence(self, obs, next_obs, new_obj, new_next_obj, mask):
        new_obj = new_obj[mask]
        new_next_obj = new_next_obj[mask]

        diff = np.abs((obs[mask, :3] - new_obj))
        next_diff = np.abs((next_obs[mask, :3] - new_next_obj))

        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)

        # Stop sampling when new_obj is independent.
        return is_independent, next_is_independent

    def _check_at_goal(self, new_next_obj, desired_goal, mask):
        at_goal = self.env.task.is_success(new_next_obj[mask], desired_goal[mask]).astype(bool)
        return at_goal

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

    def _make_transition(self, obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj):
        obs[:, self.obj_pos_mask] = independent_obj
        next_obs[:, self.obj_pos_mask] = independent_next_obj
        obs[:, self.obj_pos_idx:-3] = 0
        next_obs[:, self.obj_pos_idx:-3] = 0

        achieved_goal = next_obs[:, self.env.achieved_idx]
        desired_goal = next_obs[:, self.env.goal_idx]
        at_goal = self.env.task.is_success(achieved_goal, desired_goal).astype(bool)
        reward[:] = self.env.task.compute_reward(achieved_goal, desired_goal, infos)
        self._set_done_and_info(done, infos, at_goal)
        return obs, next_obs, action, reward, done, infos

    def _passes_checks(self, obs, next_obs, reward, **kwargs):
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
                 **kwargs
                 ):

        ep_length = obs.shape[0]
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

        self._make_transition(obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj)


class TranslateObject(ObjectAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_object(self, n, **kwargs):
        new_obj = self.env.task._sample_n_objects(n)
        return new_obj

class TranslateObjectProximal0(ObjectAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = 0

    def _sample_object(self, n, **kwargs):
        new_obj = self.env.task._sample_n_objects(n)
        return new_obj

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
        mask = np.ones(ep_length, dtype=bool)
        independent_obj = np.empty(shape=(ep_length, self.obj_size))
        independent_next_obj = np.empty(shape=(ep_length, self.obj_size))

        sample_new_obj = True
        while sample_new_obj:
            new_obj, new_next_obj = self._sample_objects(obs, next_obs)
            independent_obj[mask] = new_obj[mask]
            independent_next_obj[mask] = new_next_obj[mask]

            is_independent, next_is_independent = self._check_independence(obs, next_obs, new_obj, new_next_obj, mask)

            desired_goal = next_obs[:, self.env.goal_idx]
            at_goal = self._check_at_goal(new_next_obj, desired_goal, mask)
            mask[mask] = ~(is_independent & next_is_independent & ~at_goal)
            sample_new_obj = np.any(mask)

        self._make_transition(obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj)


class TranslateObjectProximal(ObjectAugmentationFunction):

    def __init__(self, env, p=0.5, **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = p

    def _passes_checks(self, obs, next_obs, reward, **kwargs):
        diff = np.abs((obs[:, :3] - obs[:, self.obj_pos_mask]))
        next_diff = np.abs((next_obs[:, :3] - next_obs[:,self.obj_pos_mask]))
        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)
        observed_is_independent = is_independent & next_is_independent

        if np.all(observed_is_independent):
            desired_goal = next_obs[:, self.env.goal_idx]
            ee_dist = (obs[:, :3] - desired_goal)
            ee_next_dist = (next_obs[:, :3] - desired_goal)

            # ee is too close to goal to generate reward signal
            if np.all(ee_dist < self.aug_threshold) or np.all(ee_next_dist < self.aug_threshold):
                return False
            else:
                return True
        return False

    def _sample_object(self, n):
        new_obj = self.env.task._sample_n_objects(n)
        return new_obj

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
        mask = np.ones(ep_length, dtype=bool)
        independent_obj = np.empty(shape=(ep_length, self.obj_size))
        independent_next_obj = np.empty(shape=(ep_length, self.obj_size))

        if np.random.random() < self.p:
            desired_goal = next_obs[:, self.env.goal_idx]
            independent_obj = desired_goal
            obj_pos_diff = next_obs[:, self.obj_pos_mask] - obs[:, self.obj_pos_mask]
            independent_next_obj = independent_obj + obj_pos_diff

        else:
            sample_new_obj = True
            while sample_new_obj:
                new_obj, new_next_obj = self._sample_objects(obs, next_obs)
                independent_obj[mask] = new_obj
                independent_next_obj[mask] = new_next_obj

                is_independent, next_is_independent = self._check_independence(obs, next_obs, new_obj, new_next_obj, mask)
                mask[mask] = ~(is_independent & next_is_independent)
                sample_new_obj = np.any(mask)

        self._make_transition(obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj)

#######################################################################################################################
#######################################################################################################################

class CoDA(ObjectAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def _sample_goals(self, next_obs, **kwargs):
        ep_length = next_obs.shape[0]
        return self.env.task._sample_n_goals(ep_length)

    def _passes_checks(self, obs, next_obs, reward, replay_buffer=None, **kwargs):
        if replay_buffer.size() < 1000:
            return False

        diff = np.abs((obs[:, :3] - obs[:, self.obj_pos_mask]))
        next_diff = np.abs((next_obs[:, :3] - next_obs[:,self.obj_pos_mask]))
        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)
        observed_is_independent = is_independent & next_is_independent
        return np.all(observed_is_independent)


    def augment(self,
                 aug_n: int,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 replay_buffer=None,
                 **kwargs,):

        if not self._passes_checks(obs, next_obs, reward, replay_buffer=replay_buffer):
            return None, None, None, None, None, None

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = \
            self._deepcopy_transition(aug_n, obs, next_obs, action, reward, done, infos)

        for i in range(aug_n):
            self._augment(aug_obs[i], aug_next_obs[i], aug_action[i], aug_reward[i][0], aug_done[i][0], aug_infos[i], replay_buffer=replay_buffer, **kwargs)

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
                 p=None,
                 replay_buffer=None,
                 **kwargs
                 ):

        ep_length = obs.shape[0]
        mask = np.ones(ep_length, dtype=bool)
        independent_obj = np.empty(shape=(ep_length, self.env.obj_idx.shape[0]))
        independent_next_obj = np.empty(shape=(ep_length, self.env.obj_idx.shape[0]))

        # new_goal = np.empty(shape=(ep_length, self.obj_size))

        sample_new_obj = True
        while sample_new_obj:
            new_obs, _, new_next_obs, new_reward, new_done, new_info = replay_buffer.sample_array(batch_size=ep_length)
            independent_obj[mask] = new_obs[mask]
            independent_next_obj[mask] = new_next_obs[mask]
            # reward[mask] = new_reward[mask]
            # done[mask] = new_done
            # infos[mask] = new_done


            diff = np.abs((obs[:, :3] - new_obs[:, self.obj_pos_mask])[mask])
            next_diff = np.abs((next_obs[:, :3] - new_next_obs[:, self.obj_pos_mask])[mask])

            is_independent = np.any(diff > self.aug_threshold, axis=-1)
            next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)

            mask[mask] = ~(is_independent & next_is_independent)
            sample_new_obj = np.any(mask)

        obs[:] = independent_obj
        next_obs[:] = independent_next_obj

        achieved_goal = next_obs[:, self.env.achieved_idx]
        desired_goal = next_obs[:, self.env.goal_idx]
        at_goal = self.env.task.is_success(achieved_goal, desired_goal).astype(bool)
        reward[:] = self.env.task.compute_reward(achieved_goal, desired_goal, infos)
        self._set_done_and_info(done, infos, at_goal)
        return obs, next_obs, action, reward, done, infos


class CoDAProximal0(CoDA):

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

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
        independent_obj = np.empty(shape=(ep_length, self.obj_size))
        independent_next_obj = np.empty(shape=(ep_length, self.obj_size))
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


class CoDAProximal(ObjectAugmentationFunction):
    def __init__(self, env, p=0.5, **kwargs):
        super().__init__(env, **kwargs)
        self.CoDA0 = CoDAProximal0(env, **kwargs)
        self.TranslateGoalProximal1 = TranslateGoalProximal(env, p=1, **kwargs)
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
            return self.TranslateGoalProximal1._augment(obs, next_obs, action, reward, done, infos, **kwargs,)
        else:
            return self.CoDA0._augment(obs, next_obs, action, reward, done, infos, p, **kwargs)


class TranslateObjectAndGoal(ObjectAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.delta = 0.05

        self.obj_pos_mask = np.zeros_like(self.env.obj_idx, dtype=bool)
        obj_pos_idx = np.argmax(self.env.obj_idx)
        self.obj_pos_mask[obj_pos_idx:obj_pos_idx + 3] = True

        self.obj_size = 3

    def _sample_goals(self, next_obs, **kwargs):
        ep_length = next_obs.shape[0]
        return self.env.task._sample_n_goals(ep_length)

    def _sample_object(self, n):
        new_obj = self.env.task._sample_n_objects(n)
        return new_obj

    def _sample_objects(self, obs, next_obs):
        n = obs.shape[0]
        new_next_obj = self._sample_object(n)
        obj_pos_diff = next_obs[:, self.obj_pos_mask] - obs[:, self.obj_pos_mask]
        new_obj = new_next_obj - obj_pos_diff # new_obj + obj_pos_diff = new_next_obj
        return new_obj, new_next_obj

    def _check_independence(self, obs, next_obs, new_obj, new_next_obj, mask):
        new_obj = new_obj[mask]
        new_next_obj = new_next_obj[mask]

        diff = np.abs((obs[mask, :3] - new_obj))
        next_diff = np.abs((next_obs[mask, :3] - new_next_obj))

        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)

        # Stop sampling when new_obj is independent.
        return is_independent, next_is_independent

    def _check_at_goal(self, new_next_obj, desired_goal, mask):
        at_goal = self.env.task.is_success(new_next_obj[mask], desired_goal[mask]).astype(bool)
        return at_goal

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True, 'is_success': False}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False, 'is_success': False}]

    def _make_transition(self, obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj):
        obs[:, self.obj_pos_mask] = independent_obj
        next_obs[:, self.obj_pos_mask] = independent_next_obj

        new_goal = self._sample_goals(next_obs)
        obs[:, self.env.goal_idx] = new_goal
        next_obs[:, self.env.goal_idx] = new_goal

        achieved_goal = next_obs[:, self.env.achieved_idx]
        at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        reward[:] = self.env.task.compute_reward(achieved_goal, new_goal, infos)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

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

        self._make_transition(obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj)

########################################################################################################################
########################################################################################################################

class TranslateObjectJitter(TranslateObject):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

        self.lo = np.array([-0.02, -0.02, 0])
        self.hi = np.array([0.02, 0.02, 0])

        self.min = np.array([-0.15, -0.15, 0])
        self.max = np.array([0.15, 0.15, 0.18])

    def _sample_objects(self, obs, next_obs):
        n = obs.shape[0]
        noise = np.random.uniform(self.lo, self.hi, size=(n, self.obj_size))
        new_next_obj = next_obs[:, self.obj_pos_mask] + noise
        new_next_obj = np.clip(new_next_obj, self.min, self.max)
        obj_pos_diff = next_obs[:, self.obj_pos_mask] - obs[:, self.obj_pos_mask]
        new_obj = new_next_obj - obj_pos_diff # new_obj + obj_pos_diff = new_next_obj
        return new_obj, new_next_obj

    def _check_independence(self, obs, next_obs, new_obj, new_next_obj, mask):
        new_obj = new_obj[mask]
        new_next_obj = new_next_obj[mask]

        diff = np.abs((obs[mask, :3] - new_obj))
        next_diff = np.abs((next_obs[mask, :3] - new_next_obj))

        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)

        # Stop sampling when new_obj is independent.
        return is_independent, next_is_independent

    def _check_at_goal(self, new_next_obj, desired_goal, mask):
        at_goal = self.env.task.is_success(new_next_obj[mask], desired_goal[mask]).astype(bool)
        return at_goal

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

    def _make_transition(self, obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj):
        obs[:, self.obj_pos_mask] = independent_obj
        next_obs[:, self.obj_pos_mask] = independent_next_obj

        achieved_goal = next_obs[:, self.env.achieved_idx]
        desired_goal = next_obs[:, self.env.goal_idx]
        at_goal = self.env.task.is_success(achieved_goal, desired_goal).astype(bool)
        reward[:] = self.env.task.compute_reward(achieved_goal, desired_goal, infos)
        self._set_done_and_info(done, infos, at_goal)
        return obs, next_obs, action, reward, done, infos

    def _passes_checks(self, obs, next_obs, reward, **kwargs):
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
                 **kwargs
                 ):

        ep_length = obs.shape[0]
        mask = np.ones(ep_length, dtype=bool)
        independent_obj = np.empty(shape=(ep_length, self.obj_size))
        independent_next_obj = np.empty(shape=(ep_length, self.obj_size))
        successful_aug = np.ones(ep_length, dtype=bool)

        sample_new_obj = True
        its = 0
        while sample_new_obj:
            its += 1
            new_obj, new_next_obj = self._sample_objects(obs, next_obs)

            independent_obj[mask] = new_obj[mask]
            independent_next_obj[mask] = new_next_obj[mask]

            is_independent, next_is_independent = self._check_independence(obs, next_obs, new_obj, new_next_obj, mask)
            mask[mask] = ~(is_independent & next_is_independent)
            sample_new_obj = np.any(mask)

            if its > 5:
                successful_aug[mask] = 0
                break

        self._make_transition(obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj)
        return successful_aug


    def augment(self,
                 aug_n: int,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):

        if not self._passes_checks(obs, next_obs, reward):
            return None, None, None, None, None, None

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = \
            self._deepcopy_transition(aug_n, obs, next_obs, action, reward, done, infos)

        success_mask = np.ones(aug_n, dtype=bool)

        for i in range(aug_n):
            successful_aug = self._augment(aug_obs[i], aug_next_obs[i], aug_action[i], aug_reward[i][0], aug_done[i][0], aug_infos[i], **kwargs)
            success_mask[i] = successful_aug

        if np.all(~success_mask):
            return None, None, None, None, None, None

        aug_obs = aug_obs.reshape((-1, aug_obs.shape[-1]))
        aug_next_obs = aug_next_obs.reshape((-1, aug_next_obs.shape[-1]))
        aug_action = aug_action.reshape((-1, aug_action.shape[-1]))
        aug_reward = aug_reward.reshape(-1)
        aug_done = aug_done.reshape(-1)
        aug_infos = aug_infos.reshape((-1,1))

        aug_obs = aug_obs[success_mask]
        aug_next_obs = aug_next_obs[success_mask]
        aug_action = aug_action[success_mask]
        aug_reward = aug_reward[success_mask]
        aug_done = aug_done[success_mask]
        aug_infos = aug_infos[success_mask]

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos


########################################################################################################################
########################################################################################################################

class SafeObjectAugmentationFunction(ObjectAugmentationFunction):
    def __init__(self, env, max_theta=45, max_norm=0.1, **kwargs):
        super().__init__(env=env, **kwargs)
        self.delta = 0.05
        self.aug_threshold = np.array([0.06, 0.06, 0.06])  # largest distance from center to block edge = 0.02

        self.obj_pos_mask = np.zeros_like(self.env.obj_idx, dtype=bool)
        obj_pos_idx = np.argmax(self.env.obj_idx)
        self.obj_pos_mask[obj_pos_idx:obj_pos_idx + 3] = True

        self.obj_size = 3
        self.op_threshold = np.cos(max_theta*np.pi/180)
        self.max_norm = max_norm

        # self.lo = np.array([-0.02, -0.02, 0])
        # self.hi = np.array([0.02, 0.02, 0])
        #
        # self.min = np.array([-0.15, -0.15, 0.2])
        # self.max = np.array([0.15, 0.15, 0.2])

    def _sample_object(self, n, **kwargs):
        raise NotImplementedError()

    def _sample_objects(self, obs, next_obs):
        n = obs.shape[0]
        new_obj = self._sample_object(n)
        # obj_pos_diff = next_obs[:, self.obj_pos_mask] - obs[:, self.obj_pos_mask]
        new_next_obj = new_obj #+ obj_pos_diff
        return new_obj, new_next_obj

    def _check_independence(self, obs, next_obs, new_obj, new_next_obj, mask):
        new_obj = new_obj[mask]
        new_next_obj = new_next_obj[mask]

        diff = np.abs((obs[mask, :3] - new_obj))
        next_diff = np.abs((next_obs[mask, :3] - new_next_obj))

        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)

        # Stop sampling when new_obj is independent.
        return is_independent, next_is_independent

    def _check_at_goal(self, new_next_obj, desired_goal, mask):
        at_goal = self.env.task.is_success(new_next_obj[mask], desired_goal[mask]).astype(bool)
        return at_goal

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

    def _make_transition(self, obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj):
        obs[:, self.obj_pos_mask] = independent_obj
        next_obs[:, self.obj_pos_mask] = independent_next_obj
        obs[:, self.obj_pos_idx:-3] = 0
        next_obs[:, self.obj_pos_idx:-3] = 0

        achieved_goal = next_obs[:, self.env.achieved_idx]
        desired_goal = next_obs[:, self.env.goal_idx]
        at_goal = self.env.task.is_success(achieved_goal, desired_goal).astype(bool)
        reward[:] = self.env.task.compute_reward(achieved_goal, desired_goal, infos)
        self._set_done_and_info(done, infos, at_goal)
        return obs, next_obs, action, reward, done, infos

    def _passes_checks(self, obs, next_obs, reward, **kwargs):
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
                 pi=None,
                 **kwargs
                 ):

        ep_length = obs.shape[0]
        mask = np.ones(ep_length, dtype=bool)
        independent_obj = np.empty(shape=(ep_length, self.obj_size))
        independent_next_obj = np.empty(shape=(ep_length, self.obj_size))

        successful_aug = np.ones(ep_length, dtype=bool)
        its = 0

        new_obs = obs.copy()
        sample_new_obj = True
        while sample_new_obj:
            new_obj, new_next_obj = self._sample_objects(obs, next_obs)
            new_obs[:, self.obj_pos_mask] = new_obj
            true_action = pi(torch.from_numpy(new_obs)).detach().numpy()
            action_norm = np.linalg.norm(action, axis=-1)
            true_action_norm = np.linalg.norm(true_action)
            inner_product = true_action.dot(action.T)/(action_norm*true_action_norm)
            while inner_product < self.op_threshold and np.abs(action_norm-true_action_norm) < self.max_norm:
                its += 1
                new_obj, new_next_obj = self._sample_objects(obs, next_obs)
                new_obs[:, self.obj_pos_mask] = new_obj
                true_action = pi(torch.from_numpy(new_obs)).detach().numpy()
                true_action_norm = np.linalg.norm(true_action)
                inner_product = true_action.dot(action.T) / (action_norm * true_action_norm)

                if its > 5:
                    successful_aug[mask] = 0
                    break

            independent_obj[mask] = new_obj[mask]
            independent_next_obj[mask] = new_next_obj[mask]

            is_independent, next_is_independent = self._check_independence(obs, next_obs, new_obj, new_next_obj, mask)
            mask[mask] = ~(is_independent & next_is_independent)
            sample_new_obj = np.any(mask)

        self._make_transition(obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj)
        return successful_aug


    def augment(self,
                 aug_n: int,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 **kwargs,):

        if not self._passes_checks(obs, next_obs, reward):
            return None, None, None, None, None, None

        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos = \
            self._deepcopy_transition(aug_n, obs, next_obs, action, reward, done, infos)

        success_mask = np.ones(aug_n, dtype=bool)

        for i in range(aug_n):
            successful_aug = self._augment(aug_obs[i], aug_next_obs[i], aug_action[i], aug_reward[i][0], aug_done[i][0], aug_infos[i], **kwargs)
            success_mask[i] = successful_aug

        if np.all(~success_mask):
            # print('no')
            return None, None, None, None, None, None

        # print('yes')

        aug_obs = aug_obs.reshape((-1, aug_obs.shape[-1]))
        aug_next_obs = aug_next_obs.reshape((-1, aug_next_obs.shape[-1]))
        aug_action = aug_action.reshape((-1, aug_action.shape[-1]))
        aug_reward = aug_reward.reshape(-1)
        aug_done = aug_done.reshape(-1)
        aug_infos = aug_infos.reshape((-1,1))

        aug_obs = aug_obs[success_mask]
        aug_next_obs = aug_next_obs[success_mask]
        aug_action = aug_action[success_mask]
        aug_reward = aug_reward[success_mask]
        aug_done = aug_done[success_mask]
        aug_infos = aug_infos[success_mask]

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_infos

class TranslateObjectSafe(SafeObjectAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_object(self, n, **kwargs):
        new_obj = self.env.task._sample_n_objects(n)
        return new_obj


# Reach, Push, Slide, PickAndPlace only
PANDA_AUG_FUNCTIONS = {
    'her': HER,
    # INDEPENDENCE CHECK SHOULD INSIDE _augment() FOR HER+TRANSLATE. RIGHT NOW IT HAPPENS IN PARENT augment()
    'her_coda': HERCoDA,
    'her_translate_object': HERTranslateObject,
    'her_translate_goal': HERTranslateGoal,
    'her_translate_goal_proximal': HERTranslateGoalProximal,
    'her_translate_goal_proximal_0': HERTranslateGoalProximal0,
    'translate_goal': TranslateGoal,
    'translate_goal_proximal': TranslateGoalProximal,
    'translate_goal_proximal_0': TranslateGoalProximal0,
    # 'translate_goal_dynamic': TranslateGoalDynamic,
    'coda': CoDA,
    'coda_proximal_0': CoDAProximal0,
    'coda_proximal': CoDAProximal,
    'translate_object': TranslateObject,
    'translate_object_proximal': TranslateObjectProximal,
    'translate_object_proximal_0': TranslateObjectProximal0,
    # 'translate_object_dynamic': TranslateObjectDynamic,
    'translate_object_and_goal': TranslateObjectAndGoal,
    'translate_object_jitter': TranslateObjectJitter,
    'translate_object_safe': TranslateObjectSafe,

}