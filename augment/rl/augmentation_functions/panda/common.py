from typing import Dict, List, Any
import numpy as np

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
        reward = self.env.task.compute_reward(achieved_goal, new_goal, infos)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

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
            if self.env.task.goal_range_high[-1] == 0:
                dz[:] = 0
            noise = np.array([dx, dy, dz]).T
            new_goal = next_obs[:, self.env.goal_idx] + noise
        else:
            # new goal results in no reward signal
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


class TranslateGoalDynamic(GoalAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_goals(self, next_obs, p=None, **kwargs):
        ep_length = next_obs.shape[0]
        if np.random.random() < p:
            r = np.random.uniform(0, self.delta, size=ep_length)
            theta = np.random.uniform(-np.pi, np.pi, size=ep_length)
            phi = np.random.uniform(-np.pi/2, np.pi/2, size=ep_length)
            dx = r*np.sin(phi)*np.cos(theta)
            dy = r*np.sin(phi)*np.sin(theta)
            dz = r*np.cos(phi)
            if self.env.task.goal_range_high[-1] == 0:
                dz[:] = 0
            noise = np.array([dx, dy, dz]).T
            new_goal = next_obs[:, self.env.goal_idx] + noise
        else:
            # new goal results in no reward signal
            new_goal = self.env.task._sample_n_goals(ep_length)
            achieved_goal = next_obs[:, self.env.achieved_idx]
            at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)

            # resample if success (rejection sampling)
            while np.any(at_goal):
                new_goal[at_goal] = self.env.task._sample_n_goals(ep_length)[at_goal]
                at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        return new_goal


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
        final_pos = next_obs[indices].copy()
        final_pos = final_pos[:, self.env.achieved_idx]
        return final_pos

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
        self.aug_threshold = np.array([0.03, 0.05, 0.05])  # largest distance from center to block edge = 0.02

        self.obj_pos_mask = np.zeros_like(self.env.obj_idx, dtype=bool)
        obj_pos_idx = np.argmax(self.env.obj_idx)
        self.obj_pos_mask[obj_pos_idx:obj_pos_idx + 3] = True

        self.obj_size = 3

    def _sample_object(self, n):
        raise NotImplementedError()

    def _sample_objects(self, obs, next_obs):
        n = obs.shape[0]
        new_obj = self._sample_object(n)
        obj_pos_diff = next_obs[:, self.obj_pos_mask] - obs[:, self.obj_pos_mask]
        new_next_obj = new_obj + obj_pos_diff
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

        achieved_goal = next_obs[:, self.env.achieved_idx]
        desired_goal = next_obs[:, self.env.goal_idx]
        at_goal = self.env.task.is_success(achieved_goal, desired_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, desired_goal, infos)
        self._set_done_and_info(done, infos, at_goal)
        # print(reward)
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
        assert ep_length == 1

        diff = np.abs((obs[:, :3] - obs[:, self.obj_pos_mask]))
        next_diff = np.abs((next_obs[:, :3] - next_obs[:,self.obj_pos_mask]))
        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)
        if not np.any(is_independent & next_is_independent):
            assert ep_length == 1  # REQUIRES LENGTH 1. POSSIBLE BUG IF YOU USE HER.
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


class TranslateObject(ObjectAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_object(self, n):
        new_obj = self.env.task._sample_n_objects(n)
        return new_obj

class TranslateObjectProximal0(ObjectAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = 0

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

        sample_new_obj = True
        while sample_new_obj:
            new_obj, new_next_obj = self._sample_objects(obs, next_obs)
            independent_obj[mask] = new_obj
            independent_next_obj[mask] = new_next_obj

            is_independent, next_is_independent = self._check_independence(obs, next_obs, new_obj, new_next_obj, mask)

            desired_goal = next_obs[:, self.env.goal_idx]
            at_goal = self._check_at_goal(new_next_obj, desired_goal, mask)
            mask[mask] = ~(is_independent & next_is_independent & ~at_goal)
            sample_new_obj = np.any(mask)

        return self._make_transition(obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj)


class TranslateObjectProximal(ObjectAugmentationFunction):

    def __init__(self, env, p=0.5, **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = p
        self.aug_threshold = np.array([0.03, 0.05, 0.05])  # largest distance from center to block edge = 0.02

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
            ee_dist = (obs[:, :3] - desired_goal)
            ee_next_dist = (next_obs[:, :3] - desired_goal)

            # ee is too close to goal to generate reward signal
            if np.all(ee_dist < self.aug_threshold) or np.all(ee_next_dist < self.aug_threshold):
                return None, None, None, None, None, None

            r = np.random.uniform(0, self.delta, size=ep_length)
            theta = np.random.uniform(-np.pi, np.pi, size=ep_length)
            phi = np.random.uniform(-np.pi / 2, np.pi / 2, size=ep_length)
            dx = r * np.sin(phi) * np.cos(theta)
            dy = r * np.sin(phi) * np.sin(theta)
            dz = r * np.cos(phi)
            if self.env.task.goal_range_high[-1] == 0:
                dz[:] = 0
            noise = np.array([dx, dy, dz]).T
            independent_obj = desired_goal + noise

            obj_pos_diff = next_obs[:, self.obj_pos_mask] - obs[:, self.obj_pos_mask]
            independent_next_obj = independent_obj + obj_pos_diff

        else:
            sample_new_obj = True
            while sample_new_obj:
                new_obj, new_next_obj = self._sample_objects(obs, next_obs)
                independent_obj[mask] = new_obj
                independent_next_obj[mask] = new_next_obj

                is_independent, next_is_independent = self._check_independence(obs, next_obs, new_obj, new_next_obj, mask)

                desired_goal = next_obs[:, self.env.goal_idx]
                at_goal = self._check_at_goal(new_next_obj, desired_goal, mask)
                mask[mask] = ~(is_independent & next_is_independent & ~at_goal)
                sample_new_obj = np.any(mask)

        return self._make_transition(obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj)

class TranslateObjectDynamic(ObjectAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.aug_threshold = np.array([0.03, 0.05, 0.05])  # largest distance from center to block edge = 0.02

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

        if np.random.random() < p:
            desired_goal = next_obs[:, self.env.goal_idx]
            ee_dist = (obs[:, :3] - desired_goal)
            ee_next_dist = (next_obs[:, :3] - desired_goal)

            # ee is too close to goal to generate reward signal
            if np.all(ee_dist < self.aug_threshold) or np.all(ee_next_dist < self.aug_threshold):
                return None, None, None, None, None, None

            r = np.random.uniform(0, self.delta, size=ep_length)
            theta = np.random.uniform(-np.pi, np.pi, size=ep_length)
            phi = np.random.uniform(-np.pi / 2, np.pi / 2, size=ep_length)
            dx = r * np.sin(phi) * np.cos(theta)
            dy = r * np.sin(phi) * np.sin(theta)
            dz = r * np.cos(phi)
            if self.env.task.goal_range_high[-1] == 0:
                dz[:] = 0
            noise = np.array([dx, dy, dz]).T
            independent_obj = desired_goal + noise

            obj_pos_diff = next_obs[:, self.obj_pos_mask] - obs[:, self.obj_pos_mask]
            independent_next_obj = independent_obj + obj_pos_diff

        else:
            sample_new_obj = True
            while sample_new_obj:
                new_obj, new_next_obj = self._sample_objects(obs, next_obs)
                independent_obj[mask] = new_obj
                independent_next_obj[mask] = new_next_obj

                is_independent, next_is_independent = self._check_independence(obs, next_obs, new_obj, new_next_obj, mask)

                desired_goal = next_obs[:, self.env.goal_idx]
                at_goal = self._check_at_goal(new_next_obj, desired_goal, mask)
                mask[mask] = ~(is_independent & next_is_independent & ~at_goal)
                sample_new_obj = np.any(mask)

        return self._make_transition(obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj)

#######################################################################################################################
#######################################################################################################################

class CoDAProximal0(ObjectAugmentationFunction):

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

        if replay_buffer.size() < 1000:
            return None, None, None, None, None, None,

        ep_length = obs.shape[0]
        mask = np.ones(ep_length, dtype=bool)
        independent_obj = np.empty(shape=(ep_length, self.obj_size))
        independent_next_obj = np.empty(shape=(ep_length, self.obj_size))
        desired_goal = next_obs[:, self.env.goal_idx]

        sample_new_obj = True
        while sample_new_obj:
            new_obs, _, new_next_obs, _, _, _ = replay_buffer.sample_array(batch_size=ep_length)
            new_obj = new_obs[:, self.obj_pos_mask]
            new_next_obj = new_next_obs[:, self.obj_pos_mask]

            independent_obj[mask] = new_obj[mask]
            independent_next_obj[mask] = new_next_obj[mask]

            diff = np.abs((obs[:, :self.obj_size] - new_obj)[mask])
            next_diff = np.abs((next_obs[:, :self.obj_size] - new_next_obj)[mask])

            is_independent = np.any(diff > self.aug_threshold, axis=-1)
            next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)

            at_goal = self.env.task.is_success(new_next_obj[mask], desired_goal[mask]).astype(bool)

            mask[mask] = ~(is_independent & next_is_independent & ~at_goal)
            sample_new_obj = np.any(mask)

        obs[:, self.obj_pos_mask] = independent_obj
        next_obs[:, self.obj_pos_mask] = independent_next_obj

        achieved_goal = next_obs[:, self.env.achieved_idx]
        at_goal = self.env.task.is_success(achieved_goal, desired_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, desired_goal, infos)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

class CoDA(ObjectAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def _sample_goals(self, next_obs, **kwargs):
        ep_length = next_obs.shape[0]
        return self.env.task._sample_n_goals(ep_length)


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

        if replay_buffer.size() < 1000:
            return None, None, None, None, None, None,

        ep_length = obs.shape[0]
        mask = np.ones(ep_length, dtype=bool)
        independent_obj = np.empty(shape=(ep_length, self.obj_size))
        independent_next_obj = np.empty(shape=(ep_length, self.obj_size))
        new_goal = np.empty(shape=(ep_length, self.obj_size))

        sample_new_obj = True
        while sample_new_obj:
            new_obs, _, new_next_obs, _, _, _ = replay_buffer.sample_array(batch_size=ep_length)
            new_obj = new_obs[:, self.obj_pos_mask]
            new_next_obj = new_next_obs[:, self.obj_pos_mask]

            independent_obj[mask] = new_obj[mask]
            independent_next_obj[mask] = new_next_obj[mask]

            diff = np.abs((obs[:, :self.obj_size] - new_obj)[mask])
            next_diff = np.abs((next_obs[:, :self.obj_size] - new_next_obj)[mask])

            is_independent = np.any(diff > self.aug_threshold, axis=-1)
            next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)

            mask[mask] = ~(is_independent & next_is_independent)
            sample_new_obj = np.any(mask)

        obs[:, self.obj_pos_mask] = independent_obj
        next_obs[:, self.obj_pos_mask] = independent_next_obj

        new_goal = self._sample_goals(next_obs)
        obs[:, self.env.goal_idx] = new_goal
        next_obs[:, self.env.goal_idx] = new_goal

        achieved_goal = next_obs[:, self.env.achieved_idx]
        at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, new_goal, infos)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

class TranslateObjectAndGoal(AugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.delta = 0.05
        self.aug_threshold = np.array([0.03, 0.05, 0.05])  # largest distance from center to block edge = 0.02

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
        new_obj = self._sample_object(n)
        obj_pos_diff = next_obs[:, self.obj_pos_mask] - obs[:, self.obj_pos_mask]
        new_next_obj = new_obj + obj_pos_diff
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
        reward = self.env.task.compute_reward(achieved_goal, new_goal, infos)
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

        return self._make_transition(obs, next_obs, action, reward, done, infos, independent_obj, independent_next_obj)


# Reach, Push, Slide, PickAndPlace only
PANDA_AUG_FUNCTIONS = {
    'her': HER,
    'her_coda': HERCoDA,
    'her_translate_object': HERTranslateObject,
    'her_translate_goal': HERTranslateGoal,
    'her_translate_goal_proximal': HERTranslateGoalProximal,
    'her_translate_goal_proximal_0': HERTranslateGoalProximal0,
    'her_translate_goal_proximal_09': HERTranslateGoalProximal09,
    'translate_goal': TranslateGoal,
    'translate_goal_proximal': TranslateGoalProximal,
    'translate_goal_proximal_0': TranslateGoalProximal0,
    'translate_goal_dynamic': TranslateGoalDynamic,
    'coda': CoDA,
    'coda_proximal_0': CoDAProximal0,
    'translate_object': TranslateObject,
    'translate_object_proximal': TranslateObjectProximal,
    'translate_object_proximal_0': TranslateObjectProximal0,
    'translate_object_dynamic': TranslateObjectDynamic,
    'translate_object_and_goal': TranslateObjectAndGoal,
}