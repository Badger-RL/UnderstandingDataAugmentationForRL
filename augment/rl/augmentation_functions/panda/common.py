from typing import Dict, List, Any
import numpy as np

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
# from augment.rl.augmentation_functions.coda import CoDAPanda


class ObjectAugmentationFunction(AugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.delta = 0.05
        self.aug_threshold = 0.05 # largest distance from center to block edge = 0.02

        self.obj_pos_mask = np.zeros_like(self.env.obj_idx, dtype=bool)
        obj_pos_idx = np.argmax(self.env.obj_idx)
        self.obj_pos_mask[obj_pos_idx:obj_pos_idx+3] = True

        self.obj_vel_mask = np.zeros_like(self.env.obj_idx, dtype=bool)
        self.obj_vel_mask[obj_pos_idx+3:-3] = True


    def _sample_object(self, next_obs):
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
                 **kwargs
                 ):

        ee_xy = obs[:, :2]
        next_ee_xy = next_obs[:, :2]
        # ee_z = obs[:, 2]
        # next_ee_z = next_obs[:, 2]

        new_obj = self._sample_object(next_obs)
        new_obj_xy = new_obj[:, :2]
        # new_obj_z = new_obj[:, :2]

        # if np.abs(new_obj_z - ee_z) < self.aug_threshold or np.abs(new_obj_z - next_ee_z) < self.aug_threshold:
        while np.linalg.norm(ee_xy-new_obj_xy) < self.aug_threshold or np.linalg.norm(next_ee_xy-new_obj_xy) < self.aug_threshold:
            new_obj = self._sample_object(next_obs)
            new_obj_xy = new_obj[:, :2]

        obs[:, self.obj_pos_mask] = new_obj
        next_obs[:, self.obj_pos_mask] = new_obj

        achieved_goal = next_obs[:, self.env.achieved_idx]
        desired_goal = next_obs[:, self.env.goal_idx]
        at_goal = self.env.task.is_success(achieved_goal, desired_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, desired_goal, infos)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

class TranslateObject(ObjectAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_object(self, next_obs):
        ep_length = next_obs.shape[0]
        new_obj = self.env.task._sample_n_objects(ep_length)
        return new_obj


class TranslateObjectProximal(ObjectAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_object(self, next_obs):
        ep_length = next_obs.shape[0]
        new_obj = self.env.task._sample_n_objects(ep_length)
        return new_obj

class GoalAugmentationFunction(AugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.delta = 0.05

        self.goal_length = self.env.goal_idx.shape[-1]

    def _sample_goals(self, next_obs):
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

        new_goal = self._sample_goals(next_obs)
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

    def _sample_goals(self, next_obs):
        ep_length = next_obs.shape[0]
        return self.env.task._sample_n_goals(ep_length)

class TranslateGoalProximal(GoalAugmentationFunction):

    def __init__(self, env, p=0.5,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = p

    def _sample_goals(self, next_obs):
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

    def _sample_goals(self, next_obs):
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
                 ):

        if np.random.random() < self.q:
            return self.HER._augment(obs, next_obs, action, reward, done, infos, p)
        else:
            return self.aug_function._augment(obs, next_obs, action, reward, done, infos, p)

class HERTranslateGoal(HERMixed):
    def __init__(self, env, strategy='future', q=0.5, **kwargs):
        super().__init__(env=env, aug_function=TranslateGoal, strategy=strategy, q=q, **kwargs)

class HERTranslateObject(HERMixed):
    def __init__(self, env, strategy='future', q=0.5, **kwargs):
        super().__init__(env=env, aug_function=TranslateObject, strategy=strategy, q=q, **kwargs)

class HERCoDA(HERMixed):
    def __init__(self, env, strategy='future', q=0.5, **kwargs):
        super().__init__(env=env, aug_function=CoDAPanda, strategy=strategy, q=q, **kwargs)

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

class HERReflect(HERMixed):
    def __init__(self, env, strategy='future', p=0.5, **kwargs):
        super().__init__(env=env, aug_function=Reflect, strategy=strategy, p=p, **kwargs)

class RobotAugmentationFunction(AugmentationFunction):
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


class CoDAPanda(ObjectAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.aug_threshold = 0.05
        # self.replay_buffer = replay_buffer


    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 replay_buffer=None,
                 p=None,
                 **kwargs,
                 ):
        '''
        robot obs, action -> comes from obs1
        object obs, goal, reward, done -> comes from obs2

        :param obs1:
        :param obs2:
        :return:
        '''

        is_independent = [False]
        next_is_independent = [False]

        if replay_buffer.size() < 1000:
            return None, None, None, None, None, None,

        while not(is_independent[0] and next_is_independent[0]):

            obs2, action2, next_obs2, reward2, done2, timeout2 = replay_buffer.sample_array(batch_size=1)

            ee_pos = obs[:, :3]
            obj_obs2 = obs2[:, self.env.obj_idx]
            obj_pos2 = obj_obs2[:, :3]

            next_ee_pos = next_obs[:, :3]
            next_obj_obs2 = next_obs2[:, self.env.obj_idx]
            next_obj_pos2 = next_obj_obs2[:, :3]

            # Use 0.1 as the threshold, since the goal threshold is 0.05 and the arm can move at most 0.05 along any axis.
            is_independent = (np.abs(ee_pos[:, 0] - obj_pos2[:, 0])) > 0.03 \
                            and (np.abs(ee_pos[:, 1] - obj_pos2[:, 1])) > 0.05

            next_is_independent = (np.abs(next_ee_pos[:, 0] - next_obj_pos2[:, 0])) > 0.03 \
                                 and (np.abs(next_ee_pos[:, 1] - next_obj_pos2[:, 1])) > 0.05


        # is_indepedent = np.linalg.norm(ee_pos1 - obj_pos2, axis=-1) > 0.1
        # next_is_indepedent = np.linalg.norm(next_ee_pos1 - next_obj_pos2, axis=-1) > 0.1
        # mask = (is_indepedent and next_is_indepedent).astype(bool)

        goal2 = obs2[:, self.env.goal_idx].copy()
        next_goal2 = next_obs2[:, self.env.goal_idx].copy()

        obs[:,self.env.obj_idx] = obj_obs2.copy()
        obs[:,self.env.goal_idx] = goal2.copy()

        next_obs[:,self.env.obj_idx] = next_obj_obs2.copy()
        next_obs[:,self.env.goal_idx] = next_goal2.copy()

        achieved_goal = next_obs[:, self.env.achieved_idx]
        desired_goal = next_obs[:, self.env.goal_idx]
        at_goal = self.env.task.is_success(achieved_goal, desired_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, desired_goal, infos)
        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos


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
    'coda': CoDAPanda,
    'translate_object': TranslateObject,
}