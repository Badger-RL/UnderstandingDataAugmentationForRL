import numpy as np

from augment.rl.augmentation_functions.augmentation_function import GoalAugmentationFunction, ObjectAugmentationFunction


#######################################################################################################################
#######################################################################################################################

class FetchGoalAugmentationFunction(GoalAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.achieved_goal_mask = self.env.achieved_goal_mask
        self.desired_goal_mask = self.env.desired_goal_mask

    def _is_at_goal(self, achieved_goal, desired_goal, **kwargs):
        return self.env.compute_reward(achieved_goal, desired_goal, info=None).astype(bool)

    def _compute_reward(self, achieved_goal, desired_goal, **kwargs):
        return self.env.compute_reward(achieved_goal, desired_goal, info=None)

class TranslateGoal(FetchGoalAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_goals(self, next_obs, **kwargs):
        n = next_obs.shape[0]
        return self.env.sample_n_goals(n)


class HER(FetchGoalAugmentationFunction):
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

    def _sample_goal_noise(self, n, **kwargs):
        r = np.random.uniform(0, self.env.distance_threshold, size=n)
        theta = np.random.uniform(-np.pi, np.pi, size=n)
        phi = np.random.uniform(-np.pi / 2, np.pi / 2, size=n)
        dx = r * np.sin(phi) * np.cos(theta)
        dy = r * np.sin(phi) * np.sin(theta)
        dz = r * np.cos(phi)
        dz[:] = 0
        noise = np.array([dx, dy, dz]).T
        return noise

    def _sample_future(self, next_obs):
        n = next_obs.shape[0]
        low = np.arange(n)
        indices = np.random.randint(low=low, high=n)
        final_pos = next_obs[indices]
        final_pos = final_pos[:, self.achieved_goal_mask]

        noise = self._sample_goal_noise(n)
        new_goal = final_pos + noise
        return new_goal

    def _sample_last(self, next_obs):
        final_pos = next_obs[:, self.env.achieved_idx].copy()
        return final_pos

    def _sample_random(self, next_obs):
        raise NotImplementedError

    def _sample_goals(self, next_obs, **kwargs):
        return self.goal_sampler(next_obs)


class FetchObjectAugmentationFunction(ObjectAugmentationFunction):
    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.aug_threshold = np.array([0.06, 0.06, 0.06])  # largest distance from center to block edge = 0.02

        self.achieved_goal_mask = self.env.achieved_goal_mask
        self.desired_goal_mask = self.env.desired_goal_mask
        self.object_pos_mask = self.env.object_pos_mask
        self.obj_size = 3

    def _is_at_goal(self, achieved_goal, desired_goal, **kwargs):
        return self.env.compute_reward(achieved_goal, desired_goal, info=None).astype(bool)

    def _compute_reward(self, achieved_goal, desired_goal, **kwargs):
        return self.env.compute_reward(achieved_goal, desired_goal, info=None)

    def _sample_objects(self, obs, next_obs, **kwargs):
        n = obs.shape[0]

        mask = np.ones(n, dtype=bool)
        independent_obj = np.empty(shape=(n, self.obj_size))
        independent_next_obj = np.empty(shape=(n, self.obj_size))

        sample_new_obj = True
        while sample_new_obj:
            new_obj = self._sample_object(n, **kwargs)
            # new_next_obj = new_obj
            obj_pos_diff = next_obs[:, self.object_mask] - obs[:, self.object_mask]
            new_next_obj = new_obj + obj_pos_diff
            independent_obj[mask] = new_obj[mask]
            independent_next_obj[mask] = new_next_obj[mask]

            is_independent, next_is_independent = self._check_independence(obs, next_obs, new_obj, new_next_obj, mask)
            mask[mask] = ~(is_independent & next_is_independent)
            sample_new_obj = np.any(mask)

        return independent_obj, independent_next_obj

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

    def _check_observed_constraints(self, obs, next_obs, reward, **kwargs):
        diff = np.abs((obs[:, :3] - obs[:, self.object_mask]))
        next_diff = np.abs((next_obs[:, :3] - next_obs[:,self.object_mask]))
        is_independent = np.any(diff > self.aug_threshold, axis=-1)
        next_is_independent = np.any(next_diff > self.aug_threshold, axis=-1)
        observed_is_independent = is_independent & next_is_independent
        return np.all(observed_is_independent)

class TranslateObject(FetchObjectAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)

    def _sample_object(self, n, **kwargs):
        new_obj = self.env.task._sample_n_objects(n)
        return new_obj

FETCH_AUG_FUNCTIONS = {
    'random_goal': TranslateGoal,
    'random_object': TranslateObject,
    'her': HER,

}