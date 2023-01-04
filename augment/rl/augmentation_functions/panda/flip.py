import copy

import numpy as np

from augment.rl.augmentation_functions.panda.common import GoalAugmentationFunction, PANDA_AUG_FUNCTIONS


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
            a = np.arccos(achieved_goal[0])
            theta = np.random.uniform(-0.927, +0.927) # arccos(0.6) ~= +/-0.927
            q_rotation = np.array([
                np.cos(theta / 2),
                a[0] * np.sin(theta / 2),
                a[1] * np.sin(theta / 2),
                a[2] * np.sin(theta / 2),
            ])
            new_goal = self.quaternion_multiply(achieved_goal, q_rotation)
        else:
            # new goal results in no reward signal
            new_goal = self.env.task._sample_n_goals(n)
            at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)

            # resample if success (rejection sampling)
            while at_goal:
                new_goal = self.env.task._sample_n_goals(n)
                at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        return new_goal

PANDA_FLIP_AUG_FUNCTIONS = copy.deepcopy(PANDA_AUG_FUNCTIONS)
PANDA_FLIP_AUG_FUNCTIONS.update(
    {
    'translate_goal_proximal': TranslateGoalProximal,
    })
