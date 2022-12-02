from typing import Dict, List, Any
import numpy as np

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

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
        for i in range(n):
            obs[i]['desired_goal'] = new_goal
            next_obs[i]['desired_goal'] = new_goal

        achieved_goal = next_obs[i]['achieved_goal']

        at_goal = self.env.is_success(achieved_goal, new_goal).astype(bool)
        reward = self.env.compute_reward(achieved_goal, new_goal, infos)

        self._set_done_and_info(done, infos, at_goal)

        end = np.argmax(done)+1
        obs = obs[:end]
        next_obs = next_obs[:end]
        action = action[:end]
        reward = reward[:end]
        done = done[:end]
        infos = infos[:end]


        return obs, next_obs, action, reward, done, infos

PANDA_AUG_FUNCTIONS = {
    'translate_goal': TranslateGoal,
    # 'TranslateGoalProximal': TranslateGoalProximal,
}