from typing import Dict, List, Any
import numpy as np

from augment.rl.augmentation_functions.panda.common import RobotAugmentationFunction

class ReachReflect(RobotAugmentationFunction):

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
        obs[:, 4] *= -1 # xvel

    def _reflect_object_obs(self, obs):
        # y reflection
        pass

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

class Translate(RobotAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.env = env
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

        obs[:, self.env.achieved_idx[0]] *= -1
        next_obs[:, self.env.achieved_idx[0]] *= -1
        next_obs[:, self.env.goal_idx[0]] *= -1

        action[:, 0] *= -1

        return obs, next_obs, action, reward, done, infos
