from copy import deepcopy
from typing import Dict, List, Any

import numpy as np

class CoDAPanda:

    def __init__(self, env, **kwargs):
        self.env = env
        self.aug_threshold = 0.05

    def augment(
            self,
            obs1: np.ndarray,
            next_obs1: np.ndarray,
            action1: np.ndarray,
            # reward1: np.ndarray,
            # done1: np.ndarray,
            # info1: List[Dict[str, Any]],

            obs2: np.ndarray,
            next_obs2: np.ndarray,
            # action2: np.ndarray,
            reward2: np.ndarray,
            done2: np.ndarray,
            terminated2: np.ndarray,
    ):
        '''
        robot obs, action -> comes from obs1
        object obs, goal, reward, done -> comes from obs2

        :param obs1:
        :param obs2:
        :return:
        '''
        ee_pos1 = obs1[:, :3]
        obj_obs2 = obs2[:, self.env.obj_idx]
        obj_pos2 = obj_obs2[:, :3]

        next_ee_pos1 = next_obs1[:, :3]
        next_obj_obs2 = next_obs2[:, self.env.obj_idx]
        next_obj_pos2 = next_obj_obs2[:, :3]

        # Use 0.1 as the threshold, since the goal threshold is 0.05 and the arm can move at most 0.05 along any axis.
        is_indepedent = (np.abs(ee_pos1[:, 0] - obj_pos2[:, 0])) > 0.03 \
                        and (np.abs(ee_pos1[:, 1] - obj_pos2[:, 1])) > 0.05

        next_is_indepedent = (np.abs(next_ee_pos1[:, 0] - next_obj_pos2[:, 0])) > 0.03 \
                             and (np.abs(next_ee_pos1[:, 1] - next_obj_pos2[:, 1])) > 0.05


        # is_indepedent = np.linalg.norm(ee_pos1 - obj_pos2, axis=-1) > 0.1
        # next_is_indepedent = np.linalg.norm(next_ee_pos1 - next_obj_pos2, axis=-1) > 0.1
        # mask = (is_indepedent and next_is_indepedent).astype(bool)

        if (is_indepedent and next_is_indepedent):
            aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info =\
                self._deepcopy_transition(
                    obs=obs1,
                    next_obs=next_obs1,
                    action=action1,
                    reward=reward2,
                    done=done2,
                    terminated=terminated2
                )


            goal2 = obs2[:, self.env.goal_idx].copy()
            next_goal2 = next_obs2[:, self.env.goal_idx].copy()

            aug_obs[:,self.env.obj_idx] = obj_obs2.copy()
            aug_obs[:,self.env.goal_idx] = goal2.copy()

            aug_next_obs[:,self.env.obj_idx] = next_obj_obs2.copy()
            aug_next_obs[:,self.env.goal_idx] = next_goal2.copy()

            return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info
        else:
            return None, None, None, None, None, None,

    def _deepcopy_transition(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            terminated: np.ndarray,
    ):
        aug_obs = obs.copy().reshape(1,-1)
        aug_next_obs = next_obs.copy().reshape(1,-1)
        aug_action = action.copy().reshape(1,-1)
        aug_reward = reward.copy().reshape(1,-1)
        aug_done = done.copy().reshape(1,-1)
        aug_info = [{'TimeLimit.truncated': terminated}]

        return aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info


if __name__ == "__main__":
    obs1 = np.ones(21) * -1


    obs2 = np.ones(21)

    CoDAPanda()