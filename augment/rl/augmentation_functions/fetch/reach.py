from typing import Dict, List, Any
import numpy as np

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction

'''
    - `observation`: its value is an `ndarray` of shape `(10,)`. It consists of kinematic information of the end effector. The elements of the array correspond to the following:
        | Num | Observation                                                                                                                           | Min    | Max    | Site Name (in corresponding XML file) | Joint Name (in corresponding XML file) |Joint Type| Unit                     |
        |-----|---------------------------------------------------------------------------------------------------------------------------------------|--------|--------|---------------------------------------|----------------------------------------|----------|--------------------------|
        | 0   | End effector x position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 1   | End effector y position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 2   | End effector z position in global coordinates                                                                                         | -Inf   | Inf    | robot0:grip                           |-                                       |-         | position (m)             |
        | 3   | Joint displacement of the right gripper finger                                                                                        | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | position (m)             |
        | 4   | Joint displacement of the left gripper finger                                                                                         | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | position (m)             |
        | 5   | End effector linear velocity x direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 6   | End effector linear velocity y direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 7   | End effector linear velocity z direction                                                                                              | -Inf   | Inf    | robot0:grip                           |-                                       |-         | velocity (m/s)           |
        | 8   | Right gripper finger linear velocity                                                                                                  | -Inf   | Inf    |-                                      | robot0:r_gripper_finger_joint          | hinge    | velocity (m/s)           |
        | 9   | Left gripper finger linear velocity                                                                                                   | -Inf   | Inf    |-                                      | robot0:l_gripper_finger_joint          | hinge    | velocity (m/s)           |
        
'''

class FetchReachAugmentationFunction(AugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.env = env
        self.lo = env.initial_gripper_xpos[:3] - env.target_range
        self.hi = env.initial_gripper_xpos[:3] + env.target_range
        self.delta = 0.05
        self.desired_mask = env.desired_mask
        self.achieved_mask = env.achieved_mask

    def _set_done_and_info(self, done, infos, at_goal):
        done |= at_goal
        infos[done & ~at_goal] = [{'TimeLimit.truncated': True}]
        infos[done & at_goal] = [{'TimeLimit.truncated': False}]

class FetchReachHER(FetchReachAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        obs[:, self.desired_mask] = obs[-1, self.achieved_mask]
        next_obs[:, self.desired_mask] = next_obs[-1, self.achieved_mask]
        at_goal = self.env.is_success(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask]).astype(bool)
        reward = self.env.compute_reward(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask], infos)

        self._set_done_and_info(done, infos, at_goal)

        end = np.argmax(done)+1
        obs = obs[:end]
        next_obs = next_obs[:end]
        action = action[:end]
        reward = reward[:end]
        done = done[:end]
        infos = infos[:end]


        return obs, next_obs, action, reward, done, infos

class FetchReachTranslateGoal(FetchReachAugmentationFunction):

    def __init__(self, env,  **kwargs):
        super().__init__(env=env, **kwargs)

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        new_goal = np.random.uniform(-self.lo, self.hi)

        obs[:, -3:] = new_goal
        next_obs[:, -3:] = new_goal

        at_goal = self.env.is_success(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask]).astype(bool)
        reward = self.env.compute_reward(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask], infos)

        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos


class FetchReachTranslateGoalProximal(FetchReachAugmentationFunction):

    def __init__(self, env, p=0.5, **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = p

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        if np.random.random() < self.p:
            r = np.random.uniform(0, self.delta)
            theta = np.random.uniform(-np.pi, np.pi)
            phi = np.random.uniform(-np.pi/2, np.pi/2)
            dx = r*np.sin(phi)*np.cos(theta)
            dy = r*np.sin(phi)*np.sin(theta)
            dz = r*np.cos(phi)
            new_goal = obs[:, -3:] + np.array([dx, dy, dz])
        else:
            new_goal = np.random.uniform(self.lo, self.hi)

        obs[:, -3:] = new_goal
        next_obs[:, -3:] = new_goal

        at_goal = self.env.is_success(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask]).astype(bool)
        reward = self.env.compute_reward(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask], infos)

        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

class FetchReachTranslate(FetchReachAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.env = env
        self.lo = env.initial_gripper_xpos[:3] - env.target_range
        self.hi = env.initial_gripper_xpos[:3] + env.target_range
        self.delta = 0.05
        self.distance_threshold = self.env.distance_threshold

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        v = np.random.uniform(self.lo, self.hi)
        delta_pos = next_obs[:, :3] - obs[:, :3]

        obs[:, :3] = v
        next_obs[:, :3] = v + delta_pos

        at_goal = self.env.is_success(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask]).astype(bool)
        reward = self.env.compute_reward(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask], infos)

        self._set_done_and_info(done, infos, at_goal)


        return obs, next_obs, action, reward, done, infos

class FetchReachTranslateProximal(FetchReachAugmentationFunction):

    def __init__(self, env, p=0.5, **kwargs):
        super().__init__(env=env, **kwargs)
        self.env = env
        self.lo = env.initial_gripper_xpos[:3] - env.target_range
        self.hi = env.initial_gripper_xpos[:3] + env.target_range
        self.delta = 0.05
        self.distance_threshold = self.env.distance_threshold
        self.p = p

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        if np.random.random() < self.p:


            r = np.random.uniform(0, self.delta)
            theta = np.random.uniform(-np.pi, np.pi)
            phi = np.random.uniform(-np.pi/2, np.pi/2)
            dx = r*np.sin(phi)*np.cos(theta)
            dy = r*np.sin(phi)*np.sin(theta)
            dz = r*np.cos(phi)

            delta_pos = next_obs[:, self.achieved_mask] - obs[:, self.achieved_mask]
            obs[:, self.achieved_mask] = (next_obs[:, self.desired_mask] + np.array([[dx, dy, dz]])) - delta_pos
            next_obs[:, self.achieved_mask] = obs[:, self.achieved_mask] + delta_pos

            at_goal = self.env.is_success(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask]).astype(bool)
        else:
            while True:
                v = np.random.uniform(self.lo, self.hi)
                obs[:, :3] = v
                next_obs[:, :3] = v + action[:, :3] * self.delta
                at_goal = self.env.is_success(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask]).astype(bool)
                if not np.any(at_goal):
                    break


        reward = self.env.compute_reward(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask], infos)

        self._set_done_and_info(done, infos, at_goal)


        return obs, next_obs, action, reward, done, infos

class FetchReachReflect(FetchReachAugmentationFunction):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.env = env
        self.lo = env.initial_gripper_xpos[:3] - env.target_range
        self.hi = env.initial_gripper_xpos[:3] + env.target_range
        self.delta = 0.05
        self.distance_threshold = self.env.distance_threshold

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 ):

        action *= -1
        obs[:,:3] *= -1
        next_obs[:,:3] *= -1
        obs[:,5:8] *= -1
        next_obs[:,5:8] *= -1
        obs[:,-3:] *= -1
        next_obs[:,-3:] *= -1

        # next_obs[:, self.achieved_mask] = -2*(next_obs[:, self.achieved_mask] - obs[:, self.achieved_mask])
        at_goal = self.env.is_success(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask]).astype(bool)
        reward = self.env.compute_reward(next_obs[:, self.achieved_mask], next_obs[:, self.desired_mask], infos)

        self._set_done_and_info(done, infos, at_goal)


        return obs, next_obs, action, reward, done, infos