from typing import Dict, List, Any
import numpy as np

from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction, HERAugmentationFunction

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
        achieved_goal = next_obs[:, self.env.achieved_idx]
        new_goal = np.random.uniform(low=self.lo, high=self.hi, size=(n,3))
        obs[:, -3:] = new_goal
        next_obs[:, -3:] = new_goal

        at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, new_goal, infos)

        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

class TranslateGoalProximal(AugmentationFunction):

    def __init__(self, env, p,  **kwargs):
        super().__init__(env=env, **kwargs)
        self.env = env
        self.lo = env.task.goal_range_low
        self.hi = env.task.goal_range_high
        self.delta = 0.05
        self.p = p

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
        achieved_goal = next_obs[:, self.env.achieved_idx]

        if np.random.random() < self.p:
            r = np.random.uniform(0, self.delta)
            theta = np.random.uniform(-np.pi, np.pi)
            phi = np.random.uniform(-np.pi/2, np.pi/2)
            dx = r*np.sin(phi)*np.cos(theta)
            dy = r*np.sin(phi)*np.sin(theta)
            dz = r*np.cos(phi)
            if self.hi[-1] == 0:
                dz = 0
            new_goal = obs[:, -3:] + np.array([dx, dy, dz])
        else:
            new_goal = np.random.uniform(low=self.lo, high=self.hi, size=(n,3))
        obs[:, -3:] = new_goal
        next_obs[:, -3:] = new_goal

        at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, new_goal, infos)

        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos

class HER(HERAugmentationFunction):
    def __init__(self, env, strategy='future', **kwargs):
        super().__init__(env=env, **kwargs)
        self.lo = env.task.goal_range_low
        self.hi = env.task.goal_range_high
        self.delta = 0.05
        self.strategy = strategy
        if self.strategy == 'future':
            self.sampler = self._sample_future
        else:
            self.sampler = self._sample_last

    def _sample_future(self, next_obs):
        n = next_obs.shape[0]
        low = np.arange(n)
        indices = np.random.randint(low=low, high=n)
        final_pos = next_obs[indices].copy()
        final_pos = final_pos[:, self.env.achieved_idx]
        return final_pos

    def _sample_last(self, next_obs):
        final_pos = next_obs[:, :3].copy()
        return final_pos

    def _sample_goals(self, next_obs):
        return self.sampler(next_obs)

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

        achieved_goal = next_obs[:, self.env.achieved_idx]
        new_goal = self._sample_goals(next_obs)
        obs[:, -3:] = new_goal
        next_obs[:, -3:] = new_goal

        at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, new_goal, infos)

        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos




class Reflect(AugmentationFunction):

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
        achieved_goal = next_obs[:, self.env.achieved_idx]
        new_goal = np.random.uniform(low=self.lo, high=self.hi, size=(n,3))
        obs[:, -3:] = new_goal
        next_obs[:, -3:] = new_goal

        at_goal = self.env.task.is_success(achieved_goal, new_goal).astype(bool)
        reward = self.env.task.compute_reward(achieved_goal, new_goal, infos)

        self._set_done_and_info(done, infos, at_goal)

        return obs, next_obs, action, reward, done, infos


PANDA_AUG_FUNCTIONS = {
    'her': HER,
    'translate_goal': TranslateGoal,
    'translate_goal_proximal': TranslateGoalProximal,
}
#
#
# def tmp():
#
#     env = gym.make('Humanoid-v4', reset_noise_scale=0)
#     f = HumanoidReflect()
#
#     for k in range(1,2):
#         action = np.zeros(17, dtype=np.float32).reshape(1,-1)
#         action[:, k] = 1
#         # action[:, 3:6] = 1
#         # action[:, 11:13] = 1
#
#         env.reset()
#         # f.reflect_action(action)
#         print(action)
#         for i in range(200):
#             next_obs, reward, terminated, truncated, info = env.step(action[0])
#         true = next_obs.copy()
#         aug = next_obs.copy().reshape(1,-1)
#         f.reflect_obs(aug)
#         aug.reshape(-1)
#
#         env.reset()
#         f.reflect_action(action)
#         # action = np.zeros(17, dtype=np.float32).reshape(1,-1)
#         # action[:, 16] = 1
#         print(action)
#         for i in range(200):
#             next_obs, reward, terminated, truncated, info = env.step(action[0])
#         true_reflect = next_obs.copy()
#
#         print(f'{i}\ttrue\t\ttrue_reflect\taug\tis_close')
#         is_close = np.isclose(true_reflect,aug[0])
#         for i in range(45):
#             print(f'{i}\t{true[i]:.8f}\t{true_reflect[i]:.8f}\t{aug[0][i]:.8f}\t{is_close[i]}')
#         print(np.all(is_close))
#     # print()
#     # time.sleep(2)
#
# def check_valid(env, aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info):
#
#     # set env to aug_obs
#     # env = gym.make('Walker2d-v4', render_mode='human')
#
#     # env.reset()
#     qpos, qvel = aug_obs[:21+1], aug_obs[22:]
#     x = aug_info['x_position']
#     y = aug_info['y_position']
#     qpos = np.concatenate((np.array([0,0]), qpos))
#     env.set_state(qpos, qvel)
#
#     # determine ture next_obs, reward
#     next_obs_true, reward_true, terminated_true, truncated_true, info_true = env.step(aug_action)
#     print(aug_next_obs[22:23+1])
#     print(next_obs_true[22:23+1])
#     print(aug_next_obs - next_obs_true)
#     print('here', aug_reward-reward_true)
#     print(aug_reward, aug_info)
#     print(reward_true, info_true)
#     assert np.allclose(aug_next_obs, next_obs_true)
#     assert np.allclose(aug_reward, reward_true)
#
# if __name__ == "__main__":
#     # tmp()
#     '''
#
#     '''
#     env = gym.make('Humanoid-v4', reset_noise_scale=0)
#     aug_func = Reflect(env=env)
#     validate_augmentation(env, aug_func, check_valid)