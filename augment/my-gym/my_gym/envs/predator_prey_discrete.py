import gym
import numpy as np
import scipy.linalg

from matplotlib import pyplot as plt

from augment.rl.algs.td3 import TD3

class PredatorPreyDiscreteEnv(gym.Env):
    def __init__(
            self,
            n=2,
            sigma=0.00,
    ):

        self.n = n
        # self.action_space = gym.spaces.Box(-1, +1, shape=(n,))
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.MultiDiscrete((20,20,20,20))
        self.step_num = 0
        self.horizon = 50

        self.sparse = True
        self.state_pos = np.zeros(shape=(20,20))
        self.state_goal = np.zeros(shape=(20,20))

    def step(self, u):
        self.step_num += 1

        if u == 0 and self.x[0] > 0: # left
            self.x[0] -= 1
        elif u == 1 and self.x[0] < 19: # right
            self.x[0] += 1
        elif u == 2 and self.x[1] > 0:  # down
            self.x[1] -= 1
        elif u == 3 and self.x[1] < 19:  # up
            self.x[1] += 1

        done = False
        reward = -0.1

        if np.allclose(self.x, self.goal):
            reward = +1
            done = True

        info = {}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.concatenate((self.x, self.goal))

    # def _get_obs(self):
    #     self.state_pos[:,:] = 0
    #     self.state_goal[:,:] = 0
    #
    #     self.state_pos[self.x] = 1
    #     self.state_goal[self.goal] = 1
    #
    #     return np.concatenate((self.state_pos.flatten(), self.state_goal.flatten()))

    def reset(self):
        self.x = np.random.randint(low=0, high=20, size=(2,))
        self.goal = np.random.randint(low=0, high=20, size=(2,))
        return self._get_obs()

    def set_state(self, pos, goal):
        self.x = pos
        self.goal = goal

if __name__ == "__main__":

    T = 200
    env = LQRGoalEnv(n=2)
    obs = env.reset()

    pos = []
    goal = []

    model = TD3.load('../../../local/results/LQR-v0/td3/no_aug/run_1/best_model.zip', env, custom_objects={})

    for t in range(T):
        if model:
            u, _ = model.predict(obs)
        else:
            u = np.random.uniform(-1, +1, size=(2,))
        obs, reward, done, info = env.step(u)
        pos.append(obs[:2])
        goal.append(obs[2:])

    t = np.arange(T)
    pos = np.array(pos)
    goal = np.array(goal)
    plt.scatter(pos[:, 0], pos[:, 1], c=t)
    plt.scatter(goal[:, 0], goal[:, 1])
    plt.axis('equal')
    plt.show()



