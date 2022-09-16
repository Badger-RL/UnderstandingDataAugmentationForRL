import gym
import numpy as np
import scipy.linalg

from matplotlib import pyplot as plt

from augment.rl.algs.td3 import TD3

class PredatorPreyEnv(gym.Env):
    def __init__(
            self,
            n=2,
            sigma=0.00,
            delta=0.05,
            horizon=100
    ):

        self.n = n
        # self.action_space = gym.spaces.Box(-1, +1, shape=(n,))
        self.action_space = gym.spaces.Box(low=np.zeros(2), high=np.array([1, 2 * np.pi]), shape=(n,))
        self.observation_space = gym.spaces.Box(-1, +1, shape=(2 * n,))
        self.step_num = 0
        self.horizon = horizon
        self.sigma = sigma
        self.delta = delta

        self.sparse = True


    def step(self, u):
        self.step_num += 1
        ux = u[0] * np.cos(u[1])
        uy = u[0] * np.sin(u[1])
        u = np.array([ux, uy])

        self.x += u*self.delta
        self.x = np.clip(self.x, -1, +1) # clipping makes dynamics nonlinear

        done = False
        reward = -0.1

        if self.sparse and np.linalg.norm(self.x - self.goal) < 0.05:
            reward = +1.0
            done = True

        # if self.step_num == self.horizon:
        #     done = True

        info = {}
        self.obs = np.concatenate((self.x, self.goal))
        return self.obs, reward, done, info

    def reset(self):
        self.step_num = 0
        # self.x = np.zeros(self.n)
        self.x = np.random.uniform(-1, 1, size=(self.n,))

        theta = np.random.uniform(-np.pi, +np.pi)
        r = np.random.uniform(-1, +1)
        self.goal = np.array([r * np.cos(theta), r * np.sin(theta)])
        self.goal = np.random.uniform(-1, 1, size=(self.n,))

        self.obs = np.concatenate((self.x, self.goal))
        return self.obs

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



