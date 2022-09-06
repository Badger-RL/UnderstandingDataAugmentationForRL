import gym
import numpy as np
import scipy.linalg

from matplotlib import pyplot as plt

from augment.rl.algs.td3 import TD3


def dlqr(A,B,Q,R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # first, solve the ricatti equation
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    # compute the LQR gain
    K = scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A)
    return -K

class LQREnv(gym.Env):
    def __init__(
            self,
            n=2,
            sigma=0.00,
    ):

        self.n = n
        self.action_space = gym.spaces.Box(-1, +1, shape=(n,))
        self.observation_space = gym.spaces.Box(-1, +1, shape=(2*n,))
        self.step_num = 0
        self.horizon = 200
        self.sigma = sigma
        self.delta = 0.05

        self.A = np.eye(n)
        self.B = np.eye(n)*self.delta
        self.Q = np.eye(n)
        self.R = np.eye(n)*0.1

        self.trajectory = self.get_trajectory()

    def get_trajectory(self):

        t = np.linspace(0, 2, self.horizon)

        # fixed = 1
        # if fixed:
        #     t = np.ones(self.horizon)

        rx, ry = 1, 1
        trajectory = np.array([
            rx * np.cos(t*np.pi),
            ry * np.sin(t*np.pi),
        ])

        m = trajectory.shape[0]
        trajectory = np.concatenate((trajectory, np.zeros(shape=(self.n-m, self.horizon))))
        trajectory = trajectory.T

        return trajectory

    def cost(self, x, u):
        # Wikipedia has an extra dot product at the end, but it doesn't make sense since x and u don't
        # necessarily share the same dimensionality. All other sources I've looked at don't include this cross term.
        # https://stanford.edu/class/ee363/lectures/dlqr.pdf
        # http://underactuated.mit.edu/lqr.html
        # https://lewisgroup.uta.edu/ee5321/2013%20notes/2%20lqr%20dt%20and%20sampling.pdf
        # ubar = np.linalg.inv(self.B) @ (-self.A @ x + self.goal)

        dist_x = x-self.goal
        # dist_u = u-ubar
        dist_u = u

        cost_goal = dist_x.T @ self.Q @ dist_x
        cost_ctrl = dist_u.T @ self.R @ dist_u
        return cost_goal, cost_ctrl

    def next_state(self, x, u):
        noise = np.random.normal(0, self.sigma)
        next_state = self.A @ x + self.B @ u + noise
        return next_state
    
    def step(self, u):
        self.step_num += 1

        self.x = self.next_state(self.x, u)
        # self.x = np.clip(self.x, -1, +1) # clipping makes dynamics nonlinear
        cost_goal, cost_ctrl = self.cost(self.x, u)
        cost = cost_goal + cost_ctrl

        done = False
        if self.step_num == self.horizon:
            self.step_num = 0
            done = True

        info = {"cost_goal": cost_goal, "cost_ctrl": cost_ctrl}
        
        self.goal = self.trajectory[self.step_num]
        self.obs = np.concatenate((self.x, self.goal))
        return self.obs, -cost, done, info

    def reset(self):

        self.x = np.zeros(self.n)
        # self.x = np.random.uniform(-1, 1, size=(self.n,))
        self.goal = self.trajectory[0]
        self.obs = np.concatenate((self.goal, self.x))
        return self.obs

if __name__ == "__main__":

    T = 200
    env = LQREnv(n=2)
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
    plt.scatter(pos[:,0], pos[:,1], c=t)
    # plt.scatter(goal[:,0], goal[:,1])
    plt.axis('equal')
    plt.show()



