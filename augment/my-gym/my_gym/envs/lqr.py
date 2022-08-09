import gym
import numpy as np
import scipy.linalg

from matplotlib import pyplot as plt


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
            n=1,
            sigma=0.00,
            exponent=0.33,
            hardtanh=None,
    ):

        self.n = n
        self.action_space = gym.spaces.Box(-1, +1, shape=(n,))
        self.observation_space = gym.spaces.Box(-1, +1, shape=(2*n,))
        self.step_num = 0
        self.horizon = 200
        self.sigma = sigma
        self.hardtanh = hardtanh
        self.exponent = exponent
        self.discontinuous = False

        self.delta = 0.05

        np.random.seed(0)
        self.A = np.eye(n)
        self.B = np.eye(n)*self.delta
        # self.Q = np.eye(n)
        self.Q = np.eye(n)
        self.R = np.eye(n)*0.1

        self.trajectory = self.get_trajectory()

    def get_trajectory(self):

        t = np.linspace(0, 4, self.horizon)


        fixed = 1
        if fixed:
            t = np.ones(self.horizon)
            # t[self.horizon//2:] = -1
        rx = 1
        ry = 0.5


        # t = np.ones(100)

        trajectory = np.array([
            t,
            # ry * np.sin(t*np.pi/2),
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

        # if x > -0.5 and x < 0.5:
        #     cost_goal = 5

        return cost_goal, cost_ctrl

    def next_state(self, x, u):
        noise = np.random.normal(0, self.sigma)
        # x -= self.goal

        next_state = self.A @ x + self.B @ u + noise
        return next_state


    def step(self, u):

        u = np.clip(u, -1, +1)
        # u -= 0.5
        u = self._action_map(u)

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
        self.state = np.concatenate((self.goal, self.x))
        return self.state, -cost, done, info

    def reset(self):

        self.x = np.zeros(self.n)
        self.x[0] = np.random.uniform(-1, 1)
        self.x[0] = -1
        # if self.x[0] < 0:
        #     self.trajectory *= -1
        self.goal = self.trajectory[0]
        self.state = np.concatenate((self.goal, self.x))
        return self.state

    # def optimal(self):

    def _action_map(self, a):

        if self.hardtanh:
            mask_lo  = a < -self.hardtanh
            mask_hi  = a > +self.hardtanh
            mask_mid = ~mask_lo & ~mask_hi

            a_new = np.empty_like(a)

            a_new[mask_hi] = 1
            a_new[mask_lo] = -1
            a_new[mask_mid] = a[mask_mid]/self.hardtanh
        elif self.exponent:
            neg = a < 0
            pos = a > 0
            a[pos] = a[pos]**self.exponent
            a[neg] = -(-a[neg])**self.exponent
            a_new = a
            # a_new = a**self.exponent
        elif self.discontinuous:
            mask_lo  = a < 0
            mask_hi  = a > 0

            a_new = np.copy(a)
            a_new[mask_lo] += 1
            a_new[mask_hi] -= 1
        else:
            a_new = a

        return a_new








