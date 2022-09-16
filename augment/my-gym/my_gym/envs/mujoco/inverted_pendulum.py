import os

import gym.spaces
import numpy as np

from gym import utils
from gym.envs.mujoco import InvertedPendulumEnv as InvertedPendulumEnv_original

from my_gym.envs.mujoco import mujoco_env


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    ### Description

    This environment is the cartpole environment based on the work done by
    Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can
    solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
    just like in the classic environments but now powered by the Mujoco physics simulator -
    allowing for more complex experiments (such as varying the effects of gravity).
    This environment involves a cart that can moved linearly, with a pole fixed on it
    at one end and having another end free. The cart can be pushed left or right, and the
    goal is to balance the pole on the top of the cart by applying forces on the cart.

    ### Action Space
    The agent take a 1-element vector for actions.

    The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
    the numerical force applied to the cart (with magnitude representing the amount of
    force and sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |

    ### Observation Space

    The state space consists of positional values of different body parts of
    the pendulum system, followed by the velocities of those individual parts (their derivatives)
    with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(4,)` where the elements correspond to the following:

    | Num | Observation           | Min                  | Max                | Name (in corresponding XML file) | Joint| Unit |
    |-----|-----------------------|----------------------|--------------------|----------------------|--------------------|--------------------|
    | 0   | position of the cart along the linear surface | -Inf                 | Inf                | slider | slide | position (m) |
    | 1   | vertical angle of the pole on the cart        | -Inf                 | Inf                | hinge | hinge | angle (rad) |
    | 2   | linear velocity of the cart                   | -Inf                 | Inf                | slider | slide | velocity (m/s) |
    | 3   | angular velocity of the pole on the cart      | -Inf                 | Inf                | hinge | hinge | anglular velocity (rad/s) |


    ### Rewards

    The goal is to make the inverted pendulum stand upright (within a certain angle limit)
    as long as possible - as such a reward of +1 is awarded for each timestep that
    the pole is upright.

    ### Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0) with a uniform noise in the range
    of [-0.01, 0.01] added to the values for stochasticity.

    ### Episode Termination
    The episode terminates when any of the following happens:

    1. The episode duration reaches 1000 timesteps.
    2. Any of the state space values is no longer finite.
    3. The absolutely value of the vertical angle between the pole and the cart is greater than 0.2 radian.

    ### Arguments

    No additional arguments are currently supported.

    ```
    env = gym.make('InvertedPendulum-v2')
    ```
    There is no v3 for InvertedPendulum, unlike the robot environments where a
    v3 and beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.


    ### Version History

    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks (including inverted pendulum)
    * v0: Initial versions release (1.0.0)

    """

    def __init__(self, init_pos=None, rbf_n=None, discrete=False):

        self.init_pos = init_pos
        print(f'init_pos = {init_pos}')

        self.discrete = discrete
        if self.discrete:
            self.action_map = [-0.3, -0.2, -0.1, -0.01, -0.001, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001]
            self.action_space = gym.spaces.Discrete(len(self.action_map))

        self.rbf_n = rbf_n
        if self.rbf_n:
            self.observation_space = gym.spaces.Box(low=-1, high=+1, shape=(self.rbf_n,))
            # self.ob = self.observation_space.sample()

            self.P = np.random.normal(loc=0, scale=1, size=(self.rbf_n,4))
            self.phi = np.random.uniform(low=-np.pi, high=np.pi, size=(self.rbf_n,))

        utils.EzPickle.__init__(self)
        fullpath = os.path.join(os.path.dirname(__file__), "assets", "inverted_pendulum.xml")
        mujoco_env.MujocoEnv.__init__(self, fullpath, 2)


    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        notdone = np.isfinite(obs).all() and (np.abs(self.sim.data.qpos[1]) <= 0.2)
        # ob = ob.reshape(1, -1)
        done = not notdone
        # print(ob)
        reward = 1.0 #- 10*self.sim.expert_data.qvel[0]**2
        return obs, reward, done, {}

    def reset_model(self):
        if self.init_pos:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-0.01, high=0.01
            )
            qpos[0] = self.init_qpos[0] + self.np_random.uniform(low=self.init_pos[0], high=self.init_pos[1])
        else:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-0.01, high=0.01
            )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
        if self.rbf_n:
            obs = self._rbf(obs)
        return obs

    def get_obs(self):
        return self._get_obs()

    def obs_to_q(self, obs):
        print(self.model.nq, self.model.nv)
        qpos = obs[:2]
        qvel = obs[2:]
        return qpos, qvel

    def _rbf(self, obs):
        return np.sin(self.P.dot(obs)/1 + self.phi)
