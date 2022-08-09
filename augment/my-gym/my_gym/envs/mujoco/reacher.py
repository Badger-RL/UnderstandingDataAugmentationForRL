import numpy as np

from gym import utils
from my_gym.envs.mujoco import mujoco_env


class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    ### Description
    "Reacher" is a two-jointed robot arm. The goal is to move the robot's end effector (called *fingertip*) close to a
    target that is spawned at a random position.

    ### Action Space
    The action space is a `Box(-1, 1, (2,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.

    | Num | Action                                                                          | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
    |-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------|-------|------|
    | 0   | Torque applied at the first hinge (connecting the link to the point of fixture) | -1 | 1 | joint0  | hinge | torque (N m) |
    | 1   |  Torque applied at the second hinge (connecting the two links)                  | -1 | 1 | joint1  | hinge | torque (N m) |

    ### Observation Space

    Observations consist of

    - The cosine of the angles of the two arms
    - The sine of the angles of the two arms
    - The coordinates of the target
    - The angular velocities of the arms
    - The vector between the target and the reacher's fingertip (3 dimensional with the last element being 0)

    The observation is a `ndarray` with shape `(11,)` where the elements correspond to the following:

    | Num | Observation           | Min                  | Max                | Name (in corresponding XML file) | Joint| Unit |
    |-----|-----------------------|----------------------|--------------------|----------------------|--------------------|--------------------|
    | 0   | cosine of the angle of the first arm                                                            | -Inf                 | Inf                | cos(joint0) | hinge | unitless |
    | 1   | cosine of the angle of the second arm                                                           | -Inf                 | Inf                | cos(joint1) | hinge | unitless |
    | 2   | sine of the angle of the first arm                                                              | -Inf                 | Inf                | cos(joint0) | hinge | unitless |
    | 3   | sine of the angle of the second arm                                                             | -Inf                 | Inf                | cos(joint1) | hinge | unitless |
    | 4   |  x-coorddinate of the target                                                                    | -Inf                 | Inf                | target_x | slide | position (m) |
    | 5   |  y-coorddinate of the target                                                                    | -Inf                 | Inf                | target_y | slide | position (m) |
    | 6   | angular velocity of the first arm                                                               | -Inf                 | Inf                | joint0 | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of the second arm                                                              | -Inf                 | Inf                | joint1 | hinge | angular velocity (rad/s) |
    | 8   | x-value of position_fingertip - position_target                                                 | -Inf                 | Inf                | NA | slide | position (m) |
    | 9   | y-value of position_fingertip - position_target                                                 | -Inf                 | Inf                | NA | slide | position (m) |
    | 10  | z-value of position_fingertip - position_target (0 since reacher is 2d and z is same for both)  | -Inf                 | Inf                | NA | slide | position (m) |


    Most Gym environments just return the positions and velocity of the
    joints in the `.xml` file as the state of the environment. However, in
    reacher the state is created by combining only certain elements of the
    position and velocity, and performing some function transformations on them.
    If one is to read the `.xml` for reacher then they will find 4 joints:

    | Num | Observation                 | Min      | Max      | Name (in corresponding XML file) | Joint | Unit               |
    |-----|-----------------------------|----------|----------|----------------------------------|-------|--------------------|
    | 0   | angle of the first arm      | -Inf     | Inf      | joint0                           | hinge | angle (rad)        |
    | 1   | angle of the second arm     | -Inf     | Inf      | joint1                           | hinge | angle (rad)        |
    | 2   | x-coordinate of the target  | -Inf     | Inf      | target_x                         | slide | position (m)       |
    | 3   | y-coordinate of the target  | -Inf     | Inf      | target_y                         | slide | position (m)       |


    ### Rewards
    The reward consists of two parts:
    - *reward_distance*: This reward is a measure of how far the *fingertip*
    of the reacher (the unattached end) is from the target, with a more negative
    value assigned for when the reacher's *fingertip* is further away from the
    target. It is calculated as the negative vector norm of (position of
    the fingertip - position of target), or *-norm("fingertip" - "target")*.
    - *reward_control*: A negative reward for penalising the walker if
    it takes actions that are too large. It is measured as the negative squared
    Euclidean norm of the action, i.e. as *- sum(action<sup>2</sup>)*.

    The total reward returned is ***reward*** *=* *reward_distance + reward_control*

    Unlike other environments, Reacher does not allow you to specify weights for the individual reward terms.
    However, `info` does contain the keys *reward_dist* and *reward_ctrl*. Thus, if you'd like to weight the terms,
    you should create a wrapper that computes the weighted reward from `info`.


    ### Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with a noise added for stochasticity. A uniform noise in the range
    [-0.1, 0.1] is added to the positional attributes, while the target position
    is selected uniformly at random in a disk of radius 0.2 around the origin.
    Independent, uniform noise in the
    range of [-0.005, 0.005] is added to the velocities, and the last
    element ("fingertip" - "target") is calculated at the end once everything
    is set. The default setting has a framerate of 2 and a *dt = 2 * 0.01 = 0.02*

    ### Episode Termination

    The episode terminates when any of the following happens:

    1. The episode duration reaches a 50 timesteps (with a new random target popping up if the reacher's fingertip reaches it before 50 timesteps)
    2. Any of the state space values is no longer finite.

    ### Arguments

    No additional arguments are currently supported (in v2 and lower),
    but modifications can be made to the XML file in the assets folder
    (or by changing the path to a modified XML file in another folder)..

    ```
    env = gym.make('Reacher-v2')
    ```

    There is no v3 for Reacher, unlike the robot environments where a v3 and
    beyond take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.


    ### Version History

    * v2: All continuous control environments now use mujoco_py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks (not including reacher, which has a max_time_steps of 50). Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)

    """

    def __init__(self,
                 latent_dim=-1,
                 model_class=None
                 ):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "reacher.xml", 2, id='Reacher-v2', latent_dim=latent_dim, model_class=model_class)

    def _get_robot_state(self):
        obs = self._get_obs()
        return obs[:2]

    def step(self, a):
        if self.model_class is not None:
            state = None
            if 'C' in self.model_class:
                state = self._get_obs()
            a = self.latent_to_native_mapping(s=state, z=a)

        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )
