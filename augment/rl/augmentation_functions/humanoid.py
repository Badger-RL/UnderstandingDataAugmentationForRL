"""
### Description

This environment is based on the environment introduced by Tassa, Erez and Todorov
in ["Synthesis and stabilization of complex behaviors through online trajectory optimization"](https://ieeexplore.ieee.org/document/6386025).
The 3D bipedal robot is designed to simulate a human. It has a torso (abdomen) with a pair of
legs and arms. The legs each consist of two links, and so the arms (representing the knees and
elbows respectively). The goal of the environment is to walk forward as fast as possible without falling over.

### Action Space
The action space is a `Box(-1, 1, (17,), float32)`. An action represents the torques applied at the hinge joints.

| Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
|-----|----------------------|---------------|----------------|---------------------------------------|-------|------|
| 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4 | 0.4 | hip_1 (front_left_leg)      | hinge | torque (N m) |
| 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4 | 0.4 | angle_1 (front_left_leg)    | hinge | torque (N m) |
| 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4 | 0.4 | hip_2 (front_right_leg)     | hinge | torque (N m) |
| 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4 | 0.4 | right_hip_x (right_thigh)   | hinge | torque (N m) |
| 4   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4 | 0.4 | right_hip_z (right_thigh)   | hinge | torque (N m) |
| 5   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4 | 0.4 | right_hip_y (right_thigh)   | hinge | torque (N m) |
| 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4 | 0.4 | right_knee                  | hinge | torque (N m) |
| 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4 | 0.4 | left_hip_x (left_thigh)     | hinge | torque (N m) |
| 8   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4 | 0.4 | left_hip_z (left_thigh)     | hinge | torque (N m) |
| 9   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4 | 0.4 | left_hip_y (left_thigh)     | hinge | torque (N m) |
| 10   | Torque applied on the rotor between the left hip/thigh and the left shin          | -0.4 | 0.4 | left_knee                   | hinge | torque (N m) |
| 11   | Torque applied on the rotor between the torso and right upper arm (coordinate -1) | -0.4 | 0.4 | right_shoulder1             | hinge | torque (N m) |
| 12   | Torque applied on the rotor between the torso and right upper arm (coordinate -2) | -0.4 | 0.4 | right_shoulder2             | hinge | torque (N m) |
| 13   | Torque applied on the rotor between the right upper arm and right lower arm       | -0.4 | 0.4 | right_elbow                 | hinge | torque (N m) |
| 14   | Torque applied on the rotor between the torso and left upper arm (coordinate -1)  | -0.4 | 0.4 | left_shoulder1              | hinge | torque (N m) |
| 15   | Torque applied on the rotor between the torso and left upper arm (coordinate -2)  | -0.4 | 0.4 | left_shoulder2              | hinge | torque (N m) |
| 16   | Torque applied on the rotor between the left upper arm and left lower arm         | -0.4 | 0.4 | left_elbow                  | hinge | torque (N m) |

### Observation Space

Observations consist of positional values of different body parts of the Humanoid,
 followed by the velocities of those individual parts (their derivatives) with all the
 positions ordered before all the velocities.

By default, observations do not include the x- and y-coordinates of the torso. These may
be included by passing `exclude_current_positions_from_observation=False` during construction.
In that case, the observation space will have 378 dimensions where the first two dimensions
represent the x- and y-coordinates of the torso.
Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the x- and y-coordinates
will be returned in `info` with keys `"x_position"` and `"y_position"`, respectively.

However, by default, the observation is a `ndarray` with shape `(376,)` where the elements correspond to the following:

| Num | Observation                                                                                                     | Min  | Max | Name (in corresponding XML file) | Joint | Unit                       |
| --- | --------------------------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | -------------------------- |
| 0   | z-coordinate of the torso (centre)                                                                              | -Inf | Inf | root                             | free  | position (m)               |
| 1   | x-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 2   | y-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 3   | z-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 4   | w-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
| 5   | z-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_z                        | hinge | angle (rad)                |
| 6   | y-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_y                        | hinge | angle (rad)                |
| 7   | x-angle of the abdomen (in pelvis)                                                                              | -Inf | Inf | abdomen_x                        | hinge | angle (rad)                |
| 8   | x-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_x                      | hinge | angle (rad)                |
| 9   | z-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_z                      | hinge | angle (rad)                |
| 19  | y-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_y                      | hinge | angle (rad)                |
| 11  | angle between right hip and the right shin (in right_knee)                                                      | -Inf | Inf | right_knee                       | hinge | angle (rad)                |
| 12  | x-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_x                       | hinge | angle (rad)                |
| 13  | z-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_z                       | hinge | angle (rad)                |
| 14  | y-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_y                       | hinge | angle (rad)                |
| 15  | angle between left hip and the left shin (in left_knee)                                                         | -Inf | Inf | left_knee                        | hinge | angle (rad)                |
| 16  | coordinate-1 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder1                  | hinge | angle (rad)                |
| 17  | coordinate-2 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder2                  | hinge | angle (rad)                |
| 18  | angle between right upper arm and right_lower_arm                                                               | -Inf | Inf | right_elbow                      | hinge | angle (rad)                |
| 19  | coordinate-1 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder1                   | hinge | angle (rad)                |
| 20  | coordinate-2 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder2                   | hinge | angle (rad)                |
| 21  | angle between left upper arm and left_lower_arm                                                                 | -Inf | Inf | left_elbow                       | hinge | angle (rad)                |
| 22  | x-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 23  | y-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 24  | z-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
| 25  | x-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | anglular velocity (rad/s)  |
| 26  | y-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | anglular velocity (rad/s)  |
| 27  | z-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | anglular velocity (rad/s)  |
| 28  | z-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_z                        | hinge | anglular velocity (rad/s)  |
| 29  | y-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_y                        | hinge | anglular velocity (rad/s)  |
| 30  | x-coordinate of angular velocity of the abdomen (in pelvis)                                                     | -Inf | Inf | abdomen_x                        | hinge | aanglular velocity (rad/s) |
| 31  | x-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_x                      | hinge | anglular velocity (rad/s)  |
| 32  | z-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_z                      | hinge | anglular velocity (rad/s)  |
| 33  | y-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_y                      | hinge | anglular velocity (rad/s)  |
| 34  | angular velocity of the angle between right hip and the right shin (in right_knee)                              | -Inf | Inf | right_knee                       | hinge | anglular velocity (rad/s)  |
| 35  | x-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_x                       | hinge | anglular velocity (rad/s)  |
| 36  | z-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_z                       | hinge | anglular velocity (rad/s)  |
| 37  | y-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_y                       | hinge | anglular velocity (rad/s)  |
| 38  | angular velocity of the angle between left hip and the left shin (in left_knee)                                 | -Inf | Inf | left_knee                        | hinge | anglular velocity (rad/s)  |
| 39  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder1                  | hinge | anglular velocity (rad/s)  |
| 40  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder2                  | hinge | anglular velocity (rad/s)  |
| 41  | angular velocity of the angle between right upper arm and right_lower_arm                                       | -Inf | Inf | right_elbow                      | hinge | anglular velocity (rad/s)  |
| 42  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder1                   | hinge | anglular velocity (rad/s)  |
| 43  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder2                   | hinge | anglular velocity (rad/s)  |
| 44  | angular velocitty of the angle between left upper arm and left_lower_arm                                        | -Inf | Inf | left_elbow                       | hinge | anglular velocity (rad/s)  |

Additionally, after all the positional and velocity based values in the table,
the observation contains (in order):
- *cinert:* Mass and inertia of a single rigid body relative to the center of mass
(this is an intermediate result of transition). It has shape 14*10 (*nbody * 10*)
and hence adds to another 140 elements in the state space.
- *cvel:* Center of mass based velocity. It has shape 14 * 6 (*nbody * 6*) and hence
adds another 84 elements in the state space
- *qfrc_actuator:* Constraint force generated as the actuator force. This has shape
`(23,)`  *(nv * 1)* and hence adds another 23 elements to the state space.
- *cfrc_ext:* This is the center of mass based external force on the body.  It has shape
14 * 6 (*nbody * 6*) and hence adds to another 84 elements in the state space.
where *nbody* stands for the number of bodies in the robot and *nv* stands for the
number of degrees of freedom (*= dim(qvel)*)

The (x,y,z) coordinates are translational DOFs while the orientations are rotational
DOFs expressed as quaternions. One can read more about free joints on the
[Mujoco Documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).

**Note:** Humanoid-v4 environment no longer has the following contact forces issue.
If using previous Humanoid versions from v4, there have been reported issues that using a Mujoco-Py version > 2.0
results in the contact forces always being 0. As such we recommend to use a Mujoco-Py
version < 2.0 when using the Humanoid environment if you would like to report results
with contact forces (if contact forces are not used in your experiments, you can use
version > 2.0).

### Rewards
The reward consists of three parts:
- *healthy_reward*: Every timestep that the humanoid is alive (see section Episode Termination for definition), it gets a reward of fixed value `healthy_reward`
- *forward_reward*: A reward of walking forward which is measured as *`forward_reward_weight` *
(average center of mass before action - average center of mass after action)/dt*.
*dt* is the time between actions and is dependent on the frame_skip parameter
(default is 5), where the frametime is 0.003 - making the default *dt = 5 * 0.003 = 0.015*.
This reward would be positive if the humanoid walks forward (in positive x-direction). The calculation
for the center of mass is defined in the `.py` file for the Humanoid.
- *ctrl_cost*: A negative reward for penalising the humanoid if it has too
large of a control force. If there are *nu* actuators/controls, then the control has
shape  `nu x 1`. It is measured as *`ctrl_cost_weight` * sum(control<sup>2</sup>)*.
- *contact_cost*: A negative reward for penalising the humanoid if the external
contact force is too large. It is calculated by clipping
*`contact_cost_weight` * sum(external contact force<sup>2</sup>)* to the interval specified by `contact_cost_range`.

The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - contact_cost* and `info` will also contain the individual reward terms

### Starting State
All observations start in state
(0.0, 0.0,  1.4, 1.0, 0.0  ... 0.0) with a uniform noise in the range
of [-`reset_noise_scale`, `reset_noise_scale`] added to the positional and velocity values (values in the table)
for stochasticity. Note that the initial z coordinate is intentionally
selected to be high, thereby indicating a standing up humanoid. The initial
orientation is designed to make it face forward as well.

### Episode End
The humanoid is said to be unhealthy if the z-position of the torso is no longer contained in the
closed interval specified by the argument `healthy_z_range`.

If `terminate_when_unhealthy=True` is passed during construction (which is the default),
the episode ends when any of the following happens:

1. Truncation: The episode duration reaches a 1000 timesteps
3. Termination: The humanoid is unhealthy

If `terminate_when_unhealthy=False` is passed, the episode is ended only when 1000 timesteps are exceeded.

### Arguments

No additional arguments are currently supported in v2 and lower.

```
env = gym.make('Humanoid-v4')
```

v3 and v4 take gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc.

```
env = gym.make('Humanoid-v4', ctrl_cost_weight=0.1, ....)
```

| Parameter                                    | Type      | Default          | Description                                                                                                                                                               |
| -------------------------------------------- | --------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `xml_file`                                   | **str**   | `"humanoid.xml"` | Path to a MuJoCo model                                                                                                                                                    |
| `forward_reward_weight`                      | **float** | `1.25`           | Weight for _forward_reward_ term (see section on reward)                                                                                                                  |
| `ctrl_cost_weight`                           | **float** | `0.1`            | Weight for _ctrl_cost_ term (see section on reward)                                                                                                                       |
| `contact_cost_weight`                        | **float** | `5e-7`           | Weight for _contact_cost_ term (see section on reward)                                                                                                                    |
| `healthy_reward`                             | **float** | `5.0`            | Constant reward given if the humanoid is "healthy" after timestep                                                                                                         |
| `terminate_when_unhealthy`                   | **bool**  | `True`           | If true, issue a done signal if the z-coordinate of the torso is no longer in the `healthy_z_range`                                                                       |
| `healthy_z_range`                            | **tuple** | `(1.0, 2.0)`     | The humanoid is considered healthy if the z-coordinate of the torso is in this range                                                                                      |
| `reset_noise_scale`                          | **float** | `1e-2`           | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                            |
| `exclude_current_positions_from_observation` | **bool**  | `True`           | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |

### Version History

* v4: all mujoco environments now use the mujoco bindings in mujoco>=2.1.3
* v3: support for gym.make kwargs such as xml_file, ctrl_cost_weight, reset_noise_scale etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
* v2: All continuous control environments now use mujoco_py >= 1.50
* v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
* v0: Initial versions release (1.0.0)
"""
import time
from typing import Dict, List, Any

import gym
import numpy as np

import my_gym
from augment.rl.augmentation_functions.augmentation_function import AugmentationFunction
from augment.rl.augmentation_functions.validate import validate_augmentation


class HumanoidReflect(AugmentationFunction):

    def __init__(self,  noise='uniform', **kwargs):
        super().__init__()
        self.obs_permute = np.arange(45)
        # leg angles
        self.obs_permute[8] = 12
        self.obs_permute[9] = 13
        self.obs_permute[10] = 14
        self.obs_permute[11] = 15
        self.obs_permute[12] = 8
        self.obs_permute[13] = 9
        self.obs_permute[14] = 10
        self.obs_permute[15] = 11
        self.obs_permute[16] = 19
        self.obs_permute[17] = 20
        self.obs_permute[18] = 21
        self.obs_permute[19] = 16
        self.obs_permute[20] = 17
        self.obs_permute[21] = 18

        # joint vels
        self.obs_permute[31] = 35
        self.obs_permute[32] = 36
        self.obs_permute[33] = 37
        self.obs_permute[34] = 38
        self.obs_permute[35] = 31
        self.obs_permute[36] = 32
        self.obs_permute[37] = 33
        self.obs_permute[38] = 34
        self.obs_permute[39] = 42
        self.obs_permute[40] = 43
        self.obs_permute[41] = 44
        self.obs_permute[42] = 39
        self.obs_permute[43] = 40
        self.obs_permute[44] = 41

        self.obs_reflect = np.zeros(45, dtype=bool)
        # self.obs_reflect[1] = True # x orientation of torso
        self.obs_reflect[2] = True
        self.obs_reflect[4:5+1] = True
        self.obs_reflect[7] = True # x angle of abdomen
        self.obs_reflect[16:17+1] = True
        self.obs_reflect[19:20+1] = True

        self.obs_reflect[23] = True
        self.obs_reflect[25] = True
        self.obs_reflect[27] = True
        self.obs_reflect[28] = True
        self.obs_reflect[30] = True
        self.obs_reflect[39] = True
        self.obs_reflect[40] = True
        self.obs_reflect[42] = True
        self.obs_reflect[43] = True

        self.action_permute = np.arange(17)
        self.action_permute[3:6+1] = np.array([7,8,9,10])
        self.action_permute[7:10+1] = np.array([3,4,5,6])

        self.action_permute[11:13+1] = np.array([14,15,16])
        self.action_permute[14:16+1] = np.array([11,12,13])

        self.action_reflect = np.zeros(17, dtype=bool)
        self.action_reflect[1:2+1] = True
        self.action_reflect[11:12+1] = True
        self.action_reflect[14:15+1] = True

    def reflect_action(self, action):
        action[:, :] = action[:, self.action_permute]
        action[:, self.action_reflect] *= -1

    def reflect_obs(self, obs):
        obs[:, :] = obs[:, self.obs_permute]
        obs[:, self.obs_reflect] *= -1

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None
                 ):
        self.reflect_obs(obs)
        self.reflect_obs(next_obs)
        self.reflect_action(action)

        return obs, next_obs, action, reward, done, infos



class HumanoidRotate(AugmentationFunction):

    def __init__(self, noise_scale=np.pi/4, **kwargs):
        super().__init__(**kwargs)
        self.noise_scale = noise_scale
        self.forward_reward_weight = 1.25

    def quat_mul(self, quat0, quat1):
        assert quat0.shape == quat1.shape
        assert quat0.shape[-1] == 4

        # mujoco stores quats as (qw, qx, qy, qz)
        w0 = quat0[..., 3]
        x0 = quat0[..., 0]
        y0 = quat0[..., 1]
        z0 = quat0[..., 2]

        w1 = quat1[..., 3]
        x1 = quat1[..., 0]
        y1 = quat1[..., 1]
        z1 = quat1[..., 2]

        w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
        z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
        quat = np.stack([x, y, z, w], axis=-1)

        assert quat.shape == quat0.shape
        return quat

    def _rotate_torso(self, obs, quat_rotate_by):
        quat_curr = obs[0, 1:4+1]
        quat_result = self.quat_mul(quat0=quat_curr, quat1=quat_rotate_by)
        # quat already normalized
        obs[0, 1:4+1] = quat_result

    def _rotate_vel(self, obs, sin, cos):
        x = obs[:, 22].copy()
        y = obs[:, 23].copy()
        obs[:, 22] = x * cos - y * sin
        obs[:, 23] = x * sin + y * cos

    def _augment(self,
                obs: np.ndarray,
                next_obs: np.ndarray,
                action: np.ndarray,
                reward: np.ndarray,
                done: np.ndarray,
                infos: List[Dict[str, Any]],
                delta = None,
                p=None
                ):

        assert obs.shape[0] == 1 # for now.
        alpha = np.random.uniform(low=-self.noise_scale, high=+self.noise_scale)
        sin = np.sin(alpha/2)
        cos = np.cos(alpha/2)

        # mujoco stores quats as (qw, qx, qy, qz)
        quat_rotate_by = np.array([sin, 0, 0, cos])

        self._rotate_torso(obs, quat_rotate_by)
        self._rotate_torso(next_obs, quat_rotate_by)

        # Not sure why we need -alpha here...
        sin = np.sin(-alpha)
        cos = np.cos(-alpha)
        self._rotate_vel(obs, sin, cos)
        self._rotate_vel(next_obs, sin, cos)
        # self._rotate_vel(obs, sin, cos)
        # self._rotate_vel(next_obs, sin, cos)

        vx = infos[0][0]['x_velocity']
        vy = infos[0][0]['y_velocity']
        reward_forward = infos[0][0]['reward_linvel']

        reward[:] -= reward_forward
        reward[:] += self.forward_reward_weight*(vx*cos - vy*sin)

        return obs, next_obs, action, reward, done, infos

HUMANOID_AUG_FUNCTIONS = {
    'rotate': HumanoidRotate,
    'reflect': HumanoidReflect,
}
def tmp():

    env = gym.make('Humanoid-v4', reset_noise_scale=0)
    f = HumanoidReflect()

    for k in range(1,2):
        action = np.zeros(17, dtype=np.float32).reshape(1,-1)
        action[:, k] = 1
        # action[:, 3:6] = 1
        # action[:, 11:13] = 1

        env.reset()
        # f.reflect_action(action)
        print(action)
        for i in range(200):
            next_obs, reward, terminated, truncated, info = env.step(action[0])
        true = next_obs.copy()
        aug = next_obs.copy().reshape(1,-1)
        f.reflect_obs(aug)
        aug.reshape(-1)

        env.reset()
        f.reflect_action(action)
        # action = np.zeros(17, dtype=np.float32).reshape(1,-1)
        # action[:, 16] = 1
        print(action)
        for i in range(200):
            next_obs, reward, terminated, truncated, info = env.step(action[0])
        true_reflect = next_obs.copy()

        print(f'{i}\ttrue\t\ttrue_reflect\taug\tis_close')
        is_close = np.isclose(true_reflect,aug[0])
        for i in range(45):
            print(f'{i}\t{true[i]:.8f}\t{true_reflect[i]:.8f}\t{aug[0][i]:.8f}\t{is_close[i]}')
        print(np.all(is_close))
    # print()
    # time.sleep(2)

def check_valid(env, aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info):

    # set env to aug_obs
    # env = gym.make('Walker2d-v4', render_mode='human')

    # env.reset()
    qpos, qvel = aug_obs[:21+1], aug_obs[22:]
    x = aug_info['x_position']
    y = aug_info['y_position']
    qpos = np.concatenate((np.array([0,0]), qpos))
    env.set_state(qpos, qvel)

    # determine ture next_obs, reward
    next_obs_true, reward_true, terminated_true, truncated_true, info_true = env.step(aug_action)
    print(aug_next_obs[22:23+1])
    print(next_obs_true[22:23+1])
    print(aug_next_obs - next_obs_true)
    print('here', aug_reward-reward_true)
    print(aug_reward, aug_info)
    print(reward_true, info_true)
    assert np.allclose(aug_next_obs, next_obs_true)
    assert np.allclose(aug_reward, reward_true)

if __name__ == "__main__":
    # tmp()
    '''

    '''
    env = gym.make('Humanoid-v4', reset_noise_scale=0)
    aug_func = HumanoidReflect(env=env)
    validate_augmentation(env, aug_func, check_valid)