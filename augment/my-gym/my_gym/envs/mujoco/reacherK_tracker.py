import gym
import numpy as np
from gym import utils

from my_gym.envs.mujoco.mujoco_env import MujocoEnv


class ReacherTrackerEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, num_links=10, goal=None,
                 latent_dim=-1,
                 model_class=None,
                 ):
        self.num_links = num_links
        self.goal = np.array(goal)
        self.randomize_goal = goal is None
        self.step_number = 0

        self.goal_init = np.array([0,0]) # some bogus goal for initialization

        self.max_norm = 0
        utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, model_path=f"reacher_{num_links}dof.xml", frame_skip=2,
                           id=f'ReacherTracker{num_links}-v3', latent_dim=latent_dim, model_class=model_class)

    def _step_goal(self):
        t = self.step_number
        center = np.array([0.7, 0.4]) * self.num_links/10
        rx = 0.1 * self.num_links/10
        ry = 0.2 * self.num_links/10

        x = rx * np.cos(2*np.pi/200 * t) + center[0]
        y = ry * np.sin(2*np.pi/200 * t) + center[1]
        self.goal = np.array([x, y])
        self.sim.data.qpos[-2:] = self.goal
        self.sim.data.qvel[-2:] = np.zeros(2)

        # norm = np.linalg.norm(self.goal)
        # if norm > self.max_norm:
        #     self.max_norm = norm
        #     print(self.max_norm)

    def _get_robot_state(self):
        obs = self._get_obs()
        return obs[:-2]

    def step(self, a):
        if self.model_class is not None:
            state = None
            if 'C' in self.model_class:
                state = self._get_obs()
            a = self.latent_to_native_mapping(s=state, z=a)

        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        # reward_dist = np.exp(-np.linalg.norm(vec))
        # reward_ctrl = 0
        reward_dist = -np.linalg.norm(vec) * self.num_links
        reward_ctrl = -np.square(a).sum()
        # print(a)
        # print(reward_ctrl, reward_dist, sep='\t')
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False # I think the max_steps in teh registration handles the horizon.

        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        # print(info)

        self.step_number += 1
        self._step_goal()

        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1 # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 0.4  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 0.0  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.0
        self.viewer.cam.lookat[2] += 0.0
        self.viewer.cam.elevation = -90  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 0  # camera rotation around the camera's vertical axis

    def reset_model(self):
        self.step_number = 0

        qpos = (
            self.np_random.uniform(low=-0.2, high=0.2, size=self.model.nq)
            + self.init_qpos
        )

        # reset goal
        self.goal = self.goal_init
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
                self.sim.data.qpos.flat[self.num_links:],
                self.sim.data.qvel.flat[:self.num_links],
                self.get_body_com("fingertip") - self.get_body_com("target"),
            ]
        )