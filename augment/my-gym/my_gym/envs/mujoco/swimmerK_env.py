import gym
import numpy as np
import torch
from gym import utils
#from lxml import etree
from my_gym.envs.mujoco.mujoco_env import MujocoEnv


class SwimmerKEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, num_links=6,
                 latent_dim=-1,
                 model_class=None
                 ):
        self.num_links = num_links
        self.latent_dim = latent_dim
        self.action_dim = self.latent_dim if self.latent_dim>0 else self.num_links-1
        self.action_space = gym.spaces.Box(-1, +1, shape=(self.action_dim,))

        MujocoEnv.__init__(self, f"swimmer{num_links}.xml", 4)

        utils.EzPickle.__init__(self)

    def _get_robot_state(self):
        obs = self._get_obs()
        return obs

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()

        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        info =dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)
        # print(info)
        return ob, reward, False, info

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def get_obs(self):
        return self._get_obs()

    def reset_model(self):
        vel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq),
            vel,
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0 # id of the body to track ()
        self.viewer.cam.distance = self.model.stat.extent * 0.3  # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.lookat[0] += 10.0  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] += 0.0
        self.viewer.cam.lookat[2] += 0.0
        self.viewer.cam.elevation = -20  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        self.viewer.cam.azimuth = 160  # camera rotation around the camera's vertical axis

    def obs_to_q(self, obs):
        qpos = np.zeros(self.num_links+2)
        qpos[2:] = obs[:self.num_links]
        qvel = obs[self.num_links:]
        return qpos, qvel