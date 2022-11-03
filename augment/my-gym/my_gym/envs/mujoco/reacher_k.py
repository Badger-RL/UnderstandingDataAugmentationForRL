import os

import gym
import numpy as np
from gym import utils
from my_gym.envs.mujoco.mujoco_env import MujocoEnv



class ReacherEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, num_links=2, goal=None, sparse=0, rand_central_angle=False, rbf_n=500):
        self.num_links = num_links
        self.goal = np.array(goal)
        self.randomize_goal = goal is None
        self.sparse = sparse
        self.rand_central_angle = rand_central_angle
        self.rbf_n = None

        utils.EzPickle.__init__(self)
        fullpath = os.path.join(os.path.dirname(__file__), "assets", f"reacher_{num_links}dof.xml")
        MujocoEnv.__init__(self, model_path=fullpath, frame_skip=2)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        theta = self.sim.data.qpos.flat[:self.num_links]
        if self.sparse:
            reward_dist = np.linalg.norm(vec) < 0.05
            reward_ctrl = 0
            reward = reward_dist
        else:
            reward_dist = -np.linalg.norm(vec) * self.num_links
            reward_ctrl = -np.square(a).sum()
            reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False

        theta_next = self.sim.data.qpos.flat[:self.num_links]
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, theta=theta, theta_next=theta_next)
        return ob, reward, done, info

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )

        if self.rand_central_angle:
            qpos[0] = np.random.uniform(-np.pi, np.pi)

        if self.randomize_goal:
            while True:
                r = self.num_links*0.1
                self.goal = np.random.uniform(-r, r, size=(2,))
                if np.linalg.norm(self.goal) < 1: break
            qpos[-2:] = self.goal
        else:
            qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:self.num_links]
        ob = np.concatenate(
            [
                np.cos(theta), # 4 cos(joint angles)
                np.sin(theta), # 4 sin(joint angles
                self.sim.data.qpos.flat[self.num_links:], # 2 target
                self.sim.data.qvel.flat[:self.num_links], # 4 joint velocities
                self.get_body_com("fingertip") - self.get_body_com("target"), # (x,y,z) distance
            ]
        )
        if self.rbf_n:
            ob = self._rbf(ob)
        return ob

    def obs_to_q(self, obs):
        k = self.num_links

        qpos = np.arctan2(obs[k:2*k], obs[:k]) # joint angles = arctan(sin/cos)
        qpos = np.concatenate((qpos, obs[2*k:2*k+2])) # (joint angles, target pos)
        qvel = np.concatenate((obs[2*k+2:-3], np.zeros(2))) # (joint vel, target vel)

        return qpos, qvel

    def get_obs(self):
        return self._get_obs()