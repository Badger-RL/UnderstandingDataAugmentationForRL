import copy
import os

import gym
import mujoco_py
import numpy as np
import torch
from gym import spaces
from gym.envs.mujoco.mujoco_env import convert_observation_to_space, DEFAULT_SIZE, MujocoEnv as MujocoEnv_og
from gym.utils import seeding

class MujocoEnv(MujocoEnv_og):
    """
    Superclass for all MuJoCo environments.

    User can pass an optional 'local_xml' argument to the constructor. If local_xml=True,
    we search for the xml file from the local xml directory.
    """

    def __init__(self, model_path, frame_skip):
        local_model_path = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if os.path.exists(local_model_path):
            model_path = local_model_path
        super().__init__(model_path=model_path, frame_skip=frame_skip)

    def _set_action_space(self):
        if self.action_space is None:
            bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
            low, high = bounds.T
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        if self.observation_space is None:
            self.observation_space = convert_observation_to_space(observation)
        return self.observation_space