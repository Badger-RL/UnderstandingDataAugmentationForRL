from typing import Any, Dict, List, Union, Optional
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer as ReplayBuffer_sb3
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

class ReplayBuffer(ReplayBuffer_sb3):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination)

        self.n_bins = 100
        self.bin_width = 2/self.n_bins
        # self.hist = np.zeros((self.n_bins,)*self.observation_space.shape[-1])
        self.marginal_hist = np.zeros((self.observation_space.shape[-1], self.n_bins))

        self.num_states = 0

    def _which_bin(self, x):
        return np.clip(int((x+1)/self.bin_width), 0, 99)

    def update_hists(self, states):
        assert states.shape[0] == 1

        # bin = self._which_bin(states[:,0])
        # self.state_counts[bin] += 1
        # self.num_states += len(states)

        bins = []
        for i in range(self.observation_space.shape[-1]):
            bin = self._which_bin(states[:,i])
            self.marginal_hist[i, bin] += 1
            bins.append(bin)
        self.num_states += 1

        # self.hist[bins[0], bins[1], bins[2], bins[3]] += 1

        #
        # bins = np.array(bins, dtype=int)
        # self.state_counts[bins] += 1
        # x = self.state_counts[bins]
        # self.num_states += 1

        # counts, bins = np.histogramdd(states, bins=self.n_bins, range=[[-1,1] for i in range(4)],)
        # self.state_counts += counts
        # self.num_states += len(states)
