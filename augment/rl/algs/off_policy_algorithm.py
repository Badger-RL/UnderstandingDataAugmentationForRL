import io
import pathlib
import sys
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
# from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from augment.rl.algs.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, \
    TrainFrequencyUnit, ReplayBufferSamples
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from augment.rl.augmentation_functions.coda import CoDAPanda

OffPolicyAlgorithmAugmentSelf = TypeVar("OffPolicyAlgorithmAugmentSelf", bound="OffPolicyAlgorithmAugment")


class OffPolicyAlgorithmAugment(OffPolicyAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: Whether the model support gSDE or not
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Union[str,
        Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int,
        Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = ReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        aug_function: Optional = None,
        aug_ratio: Optional[Union[float, Schedule]] = None,
        aug_n: Optional[int] = 1,
        aug_buffer: Optional[bool] = True,
        aug_constraint: Optional[float] = None,
        aug_freq: Optional[Union[int, str]] = 1,
        coda_function: Optional = None,
        coda_n: Optional[int] = 1,
    ):

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            policy_kwargs,
            tensorboard_log,
            verbose,
            device,
            support_multi_env,
            monitor_wrapper,
            seed,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            sde_support,
            supported_action_spaces)

        self.aug_function = aug_function
        self.aug_ratio = aug_ratio
        self.aug_n = aug_n
        self.aug_n_floor = int(np.floor(aug_n))
        self.aug_prob = aug_n - self.aug_n_floor
        self.aug_freq = aug_freq
        self.separate_aug_buffer = aug_buffer
        self.aug_constraint = aug_constraint
        self.coda_function = coda_function
        self.coda_n = coda_n
        self.use_coda = self.coda_function is not None

        self.use_aug = self.aug_function is not None
        if self.use_aug or self.use_coda:
            assert self._vec_normalize_env is None
            assert not self.optimize_memory_usage
            self._setup_augmented_replay_buffer()

            self.aug_indices = []
            self.past_infos = []

    def _setup_augmented_replay_buffer(self):
        if self.use_coda and self.use_aug:
            aug_buffer_size = int(self.buffer_size * (self.aug_n+self.coda_n))
        elif self.use_aug:
            aug_buffer_size = int(self.buffer_size * self.aug_n)
        elif self.use_coda:
            aug_buffer_size = int(self.buffer_size * self.coda_n)

        self.aug_replay_buffer = ReplayBuffer(
            aug_buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **self.replay_buffer_kwargs,
        )

    def augment_transition(self, obs, next_obs, action, reward, done, info):
        normalization_constant = 1
        dist = None
        if self.aug_constraint is not None:
            dist = self.replay_buffer.marginal_hist + self.aug_constraint*self.replay_buffer.num_states
            dist /= dist.sum()

        aug_n = self.aug_n_floor + int(np.random.random() < self.aug_prob)
        if aug_n < 1: return None, None, None, None, None, None
        aug_transition = self.aug_function.augment(
            aug_n,
            obs,
            next_obs,
            action,
            reward,
            done,
            info,
            p=dist
            # p=self.replay_buffer.state_counts/self.replay_buffer.num_states
        )

        return aug_transition

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["aug_replay_buffer"]

    def save_aug_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.aug_replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.aug_replay_buffer, self.verbose)

    def load_aug_replay_buffer(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.aug_replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.aug_replay_buffer,
                          ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.aug_replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.aug_replay_buffer.handle_timeout_termination = False
            self.aug_replay_buffer.timeouts = np.zeros_like(self.aug_replay_buffer.dones)

        if isinstance(self.aug_replay_buffer, HerReplayBuffer):
            assert self.env is not None, "You must pass an environment at load time when using `HerReplayBuffer`"
            self.aug_replay_buffer.set_env(self.get_env())
            if truncate_last_traj:
                self.aug_replay_buffer.truncate_last_trajectory()

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).
        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            if self.use_coda:
                self._coda()

            if self.use_aug:
                self.aug_indices.append(self.replay_buffer.size()-1)
                self.past_infos.append(infos)
                if self.aug_freq == 'episode':
                    do_aug = dones.all()
                else:
                    do_aug = (self.num_timesteps % self.aug_freq == 0) or dones.all()

                if do_aug:
                    env_indices = np.random.randint(0, high=self.n_envs, size=(len(self.aug_indices),))
                    obs = self.replay_buffer.observations[self.aug_indices, env_indices, :]
                    next_obs = self.replay_buffer.next_observations[self.aug_indices, env_indices, :]
                    actions = self.replay_buffer.actions[self.aug_indices, env_indices, :]
                    dones = self.replay_buffer.dones[self.aug_indices, env_indices]
                    # timeouts = self.replay_buffer.timeouts[self.aug_indices, env_indices]
                    rewards = self.replay_buffer.rewards[self.aug_indices, env_indices]

                    unscaled_actions = self.policy.unscale_action(actions)
                    aug_obs, aug_next_obs, aug_unscaled_action, aug_reward, aug_done, aug_info = self.augment_transition(
                        obs,
                        next_obs,
                        unscaled_actions,
                        rewards,
                        dones,
                        self.past_infos,
                    )
                    if aug_obs is not None: # When aug_n < 1, we only augment a fraction of transitions.
                        aug_action = self.policy.scale_action(aug_unscaled_action)
                        self.aug_replay_buffer.extend(aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info)
                    self.aug_indices.clear()
                    self.past_infos.clear()

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def sample_replay_buffers(self):
        alpha = 0
        if self.use_aug:
            alpha = self.aug_ratio(self._current_progress_remaining, self.num_timesteps)

        if alpha >= 0:
            observed_batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
            if alpha == 0:
                return observed_batch, None, None

        aug_batch_size = int(np.abs(alpha) * self.batch_size)
        if aug_batch_size == 0:
            aug_batch_size = 1 if np.random.uniform(0, 1) < np.abs(alpha) else 0
        aug_batch = self.aug_replay_buffer.sample(aug_batch_size, env=self._vec_normalize_env)
        if alpha < 0:
            return aug_batch

        observations = th.concat((observed_batch.observations, aug_batch.observations))
        actions = th.concat((observed_batch.actions, aug_batch.actions))
        next_observations = th.concat((observed_batch.next_observations, aug_batch.next_observations))
        rewards = th.concat((observed_batch.rewards, aug_batch.rewards))
        dones = th.concat((observed_batch.dones, aug_batch.dones))

        return ReplayBufferSamples(observations, actions, next_observations, dones, rewards), observed_batch, aug_batch

    def _coda(self):
        num_coda_samples_made = 0
        while num_coda_samples_made < self.coda_n:
            observations, actions, next_observations, rewards, dones, timeouts = self.replay_buffer.sample_array(
                batch_size=2)


            aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info = self.coda_function.augment(
                observations[0],
                next_observations[0],
                actions[0],
                observations[1],
                next_observations[1],
                rewards[1],
                dones[1],
                timeouts[1]
            )

            if aug_obs is not None:  # When aug_n < 1, we only augment a fraction of transitions.
                self.aug_replay_buffer.add(aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info)
                num_coda_samples_made += 1