import io
import pathlib
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

# from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import save_to_pkl, load_from_pkl
from stable_baselines3.common.type_aliases import GymEnv, Schedule, ReplayBufferSamples, RolloutReturn, \
    TrainFrequencyUnit, TrainFreq
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecNormalize, VecEnv

# from augment.rl.algs.buffers import ReplayBuffer
from augment.rl.augmentation_functions import AugmentationFunction


class OffPolicyAlgorithmAugment(OffPolicyAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: Policy object
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
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
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
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        aug_function: Optional[AugmentationFunction] = None,
        aug_ratio: Optional[Union[float, Schedule]] = None,
        aug_n: Optional[int] = 1,
        aug_buffer: Optional[bool] = True,
        aug_constraint: Optional[float] = None,
        aug_freq: Optional[Union[int, str]] = 1,
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            sde_support=sde_support,
            supported_action_spaces=supported_action_spaces,
        )

        self.aug_function = aug_function
        self.aug_ratio = aug_ratio
        self.aug_n = aug_n
        self.aug_n_floor = int(np.floor(aug_n))
        self.aug_prob = aug_n - self.aug_n_floor
        self.aug_freq = aug_freq
        self.separate_aug_buffer = aug_buffer
        self.aug_constraint = aug_constraint

        self.use_aug = self.aug_function is not None
        if self.use_aug:
            assert self._vec_normalize_env is None
            assert not self.optimize_memory_usage
            self._setup_augmented_replay_buffer()

    def _setup_augmented_replay_buffer(self):
        self.aug_replay_buffer = ReplayBuffer(
            int(self.buffer_size * self.aug_n),
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

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ):
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
        # self.replay_buffer.update_hists(next_obs)

        # if self.use_aug:
        #     unscaled_action = self.policy.unscale_action(buffer_action)
        #     aug_obs, aug_next_obs, aug_unscaled_action, aug_reward, aug_done, aug_info = self.augment_transition(
        #         self._last_original_obs,
        #         next_obs,
        #         unscaled_action,
        #         reward_,
        #         dones,
        #         infos,
        #     )
        #     if aug_obs is not None: # When aug_n < 1, we only augment a fraction of transitions.
        #         aug_action = self.policy.scale_action(aug_unscaled_action)
        #         self.aug_replay_buffer.extend(aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info)

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

        return self._last_original_obs, next_obs, buffer_action, reward_, dones, infos

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "env",
            "eval_env",
            "replay_buffer",
            "aug_replay_buffer",
            "rollout_buffer",
            "_vec_normalize_env",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]

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
        assert isinstance(self.aug_replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

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

    def sample_replay_buffers(self):
        alpha = 0
        if self.use_aug:
            alpha = self.aug_ratio(self._current_progress_remaining)

        if alpha >= 0:
            observed_batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
            if alpha == 0:
                return observed_batch, None, None

        aug_batch_size = int(np.abs(alpha) * self.batch_size)
        aug_batch = self.aug_replay_buffer.sample(aug_batch_size, env=self._vec_normalize_env)
        if alpha < 0:
            return aug_batch

        observations = th.concat((observed_batch.observations, aug_batch.observations))
        actions = th.concat((observed_batch.actions, aug_batch.actions))
        next_observations = th.concat((observed_batch.next_observations, aug_batch.next_observations))
        rewards = th.concat((observed_batch.rewards, aug_batch.rewards))
        dones = th.concat((observed_batch.dones, aug_batch.dones))

        return ReplayBufferSamples(observations, actions, next_observations, dones, rewards), observed_batch, aug_batch

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
        aug_indices = []

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
            # buf_obs, buf_next_obs, buf_action, buf_reward, buf_dones, buf_infos = self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            aug_indices.append(self.replay_buffer.size()-1)
            if self.aug_freq == 'episode':
                do_aug = self.use_aug and dones.all()
            else:
                do_aug = self.use_aug and (num_collected_steps % self.aug_freq == 0)

            if do_aug:
                env_indices = np.random.randint(0, high=self.n_envs, size=(len(aug_indices),))
                obs = self.replay_buffer.observations[aug_indices, env_indices, :]
                next_obs = self.replay_buffer.next_observations[aug_indices, env_indices, :]
                actions = self.replay_buffer.actions[aug_indices, env_indices, :]
                dones = self.replay_buffer.dones[aug_indices, env_indices]
                timeouts = self.replay_buffer.timeouts[aug_indices, env_indices]
                rewards = self.replay_buffer.rewards[aug_indices, env_indices]
                infos = [[{'TimeLimit.Truncated': True}] if truncated else [{}] for truncated in timeouts]

                unscaled_actions = self.policy.unscale_action(actions)
                aug_obs, aug_next_obs, aug_unscaled_action, aug_reward, aug_done, aug_info = self.augment_transition(
                    obs,
                    next_obs,
                    unscaled_actions,
                    rewards,
                    dones,
                    infos,
                )
                if aug_obs is not None: # When aug_n < 1, we only augment a fraction of transitions.
                    aug_action = self.policy.scale_action(aug_unscaled_action)
                    self.aug_replay_buffer.extend(aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info)
                aug_indices.clear()

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
