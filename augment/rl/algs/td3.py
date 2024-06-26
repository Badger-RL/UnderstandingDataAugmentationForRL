import copy
import io
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import gymnasium.spaces.box
import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithmSelf
from stable_baselines3.common.save_util import recursive_setattr, load_from_zip_file
from torch.nn import functional as F

# from stable_baselines3.common.buffers import ReplayBuffer
from augment.rl.algs.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update, check_for_correct_spaces, \
    get_system_info
from stable_baselines3.td3.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy

from augment.rl.algs.off_policy_algorithm import OffPolicyAlgorithmAugment

TD3Self = TypeVar("TD3Self", bound="TD3")


class TD3(OffPolicyAlgorithmAugment):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.
    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
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
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 10000,
        batch_size: int = 128,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        extra_collect_info:  Tuple[int, int] = (0, 0),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        random_action_prob: Optional[float] = 0,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = ReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        aug_function: Optional = None,
        aug_ratio: Optional[Union[float, Schedule]] = None,
        aug_n: Optional[int] = 1,
        aug_freq: Optional[Union[int, str]] = 1,
        aug_buffer: Optional[bool] = True,
        aug_buffer_size: Optional[int] = None,
        aug_constraint: Optional[float] = 0,
        actor_data_source: Optional[str] = 'both',
        critic_data_source: Optional[str] = 'both',
        # freeze_layers: Optional[List[str]] = ('both', 'both', 'both'),
        obs_active_layer_mask: Optional[List[int]] = (),
        aug_active_layer_mask: Optional[List[int]] = (),
        separate_aug_critic: Optional[bool] = False,
        coda_function: Optional = None,
        coda_n: Optional = 1,
        critic_clip: Optional = (-50, 0),
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
            extra_collect_info,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box, gymnasium.spaces.Box),
            support_multi_env=True,
            aug_function=aug_function,
            aug_ratio=aug_ratio,
            aug_n=aug_n,
            aug_buffer=aug_buffer,
            aug_buffer_size=aug_buffer_size,
            aug_constraint=aug_constraint,
            aug_freq=aug_freq,
            coda_function=coda_function,
            coda_n=coda_n
        )

        self.random_action_prob = random_action_prob
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.critic_clip = critic_clip
        self.opmse_obs = []
        self.opmse_aug = []

        print(actor_data_source, critic_data_source)
        assert actor_data_source == 'obs' or actor_data_source == 'aug' or actor_data_source == 'both'
        assert critic_data_source == 'obs' or critic_data_source == 'aug' or critic_data_source == 'both'
        if actor_data_source != 'both' or critic_data_source != 'both':
            assert self.use_aug

        self.actor_data_source = actor_data_source
        self.critic_data_source = critic_data_source

        self.obs_active_layer_mask = []
        self.aug_active_layer_mask = []
        for i, j in zip(obs_active_layer_mask, aug_active_layer_mask):
            self.obs_active_layer_mask.extend([int(i), int(i)])
            self.aug_active_layer_mask.extend([int(j), int(j)])
        # 3 layers with 3 biases
        # self.actor_freeze_layers = []
        # self.critic_freeze_layers = []
        # for i, j in zip(actor_freeze_layers, critic_freeze_layers):
        #     self.actor_freeze_layers.extend([2*i, 2*i+1])
        #     self.critic_freeze_layers.extend([2*j, 2*j+1])
        self.freeze_layers = len(self.obs_active_layer_mask) > 0 or len(self.aug_active_layer_mask) > 0
        if self.freeze_layers:
            assert self.use_aug == True

        if _init_setup_model:
            self._setup_model()

        self.separate_aug_critic = separate_aug_critic
        self.aug_critic = None
        if self.separate_aug_critic:
            self.policy.aug_critic = copy.deepcopy(self.critic)
            self.policy.aug_critic_target = copy.deepcopy(self.critic_target)

            # alias
            self.aug_critic = self.policy.aug_critic
            self.aug_critic_target = self.policy.aug_critic_target

            self.actor_data_source = 'obs'
            self.critic_data_source = 'obs'


    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target


    def _critic_loss(self, replay_data, ):
        with th.no_grad():
            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            # target_q_values.clamp(*self.critic_clip)

        # Get current Q-values estimates for each critic network
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        return sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)

    def _critic_q(self, replay_data, ):
        with th.no_grad():
            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

        # Get current Q-values estimates for each critic network
        current_q_values = self.critic(replay_data.observations, replay_data.actions)

        # Compute critic loss
        return current_q_values, target_q_values


    def _update_freeze(self, replay_data_observed, replay_data_aug, actor_losses, critic_losses):
        # zero gradients
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        # uncomment to test freezing
        # actor_prev = copy.deepcopy(self.actor)
        # critic_prev = copy.deepcopy(self.critic)

        # accumulate observed critic gradients
        self._freeze_critic_features(active_layer_mask=self.obs_active_layer_mask)
        current_q_values_obs, target_q_values_obs = self._critic_q(replay_data_observed) # comment out to test freezing
        self._unfreeze(self.critic)

        # accumulate aug critic gradients
        self._freeze_critic_features(active_layer_mask=self.aug_active_layer_mask)
        current_q_values_aug, target_q_values_aug = self._critic_q(replay_data_aug)
        self._unfreeze(self.critic)

        current_q_values = ( # comment out to test freezing
            th.concat([current_q_values_obs[0], current_q_values_aug[0]]),
            th.concat([current_q_values_obs[1], current_q_values_aug[1]]),
        )
        target_q_values = th.concat([target_q_values_obs, target_q_values_aug]) # comment out to test freezing
        # current_q_values = current_q_values_aug # uncomment to test freezing
        # target_q_values = target_q_values_aug # # uncomment to test freezing
        critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        critic_loss.backward()
        self.critic.optimizer.step()
        critic_losses.append(critic_loss.item())

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            # accumulate obs actor gradients
            self._freeze_actor_features(active_layer_mask=self.obs_active_layer_mask)
            actor_loss_obs = -self.critic.q1_forward(replay_data_observed.observations, self.actor(replay_data_observed.observations))
            self._unfreeze(self.actor)

            # accumulate aug actor gradients
            self._freeze_actor_features(active_layer_mask=self.aug_active_layer_mask)
            actor_loss_aug = -self.critic.q1_forward(replay_data_aug.observations, self.actor(replay_data_aug.observations))
            self._unfreeze(self.actor)
            actor_loss = th.concat([actor_loss_obs, actor_loss_aug]).mean()
            # actor_loss = actor_loss_aug.mean()
            actor_loss.backward()
            self.actor.optimizer.step()
            actor_losses.append(actor_loss.item())

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)
            # print('Actor')
            # for prev, curr in zip(actor_prev.parameters(), self.actor.parameters()):
            #     print(prev.shape, curr.shape, th.allclose(prev, curr))
            # print()
            # print('Critic')
            # for prev, curr in zip(critic_prev.parameters(), self.critic.parameters()):
            #     print(prev.shape, curr.shape, th.allclose(prev, curr))
            # print()

    def _update(self, actor_replay_data, critic_replay_data, actor_losses, critic_losses):
        # zero gradients
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        # critic update
        critic_loss = self._critic_loss(critic_replay_data)
        critic_losses.append(critic_loss.item())
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            pi = self.actor(actor_replay_data.observations)
            actor_loss = -self.critic.q1_forward(actor_replay_data.observations, pi).mean()
            actor_losses.append(actor_loss.item())
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

            # Copy running stats, see GH issue #996
            polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

            return pi

    def _update_aug_critic(self, critic_replay_data):
        self.aug_critic.optimizer.zero_grad()

        # critic update
        critic_loss = self._critic_loss(critic_replay_data)
        critic_loss.backward()
        self.aug_critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            polyak_update(self.aug_critic.parameters(), self.aug_critic_target.parameters(), self.tau)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample observed and augmented replay buffers
            both_replay_data, observed_replay_data, aug_replay_data = self.sample_replay_buffers()

            actor_data = both_replay_data
            critic_data = both_replay_data

            if self.critic_data_source == 'both':
                critic_data = both_replay_data
            elif self.critic_data_source == 'obs':
                critic_data = observed_replay_data
            elif self.critic_data_source == 'aug':
                critic_data = aug_replay_data

            if self.actor_data_source == 'both':
                actor_data = both_replay_data
            elif self.actor_data_source == 'obs':
                actor_data = observed_replay_data
            elif self.actor_data_source == 'aug':
                actor_data = aug_replay_data

            if self.freeze_layers and aug_replay_data is not None:
                self._update_freeze(observed_replay_data, aug_replay_data, actor_losses, critic_losses)
            else:
                pi = self._update(actor_replay_data=actor_data, critic_replay_data=critic_data,
                             actor_losses=actor_losses, critic_losses=critic_losses)

                if self.policy_delay <= 1:
                    with th.no_grad():
                        pi_obs = pi[:self.batch_size]
                        a_obs = observed_replay_data.actions
                        opmse_obs = ((pi_obs - a_obs) ** 2).mean()
                        self.opmse_obs.append(opmse_obs.cpu())
    
                        if self.use_aug:
                            if self.actor_data_source == 'both' and aug_replay_data is not None:
                                pi_obs = pi[:self.batch_size]
                                a_obs = observed_replay_data.actions
                                opmse_obs = ((pi_obs - a_obs) ** 2).mean()
                                self.opmse_obs.append(opmse_obs.cpu())

                                pi_aug = pi[self.batch_size:]
                                a_aug = aug_replay_data.actions
                                opmse_aug = ((pi_aug - a_aug) ** 2).mean()
                                self.opmse_aug.append(opmse_aug.cpu())
                            else:
                                pi_aug = self.actor(aug_replay_data.observations)
                                a_aug = aug_replay_data.actions
                                opmse_aug = ((pi_aug - a_aug) ** 2).mean()
                                self.opmse_aug.append(opmse_aug.cpu())


                # print(opmse_obs, opmse_aug)

            if self.separate_aug_critic:
                self._update_aug_critic(both_replay_data)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

    def _freeze_actor_features(self, active_layer_mask):
        for is_active, param in zip(active_layer_mask, self.actor.parameters()):
            if not is_active:
                param.requires_grad = False

    def _freeze_critic_features(self, active_layer_mask):
        for is_active, qf0_param, qf1_param in zip(active_layer_mask, self.critic.qf0.parameters(), self.critic.qf1.parameters()):
            if not is_active:
                qf0_param.requires_grad = False
                qf1_param.requires_grad = False

    def _unfreeze(self, model):
        for parameter in model.parameters():
            parameter.requires_grad = True