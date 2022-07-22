import argparse
import glob
import importlib
import os.path
from collections import OrderedDict
from pprint import pprint
from typing import Tuple, Dict, Any, Optional, Callable, List

import gym
import numpy as np
import yaml

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import VecEnv

from augment.rl.callbacks import SaveVecNormalizeCallback

ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)

def get_latest_run_id(save_dir: str) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(save_dir, 'run_[0-9]*')):
        filename = os.path.basename(path)
        ext = filename.split('_')[-1]
        if ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def read_hyperparameters(env_id, algo) -> Dict[str, Any]:
    # Load hyperparameters from yaml file
    with open(f"hyperparams/{algo}.yml") as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        # elif _is_atari:
        #     hyperparams = hyperparams_dict["atari"]
        else:
            raise ValueError(f"Hyperparameters not found for {algo}-{env_id}")

    # if self.custom_hyperparams is not None:
    #     # Overwrite hyperparams if needed
    #     hyperparams.update(self.custom_hyperparams)
    # # Sort hyperparams that will be saved
    # saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
    #
    # if self.verbose > 0:
    #     print("Default hyperparameters for environment (ones being tuned will be overridden):")
    #     pprint(saved_hyperparams)

    for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
        if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
            hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

    return hyperparams#, saved_hyperparams

def _preprocess_hyperparams(
    self, hyperparams: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback], Optional[Callable]]:
    self.n_envs = hyperparams.get("n_envs", 1)

    if self.verbose > 0:
        print(f"Using {self.n_envs} environments")

    # Convert schedule strings to objects
    hyperparams = self._preprocess_schedules(hyperparams)

    # Pre-process train_freq
    if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
        hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

    # Should we overwrite the number of timesteps?
    if self.n_timesteps > 0:
        if self.verbose:
            print(f"Overwriting n_timesteps with n={self.n_timesteps}")
    else:
        self.n_timesteps = int(hyperparams["n_timesteps"])

    # Derive n_evaluations from number of timesteps if needed
    if self.n_evaluations is None and self.optimize_hyperparameters:
        self.n_evaluations = max(1, self.n_timesteps // int(1e5))
        print(
            f"Doing {self.n_evaluations} intermediate evaluations for pruning based on the number of timesteps."
            " (1 evaluation every 100k timesteps)"
        )

    # Pre-process normalize config
    hyperparams = self._preprocess_normalization(hyperparams)

    # Pre-process policy/buffer keyword arguments
    # Convert to python object if needed
    for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
        if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
            hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

    # Delete keys so the dict can be pass to the model constructor
    if "n_envs" in hyperparams.keys():
        del hyperparams["n_envs"]
    del hyperparams["n_timesteps"]

    if "frame_stack" in hyperparams.keys():
        self.frame_stack = hyperparams["frame_stack"]
        del hyperparams["frame_stack"]

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    # Same for VecEnvWrapper
    vec_env_wrapper = get_wrapper_class(hyperparams, "vec_env_wrapper")
    if "vec_env_wrapper" in hyperparams.keys():
        del hyperparams["vec_env_wrapper"]

    callbacks = get_callback_list(hyperparams)
    if "callback" in hyperparams.keys():
        self.specified_callbacks = hyperparams["callback"]
        del hyperparams["callback"]

    return hyperparams, env_wrapper, callbacks, vec_env_wrapper


def _preprocess_action_noise(
    self, hyperparams: Dict[str, Any], saved_hyperparams: Dict[str, Any], env: VecEnv
) -> Dict[str, Any]:
    # Parse noise string
    # Note: only off-policy algorithms are supported
    if hyperparams.get("noise_type") is not None:
        noise_type = hyperparams["noise_type"].strip()
        noise_std = hyperparams["noise_std"]

        # Save for later (hyperparameter optimization)
        self.n_actions = env.action_space.shape[0]

        if "normal" in noise_type:
            hyperparams["action_noise"] = NormalActionNoise(
                mean=np.zeros(self.n_actions),
                sigma=noise_std * np.ones(self.n_actions),
            )
        elif "ornstein-uhlenbeck" in noise_type:
            hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(self.n_actions),
                sigma=noise_std * np.ones(self.n_actions),
            )
        else:
            raise RuntimeError(f'Unknown noise type "{noise_type}"')

        print(f"Applying {noise_type} noise with std {noise_std}")

        del hyperparams["noise_type"]
        del hyperparams["noise_std"]

    return hyperparams

def create_log_folder(self):
    os.makedirs(self.params_path, exist_ok=True)

def create_callbacks(self):

    if self.save_freq > 0:
        # Account for the number of parallel environments
        self.save_freq = max(self.save_freq // self.n_envs, 1)
        self.callbacks.append(
            CheckpointCallback(
                save_freq=self.save_freq,
                save_path=self.save_path,
                name_prefix="rl_model",
                verbose=1,
            )
        )

    # Create test env if needed, do not normalize reward
    if self.eval_freq > 0 and not self.optimize_hyperparameters:
        # Account for the number of parallel environments
        self.eval_freq = max(self.eval_freq // self.n_envs, 1)

        if self.verbose > 0:
            print("Creating test environment")

        save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=self.params_path)
        eval_callback = EvalCallback(
            self.create_envs(self.n_eval_envs, eval_env=True),
            callback_on_new_best=save_vec_normalize,
            best_model_save_path=self.save_path,
            n_eval_episodes=self.n_eval_episodes,
            log_path=self.save_path,
            eval_freq=self.eval_freq,
            deterministic=self.deterministic_eval,
        )

        self.callbacks.append(eval_callback)



def get_wrapper_class(hyperparams: Dict[str, Any]) -> Optional[Callable[[gym.Env], gym.Env]]:
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    for multiple, specify a list:

    env_wrapper:
        - utils.wrappers.PlotActionWrapper
        - utils.wrappers.TimeFeatureWrapper


    :param hyperparams:
    :return: maybe a callable to wrap the environment
        with one or multiple gym.Wrapper
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if "env_wrapper" in hyperparams.keys():
        wrapper_name = hyperparams.get("env_wrapper")

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {wrapper_name}. "
                    "You should check the indentation."
                )
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env: gym.Env) -> gym.Env:
            """
            :param env:
            :return:
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None


def get_callback_list(hyperparams: Dict[str, Any]) -> List[BaseCallback]:
    """
    Get one or more Callback class specified as a hyper-parameter
    "callback".
    e.g.
    callback: stable_baselines3.common.callbacks.CheckpointCallback

    for multiple, specify a list:

    callback:
        - utils.callbacks.PlotActionWrapper
        - stable_baselines3.common.callbacks.CheckpointCallback

    :param hyperparams:
    :return:
    """

    def get_module_name(callback_name):
        return ".".join(callback_name.split(".")[:-1])

    def get_class_name(callback_name):
        return callback_name.split(".")[-1]

    callbacks = []

    if "callback" in hyperparams.keys():
        callback_name = hyperparams.get("callback")

        if callback_name is None:
            return callbacks

        if not isinstance(callback_name, list):
            callback_names = [callback_name]
        else:
            callback_names = callback_name

        # Handle multiple wrappers
        for callback_name in callback_names:
            # Handle keyword arguments
            if isinstance(callback_name, dict):
                assert len(callback_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {callback_name}. "
                    "You should check the indentation."
                )
                callback_dict = callback_name
                callback_name = list(callback_dict.keys())[0]
                kwargs = callback_dict[callback_name]
            else:
                kwargs = {}
            callback_module = importlib.import_module(get_module_name(callback_name))
            callback_class = getattr(callback_module, get_class_name(callback_name))
            callbacks.append(callback_class(**kwargs))

    return callbacks

