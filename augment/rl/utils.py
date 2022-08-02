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
from augment.rl.algs.ddpg import DDPG
from augment.rl.algs.ppo import PPO
from augment.rl.algs.td3 import TD3
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

def create_log_folder(self):
    os.makedirs(self.params_path, exist_ok=True)

def read_hyperparameters(env_id, algo) -> Dict[str, Any]:
    # Load hyperparameters from yaml file
    with open(f"hyperparams/{algo}.yml") as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        else:
            raise ValueError(f"Hyperparameters not found for {algo}-{env_id}")

    for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
        if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
            hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

    return hyperparams#, saved_hyperparams

def get_latest_run_id(save_dir: str) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(save_dir, 'run_[0-9]*')):
        filename = os.path.basename(path)
        ext = filename.split('_')[-1]
        if ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def get_save_dir(log_folder, env_id, algo, run_id, exp=""):
    # set save directory
    save_dir = f'{log_folder}/{env_id}/{algo}/{exp}'
    if run_id:
        save_dir += f'/run_{run_id}'
    else:
        save_dir += f'/run_{get_latest_run_id(save_dir) + 1}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f'Results will be saved to {save_dir}')

    return save_dir

def preprocess_action_noise(hyperparams: Dict[str, Any], env: VecEnv) -> Dict[str, Any]:
    # Parse noise string
    # Note: only off-policy algorithms are supported
    if hyperparams.get("noise_type") is not None:
        noise_type = hyperparams["noise_type"].strip()
        noise_std = hyperparams["noise_std"]

        # Save for later (hyperparameter optimization)
        n_actions = env.action_space.shape[0]

        if "normal" in noise_type:
            hyperparams["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions),
            )
        elif "ornstein-uhlenbeck" in noise_type:
            hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions),
            )
        else:
            raise RuntimeError(f'Unknown noise type "{noise_type}"')

        print(f"Applying {noise_type} noise with std {noise_std}")

        del hyperparams["noise_type"]
        del hyperparams["noise_std"]

    return hyperparams
