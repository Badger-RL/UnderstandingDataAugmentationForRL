import argparse
import difflib
import os.path
import uuid
from operator import itemgetter

import gym, my_gym
import numpy as np
import torch
import yaml
from typing import Union
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from augment.rl.augmentation_functions import AUGMENTATION_FUNCTIONS
from augment.rl.utils import ALGOS, StoreDict, get_save_dir, preprocess_action_noise, read_hyperparameters, SCHEDULES
from stable_baselines3.common.utils import set_random_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--algo", help="RL Algorithm", default="td3", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--env", type=str, default="InvertedPendulum-v2", help="environment ID")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    parser.add_argument("-n", "--n-timesteps", help="Overwrite the number of timesteps", default=int(1e5), type=int)
    parser.add_argument("--eval-freq", help="Evaluate the agent every n steps (if negative, no evaluation).", default=10000, type=int,)
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=10, type=int)
    parser.add_argument("-i", "--trained-agent", help="Path to a pretrained agent to continue training", default="", type=str)

    # parameters
    parser.add_argument("--env-kwargs", type=str, nargs="*", action=StoreDict, default={}, help="Optional keyword argument to pass to the env constructor")
    parser.add_argument("-params", "--hyperparams", type=str, nargs="+", action=StoreDict, help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)" )

    # augmentation
    parser.add_argument("--aug-function", type=str, default=None)
    parser.add_argument("--aug-function-kwargs", type=str, nargs="*", action=StoreDict, default={})
    parser.add_argument("--aug-n", type=int, default=1)
    parser.add_argument("--aug-ratio", type=float, default=1)
    parser.add_argument("--aug-schedule", type=str, default="constant")
    # parser.add_argument("-aug-ratio-final", "--augmentation-ratio-final", type=Union[float, str], default=1)
    parser.add_argument("--aug-buffer", type=bool, default=True)
    parser.add_argument("--aug-constraint", type=bool, default=None)
    parser.add_argument("--add-policy-kwargs", type=str, nargs="*", action=StoreDict, default={},
                        help="Optional ADDITIONAL keyword argument to pass to the policy constructor")

    # saving
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="results")
    parser.add_argument("--save-best-model", default=False, type=bool)
    parser.add_argument("-exp", "--experiment-name", help="<log folder>/<env_id>/<algo>/<experiment name>/run_<run_id>", type=str, default="")
    parser.add_argument("--run-id", help="Run id to append to env save directory", default=None, type=int)

    # extra
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=0, type=int)
    parser.add_argument("--random-hyperparameters", default=False, help="Sample random hyperparameters for a single run.")

    args = parser.parse_args()

    ####################################################################################################################

    env_id = args.env
    algo = args.algo
    n_timesteps = args.n_timesteps

    ####################################################################################################################
    # Assertions

    # If the environment is not found, suggest the closest match
    registered_envs = set(gym.envs.registry.env_specs.keys())  # pytype: disable=module-attr
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(args.trained_agent), "The trained_agent must be a valid path to a .zip file"

    ####################################################################################################################
    # Preprocess args

    save_dir = get_save_dir(args.log_folder, env_id, algo, args.run_id, args.experiment_name)
    best_model_save_dir = save_dir if args.save_best_model else None

    # update hyperparams
    hyperparams = read_hyperparameters(env_id, algo)
    if args.hyperparams is not None:
        hyperparams.update(args.hyperparams)

    # set seed
    if args.run_id:
        args.seed=args.run_id
    elif args.seed < 0:
        args.seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()
    set_random_seed(args.seed)

    # set n_timesteps
    if args.n_timesteps > 0:
        print(f"Overwriting n_timesteps with n={n_timesteps}")
        del hyperparams["n_timesteps"]
    else:
        n_timesteps = int(hyperparams.pop("n_timesteps"))

    # set train_freq
    if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
        hyperparams["train_freq"] = tuple(hyperparams["train_freq"])
    # hyperparams['buffer_size'] = int(hyperparams['buffer_size'])

    # augmentation
    if args.aug_function:
        # print(f'Automatically scaling replay buffer')
        # print('BUFFER SCALING CURRENTLY DISABLED')

        aug_buffer = args.aug_buffer
        aug_ratio = args.aug_ratio
        aug_schedule = args.aug_schedule #
        aug_n = args.aug_n
        aug_func = args.aug_function #
        aug_func_kwargs = args.aug_function_kwargs

        aug_func_class = AUGMENTATION_FUNCTIONS[env_id][aug_func]
        # buffer_scale = int(1+aug_ratio*aug_n)
        # hyperparams['buffer_size'] = hyperparams['buffer_size'] * buffer_scale
        hyperparams['aug_ratio'] = SCHEDULES[aug_schedule](initial_value=aug_ratio)
        hyperparams['aug_function'] = aug_func_class(aug_n, aug_func_kwargs)
        hyperparams['aug_constraint'] = args.aug_constraint


    ####################################################################################################################
    # More preprocessing that depends on the env object

    env = Monitor(gym.make(env_id, **args.env_kwargs),)
    preprocess_action_noise(hyperparams=hyperparams, env=env)

    algo_class = ALGOS[algo]
    model = algo_class(env=env, **hyperparams)

    # save hyperparams and args
    with open(os.path.join(save_dir, "config.yml"), "w") as f:
        yaml.dump(hyperparams, f, sort_keys=True)
    with open(os.path.join(save_dir, "args.yml"), "w") as f:
        yaml.dump(args, f, sort_keys=True)

    # Setting num threads to 1 makes things run faster on cpu
    torch.set_num_threads(1)
    env_eval = Monitor(gym.make(env_id, **args.env_kwargs), filename=save_dir)
    eval_callback = EvalCallback(eval_env=env_eval, n_eval_episodes=args.eval_episodes, eval_freq=args.eval_freq, log_path=save_dir, best_model_save_path=best_model_save_dir)
    model.learn(total_timesteps=int(n_timesteps), callback=eval_callback)
    # model.save_replay_buffer(f"{save_dir}/buffer")

    print(f'Results saved to {save_dir}')
