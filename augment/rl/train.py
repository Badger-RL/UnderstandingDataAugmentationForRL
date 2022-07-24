import argparse
import difflib
import os.path
import uuid

import gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from augment.rl.utils import get_latest_run_id, ALGOS, read_hyperparameters, StoreDict, preprocess_action_noise
from stable_baselines3.common.utils import set_random_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="RL Algorithm", default="ddpg", type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("--env", type=str, default="InvertedPendulum-v2", help="environment ID")
    parser.add_argument("--run-id", help="Run id to append to env save directory", default=None, type=int)
    parser.add_argument("-i", "--trained-agent", help="Path to a pretrained agent to continue training", default="", type=str)
    parser.add_argument("-n", "--n-timesteps", help="Overwrite the number of timesteps", default=int(1e4), type=int)
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation). "
        "During hyperparameter optimization n-evaluations is used instead",
        default=10000,
        type=int,
    )
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=10, type=int)
    parser.add_argument("--save-freq", help="Save the model every n steps (if negative, no checkpoint)", default=-1, type=int)
    parser.add_argument("-f", "--log-folder", help="Log folder", type=str, default="results")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    parser.add_argument("--random-hyperparameters", default=False, help="Sample random hyperparameters for a single run.")
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=0, type=int)
    parser.add_argument("--env-kwargs", type=str, nargs="*", action=StoreDict, default={}, help="Optional keyword argument to pass to the env constructor")
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument(
        "--add-policy-kwargs", type=str, nargs="*", action=StoreDict, default={}, help="Optional ADDITIONAL keyword argument to pass to the policy constructor"
    )
    parser.add_argument("-uuid", "--uuid", action="store_true", default=False, help="Ensure that the run has a unique ID")
    args = parser.parse_args()

    ###################################################################################################################

    env_id = args.env
    algo = args.algo
    n_timesteps = args.n_timesteps

    registered_envs = set(gym.envs.registry.env_specs.keys())  # pytype: disable=module-attr
    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"


    set_random_seed(args.seed)

    save_dir = f'{args.log_folder}/{algo}/{env_id}'
    save_dir += f'/run_{get_latest_run_id(save_dir) + 1}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f'Results will be saved to {save_dir}')

    ####################################################################################################################

    env = Monitor(gym.make(env_id, **args.env_kwargs),)
    algo_class = ALGOS[algo]

    hyperparams = read_hyperparameters(env_id, algo)
    preprocess_action_noise(hyperparams=hyperparams, env=env)

    if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
        hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

    # # Should we overwrite the number of timesteps?
    if n_timesteps > 0:
        print(f"Overwriting n_timesteps with n={n_timesteps}")
        del hyperparams["n_timesteps"]
    else:
        n_timesteps = int(hyperparams.pop("n_timesteps"))
    # n_timesteps = int(hyperparams.pop("n_timesteps"))

    model = algo_class(env=env, **hyperparams)

    # Setting num threads to 1 makes things run faster on cpu
    torch.set_num_threads(1)

    env_eval = Monitor(gym.make(env_id, **args.env_kwargs), filename=save_dir)
    eval_callback = EvalCallback(eval_env=env_eval, n_eval_episodes=args.eval_episodes, eval_freq=args.eval_freq, log_path=save_dir, best_model_save_path=save_dir)
    model.learn(total_timesteps=int(n_timesteps), callback=eval_callback)

    print(f'Results saved to {save_dir}')


    custom_objects = {}
    algo_class = ALGOS[algo]
    model_path = f'{save_dir}/best_model.zip'
    model = algo_class.load(path=model_path, custom_objects=custom_objects, env=env)

    evaluate_policy(model=model, env=env_eval, n_eval_episodes=10, deterministic=True, render=True)
