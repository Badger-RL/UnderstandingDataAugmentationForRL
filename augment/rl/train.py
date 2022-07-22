import os.path

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from augment.rl.utils import get_latest_run_id, ALGOS, read_hyperparameters

if __name__ == '__main__':

    env_id = 'CartPole-v1'
    algo = 'dqn'
    save_dir = f'results/{algo}/{env_id}'
    save_dir += f'/run_{get_latest_run_id(save_dir) + 1}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f'Results will be saved to {save_dir}')

    env_kwargs = {}
    env = Monitor(gym.make(env_id, **env_kwargs),)
    algo_class = ALGOS[algo]

    hyperparams = read_hyperparameters(env_id, algo)
    num_timesteps = hyperparams.pop('n_timesteps')

    model = algo_class(
        env=env,
        **hyperparams)

    num_timesteps= int(1e5)

    env_eval = Monitor(gym.make(env_id, **env_kwargs), filename=save_dir)
    eval_callback = EvalCallback(eval_env=env_eval, n_eval_episodes=10, eval_freq=int(10e3), log_path=save_dir, best_model_save_path=save_dir)
    model.learn(total_timesteps=int(num_timesteps), callback=eval_callback)
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/


    print(f'Results saved to {save_dir}')


    # custom_objects = {}
    # algo_class = ALGOS[algo]
    # model_path = f'{save_dir}/best_model.zip'
    # model = algo_class.load(path=model_path, custom_objects=custom_objects, env=env)
    #
    # evaluate_policy(model=model, env=env_eval, n_eval_episodes=1, deterministic=True, render=True)
