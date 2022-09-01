"""This is a simple example demonstrating how to clone the behavior of an expert.
Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import argparse
import os

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from augment.rl.utils import StoreDict

def augment_transition(transition):
    # Transition
    pass


def train_expert():
    print("Training a expert.")
    expert = PPO(
        policy='MlpPolicy',
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(100)  # Note: change this to 100000 to trian a decent expert.
    return expert


def sample_expert_transitions(expert):

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=1),
        verbose=0
    )
    return rollout.flatten_trajectories(rollouts)

if __name__ == "__main__":

    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default=f"CartPole-v1")
    parser.add_argument("--env-kwargs", type=str, nargs="*", action=StoreDict, default={})
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--algo", type=str, default="ppo")
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--num-actions", type=int, default=int(10))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--random', action='store_true', default=False)
    args = parser.parse_args()

    env_id = args.env_id
    env = gym.make(env_id, **args.env_kwargs)
    env.seed(args.seed)

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    model_path = f'rl/results/{args.algo}/{args.env_id}/run_{args.run_id}/best_model.zip'
    expert = PPO.load(model_path, env=env, custom_objects=custom_objects)
    # expert = train_expert()
    env = gym.make("CartPole-v1")
    transitions = sample_expert_transitions(expert)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
    )

    reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=10, render=False)
    print(f"Reward before training: {reward}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=10, log_interval=np.inf, progress_bar=True)

    reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=10, render=False)
    print(f"Reward after training: {reward}")