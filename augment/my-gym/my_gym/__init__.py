import copy

import gym
from gym.envs.registration import register

# unregister gym's env so I can use the same name
# envs_to_unregister = ['Ant-v3', 'HalfCheetah-v3', 'Humanoid-v3', 'Walker2d-v3']
envs_to_unregister = ['CartPole-v1', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2']
# for env_id in envs_to_unregister:
#     if env_id in gym.envs.registry.env_specs:
#         del gym.envs.registry.env_specs[env_id]
env_ids = list(gym.envs.registry.env_specs.keys())
for env_id in envs_to_unregister:
        del gym.envs.registry.env_specs[env_id]

###########################################################################

register(
    id="CartPole-v1",
    entry_point="my_gym.envs.cartpole:CartPoleEnv",
    max_episode_steps=500,
)

# Mujoco
# ----------------------------------------

# 2D
#
# register(
#     id="Reacher-v2",
#     entry_point="my_gym.envs.mujoco:ReacherEnv",
#     max_episode_steps=50,
#     reward_threshold=-3.75,
# )
#
# register(
#     id="Pusher-v2",
#     entry_point="my_gym.envs.mujoco:PusherEnv",
#     max_episode_steps=100,
#     reward_threshold=0.0,
# )

register(
    id="InvertedPendulum-v2",
    entry_point="my_gym.envs.mujoco.inverted_pendulum:InvertedPendulumEnv",
    max_episode_steps=1000,
)
#
register(
    id="InvertedDoublePendulum-v2",
    entry_point="my_gym.envs.mujoco.inverted_double_pendulum:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id="HalfCheetah-v3",
    entry_point="my_gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="Hopper-v3",
    entry_point="my_gym.envs.mujoco.hopper_v3:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="Swimmer-v3",
    entry_point="my_gym.envs.mujoco.swimmer_v3:SwimmerEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id="Walker2d-v3",
    max_episode_steps=1000,
    entry_point="my_gym.envs.mujoco.walker2d_v3:Walker2dEnv",
)

register(
    id="Ant-v3",
    entry_point="my_gym.envs.mujoco.ant_v3:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Humanoid-v3",
    entry_point="my_gym.envs.mujoco.humanoid_v3:HumanoidEnv",
    max_episode_steps=1000,
)

register(
    id="HumanoidStandup-v2",
    entry_point="my_gym.envs.mujoco:HumanoidStandupEnv",
    max_episode_steps=1000,
)
#
# ###########################################################################

register(
    id=f'Swimmer20-v3',
    entry_point='my_gym.envs.mujoco:SwimmerKEnv',
    kwargs={'num_links': 20},
    max_episode_steps=1000,
)

for k in [4,8,12,16,20]:
    # register(
    #     id=f'ReacherTracker{k}-v3',
    #     entry_point='my_gym.envs.mujoco:ReacherTrackerEnv',
    #     kwargs={'num_links': k},
    #     max_episode_steps=200,
    # )

    register(
        id=f'Reacher{k}-v3',
        entry_point='my_gym.envs.mujoco:ReacherEnv',
        kwargs={'num_links': k},
        max_episode_steps=100,
    )
#
# ###########################################################################
#
# register(
#     id="LQR-v0",
#     entry_point="my_gym.envs:LQREnv",
#     max_episode_steps=50,
# )
#
# register(
#     id="Bandit-v0",
#     entry_point="my_gym.envs:BanditEnv",
#     max_episode_steps=100,
# )
#
# register(
#     id="Chain-v0",
#     entry_point="my_gym.envs:ChainEnv",
#     max_episode_steps=100,
# )
#
# register(
#     id="Chain-Continuous-v0",
#     entry_point="my_gym.envs:ChainContinuousEnv",
#     max_episode_steps=100,
# )