import copy
import os

import gym
from gym.envs.registration import register

ENVS_DIR = os.path.join(os.path.dirname(__file__), 'envs')


# unregister gym's env so I can use the same name
# envs_to_unregister = ['Ant-v3', 'HalfCheetah-v3', 'Humanoid-v3', 'Walker2d-v3']
# envs_to_unregister = [
#     'CartPole-v1', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2',
#     'Ant-v3', 'HalfCheetah-v3', 'Humanoid-v3', 'Walker2d-v3', 'Hopper-v3', 'Swimmer-v3', 'HumanoidStandup-v2']
# for env_id in envs_to_unregister:
#     if env_id in gym.envs.registry.env_specs:
#         del gym.envs.registry.env_specs[env_id]
# env_ids = list(gym.envs.registry.env_specs.keys())
# for env_id in env_ids:
#         del gym.envs.registry.env_specs[env_id]

###########################################################################
### Classic control

register(
    id="CartPole-v1",
    entry_point="my_gym.envs.cartpole:CartPoleEnv",
    max_episode_steps=500,
)

###########################################################################
### Mujoco

register(
    id="Walker2d-v4",
    max_episode_steps=1000,
    entry_point="my_gym.envs.mujoco:Walker2dEnv",
)
register(
    id="Humanoid-v4",
    entry_point="my_gym.envs.mujoco.humanoid_v4:HumanoidEnv",
    max_episode_steps=1000,
)

# ----------------------------------------

# 2D
#
register(
    id="Reacher-v4",
    entry_point="my_gym.envs.mujoco:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)
#
# register(
#     id="Pusher-v2",
#     entry_point="my_gym.envs.mujoco:PusherEnv",
#     max_episode_steps=100,
#     reward_threshold=0.0,
# )
#
register(
    id="InvertedPendulum-v4",
    entry_point="my_gym.envs.mujoco.inverted_pendulum_v4:InvertedPendulumEnv",
    max_episode_steps=1000,
)
#
register(
    id="InvertedDoublePendulum-v4",
    entry_point="my_gym.envs.mujoco.inverted_double_pendulum_v4:InvertedDoublePendulumEnv",
    max_episode_steps=1000,
    reward_threshold=9100.0,
)
#
# register(
#     id="InvertedPendulumWide-v2",
#     entry_point="my_gym.envs.mujoco.inverted_pendulum:InvertedPendulumEnv",
#     max_episode_steps=1000,
#     kwargs={
#         'init_pos': [-0.9, 0.9]
#     }
# )
#
# register(
#     id="InvertedDoublePendulumWide-v2",
#     entry_point="my_gym.envs.mujoco.inverted_double_pendulum:InvertedDoublePendulumEnv",
#     max_episode_steps=1000,
#     reward_threshold=9100.0,
#     kwargs={
#         'init_pos': [-0.9, 0.9]
#     }
# )
#
#
# register(
#     id="HalfCheetah-v3",
#     entry_point="my_gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv",
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
# )
#
# register(
#     id="Hopper-v3",
#     entry_point="my_gym.envs.mujoco.hopper_v3:HopperEnv",
#     max_episode_steps=1000,
#     reward_threshold=3800.0,
# )
#
# register(
#     id="Swimmer-v3",
#     entry_point="my_gym.envs.mujoco.swimmer_v3:SwimmerEnv",
#     max_episode_steps=1000,
#     reward_threshold=360.0,
# )
#
#
# register(
#     id="HumanoidStandup-v2",
#     entry_point="my_gym.envs.mujoco:HumanoidStandupEnv",
#     max_episode_steps=1000,
# )
# #
# # ###########################################################################
#
# register(
#     id=f'Swimmer10-v3',
#     entry_point='my_gym.envs.mujoco:SwimmerKEnv',
#     kwargs={'num_links': 10},
#     max_episode_steps=1000,
# )
#
# register(
#     id=f'Swimmer20-v3',
#     entry_point='my_gym.envs.mujoco:SwimmerKEnv',
#     kwargs={'num_links': 20},
#     max_episode_steps=1000,
# )
#
# for k in [2,4,8,12,16,20]:
    # register(
    #     id=f'ReacherTracker{k}-v3',
    #     entry_point='my_gym.envs.mujoco:ReacherTrackerEnv',
    #     kwargs={'num_links': k},
    #     max_episode_steps=200,
    # )

    # register(
    #     id=f'Reacher{k}-v4',
    #     entry_point='my_gym.envs.mujoco:ReacherEnv',
    #     kwargs={'num_links': k},
    #     max_episode_steps=100,
    # )
#
#     register(
#         id=f'Reacher{k}Sparse-v3',
#         entry_point='my_gym.envs.mujoco:ReacherEnv',
#         kwargs={'num_links': k, 'sparse': True},
#         max_episode_steps=100,
#     )
#
#     register(
#         id=f'Reacher{k}Rand-v3',
#         entry_point='my_gym.envs.mujoco:ReacherEnv',
#         kwargs={
#             'num_links': k,
#             'rand_central_angle': True
#         },
#         max_episode_steps=100,
#     )

############################################################################
### Toy

register(
    id="Goal2D-v0",
    entry_point="my_gym.envs:Goal2DEnv",
    max_episode_steps=100,
)
register(
    id="Goal2DKey-v0",
    entry_point="my_gym.envs:Goal2DKeyEnv",
    max_episode_steps=100,
)
register(
    id="Goal2DQuadrant-v0",
    entry_point="my_gym.envs:Goal2DQuadrantEnv",
    max_episode_steps=100,
)

###############################################################################
register(
    id="MeetUp-v0",
    entry_point="my_gym.envs:MeetUpEnv",
    max_episode_steps=100,
)

############################################

import dm_control.suite

def register_env(
        domain_name,
        task_name,
        seed=0,
        visualize_reward=True,
        from_pixels=False,
        height=84,
        width=84,
        camera_id=0,
        frame_skip=1,
        episode_length=1000,
        environment_kwargs=None,
        time_limit=None,
        channels_first=True
):
    env_id = 'dmc_%s_%s_%s-v1' % (domain_name, task_name, seed)

    if from_pixels:
        assert not visualize_reward, 'cannot use visualize reward when learning from pixels'

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if not env_id in gym.envs.registry:
        task_kwargs = {}
        if seed is not None:
            task_kwargs['random'] = seed
        if time_limit is not None:
            task_kwargs['time_limit'] = time_limit
        register(
            id=env_id,
            entry_point='my_gym.envs.dm_control.gym_wrapper:DMCWrapper',
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
            ),
            max_episode_steps=max_episode_steps,
        )

tasks = dm_control.suite.ALL_TASKS
for domain, task in tasks:
    register_env(domain, task)