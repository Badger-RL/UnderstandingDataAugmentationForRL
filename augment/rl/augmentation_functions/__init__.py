from augment.rl.augmentation_functions.fetch.reach import *
from augment.rl.augmentation_functions.fetch.push import *
from augment.rl.augmentation_functions.fetch.slide import *
from augment.rl.augmentation_functions.fetch.pick_and_place import *
from augment.rl.augmentation_functions.fetch.common import *

from augment.rl.augmentation_functions.inverted_pendulum import *
from augment.rl.augmentation_functions.lqr import LQRTranslate, LQRRotate
from augment.rl.augmentation_functions.meetup import MeetUpTranslate, MeetUpRotate, MeetUpRotateTranslate
from augment.rl.augmentation_functions.predator_prey import *
from augment.rl.augmentation_functions.reacher_k import *
from augment.rl.augmentation_functions.swimmer_k import SwimmerReflect
from augment.rl.augmentation_functions.walker2d import Walker2dReflect
from augment.simulate import simulate

predator_prey_box_augmentation_functions = {
        'rotate': Goal2DRotateRestricted,
        'translate': Goal2DTranslate,
        'translate_proximal': Goal2DTranslateProximal,
        'her': Goal2DHER,
        'rotate_her': Goal2DRotateRestrictedHER,
}

predator_prey_disk_augmentation_functions = {
        'rotate': Goal2DRotate,
        'translate': Goal2DTranslate,
        'translate_proximal': Goal2DTranslateProximal,
        'her': Goal2DHER
    }

AUGMENTATION_FUNCTIONS = {
    'InvertedPendulum': {
        'translate': InvertedPendulumTranslate,
        'reflect': InvertedPendulumReflect,
        'translate_reflect': InvertedPendulumTranslateReflect,
    },
    'InvertedDoublePendulum': {
        'translate': InvertedPendulumTranslate,
        'reflect': InvertedPendulumReflect,
        'translate_reflect': InvertedPendulumTranslateReflect,
    },
    'InvertedPendulumWide': {
        'translate': InvertedPendulumTranslate,
        'reflect': InvertedPendulumReflect,
        'translate_reflect': InvertedPendulumTranslateReflect,
    },
    'InvertedDoublePendulumWide': {
        'translate': InvertedPendulumTranslate,
        'reflect': InvertedPendulumReflect,
        'translate_reflect': InvertedPendulumTranslateReflect,
    },
    'CartPole': {
        'translate': InvertedPendulumTranslate,
        'reflect': InvertedPendulumReflect,
    },
    'LQR': {
        'translate': LQRTranslate,
        'rotate': LQRRotate,
    },
    'LQRGoal': {
        'translate': LQRTranslate,
        'rotate': LQRRotate,
    },
    'Swimmer': {
        'reflect': SwimmerReflect,
    },
    'Walker2d': {
        'reflect': Walker2dReflect,
    },
    'Goal2D': predator_prey_box_augmentation_functions,
    'Goal2DQuadrant': predator_prey_box_augmentation_functions,
    'Goal2DDense': predator_prey_box_augmentation_functions,
    'MeetUp': {
        'translate': MeetUpTranslate,
        'rotate': MeetUpRotate,
        'rotate_translate': MeetUpRotateTranslate,
    },
    'FetchReach': {
        'her': FetchHER,
        'translate': FetchReachTranslate,
        'translate_proximal': FetchReachTranslateProximal,
        'translate_goal': FetchTranslateGoal,
        'translate_goal_proximal': FetchTranslateGoalProximal,
        'reflect': FetchReachReflect,
    },
    'FetchReachDense': {
        'her': FetchHER,
        'translate_goal': FetchTranslateGoal,
        'translate_goal_proximal': FetchTranslateGoalProximal,
        'translate': FetchReachTranslate,
    },
    'FetchPush': {
        'her': FetchHER,
        'translate_goal': FetchTranslateGoal,
        'translate_goal_proximal': FetchTranslateGoalProximal,
        'translate': FetchPushTranslateGoal,
        'reflect': FetchPushReflect,
    },
    'FetchSlide': {
        'her': FetchHER,
        'translate_goal': FetchTranslateGoal,
        'translate_goal_proximal': FetchTranslateGoalProximal,
        'translate': FetchSlideTranslate,
        'reflect': FetchSlideReflect,
    },
    'FetchPickAndPlace': {
        'her': FetchHER,
        'translate_goal': FetchTranslateGoal,
        'translate_goal_proximal': FetchTranslateGoalProximal,
        'translate': FetchPickAndPlaceTranslate,
        'reflect': FetchPickAndPlaceReflect,
    },

    'dmc_cartpole_swingup_0': {
        'translate': InvertedPendulumTranslate,
        'reflect': InvertedPendulumReflect,
    },
    'dmc_cartpole_swingup_sparse_0': {
        'translate': InvertedPendulumTranslate,
        'reflect': InvertedPendulumReflect,
    },
    'dmc_cartpole_balance_0': {
        'translate': InvertedPendulumTranslate,
        'reflect': InvertedPendulumReflect,
    },
    'dmc_cartpole_balance_sparse_0': {
        'translate': InvertedPendulumTranslate,
        'reflect': InvertedPendulumReflect,
    },
}

for k in range(2,20+1):
    AUGMENTATION_FUNCTIONS[f'Reacher{k}'] = {
        'rotate': ReacherRotate,
        'reflect': ReacherReflect,
    }
    AUGMENTATION_FUNCTIONS[f'Reacher{k}Rand'] = {
        'rotate': ReacherRotate,
        'reflect': ReacherReflect,
    }
    AUGMENTATION_FUNCTIONS[f'Swimmer{k}'] = {
        'reflect': SwimmerReflect
    }

def validate_augmentation(env, aug_func, check_valid):
    observations, next_observations, actions, rewards, dones, infos = simulate(
        model=None, env=env, num_episodes=1, seed=np.random.randint(1,1000000), render=False, flatten=True, verbose=0)

    observations = np.expand_dims(observations, axis=1)
    next_observations = np.expand_dims(next_observations, axis=1)
    actions = np.expand_dims(actions, axis=1)
    rewards = np.expand_dims(rewards, axis=1)
    dones = np.expand_dims(dones, axis=1)
    infos = np.expand_dims(infos, axis=1)

    aug_n = 1

    for j in range(observations.shape[0]):
        aug_obs, aug_next_obs, aug_action, aug_reward, aug_done, aug_info = aug_func.augment(
            aug_n, observations[j], next_observations[j], actions[j], rewards[j], dones[j], infos[j])
        for k in range(aug_n):
            check_valid(env, aug_obs[k], aug_next_obs[k], aug_action[k], aug_reward[k], aug_done[k], aug_info[k])