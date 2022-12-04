from augment.rl.augmentation_functions.fetch.reach import *
from augment.rl.augmentation_functions.fetch.push import *
from augment.rl.augmentation_functions.fetch.slide import *
from augment.rl.augmentation_functions.fetch.pick_and_place import *
from augment.rl.augmentation_functions.fetch.common import *

from augment.rl.augmentation_functions.goal2d import GOAL2D_AUG_FUNCTIONS
from augment.rl.augmentation_functions.goal2dkey import GOAL2DKEY_AUG_FUNCTIONS
from augment.rl.augmentation_functions.cartpole import CARTPOLE_AUG_FUNCTIONS
from augment.rl.augmentation_functions.humanoid import HUMANOID_AUG_FUNCTIONS
from augment.rl.augmentation_functions.panda import PANDA_AUG_FUNCTIONS
from augment.rl.augmentation_functions.swimmer import SWIMMER_AUG_FUNCTIONS


from augment.rl.augmentation_functions.reacher_k import *
from augment.rl.augmentation_functions.walker2d import WALKER2D_AUG_FUNCTIONS
from augment.rl.augmentation_functions.ant import ANT_AUG_FUNCTIONS

from augment.simulate import simulate



AUGMENTATION_FUNCTIONS = {
    # Toy
    'Goal2D': GOAL2D_AUG_FUNCTIONS,
    'Goal2DQuadrant': GOAL2D_AUG_FUNCTIONS,
    'Goal2DDense': GOAL2D_AUG_FUNCTIONS,
    'Goal2DKey': GOAL2DKEY_AUG_FUNCTIONS,

    # Pendulum-like environments
    'InvertedPendulum': CARTPOLE_AUG_FUNCTIONS,
    'InvertedDoublePendulum': CARTPOLE_AUG_FUNCTIONS,
    'CartPole': CARTPOLE_AUG_FUNCTIONS,
    'dmc_cartpole_swingup_0': CARTPOLE_AUG_FUNCTIONS,
    'dmc_cartpole_swingup_sparse_0': CARTPOLE_AUG_FUNCTIONS,
    'dmc_cartpole_balance_0': CARTPOLE_AUG_FUNCTIONS,
    'dmc_cartpole_balance_sparse_0': CARTPOLE_AUG_FUNCTIONS,

    # locomotion
    'Swimmer': SWIMMER_AUG_FUNCTIONS,
    'Walker2d': WALKER2D_AUG_FUNCTIONS,
    'Ant': ANT_AUG_FUNCTIONS,
    'Humanoid': HUMANOID_AUG_FUNCTIONS,

    # robotics
    'PandaReach': PANDA_AUG_FUNCTIONS,
    'PandaPush': PANDA_AUG_FUNCTIONS,
    'PandaSlide': PANDA_AUG_FUNCTIONS,
    'PandaPickAndPlace': PANDA_AUG_FUNCTIONS,

    'FetchReach': {
        'her': FetchHER,
        'translate': FetchReachTranslate,
        'translate_proximal': FetchReachTranslateProximal,
        'translate_goal': FetchReachTranslateGoal,
        'translate_goal_proximal': FetchReachTranslateGoalProximal,
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
        'translate': FetchTranslateGoal,
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
}

# for k in range(2,20+1):
#     AUGMENTATION_FUNCTIONS[f'Reacher{k}'] = {
#         'rotate': ReacherRotate,
#         'reflect': ReacherReflect,
#     }
#     AUGMENTATION_FUNCTIONS[f'Reacher{k}Rand'] = {
#         'rotate': ReacherRotate,
#         'reflect': ReacherReflect,
#     }
#     AUGMENTATION_FUNCTIONS[f'Swimmer{k}'] = {
#         'reflect': SwimmerReflect
#     }