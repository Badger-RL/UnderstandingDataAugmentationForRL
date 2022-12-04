from augment.rl.augmentation_functions.goal2d import GOAL2D_AUG_FUNCTIONS
from augment.rl.augmentation_functions.goal2dkey import GOAL2DKEY_AUG_FUNCTIONS
from augment.rl.augmentation_functions.humanoidstandup import HUMANOIDSTANDUP_AUG_FUNCTIONS
from augment.rl.augmentation_functions.inverted_double_pendulum import INVERTED_DOUBLE_PENDULUM_AUG_FUNCTIONS
from augment.rl.augmentation_functions.inverted_pendulum import INVERTED_PENDULUM_AUG_FUNCTIONS
from augment.rl.augmentation_functions.humanoid import HUMANOID_AUG_FUNCTIONS
from augment.rl.augmentation_functions.panda import PANDA_AUG_FUNCTIONS, PANDA_PUSH_AUG_FUNCTIONS, \
    PANDA_SLIDE_AUG_FUNCTIONS
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
    'InvertedPendulum': INVERTED_PENDULUM_AUG_FUNCTIONS,
    'InvertedDoublePendulum': INVERTED_DOUBLE_PENDULUM_AUG_FUNCTIONS,

    'CartPole': INVERTED_PENDULUM_AUG_FUNCTIONS,
    'dmc_cartpole_swingup_0': INVERTED_PENDULUM_AUG_FUNCTIONS,
    'dmc_cartpole_swingup_sparse_0': INVERTED_PENDULUM_AUG_FUNCTIONS,
    'dmc_cartpole_balance_0': INVERTED_PENDULUM_AUG_FUNCTIONS,
    'dmc_cartpole_balance_sparse_0': INVERTED_PENDULUM_AUG_FUNCTIONS,

    # locomotion
    'Swimmer': SWIMMER_AUG_FUNCTIONS,
    'Walker2d': WALKER2D_AUG_FUNCTIONS,
    'Ant': ANT_AUG_FUNCTIONS,
    'Humanoid': HUMANOID_AUG_FUNCTIONS,
    'HumanoidStandup': HUMANOIDSTANDUP_AUG_FUNCTIONS,

    # robotics
    'PandaReach': PANDA_AUG_FUNCTIONS,
    'PandaPush': PANDA_PUSH_AUG_FUNCTIONS,
    'PandaSlide': PANDA_SLIDE_AUG_FUNCTIONS,
    'PandaPickAndPlace': PANDA_AUG_FUNCTIONS,
    'PandaStack': PANDA_AUG_FUNCTIONS,
    'PandaFlip': PANDA_AUG_FUNCTIONS,
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