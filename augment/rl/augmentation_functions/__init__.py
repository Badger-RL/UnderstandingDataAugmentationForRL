from augment.rl.augmentation_functions.fetch import FetchReachHER
from augment.rl.augmentation_functions.inverted_pendulum import *
from augment.rl.augmentation_functions.lqr import LQRTranslate, LQRRotate
from augment.rl.augmentation_functions.predator_prey import PredatorPreyRotate, PredatorPreyTranslate, \
    PredatorPreyTranslateProximal
from augment.rl.augmentation_functions.reacher_k import *
from augment.rl.augmentation_functions.swimmer_k import SwimmerReflect
from augment.rl.augmentation_functions.walker2d import Walker2dReflect

predator_prey_augmentation_functions = {
        'rotate': PredatorPreyRotate,
        'translate': PredatorPreyTranslate,
        'translate_proximal': PredatorPreyTranslateProximal,
    }

AUGMENTATION_FUNCTIONS = {
    'InvertedPendulum-v2': {
        'translate': InvertedPendulumTranslateUniform,
        'translate_uniform': InvertedPendulumTranslateUniform,
        'reflect': InvertedPendulumReflect,
    },
    'InvertedDoublePendulum-v2': {
        'translate': InvertedPendulumTranslateUniform,
        'translate_uniform': InvertedPendulumTranslateUniform,
        'reflect': InvertedPendulumReflect,
    },
    'CartPole-v1': {
        'translate': InvertedPendulumTranslate,
        'translate_uniform': InvertedPendulumTranslateUniform,
        'reflect': InvertedPendulumReflect,
    },
    'LQR-v0': {
        'translate': LQRTranslate,
        'rotate': LQRRotate,
    },
    'LQRGoal-v0': {
        'translate': LQRTranslate,
        'rotate': LQRRotate,
    },
    'Swimmer-v3': {
        'reflect': SwimmerReflect,
    },
    'Walker2d-v3': {
        'reflect': Walker2dReflect,
    },
    'PredatorPrey-v0': predator_prey_augmentation_functions,
    'PredatorPreyBox-v0': predator_prey_augmentation_functions,
    'PredatorPreyDense-v0': predator_prey_augmentation_functions,
    'PredatorPreyBoxDense-v0': predator_prey_augmentation_functions,
    'FetchReach-v1': {
        'her': FetchReachHER,
    }
}

for k in range(2,20+1):
    AUGMENTATION_FUNCTIONS[f'Reacher{k}-v3'] = {'rotate': Rotate}
    AUGMENTATION_FUNCTIONS[f'Swimmer{k}-v3'] = {'reflect': SwimmerReflect}