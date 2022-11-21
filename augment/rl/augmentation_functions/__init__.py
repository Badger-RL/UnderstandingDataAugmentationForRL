from augment.rl.augmentation_functions.fetch.reach import *
from augment.rl.augmentation_functions.fetch.push import *
from augment.rl.augmentation_functions.fetch.slide import *
from augment.rl.augmentation_functions.fetch.pick_and_place import *

from augment.rl.augmentation_functions.inverted_pendulum import *
from augment.rl.augmentation_functions.lqr import LQRTranslate, LQRRotate
from augment.rl.augmentation_functions.meetup import MeetUpTranslate, MeetUpRotate, MeetUpRotateTranslate
from augment.rl.augmentation_functions.predator_prey import *
from augment.rl.augmentation_functions.reacher_k import *
from augment.rl.augmentation_functions.swimmer_k import SwimmerReflect
from augment.rl.augmentation_functions.walker2d import Walker2dReflect

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
        'her': FetchReachHER,
        'translate': FetchReachTranslate,
        'translate_proximal': FetchReachTranslateProximal,
        'reflect': FetchReachReflect,
    },
    'FetchReachDense': {
        'her': FetchReachHER,
        'translate': FetchReachTranslate,
    },
    'FetchPush': {
        'her': FetchPushHER,
        'translate': FetchPushTranslate,
        'reflect': FetchPushReflect,
    },
    'FetchSlide': {
        'her': FetchSlideHER,
        'translate': FetchSlideTranslate,
        'reflect': FetchSlideReflect,
    },
    'FetchPickAndPlace': {
        'her': FetchPickAndPlaceHER,
        'translate': FetchPickAndPlaceTranslate,
        'reflect': FetchPickAndPlaceReflect,
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