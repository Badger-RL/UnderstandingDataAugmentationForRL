from augment.rl.augmentation_functions.inverted_pendulum import *
from augment.rl.augmentation_functions.reacher_k import *

AUGMENTATION_FUNCTIONS = {
    'InvertedPendulum-v2': {
        'translate': InvertedPendulumTranslate,
        'translate_uniform': InvertedPendulumTranslateUniform,
        'reflect': InvertedPendulumReflect,
    },
    'Reacher4-v3': {
        'rotate': Rotate,
    }
}