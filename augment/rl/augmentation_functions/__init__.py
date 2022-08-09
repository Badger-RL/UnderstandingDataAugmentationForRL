from augment.rl.augmentation_functions.inverted_pendulum import *

AUGMENTATION_FUNCTIONS = {
    'InvertedPendulum-v2': {
        'translate': InvertedPendulumTranslate,
        'translate_uniform': InvertedPendulumTranslateUniform,
        'reflect': InvertedPendulumReflect,
    }
}