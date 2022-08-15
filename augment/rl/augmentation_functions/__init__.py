from augment.rl.augmentation_functions.inverted_pendulum import *
from augment.rl.augmentation_functions.reacher_k import *

AUGMENTATION_FUNCTIONS = {
    'InvertedPendulum-v2': {
        'translate': InvertedPendulumTranslate,
        'translate_uniform': InvertedPendulumTranslateUniform,
        'reflect': InvertedPendulumReflect,
    },
    'InvertedDoublePendulum-v2': {
        'translate': InvertedPendulumTranslate,
        'translate_uniform': InvertedPendulumTranslateUniform,
        'reflect': InvertedPendulumReflect,
    },
}

for k in range(2,20+1):
    AUGMENTATION_FUNCTIONS[f'Reacher{k}-v3'] = {'rotate': Rotate}