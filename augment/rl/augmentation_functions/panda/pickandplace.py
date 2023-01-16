import copy
from typing import Dict, List, Any

import numpy as np

from augment.rl.augmentation_functions.panda.common import PANDA_AUG_FUNCTIONS, TranslateObject, \
    TranslateObjectProximal0, TranslateObjectProximal, CoDA


class TranslateObjectPick(TranslateObject):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.aug_threshold = np.array([0.03, 0.10, 0.05])  # largest distance from center to block edge = 0.02

class TranslateObjectProximal0Pick(TranslateObjectProximal0):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.aug_threshold = np.array([0.03, 0.1, 0.05])  # largest distance from center to block edge = 0.02


class TranslateObjectProximalPick(TranslateObjectProximal):

    def __init__(self, env, p=0.5, **kwargs):
        super().__init__(env=env, **kwargs)
        self.p = p
        self.aug_threshold = np.array([0.03, 0.10, 0.05])  # largest distance from center to block edge = 0.02

class CoDAPick(CoDA):

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.aug_threshold = np.array([0.03, 0.10, 0.05])  # largest distance from center to block edge = 0.02

PANDA_PICKANDPLACE_AUG_FUNCTIONS = copy.deepcopy(PANDA_AUG_FUNCTIONS)
# PANDA_PICKANDPLACE_AUG_FUNCTIONS.update(
#     {
#         'translate_object': TranslateObject,
#     }
# )