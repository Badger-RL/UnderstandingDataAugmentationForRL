import copy
from typing import Dict, List, Any

import numpy as np

from augment.rl.augmentation_functions.panda.common import PANDA_AUG_FUNCTIONS, TranslateObject, \
    TranslateObjectProximal0, TranslateObjectProximal, CoDA, TranslateGoalProximal


class TranslateObjectPick(TranslateObject):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.aug_threshold = np.array([0.05, 0.10, 0.05])  # largest distance from center to block edge = 0.02

class TranslateObjectProximal0Pick(TranslateObjectProximal0):

    def __init__(self, env, **kwargs):
        super().__init__(env=env, **kwargs)
        self.aug_threshold = np.array([0.05, 0.1, 0.05])  # largest distance from center to block edge = 0.02

#
# class TranslateObjectProximalPick(TranslateObjectProximal):
#
#     def __init__(self, env, p=0.5, **kwargs):
#         super().__init__(env=env, **kwargs)
#         self.p = p
#         self.aug_threshold = np.array([0.03, 0.10, 0.05])  # largest distance from center to block edge = 0.02


class TranslateObjectProximalPick(TranslateObjectPick):
    def __init__(self, env, p=0.5, **kwargs):
        super().__init__(env, **kwargs)
        self.aug_threshold = np.array([0.03, 0.10, 0.05])  # largest distance from center to block edge = 0.02
        self.TranslateObjectProximal0 = TranslateObjectProximal0Pick(env, **kwargs)
        self.TranslateGoalProximal1 = TranslateGoalProximal(env, p=1, **kwargs)
        self.q = p

    def _augment(self,
                 obs: np.ndarray,
                 next_obs: np.ndarray,
                 action: np.ndarray,
                 reward: np.ndarray,
                 done: np.ndarray,
                 infos: List[Dict[str, Any]],
                 p=None,
                 **kwargs,
                 ):

        if np.random.random() < self.q:
            return self.TranslateObjectProximal0._augment(obs, next_obs, action, reward, done, infos, p, **kwargs)
        else:
            return self.TranslateGoalProximal1._augment(obs, next_obs, action, reward, done, infos, **kwargs,)


class CoDAPick(CoDA):

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.aug_threshold = np.array([0.05, 0.10, 0.05])  # largest distance from center to block edge = 0.02

PANDA_PICKANDPLACE_AUG_FUNCTIONS = copy.deepcopy(PANDA_AUG_FUNCTIONS)
PANDA_PICKANDPLACE_AUG_FUNCTIONS.update(
    {
        'translate_object': TranslateObjectPick,
        'translate_object_proxmial_0': TranslateObjectProximal0Pick,
        'translate_object_proximal': TranslateObjectProximalPick,
    }
)