

# Reach, Push, Slide, PickAndPlace only
from augment.rl.augmentation_functions.panda.common import HER, HERTranslateGoal, HERTranslateGoalProximal, \
    TranslateGoal, TranslateGoalProximal
from augment.rl.augmentation_functions.panda.reach import ReachReflect

PANDA_AUG_FUNCTIONS = {
    'her': HER,
    'her_translate_goal': HERTranslateGoal,
    'her_translate_goal_proximal': HERTranslateGoalProximal,
    'translate_goal': TranslateGoal,
    'translate_goal_proximal': TranslateGoalProximal,
}

PANDA_REACH_AUG_FUNCTIONS = {
    'reflect': ReachReflect,
}
PANDA_REACH_AUG_FUNCTIONS.update(PANDA_AUG_FUNCTIONS)