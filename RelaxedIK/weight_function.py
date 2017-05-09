__author__ = 'drakita'

import numpy as np
import transformations as T
from abc import ABCMeta, abstractmethod, abstractproperty

class Weight_Function:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, vars):
        pass

class Hand_Vel_Weight(Weight_Function):
    def __call__(self, vars):
        velMax = vars.handVelMax
        handVel = np.linalg.norm(vars.goal_pos - vars.prev_goal_pos)
        return (max(0.0, velMax - handVel) / velMax) ** 2

class Identity_Weight(Weight_Function):
    def __call__(self, vars):
        return 1.0