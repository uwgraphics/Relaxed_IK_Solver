__author__ = 'drakita'

from abc import ABCMeta, abstractmethod

class Constraint:
    __metaclass__ = ABCMeta

    @abstractmethod
    def constraintType(self):
        'return eq for equality and ineq for inequality'
        return None

    @abstractmethod
    def func(self, x, *args):
        pass

class SingularityConstraint(Constraint):
    def constraintType(self):
        return 'ineq'

    def func(self, x, *args):
        vars = args[0]
        arm = vars.arm
        y = arm.getYoshikawaMeasure(x)
        lower_bound = 1e-18
        return y - lower_bound


