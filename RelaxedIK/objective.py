__author__ = 'drakita'

import numpy as np
import transformations as T
import math as M
from abc import ABCMeta, abstractmethod

def objective_master(x, *args):
    vars = args[0]
    vars.frames = vars.arm.getFrames(x)
    objectives = vars.objectives
    weight_funcs = vars.weight_funcs
    weight_priors = vars.weight_priors
    objective_sum = 0.0
    for i,o in enumerate(objectives):
        if o.isVelObj() and not vars.vel_objectives_on:
            continue
        weight_func = weight_funcs[i]
        term_weight = weight_priors[i]*weight_func(vars)
        objective_sum += term_weight*o(x,vars)
    return objective_sum

#################################################################################################

class Objective:
    __metaclass__ = ABCMeta

    @abstractmethod
    def isVelObj(self):
        return False

    @abstractmethod
    def __call__(self, x, vars):
        pass

class Position_Obj(Objective):
    def isVelObj(self):
        return False

    def __call__(self, x, vars):
        positions = vars.frames[0]
        eePos = positions[-1]
        goal_pos = vars.goal_pos
        return np.linalg.norm(eePos - goal_pos)**2

class Orientation_Obj(Objective):
    def isVelObj(self):
        return False

    def __call__(self, x, vars):
        frames = vars.frames[1]
        eeMat = frames[-1]
        goal_quat = vars.goal_quat
        ee_quat = T.quaternion_from_matrix(eeMat)

        q = ee_quat
        ee_quat2 = [-q[0],-q[1],-q[2],-q[3]]

        disp = np.linalg.norm(T.quaternion_disp(goal_quat,ee_quat))
        disp2 = np.linalg.norm(T.quaternion_disp(goal_quat,ee_quat2))

        return min(disp,disp2)**2

class Min_EE_Vel_Obj(Objective):
    def isVelObj(self):
        return True

    def __call__(self, x, vars):
        jtPt = vars.frames[0][-1]
        return np.linalg.norm(vars.prev_ee_pos - jtPt)**2

class Min_Jt_Vel_Obj(Objective):
    def isVelObj(self):
        return True

    def __call__(self, x, vars):
        v = x - np.array(vars.prev_state)
        return np.linalg.norm(v)**2

class Min_Rot_Vel_Obj(Objective):
    def isVelObj(self):
        return True

    def __call__(self,x,vars):
        frames = vars.frames[1]
        eeMat = frames[-1]
        ee_quat = T.quaternion_from_matrix(eeMat)

        prev_ee_quat = vars.prev_ee_quat
        disp = np.linalg.norm(T.quaternion_disp(prev_ee_quat,ee_quat))

        return disp**2


'''
if quat_disp2 < quat_disp:
    vars.prev_ee_quat = ee_quat2
    ee_quat = ee_quat2
else:
    vars.prev_ee_quat = ee_quat

if np.linalg.norm(T.quaternion_disp(prev_ee_quat,ee_quat)) > 1.57:
    ee_quat = ee_quat2
    print 'yes!'

q = ee_quat
angle = 2*M.acos(q[0])
x = q[1] / M.sqrt(1 - q[0]*q[0])
y = q[2] / M.sqrt(1 - q[0]*q[0])
z = q[3] / M.sqrt(1 - q[0]*q[0])

axis = np.array([x,y,z])
if np.dot(np.array([0,0,1]), axis) > 0:
    ee_quat = [-q[0],-q[1],-q[2],-q[3]]

angle = 2*M.acos(q[0])
x = q[1] / M.sqrt(1 - q[0]*q[0])
y = q[2] / M.sqrt(1 - q[0]*q[0])
z = q[3] / M.sqrt(1 - q[0]*q[0])

new_angle = 2*M.pi - angle
new_axis = M.sin(new_angle/2)*-np.array([x,y,z])
goal_quat2 = [0,0,0,0]
goal_quat2[0] = M.cos(new_angle/2)
goal_quat2[1] = new_axis[0]
goal_quat2[2] = new_axis[1]
goal_quat2[3] = new_axis[2]
goal_quat2 = [-q[0],-q[1],-q[2],-q[3]]
'''