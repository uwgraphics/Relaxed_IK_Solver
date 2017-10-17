__author__ = 'drakita'
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
from RelaxedIK.urdf_load import *
from RelaxedIK.objective import *
from RelaxedIK.constraint import *
from RelaxedIK.vars import *
from RelaxedIK.colors import *
import RelaxedIK.transformations as T
import scipy.optimize as O
from RelaxedIK.objective import *
from RelaxedIK.weight_function import *

class RelaxedIK:
    def __init__(self, (urdf_path, start_joint, end_joint, fixed_ee_joint),
                 full_joint_list=(),
                 init_state=6*[0],
                 rotation_mode = 'relative', # could be 'absolute' or 'relative'
                 position_mode = 'relative', # could be 'absolute' or 'relative'
                 objectives=(Position_Obj(), Orientation_Obj(), Min_EE_Vel_Obj(), Min_Jt_Vel_Obj()),
                 weight_funcs=(Identity_Weight(), Identity_Weight(), Identity_Weight(), Identity_Weight()),
                 weight_priors=(12.0, 7.0, 1.0, 3.0),
                 constraints=(),
                 bounds=()):

        # check inputs ####################################################################################################################
        if (start_joint == '' or end_joint == '') and full_joint_list == ():
            print bcolors.FAIL + 'Invalid robot info.  Must either specify start and end joints or specify full joint list.' + bcolors.ENDC
            raise Exception('Invalid robot info.')
        if not (rotation_mode == 'relative' or rotation_mode == 'absolute'):
            print bcolors.FAIL + 'Invalid rotation_mode.  Must be <relative> or <absolute>.  Exiting.' + bcolors.ENDC
            raise ValueError('Invalid rotation_mode.')
        if not (position_mode == 'relative' or position_mode == 'absolute'):
            print bcolors.FAIL + 'Invalid position_mode.  Must be <relative> or <absolute>.  Exiting.' + bcolors.ENDC
            raise ValueError('Invalid position_mode.')
        num_objs = len(objectives)
        if not (num_objs == len(weight_funcs) == len(weight_priors)):
            print bcolors.FAIL + 'Invalid Inputs.  The number of objectives ({}) must be the same as the number' \
                                 'of weight functions ({}) and weight priors ({}).  Exiting.'.format(str(num_objs),
                                                                                                    str(len(weight_funcs)),
                                                                                                    str(len(weight_priors))) + bcolors.ENDC
            raise ValueError('Invalid function arguments.')
        ###################################################################################################################################

        urdf_robot, arm, tree = urdf_load(urdf_path, start_joint, end_joint, full_joint_list, fixed_ee_joint)

        self.urdf_robot = urdf_robot
        self.arm = arm
        self.tree = tree
        self.init_state = init_state
        self.rotation_mode = rotation_mode
        self.position_mode = position_mode
        self.constraints = constraints
        self.numDOF = len(arm.axes)
        if not self.numDOF == len(init_state):
            self.init_state = self.numDOF*[0]
            self.bounds = [tuple((-1000.0,1000.0)) for i in range(0,self.numDOF)]
            print bcolors.WARNING + 'WARNING: Length of init_state does not match number of robot DOFs.  Automatically ' \
                                    'initializing init_state as {}.  This may cause errors.'.format(str(self.init_state)) + bcolors.ENDC
        self.bounds = bounds
        self.vars = Vars(self.arm, self.init_state,objectives,weight_funcs,weight_priors,self.constraints, self.bounds)
        self.constraint_dict = self.__construct_constraint_dict(constraints)


    def solve(self, goal_pos, goal_quat, prev_state=None, vel_objectives_on=True, verbose_output=False):
        '''
        :param goal_pos:
        :param goal_quat:
        :param prev_state:
        :return:
        '''

        if self.rotation_mode == 'relative':
            self.vars.goal_quat = T.quaternion_multiply(goal_quat, self.vars.init_ee_quat)
        elif self.rotation_mode == 'absolute':
            self.vars.goal_quat = goal_quat

        # flip goal quat if necessary
        disp = np.linalg.norm(T.quaternion_disp(self.vars.prev_goal_quat,self.vars.goal_quat))
        q = self.vars.goal_quat
        if disp > M.pi / 2.0:
            self.vars.goal_quat = [-q[0],-q[1],-q[2],-q[3]]

        if self.position_mode == 'relative':
            self.vars.goal_pos = np.array(goal_pos) + self.vars.init_ee_pos
        elif self.position_mode == 'absolute':
            self.vars.goal_pos = np.array(goal_pos)


        if prev_state == None:
            initSol = self.vars.prev_state
        else:
            initSol = prev_state

        self.vars.vel_objectives_on = vel_objectives_on

        # Solve #############################################################################################################
        xopt_full = O.minimize(objective_master,initSol,constraints=self.constraint_dict,bounds=self.bounds,args=(self.vars,),method='slsqp',options={'maxiter':75,'disp':verbose_output})
        #####################################################################################################################

        xopt = xopt_full.x

        if verbose_output:
            print bcolors.OKBLUE + str(xopt_full) + bcolors.ENDC + '\n'

        # check rotation convergence
        frames = self.vars.arm.getFrames(xopt)[1]
        eeMat = frames[-1]
        goal_quat = self.vars.goal_quat
        ee_quat = T.quaternion_from_matrix(eeMat)
        q = goal_quat
        goal_quat2 = [-q[0],-q[1],-q[2],-q[3]]
        disp = np.linalg.norm(T.quaternion_disp(goal_quat,ee_quat))
        disp2 = np.linalg.norm(T.quaternion_disp(goal_quat2,ee_quat))

        # log info into vars
        self.vars.prev_state = self.vars.xopt
        self.vars.xopt = xopt
        self.vars.prev_ee_pos = self.vars.arm.getFrames(xopt)[0][-1]
        self.vars.prev_ee_quat = T.quaternion_from_matrix(self.arm.getFrames(xopt)[1][-1])
        self.vars.prev_goal_pos = self.vars.goal_pos
        if disp2 < disp:
            pass
            # self.vars.prev_goal_quat = goal_quat2
        else:
            pass
            # self.vars.prev_goal_quat = goal_quat

        self.vars.all_states.append(xopt)
        self.vars.all_ee_pos.append(self.vars.prev_ee_pos)
        self.vars.all_hand_pos.append(self.vars.goal_pos)

        return xopt

    def __construct_constraint_dict(self, constraints):
        constraint_dicts = []
        for c in constraints:
            d = {
                'type': c.constraintType(),
                'fun': c.func,
                'args': (self.vars,)
            }
            constraint_dicts.append(d)

        return tuple(constraint_dicts)








