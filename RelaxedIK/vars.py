__author__ = 'drakita'

import transformations as T

class Vars:
    def __init__(self, arm, init_state, objectives, weight_funcs, weight_priors, constraints, bounds):
        # Main objects ###########################################################
        self.arm = arm
        #########################################################################


        # Solver variables #######################################################
        self.xopt = init_state
        self.goal_pos = []
        self.goal_quat = []
        self.init_state = init_state
        self.init_ee_pos = self.arm.getFrames(self.init_state)[0][-1]
        self.init_ee_quat = T.quaternion_from_matrix(self.arm.getFrames(self.init_state)[1][-1])
        self.prev_state = self.init_state
        self.prev_ee_pos = self.init_ee_pos
        self.prev_ee_quat = self.init_ee_quat
        self.prev_goal_pos =  self.init_ee_pos
        self.prev_goal_quat = self.init_ee_quat
        self.handVelMax = 0.04
        self.frames = []
        self.vel_objectives_on = True
        ##########################################################################


        # data log ###############################################################
        self.all_states = []
        self.all_ee_pos = []
        self.all_hand_pos = []
        ##########################################################################


        # function pointers #######################################################
        self.objectives = objectives
        self.weight_funcs = weight_funcs
        self.constraints = constraints
        self.bounds = bounds
        ###########################################################################


        # weights #################################################################
        self.weight_priors = weight_priors
        ###########################################################################


