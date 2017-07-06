import math
from relaxedIK import *

########################################################################################################################
# Danny Rakita 7/5/17
# This is sample code for getting the relaxed-IK solver up and running.  This script will simply generate the joint angles
# such that the end effector oscillates back and forth along the x-axis with respect to the initial configuration.  The
# orientation of the end effector should not change.


# This sample uses the ur5 robot.  The initial configuration is set to the zero configuration just for this example, this would
# rarely be useful in practice
init_state = [0,0,0,0,0,0]
robot_info = ('urdfs/ur5_robot_rotated.urdf', 'shoulder_pan_joint', 'wrist_3_joint', 'ee_fixed_joint')
ik = RelaxedIK(robot_info, init_state=init_state)

# To test that the solver is working with your robot, initialize the ik solver with your urdf file and other necessary information
# TODO: initialize your own robot
# init_state = [...]
# robot_info = (...)
# ik = RelaxedIK(robot_info, init_state=init_state)
#
# Note: if IK solver does not build successfully just by specifying the start and end joint names, you can manually specify
# the full joint name list as a separate argument.  See the documentation for details.


val = 0.0
stride = 0.001
while True:
    x_val = math.sin(val)
    print ik.solve([x_val, 0, 0], [1,0,0,0])
    val += stride
