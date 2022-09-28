import pybullet as p
import time
import pybullet_data
from math import sqrt, cos, sin, pi
import os
import numpy as np
import csv


filename = "robot_arm1"

# or p.DIRECT for non-graphical version
physicsClient = p.connect(1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

boxId = p.loadURDF(filename + ".urdf", startPos, startOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
boxId2 = p.loadURDF("cube_small.urdf", [0, 0.2, 0.025], startOrientation, useFixedBase=0, flags=p.URDF_USE_SELF_COLLISION)

p.changeDynamics(boxId, 10, lateralFriction=0.99,spinningFriction=0.02,rollingFriction=0.002)
p.changeDynamics(boxId, 11, lateralFriction=0.99,spinningFriction=0.02,rollingFriction=0.002)
p.changeDynamics(boxId2, -1, lateralFriction=0.99,spinningFriction=0.02,rollingFriction=0.002)
num_joints = p.getNumJoints(boxId)
print(num_joints)
# print(p.getLinkState(boxId,12))

#ee_link index = 9
#base_rot_joint_index = 0
#shuold_index = 1
#elbow_index = 2
#index = 3
#index = 4
#left_gripper = 7, lower="0" upper="0.03202"
#right_gripper = 8, lower="0" upper="0.03202"

ik_angle = p.calculateInverseKinematics(boxId, 9, targetPosition = [0, 0.2, 0.2],maxNumIterations = 200, targetOrientation = p.getQuaternionFromEuler([0,1.57,1.57]))

t = 0
for i in range(20000):
    p.stepSimulation()
    p.setJointMotorControlArray(boxId, [7, 8], p.POSITION_CONTROL, targetPositions=[0.032*sin(t), 0.032*sin(t)])
    t+=0.01
    time.sleep(1/240)

p.disconnect()
