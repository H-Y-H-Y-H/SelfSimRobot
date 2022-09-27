import pybullet as p
import time
import pybullet_data
from math import sqrt, cos, sin, pi
import os
import numpy as np
import csv

for m in range(1):
    savearray = []
    filename = "robot_arm"

    for n in range(1):
        # or p.DIRECT for non-graphical version
        physicsClient = p.connect(1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf")
        startPos = [0, 0, 0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])

        boxId = p.loadURDF(filename + ".urdf", startPos, startOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        # boxId2 = p.loadURDF("cube.urdf", [0, 0.2, 0], startOrientation, useFixedBase=0, flags=p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT, globalScaling = 0.05)

        # num_joints = p.getNumJoints(boxId)
        # print(num_joints)
        # print(p.getLinkState(boxId,12))

        #ee_link index = 12
        #base_rot_joint_index = 0
        #shuold_index = 1
        #elbow_index = 2
        #index = 3
        #index = 4
        #left_gripper = 10
        #right_gripper = 11

        ik_angle = p.calculateInverseKinematics(boxId, 12, targetPosition = [0.1, 0.2, 0.1])
        # print(ik_angle)
        # print(len(ik_angle))
        # p.setJointMotorControlArray(boxId, [0, 1, 2, 3, 4, 5], p.POSITION_CONTROL, targetPositions = ik_angle[0:6])
        p.setJointMotorControl2(boxId, 0, p.POSITION_CONTROL, targetPosition = 0, force=2, maxVelocity=100)
        for i in range(200):
            p.stepSimulation()

        print(p.getLinkState(boxId,12))

        for i in range(20000):
            p.stepSimulation()
            time.sleep(1/240)

        p.disconnect()

    # with open(filename + ".csv", 'w')as f:
    #    write = csv.writer(f)
    #    write.writerow(savearray)
