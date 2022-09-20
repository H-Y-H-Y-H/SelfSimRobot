import pybullet as p
import time
import pybullet_data as pd
import gym
import random
import numpy as np
import scipy.linalg as linalg


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pd.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
robotid = p.loadURDF("arm3dof/urdf/arm3dof.urdf",startPos, startOrientation,useFixedBase=1)
basePos, baseOrn = p.getBasePositionAndOrientation(robotid)  # Get model position
basePos_list = [basePos[0], basePos[1], 0.3]
p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=75, cameraPitch=-20,
                             cameraTargetPosition=basePos_list)  # fix camera onto model
force = 1.8
maxVelocity = 1.5
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)

for i in range (10000):
    pos_value = [0,-np.pi,0]
    for joint in range(3):
        p.setJointMotorControl2(robotid, joint, controlMode=p.POSITION_CONTROL, targetPosition=pos_value[joint],
                                force=force,
                                maxVelocity=maxVelocity)

    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(robotid)
print(cubePos,cubeOrn)
p.disconnect()

