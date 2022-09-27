import pybullet as p
import time
import pybullet_data as pd
import gym
import random
import numpy as np
import scipy.linalg as linalg
from ray_test import point_test, inside_data_sampling, pixel_sampling, face_sampling

force = 1.8
maxVelocity = 1.5

def angle_sim(angle_list, robot_id):
    for i in range (200):
        pos_value = angle_list * np.pi
        for joint in range(3):
            p.setJointMotorControl2(robot_id, joint, controlMode=p.POSITION_CONTROL, targetPosition=pos_value[joint],
                                    force=force,
                                    maxVelocity=maxVelocity)

        p.stepSimulation()
        time.sleep(1./240.)


if __name__ == "__main__":
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

    # p.addUserDebugLine([0.3,0.3,0], [0.3,0.3,0.4], [0.2,0,0])
    # p.addUserDebugLine([0.3,-0.3,0], [0.3,-0.3,0.4], [0.2,0,0])
    # p.addUserDebugLine([-0.3,-0.3,0], [-0.3,-0.3,0.4], [0.2,0,0])
    # p.addUserDebugLine([-0.3,0.3,0], [-0.3,0.3,0.4], [0.2,0,0])

    st = time.time()
    for idx in range(10):
        angle01 = random.random()
        angle02 = random.random() * 0.8 - 1.4
        angle03 = random.random() - 0.5
        angle_sim(np.array([angle01, angle02, angle03]), robotid)
        pixel_sampling(filename="arm-pix-{a1:.2f}-{a2:.2f}-{a3:.2f}.csv".format(a1=angle01, a2=angle02+1.4, a3=angle03+0.5))
        # face_sampling(box_len=0.4, num_points=63, filename="arm-pix%d.csv"%idx)

    et = time.time()
    print("Time: ", et-st)

    cubePos, cubeOrn = p.getBasePositionAndOrientation(robotid)
    print(cubePos,cubeOrn)
    p.disconnect()
