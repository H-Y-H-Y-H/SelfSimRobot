import pybullet as p
import time
import pybullet_data as pd
import gym
import random
import numpy as np
from PIL import Image
import scipy.linalg as linalg
from ray_test import point_test, inside_data_sampling, pixel_sampling, face_sampling, get_shadow

force = 1.8
maxVelocity = 1.5

"""camera parameters"""
width = 256
height = 256

view_matrix = p.computeViewMatrix(
    cameraEyePosition=[0, -1, 0.2],
    cameraTargetPosition=[0, 0, 0.2],
    cameraUpVector=[0, 0, 1])

projection_matrix = p.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=10.0)
"""camera parameters"""


def angle_sim(angle_list, robot_id):
    for i in range(200):
        pos_value = angle_list * np.pi
        for joint in range(3):
            p.setJointMotorControl2(robot_id, joint, controlMode=p.POSITION_CONTROL, targetPosition=pos_value[joint],
                                    force=force,
                                    maxVelocity=maxVelocity)

        p.stepSimulation()
        # get_shadow(box_len=1, num_points=1001, filename="data-02/arm-shadow%d.csv" % idx)
        time.sleep(1. / 240.)


if __name__ == "__main__":
    DATA_PATH = "image/01/"
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pd.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    # planeId = p.loadURDF("plane.urdf")
    textureId = p.loadTexture("green.png")
    WallId_front = p.loadURDF("plane.urdf", [0, 5, 0], p.getQuaternionFromEuler([1.57, 0, 0]))
    p.changeVisualShape(WallId_front, -1, textureUniqueId=textureId)
    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])

    urdf_path = 'arm3dof/urdf/arm3dof.urdf'
    # urdf_path = 'robot_arm/robot_arm1.urdf'

    robotid = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=1)
    basePos, baseOrn = p.getBasePositionAndOrientation(robotid)  # Get model position
    basePos_list = [basePos[0], basePos[1], 0.3]
    # p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=75, cameraPitch=-20,
    #                              cameraTargetPosition=basePos_list)  # fix camera onto model

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
        # pixel_sampling(filename="arm-pix-{a1:.2f}-{a2:.2f}-{a3:.2f}.csv".format(a1=angle01, a2=angle02+1.4, a3=angle03+0.5))
        # get_shadow(box_len=1, num_points=1001, filename="data-01/arm-shadow%d.csv"%idx)
        # face_sampling(box_len=0.4, num_points=63, filename="arm-pix%d.csv"%idx)
        img = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgbBuffer = img[2]

        img = Image.fromarray(rgbBuffer[:, :, :3], 'RGB')
        img.save(DATA_PATH + "%d.png" % idx)
        img.show()
        print(rgbBuffer.shape)

    et = time.time()
    print("Time: ", et - st)

    cubePos, cubeOrn = p.getBasePositionAndOrientation(robotid)
    print(cubePos, cubeOrn)
    p.disconnect()
