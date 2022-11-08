import os
import random
import numpy as np
import pybullet as p
import time
import pybullet_data as pd
import matplotlib.pyplot as plt
import cv2
render_flag = False
OFFLINE = True

force = 1.8
maxVelocity = 1.5


width = 400
height = 400


DATA_PATH = "data/white_body/"
os.makedirs(DATA_PATH,exist_ok=True)
os.makedirs(DATA_PATH + 'image',exist_ok=True)

"""camera parameters"""
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[0.8, 0, 1.106],
    cameraTargetPosition=[-1, 0, 1.106],
    cameraUpVector=[0, 0, 1])

projection_matrix = p.computeProjectionMatrixFOV(
    fov=42.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=200)


def get_musk(orig_image):
    new_image = orig_image.copy()
    for w in range(width):
        for h in range(height):
            # print(orig_image[w, h])
            if orig_image[w, h, 0] < 120 or orig_image[w, h, 2] < 120:
                new_image[w, h] = np.array([0., 0., 0.])
    return new_image


def get_img():
    img = p.getCameraImage(width, height, view_matrix, projection_matrix,
                           renderer=p.ER_BULLET_HARDWARE_OPENGL,
                           shadow=0)
    rgbBuffer = img[2][:, :, :3]
    # musk = get_musk(rgbBuffer)
    cv2.imshow('Windows',rgbBuffer)
    cv2.waitKey(1)
    return rgbBuffer


def angle_sim(angle_list, robot_id):
    link_num = 3
    joint_data = []
    for i in range(1000):
        for joint in range(link_num):
            p.setJointMotorControl2(robot_id, joint, controlMode=p.POSITION_CONTROL, targetPosition=angle_list[joint],
                                    force=force,
                                    maxVelocity=maxVelocity)

        p.stepSimulation()

        joint_list = []
        for j in range(link_num):
            joint_state = p.getJointState(robot_id, j)[0]
            joint_list.append(joint_state)
        joint_data.append(joint_list)

        if abs(joint_list[0] - angle_list[0]) + abs(joint_list[1] - angle_list[1]) + abs(
                joint_list[2] - angle_list[2]) < 0.003:
            # print("reached")
            break

        if render_flag:
            time.sleep(1. / 240.)


def angle_convertor(a1, a2, a3):
    # a1, 0-1; a2, 0-1; a3, 0-1; a3 - 0.5
    a_range = [-50., 50.]
    # print(np.array([a1, a2, a3]) * (a_range[1] - a_range[0]) + a_range[0] + np.array([90, 90, 0]))
    return (np.array([a1, a2, a3]) * (a_range[1] - a_range[0]) + a_range[0] + np.array([90, 90, 0])) * np.pi / 180


if __name__ == "__main__":

    if render_flag:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    textureId = p.loadTexture("green.png")
    WallId_front = p.loadURDF("plane.urdf", [-1, 0, 0], p.getQuaternionFromEuler([0, 1.57, 0]))
    p.changeVisualShape(WallId_front, -1, textureUniqueId=textureId)
    # p.changeVisualShape(planeId, -1, textureUniqueId=textureId)
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])

    urdf_path = 'arm3dof/urdf/arm3dof.urdf'
    # urdf_path = 'robot_arm/robot_arm1.urdf'

    robotId = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=1)
    sphereRadius = 0.05
    colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)

    basePos, baseOrn = p.getBasePositionAndOrientation(robotId)  # Get model position
    basePos_list = [basePos[0], basePos[1], .8]
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20,
                                 cameraTargetPosition=basePos_list)  # fix camera onto model

    # get_img()
    p.addUserDebugLine([0, 0, 0.106], [0, 0, 0.5], [1, 0, 0])
    angles_record = []
    # angle_sim(np.array([1, 0.5, 0.6]) * np.pi, robotId)
    line_list = np.linspace(0.0, 1.0, num=21)
    idx = 0
    for angle01 in line_list:
        for angle02 in line_list:
            for angle03 in line_list:
                a_list = angle_convertor(angle01, angle02, angle03)
                angle_sim(a_list, robotId)
                image = get_img()
                plt.imsave(DATA_PATH + "image/" + "%d.png" % idx,image)
                angles_record.append(np.array([angle01, angle02, angle03]))
                print(idx)
                idx += 1

    """fixed angles version"""
    # for

    np.savetxt(DATA_PATH + "angles.csv", angles_record)

    p.disconnect()
    # cv2.destroyWindow()
