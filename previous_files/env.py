import pybullet as p
import time
import pybullet_data as pd
from urdfpy import URDF
import gym
import random
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from PIL import Image
import scipy.linalg as linalg
from ray_test import point_test, inside_data_sampling, pixel_sampling, face_sampling, get_shadow

# DATA_PATH = "musk_data/dataset01/"
DATA_PATH = "/Users/jionglin/Downloads/vsm/vsm_data_03/"
# DATA_PATH = "data/"
force = 1.8
maxVelocity = 1.5
robot_for_fk = URDF.load('../arm3dof/urdf/arm3dof.urdf')

"""camera parameters"""
# width = 256
# height = 256
width = 128
height = 128

view_matrix = p.computeViewMatrix(
    cameraEyePosition=[0, -1, 0.2],
    cameraTargetPosition=[0, 0, 0.2],
    cameraUpVector=[0, 0, 1])

projection_matrix = p.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=3.0)
"""camera parameters"""


def angle_sim(angle_list, robot_id, file_dir, sim_only=False):
    link_num = 3
    joint_data = []
    for i in range(500):
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

        if sim_only:
            pass
        else:
            if i % 10 == 0:
                get_image_and_save_data(sub_dir=file_dir, index=i)

        if abs(joint_list[0] - angle_list[0]) + abs(joint_list[1] - angle_list[1]) + abs(
                joint_list[2] - angle_list[2]) < 0.003:
            # print("reached")
            break

        time.sleep(1. / 240.)

    return joint_data


def get_musk(rgb_data):
    mask_data = np.zeros((height, width))
    for row in range(height):
        for col in range(width):
            if rgb_data[row, col][0] < 100 and rgb_data[row, col][1] > 100 and rgb_data[row, col][2] < 100:
                pass
            else:
                mask_data[row, col] = 1

    return mask_data


def get_image_and_save_data(sub_dir, index, is_save=True, no_sub=False):
    img = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgbBuffer = img[2][:, :, :3]
    musk = get_musk(rgbBuffer)
    if no_sub:
        path = DATA_PATH + "img/"
        if not os.path.exists(path):
            os.makedirs(path)
            print("make dirs: ", path)
        if is_save:
            np.savetxt(path + sub_dir+".csv", musk)
    else:
        path = DATA_PATH + sub_dir
        if not os.path.exists(path):
            os.makedirs(path)
            print("make dirs: ", path)
        if is_save:
            np.savetxt(path + "%d.csv" % index, musk)
    # img = Image.fromarray(rgbBuffer, 'RGB')
    # img.save(DATA_PATH + "%d.png" % idx)
    # img.show()


def get_link_transform_matrix(link_id, is_save=True):
    """not use, urdfpy instead"""
    path = "transform_data/data_01/"
    link_state = p.getLinkState(robotid, link_id)
    link_pos = np.array(link_state[0]).reshape(3, 1)
    link_ori = link_state[1]
    r = R.from_quat(link_ori)
    r_m = r.as_matrix()
    transform_m = np.append(r_m, link_pos, axis=1)
    transform_m = np.append(transform_m, [[0, 0, 0, 1]], axis=0)
    print(transform_m)
    if is_save:
        np.savetxt(path + "link%d.csv" % link_id, transform_m)
    return transform_m


def get_ik(loop_id, angle_list):
    parent_dir = DATA_PATH + "transform_data/"
    sub_dir = "data_" + str(loop_id) + "/"
    path = parent_dir + sub_dir

    if not os.path.exists(path):
        os.makedirs(path)
        # print("make dirs: " + path)

    #  update fk
    fk = robot_for_fk.link_fk(cfg={
        'base_link1': angle_list[0],
        'link1_link2': angle_list[1],
        'link2_link3': angle_list[2],
    })

    # fk = robot_for_fk.link_fk()
    for i in range(1, 4):
        # print(i, fk[robot_for_fk.links[i]])
        np.savetxt(path + "link%d.csv" % i, fk[robot_for_fk.links[i]])


if __name__ == "__main__":
    # physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    textureId = p.loadTexture("green.png")
    WallId_front = p.loadURDF("plane.urdf", [0, 1, 0], p.getQuaternionFromEuler([1.57, 0, 0]))
    p.changeVisualShape(WallId_front, -1, textureUniqueId=textureId)
    p.changeVisualShape(planeId, -1, textureUniqueId=textureId)
    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])

    urdf_path = '../arm3dof/urdf/arm3dof.urdf'
    # urdf_path = 'robot_arm/robot_arm1.urdf'

    robotid = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=1)
    basePos, baseOrn = p.getBasePositionAndOrientation(robotid)  # Get model position
    basePos_list = [basePos[0], basePos[1], 0.3]
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=75, cameraPitch=-20,
                                 cameraTargetPosition=basePos_list)  # fix camera onto model

    st = time.time()
    discrete = True
    if discrete:
        """discrete angles, +-30"""
        angle_idx_list = np.linspace(-15, 15, 31)
        print(angle_idx_list)
        idx = 0
        for angle01 in angle_idx_list:
            for angle02 in angle_idx_list:
                for angle03 in angle_idx_list:
                    a_list = np.array([angle01, angle02 + 90, angle03]) * np.pi / 180
                    angle_sim(a_list, robotid, "n" + "/", sim_only=True)
                    get_image_and_save_data(sub_dir=str(angle01)+"_"+str(angle02)+"_"+str(angle03), index=idx, no_sub=True)
                    get_ik(loop_id=idx, angle_list=a_list)
                    idx += 1
                    print(idx)

    else:
        joint_path = DATA_PATH + "joint_data/"
        if not os.path.exists(joint_path):
            os.makedirs(joint_path)
        for idx in range(100):
            # 0< angle01 <1, 0.1< angle02 <0.9, -0.5< angle03 <0.5
            """random angles version"""
            angle01 = random.random()
            angle02 = random.random() * 0.8 + 0.1
            angle03 = random.random() - 0.5
            a_list = np.array([angle01, angle02, angle03]) * np.pi

            jointData = angle_sim(a_list, robotid, str(idx) + "/", sim_only=True)
            print(len(jointData))
            get_ik(loop_id=idx, angle_list=a_list)
            np.savetxt(joint_path + "%d.csv" % idx, jointData)

    et = time.time()
    print("Time: ", et - st)

    cubePos, cubeOrn = p.getBasePositionAndOrientation(robotid)
    print(cubePos, cubeOrn)
    p.disconnect()
