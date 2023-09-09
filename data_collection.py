import os

import numpy as np
import torch

from env4 import *  # if dof=4, env4
import pybullet as p
import time
import matplotlib.pyplot as plt

def rot_X(th):
    matrix = ([
        [1, 0, 0, 0],
        [0, np.cos(th), -np.sin(th), 0],
        [0, np.sin(th), np.cos(th), 0],
        [0, 0, 0, 1]])
    np.asarray(matrix)

    return matrix


def rot_Y(th):
    matrix = ([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])
    np.asarray(matrix)

    return matrix


def rot_Z(th):
    matrix = ([
        [np.cos(th), -np.sin(th), 0, 0],
        [np.sin(th), np.cos(th), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    np.asarray(matrix)

    return matrix

def transition_matrix(label, value):
    if label == "rot_x":
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(value), -np.sin(value), 0],
            [0, np.sin(value), np.cos(value), 0],
            [0, 0, 0, 1]])

    if label == "rot_y":
        return np.array([
            [np.cos(value), 0, -np.sin(value), 0],
            [0, 1, 0, 0],
            [np.sin(value), 0, np.cos(value), 0],
            [0, 0, 0, 1]])

    if label == "tran_z":
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, value],
            [0, 0, 0, 1]])

    else:
        return "wrong label"


# def c2w_matrix(theta, phi, radius):
#     c2w = transition_matrix("tran_z", radius)
#     c2w = np.dot(transition_matrix("rot_x", phi / 180. * np.pi), c2w)
#     c2w = np.dot(transition_matrix("rot_y", theta / 180. * np.pi), c2w)
#     return c2w


def w2c_matrix(theta, phi):
    # the coordinates in pybullet, camera is along X axis, but in the pts coordinates, the camera is along z axis
    full_matrix = np.dot(rot_Z(theta / 180 * np.pi),
                         rot_Y(phi / 180 * np.pi))
    full_matrix = np.linalg.inv(full_matrix)
    return full_matrix



def random_data(DOF, num_data):
    log_angle_list = []

    for i in range(num_data):
        angle_list = np.random.rand(DOF) * 2 - 1
        log_angle_list.append(angle_list)

    return np.asarray(log_angle_list)


def uniform_data(uniform_samples=10):
    theta_0_linspace = np.linspace(-1., 1., uniform_samples, endpoint=True)
    theta_1_linspace = np.linspace(-1., 1., uniform_samples, endpoint=True)
    theta_2_linspace = np.linspace(-1., 1., uniform_samples, endpoint=True)
    theta_3_linspace = np.linspace(-1., 1., uniform_samples, endpoint=True)

    log_angle_list = []
    if DOF == 2:
        for i in range(uniform_samples ** DOF):
            theta0 = theta_0_linspace[i // uniform_samples]
            theta1 = theta_1_linspace[i % uniform_samples]
            log_angle_list.append([theta0, theta1])
    if DOF == 3:
        for i in range(uniform_samples):
            theta0 = theta_0_linspace[i]
            for j in range(uniform_samples ** (DOF - 1)):
                theta1 = theta_1_linspace[j // uniform_samples]
                theta2 = theta_2_linspace[j % uniform_samples]
                log_angle_list.append([theta0, theta1, theta2])

    if DOF == 4:
        for i in range(uniform_samples ** 2):
            theta0 = theta_0_linspace[i // uniform_samples]
            theta1 = theta_1_linspace[i % uniform_samples]
            for j in range(uniform_samples ** 2):
                theta2 = theta_2_linspace[j // uniform_samples]
                theta3 = theta_3_linspace[j % uniform_samples]
                log_angle_list.append([theta0, theta1, theta2, theta3])

    log_angle_list = np.asarray(log_angle_list)

    return log_angle_list


def collect_data(my_env, path, action_lists):
    my_env.reset()
    image_record, pose_record, angle_record = [], [], []

    for i in range(len(action_lists)):
        action_l = action_lists[i]
        print(action_l)
        obs, _, _, _ = my_env.step(action_l)
        angles = obs[0] * 90.  # obs[0] -> angles of motors, size = motor num
        # trans_m = pts_trans_matrix(angles[0], angles[1])
        # inverse_matrix = my_env.full_matrix_inv

        img = 1. - obs[1] / 255.
        # img = obs[1] / 255.
        image_record.append(img[..., 0])
        # pose_record.append(trans_m)
        angle_record.append(angles)

    np.savez(path + 'dof%d_data%d_px%d.npz' % (NUM_MOTOR, len(action_lists), WIDTH),
             images=np.array(image_record),
             poses=np.array(pose_record),
             angles=np.array(angle_record),
             focal=focal)

    # keep running
    if RENDER:
        for _ in range(1000000):
            p.stepSimulation()
            time.sleep(1 / 240)
        pass


# def df_data(data_num: int, dof: int) -> np.array:
#     # data for dense field
#     if dof == 1:
#         theta_0_list = np.linspace(-1., 1., data_num, endpoint=False)  # no repeated angles
#         theta_1_list = np.ones_like(theta_0_list) * 0.3
#         theta_2_list = np.ones_like(theta_0_list) * 0.3
#         return np.stack((theta_0_list, theta_1_list, theta_2_list), -1)
#     elif dof == 3:
#         theta_0_list = np.linspace(-1., 1., data_num, endpoint=False)  # no repeated angles
#         theta_1_list = np.linspace(-1., 1., data_num, endpoint=False)
#         theta_2_list = np.linspace(-1., 1., data_num, endpoint=False)
#         return np.stack((theta_0_list, theta_1_list, theta_2_list), -1)


def matrix_visual():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=90, azim=-90)
    arm = np.array([
        [0, 0],
        [0, 2],
        [0, 0],
        [1, 1]
    ])
    orig_cam_1 = np.array([
        [1, 1, -1, -1, 1, 0],
        [-1, 1, 1, -1, -1, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1]
    ])

    orig_cam_2 = np.array([
        [1, 1, -1, -1, 1, 0],
        [1, 1, 1, 1, 1, 0],
        [-1, 1, 1, -1, -1, 0],
        [1, 1, 1, 1, 1, 1]
    ])

    ax.plot(arm[0], arm[1], arm[2])
    ax.plot(orig_cam_1[0], orig_cam_1[1], orig_cam_1[2])
    plot_new_cam(ax, orig_cam_1)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def plot_new_cam(ax, orig_cam):
    for i in range(100):
        theta = np.random.rand() * 2. - 1.
        phi = np.random.rand() * 2. - 1.
        w2c = w2c_matrix(theta=theta * 90., phi=phi * 90.)
        new_cam = np.dot(w2c, orig_cam)
        ax.plot(new_cam[0], new_cam[1], new_cam[2], c="r")


if __name__ == "__main__":
    """data collection"""
    RENDER = False
    MOV_CAM = False
    WIDTH, HEIGHT = 100, 100
    HYPER_radius_scaler = 1.  # distance between the camera and the robot arm, previous 4, scaled value, in pose matrix
    NUM_MOTOR = 4  # the number of motors
    robot_ID = 0
    sample_size = 20

    # Camera config: focal
    Camera_FOV = 42.
    camera_angle_x = Camera_FOV * np.pi / 180.
    focal = .5 * WIDTH / np.tan(.5 * camera_angle_x)
    p.connect(p.GUI) if RENDER else p.connect(p.DIRECT)

    MyEnv = FBVSM_Env(
        show_moving_cam=MOV_CAM,
        robot_ID = robot_ID,
        width=WIDTH,
        height=HEIGHT,
        render_flag=RENDER,
        num_motor=NUM_MOTOR)

    np.random.seed(2023)
    torch.manual_seed(2023)
    # prepare_data_4dof(full_env=MyEnv, path="data/arm_data/")

    # Data_collection
    log_pth = "data/data_uniform_robo%d/"%robot_ID
    os.makedirs(log_pth, exist_ok=True)

    # action_lists = uniform_data(sample_num)
    action_lists = np.loadtxt("workspace_robo%d_dof%d_size%d.csv"%(robot_ID,NUM_MOTOR,sample_size))
    print(action_lists.shape)

    collect_data(my_env=MyEnv,
                 path=log_pth,
                 action_lists=action_lists)

    """visual test"""
    # matrix_visual()