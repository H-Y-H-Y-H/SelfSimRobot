import os

import numpy as np
import torch

from env import FBVSM_Env
import pybullet as p
import time
import matplotlib.pyplot as plt


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


def c2w_matrix(theta, phi, radius):
    c2w = transition_matrix("tran_z", radius)
    c2w = np.dot(transition_matrix("rot_x", phi / 180. * np.pi), c2w)
    c2w = np.dot(transition_matrix("rot_y", theta / 180. * np.pi), c2w)
    return c2w


def w2c_matrix(theta, phi, radius):
    w2c = transition_matrix("tran_z", radius)
    w2c = np.dot(transition_matrix("rot_y", -theta / 180. * np.pi), w2c)
    w2c = np.dot(transition_matrix("rot_x", -phi / 180. * np.pi), w2c)
    return w2c


def prepare_data(my_env, path, num_data):
    my_env.reset()
    image_record, pose_record, angle_record = [], [], []

    for i in range(num_data):
        theta = np.random.uniform(-1, 1)
        phi = np.random.uniform(-1, 1)
        angle = np.random.uniform(-1, 1)
        obs, _, _, _ = my_env.step(a=np.array([theta, phi, angle]))
        angles = obs[0] * 90.  # obs[0] -> angles of motors, size = motor num
        w2c_m = w2c_matrix(angles[0], angles[1], HYPER_radius_scaler)
        image_record.append(1. - obs[1] / 255.)
        pose_record.append(w2c_m)
        angle_record.append(angle)

    if DOF == 2:
        np.savez(path + 'dof%d_data%d.npz' % (DOF, num_data),
                 images=np.array(image_record),
                 poses=np.array(pose_record),
                 focal=focal)

    elif DOF ==3:
        np.savez(path + 'dof%d_data%d.npz' % (DOF, num_data),
                 images=np.array(image_record),
                 poses=np.array(pose_record),
                 angles=np.array(angle_record),
                 focal=focal)
    else:
        print("DOF error!")
        quit()

    # keep running
    for _ in range(1000000):
        p.stepSimulation()
        time.sleep(1 / 240)
    pass


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
    # for i in range(100):
    #     theta = np.random.rand() * 4.
    #     phi = -np.random.rand()
    #     c2w = c2w_matrix(theta=theta * 90., phi=phi * 90., radius=10.)
    #     new_cam = np.dot(c2w, orig_cam)
    #     ax.plot(new_cam[0], new_cam[1], new_cam[2], c="g")

    for i in range(100):
        theta = np.random.rand() * 2. - 1.
        phi = np.random.rand() * 2. - 1.
        w2c = w2c_matrix(theta=theta * 90., phi=phi * 90., radius=10.)
        new_cam = np.dot(w2c, orig_cam)
        ax.plot(new_cam[0], new_cam[1], new_cam[2], c="r")


if __name__ == "__main__":
    """data collection"""
    RENDER = True
    MOV_CAM = False
    WIDTH, HEIGHT = 100, 100
    HYPER_radius_scaler = 4  # TBD: test 0.8 or rescaler during visualization
    DOF = 2  # the number of motors
    # Camera config: focal
    Camera_FOV = 42.
    camera_angle_x = Camera_FOV * np.pi / 180.
    focal = .5 * 100 / np.tan(.5 * camera_angle_x)
    p.connect(p.GUI) if RENDER else p.connect(p.DIRECT)

    MyEnv = FBVSM_Env(
        show_moving_cam=MOV_CAM,
        width=WIDTH,
        height=HEIGHT,
        render_flag=RENDER,
        num_motor=DOF
    )

    np.random.seed(2023)
    torch.manual_seed(2023)
    # prepare_data_4dof(full_env=MyEnv, path="data/arm_data/")

    # Data_collection
    log_pth = "data/arm_data/"
    os.makedirs(log_pth, exist_ok=True)
    prepare_data(my_env=MyEnv, path=log_pth, num_data=110)

    """visual test"""
    # matrix_visual()
