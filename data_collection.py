import os

import numpy as np
import torch

from env4 import *  # if dof=4, env4
import pybullet as p
import time
import matplotlib.pyplot as plt



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

def w2c_matrix(theta, phi):
    # the coordinates in pybullet, camera is along X axis, but in the pts coordinates, the camera is along z axis
    full_matrix = np.dot(rot_Z(theta / 180 * np.pi),
                         rot_Y(phi / 180 * np.pi))
    full_matrix = np.linalg.inv(full_matrix)
    return full_matrix



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


def collect_data(my_env, save_path, action_lists):
    my_env.reset()
    clean_action_list, image_record, pose_record, angle_record = [], [], [], []

    for i in range(len(action_lists)):
        action_l = action_lists[i]
        print(action_l)
        obs, _, done, _ = my_env.step(action_l)
        if not done:
            continue
        angles = obs[0] * 90.  # obs[0] -> angles of motors, size = motor num
        # trans_m = pts_trans_matrix(angles[0], angles[1])
        # inverse_matrix = my_env.full_matrix_inv

        img = 1. - obs[1] / 255.
        # img = obs[1] / 255.
        image_record.append(img[..., 0])
        # pose_record.append(trans_m)
        angle_record.append(angles)
        clean_action_list.append(action_l)

    np.savez(save_path ,
             images=np.array(image_record),
             poses=np.array(pose_record),
             angles=np.array(angle_record),
             focal=focal)
    
    print("Data collection done!")

    # keep running
    if RENDER:
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
    for i in range(100):
        theta = np.random.rand() * 2. - 1.
        phi = np.random.rand() * 2. - 1.
        w2c = w2c_matrix(theta=theta * 90., phi=phi * 90.)
        new_cam = np.dot(w2c, orig_cam)
        ax.plot(new_cam[0], new_cam[1], new_cam[2], c="r")


if __name__ == "__main__":
    """data collection"""
    RENDER = True
    MOV_CAM = False
    WIDTH, HEIGHT = 100, 100
    HYPER_radius_scaler = 1.  # distance between the camera and the robot arm, previous 4, scaled value, in pose matrix
    NUM_MOTOR = 4  # the number of motors
    robot_ID = 1
    # sample_size = 20

    cam_dist = 1.


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
        num_motor=NUM_MOTOR,
        init_angle = [-np.pi/2,0,0,-np.pi/2], 
        cam_dist=cam_dist)
    # -1, 0.8, 1, -0.1

    

    np.random.seed(2023)
    torch.manual_seed(2023)

    # Data_collection
    # log_pth = "data/data_uniform_robo%d/"%robot_ID
    log_pth = "data/sim_data/"
    os.makedirs(log_pth, exist_ok=True)

    # action_lists = np.loadtxt('data/action/cleaned_0531(1)_con_action_robo%d_dof4_size20.csv'%robot_ID)
    # action_lists = np.loadtxt('data/action/cleaned_con_action_robo%d_dof4_size10.csv'%robot_ID)
    action_lists = np.loadtxt('data/action/ee_workspace_10.csv')


    print(action_lists.shape)
    # log_pth += '1009(1)_con_dof%d_data.npz' % (NUM_MOTOR)
    log_pth += 'sim_data_robo%d(ee).npz' % (robot_ID)
    collect_data(my_env=MyEnv,
                 save_path=log_pth,
                 action_lists=action_lists)

    """visual test"""
    # matrix_visual()