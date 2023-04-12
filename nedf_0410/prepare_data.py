import os

from env4 import FBVSM_Env
import torch
import numpy as np
import pybullet as p


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


def w2c_matrix(theta, phi, radius):
    w2c = transition_matrix("tran_z", radius)
    w2c = np.dot(transition_matrix("rot_y", -theta / 180. * np.pi), w2c)
    w2c = np.dot(transition_matrix("rot_x", -phi / 180. * np.pi), w2c)
    return w2c


def uniform_data(uniform_samples=10):
    theta_0_linspace = np.linspace(-1., 1., uniform_samples, endpoint=True)
    theta_1_linspace = np.linspace(-1., 1., uniform_samples, endpoint=True)
    theta_2_linspace = np.linspace(-1., 1., uniform_samples, endpoint=True)
    theta_3_linspace = np.linspace(-1., 1., uniform_samples, endpoint=True)

    fix_4 = True  # 4 motor and use the first 3 dof

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
                if fix_4:
                    log_angle_list.append([theta0, theta1, theta2, 0.])
                else:
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


def data_collect(
        data_path: str,
        env: FBVSM_Env,
        action_list: np.array):
    env.reset()
    data_length = action_list.shape[0]
    save_text_len = str(int(np.log10(data_length) + 1))
    print(save_text_len)
    # todo:
    #  1. collect image,
    #  2. data loader,
    #  3. train!
    idx = 0
    angle_record = []
    for act in action_list:
        obs, _, _, _ = env.step(act)
        angles = obs[0] * 90.  # obs[0] -> angles of motors, size = motor num
        w2c_m = w2c_matrix(angles[0], angles[1], HYPER_radius_scaler)
        img = 1. - obs[1] / 255.
        # save (image, w2c, angles) by order
        # update data length if needed, %4d
        np.savetxt(data_path + "images/" + "%04d.txt" % idx, img[..., 0], fmt="%3i")  # save as int
        np.savetxt(data_path + "w2c/" + "%04d.txt" % idx, w2c_m)
        angle_record.append(angles)
        idx += 1

    np.savetxt(data_path + "angle.txt", np.array(angle_record))


if __name__ == "__main__":
    """data collection"""
    RENDER = False
    MOV_CAM = False
    WIDTH, HEIGHT = 200, 200
    HYPER_radius_scaler = 4.  # distance between the camera and the robot arm, previous 4, scaled value, in pose matrix
    DOF = 3  # the number of motors
    sample_num = 10  # separate the action space

    # Camera config: focal
    Camera_FOV = 42.
    camera_angle_x = Camera_FOV * np.pi / 180.
    focal = .5 * WIDTH / np.tan(.5 * camera_angle_x)
    p.connect(p.GUI) if RENDER else p.connect(p.DIRECT)

    MyEnv = FBVSM_Env(
        show_moving_cam=MOV_CAM,
        width=WIDTH,
        height=HEIGHT,
        render_flag=RENDER,
        num_motor=4)  # here num_motor != DOF if we fixed the last motor

    np.random.seed(2023)
    torch.manual_seed(2023)

    DATA_PATH = "./data/"
    os.makedirs(DATA_PATH + "images/", exist_ok=True)
    os.makedirs(DATA_PATH + "w2c/", exist_ok=True)

    ActionList = uniform_data(uniform_samples=sample_num)

    data_collect(
        data_path=DATA_PATH,
        env=MyEnv,
        action_list=ActionList)
