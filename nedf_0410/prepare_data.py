import os

from env4 import FBVSM_Env
import torch
import numpy as np
import pybullet as p


def data_collect(
        data_path: str,
        env: FBVSM_Env,
        action_list: np.array):

    env.reset()
    print(action_list.shape)


if __name__ == "__main__":
    """data collection"""
    RENDER = False
    MOV_CAM = False
    WIDTH, HEIGHT = 200, 200
    HYPER_radius_scaler = 4.  # distance between the camera and the robot arm, previous 4, scaled value, in pose matrix
    DOF = 3  # the number of motors
    sample_num = 20  # separate the action space

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
        num_motor=DOF)

    np.random.seed(2023)
    torch.manual_seed(2023)

    DATA_PATH = "./data"
    os.makedirs(DATA_PATH + "images", exist_ok=True)


