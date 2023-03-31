import os
import torch
from train import nerf_forward, get_fixed_camera_rays, init_models
from func import w2c_matrix, c2w_matrix
import numpy as np
from env import FBVSM_Env
import pybullet as p


# changed Transparency in urdf, line181, Mar31

def scale_points(predict_points):
    scaled_points = predict_points / 5.  # scale 4 / 0.8
    scaled_points[:, [1, 2]] = scaled_points[:, [2, 1]]
    scaled_points[:, [1, 0]] = scaled_points[:, [0, 1]]
    scaled_points[:, 0] = -scaled_points[:, 0]
    scaled_points[:, 2] += 1.106

    new_points = scaled_points
    return new_points


def load_point_cloud(angle_list: list, debug_points, logger):
    angle_lists = np.asarray([angle_list] * len(logger)) * 90  # -90 to 90
    diff = np.sum(abs(logger - angle_lists), axis=1)
    idx = np.argmin(diff)
    predict_points = np.load(data_pth + '%04d.npy' % idx)
    # scaled points

    trans_points = scale_points(predict_points)
    # test_points = np.random.rand(100, 3)
    p_rgb = np.ones_like(trans_points)
    p.removeUserDebugItem(debug_points)  # update points every step
    debug_points = p.addUserDebugPoints(trans_points, p_rgb, pointSize=2)

    return debug_points


def interact_env(
        pic_size: int = 100,
        render: bool = True,
        interact: bool = True,
        dof: int = 3):
    p.connect(p.GUI) if render else p.connect(p.DIRECT)
    logger = np.loadtxt("train_log/log_1000data_out1_img100/logger.csv")

    env = FBVSM_Env(
        show_moving_cam=False,
        width=pic_size,
        height=pic_size,
        render_flag=render,
        num_motor=dof)

    obs = env.reset()

    obs = env.reset()
    c_angle = obs[0]
    debug_points = 0

    if interact:
        # 3 dof
        m0 = p.addUserDebugParameter("motor0: Yaw", -1, 1, 0)
        m1 = p.addUserDebugParameter("motor1: pitch", -1, 1, 0)
        m2 = p.addUserDebugParameter("motor1: pitch", -1, 1, 0)

        runTimes = 10000
        for i in range(runTimes):
            c_angle[0] = p.readUserDebugParameter(m0)
            c_angle[1] = p.readUserDebugParameter(m1)
            c_angle[2] = p.readUserDebugParameter(m2)
            print([c_angle[0], c_angle[1], c_angle[2]])
            debug_points = load_point_cloud([c_angle[0], c_angle[1], c_angle[2]], debug_points, logger)
            obs, _, _, _ = env.step(c_angle)
            # print(obs[0])


if __name__ == "__main__":
    data_pth = 'train_log/log_1000data_out1_img100/best_model/pc_record/'
    interact_env()
