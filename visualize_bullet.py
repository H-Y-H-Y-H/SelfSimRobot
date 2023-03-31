import os
import torch
from train import nerf_forward, get_fixed_camera_rays, init_models
from func import w2c_matrix, c2w_matrix
import numpy as np
from env import FBVSM_Env
import pybullet as p


# changed Transparency in urdf, line181, Mar31

def load_point_cloud(angle_list: list, debug_points):
    test_points = np.random.rand(100, 3)
    p_rgb = np.ones_like(test_points)
    p.removeUserDebugItem(debug_points)  # update points every step
    debug_points = p.addUserDebugPoints(test_points, p_rgb, pointSize=5)

    return debug_points


def interact_env(
        pic_size: int = 100,
        render: bool = True,
        interact: bool = True,
        dof: int = 3):
    p.connect(p.GUI) if render else p.connect(p.DIRECT)

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
            debug_points = load_point_cloud([c_angle[0], c_angle[1], c_angle[2]], debug_points)
            obs, _, _, _ = env.step(c_angle)
            print(obs[0])


if __name__ == "__main__":
    interact_env()
