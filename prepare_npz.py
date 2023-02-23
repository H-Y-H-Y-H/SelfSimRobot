import numpy as np
from env import FBVSM_Env
import pybullet as p
import time
from rays_check import pose_spherical, camera_spherical


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

def prepare_data(my_env, path):
    my_env.reset()
    # theta = np.linspace(0.0, 4.0, 11) # 0 ~ 360
    # phi = np.linspace(0.0, -1.0, 11) # 0 ~ -90
    num_data = 10

    for i in range(num_data):
        theta = np.random.rand() * 4.
        phi = -np.random.rand()
        obs, _, _, _ = my_env.step(a=np.array([theta, phi]))
        angles = obs[0] * 90.
        pose_matrix = pose_spherical(angles[0], angles[1], 4.)
        test_matrix = c2w_matrix(angles[0], angles[1], 4.)

        print(angles)
        print(pose_matrix)
        print(test_matrix)

    # keep running
    for _ in range(1000000):
        p.stepSimulation()
        time.sleep(1 / 240)
    pass


def matrix_visual():
    pass


if __name__ == "__main__":
    RENDER = True
    MOV_CAM = False
    p.connect(p.GUI) if RENDER else p.connect(p.DIRECT)

    MyEnv = FBVSM_Env(
        show_moving_cam=MOV_CAM,
        width=100,
        height=100,
        render_flag=RENDER,
        num_motor=2
    )

    prepare_data(my_env=MyEnv, path="data/npz_data/")
