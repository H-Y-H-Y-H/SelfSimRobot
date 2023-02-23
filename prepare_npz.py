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

def w2c_matrix(theta, phi, radius):
    w2c = transition_matrix("tran_z", radius)
    w2c = np.dot(transition_matrix("rot_y", -theta / 180. * np.pi), w2c)
    w2c = np.dot(transition_matrix("rot_x", -phi / 180. * np.pi), w2c)
    return w2c

def prepare_data(my_env, path):
    # focal, 100 * 100 pixels
    camera_angle_x = 42. * np.pi / 180.
    focal = .5 * 100 / np.tan(.5 * camera_angle_x)

    my_env.reset()
    # theta = np.linspace(0.0, 4.0, 11) # 0 ~ 360
    # phi = np.linspace(0.0, -1.0, 11) # 0 ~ -90
    num_data = 110

    image_record = []
    pose_record = []

    for i in range(num_data):
        theta = np.random.rand() * 4.
        phi = -np.random.rand()
        obs, _, _, _ = my_env.step(a=np.array([theta, phi]))
        angles = obs[0] * 90.
        # pose_matrix = pose_spherical(angles[0], angles[1], 4.)
        w2c_m = w2c_matrix(angles[0], angles[1], 4.)
        image_record.append(obs[1] / 255.)
        pose_record.append(w2c_m)

    np.savez(path + 'data01.npz', images=np.array(image_record), poses=np.array(pose_record), focal=focal)

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
