import os
import numpy as np
import torch
import pybullet as p
import time
import matplotlib.pyplot as plt

# Import the simulation environment.
# Ensure that env4.py provides FBVSM_Env and DOF (degree-of-freedom) variables.
from env import FBVSM_Env


def rot_Y(theta: float) -> np.ndarray:
    """
    Return the rotation matrix for a rotation around the Y-axis by angle theta (in radians).
    """
    return np.array([
        [np.cos(theta), 0, -np.sin(theta), 0],
        [0, 1, 0, 0],
        [np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])


def rot_Z(theta: float) -> np.ndarray:
    """
    Return the rotation matrix for a rotation around the Z-axis by angle theta (in radians).
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])


def w2c_matrix(theta: float, phi: float) -> np.ndarray:
    """
    Compute the world-to-camera transformation matrix.

    Parameters:
        theta: Rotation about Z in degrees.
        phi: Rotation about Y in degrees.

    Returns:
        The inverse transformation matrix (world-to-camera).
    """
    full_matrix = np.dot(rot_Z(theta / 180 * np.pi), rot_Y(phi / 180 * np.pi))
    return np.linalg.inv(full_matrix)


def collect_data(my_env: FBVSM_Env, save_path: str, action_lists: np.ndarray, focal: float) -> None:
    """
    Collect simulation data by stepping through the provided action list.
    Saves the collected images, poses, angles, and focal length into an NPZ file.

    Parameters:
        my_env: The simulation environment.
        save_path: Full path (including filename) to save the NPZ file.
        action_lists: Array of action commands.
        focal: Focal length of the camera.
    """
    my_env.reset()
    clean_action_list, image_record, pose_record, angle_record = [], [], [], []

    for action in action_lists:
        print("Processing action:", action)
        obs, _, done, _ = my_env.step(action)
        if not done:
            continue
        # Convert normalized angles to degrees.
        angles = obs[0] * 90.
        # Normalize the image (invert intensity).
        img = 1. - obs[1] / 255.
        image_record.append(img[..., 0])  # Save one channel (assumed grayscale)
        angle_record.append(angles)
        clean_action_list.append(action)

    np.savez(save_path,
             images=np.array(image_record),
             poses=np.array(pose_record),  # Note: pose_record remains empty unless you add pose data
             angles=np.array(angle_record),
             focal=focal)

    print("Data collection done!")

    if my_env.render_flag:
        while True:
            p.stepSimulation()
            time.sleep(1 / 240)


def matrix_visual() -> None:
    """
    Visualize the arm and camera transformation matrices using matplotlib.
    """
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

    ax.plot(arm[0], arm[1], arm[2], label="Arm")
    ax.plot(orig_cam_1[0], orig_cam_1[1], orig_cam_1[2], label="Original Cam")
    plot_new_cam(ax, orig_cam_1)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()


def plot_new_cam(ax, orig_cam: np.ndarray) -> None:
    """
    Plot new camera positions on the given matplotlib axis.

    Parameters:
        ax: Matplotlib 3D axis.
        orig_cam: The original camera coordinates.
    """
    for _ in range(100):
        theta = np.random.rand() * 2 - 1
        phi = np.random.rand() * 2 - 1
        w2c = w2c_matrix(theta=theta * 90., phi=phi * 90.)
        new_cam = np.dot(w2c, orig_cam)
        ax.plot(new_cam[0], new_cam[1], new_cam[2], c="r")


def main() -> None:

    RENDER = True
    robot_ID = 1

    # Simulation parameters.
    WIDTH, HEIGHT = 100, 100
    NUM_MOTOR = 4  # number of motors
    cam_dist = 1.0
    Camera_FOV = 42.
    camera_angle_x = Camera_FOV * np.pi / 180.
    focal = 0.5 * WIDTH / np.tan(0.5 * camera_angle_x)
    p.connect(p.GUI) if RENDER else p.connect(p.DIRECT)

    # Initialize the simulation environment.
    env = FBVSM_Env(
        robot_ID=robot_ID,
        width=WIDTH,
        height=HEIGHT,
        render_flag=RENDER,
        num_motor=NUM_MOTOR,
        init_angle=[-np.pi / 2, 0, 0, -np.pi / 2],
        cam_dist=cam_dist
    )


    # Create directory for saving simulation data.
    save_dir = "data/sim_data/"
    os.makedirs(save_dir, exist_ok=True)

    # Load the action list from file.
    action_file = f"data/action/action_robot{robot_ID}.csv"
    action_lists = np.loadtxt(action_file)
    print("Action list shape:", action_lists.shape)

    # Define the save path for the collected data.
    save_path = os.path.join(save_dir, f"sim_data_robo{robot_ID}(ee).npz")

    # Collect simulation data.
    collect_data(my_env=env, save_path=save_path, action_lists=action_lists, focal=focal)


if __name__ == "__main__":
    main()
