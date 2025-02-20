import numpy as np
import cv2
from itertools import product

# Global configuration constants.
NUM_MOTOR = 4
TASK = 0


def rot_Z(theta: float) -> np.ndarray:
    """
    Return the rotation matrix about the Z-axis by angle theta (in radians).
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,              0,             1, 0],
        [0,              0,             0, 1]
    ])


def rot_Y(theta: float) -> np.ndarray:
    """
    Return the rotation matrix about the Y-axis by angle theta (in radians).
    """
    return np.array([
        [ np.cos(theta), 0, np.sin(theta), 0],
        [ 0,             1, 0,             0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [ 0,             0, 0,             1]
    ])


def green_black(img):
    """
    Convert parts of the image (based on the green channel) to white.
    """
    img = np.array(img)
    # Use a threshold on the green channel to create a mask.
    mask = cv2.inRange(img[..., 1], 100, 255)
    img[mask > 0] = (255, 255, 255)
    return img


def generate_action_list():
    """
    Generate a list of actions for each motor based on a fixed step size.
    """
    step_size = 0.1
    line_array = np.linspace(-1.0, 1.0, num=21)
    # Randomly choose target angles for each motor.
    t_angle = np.random.choice(line_array, NUM_MOTOR)
    act_list = []
    # c_angle is assumed to be provided from the environment initial observation.
    # Here we use zeros as the initial joint angles.
    c_angle = [0] * NUM_MOTOR
    for act_i in range(NUM_MOTOR):
        num_steps = round(abs((t_angle[act_i] - c_angle[act_i]) / step_size) + 1)
        act_list.append(np.linspace(c_angle[act_i], t_angle[act_i], num_steps))
    return act_list


def self_collision_check1(sample_size: int, Env, num_dof: int) -> np.array:
    """
    Sample joint configurations and record those where the 'done' flag is True.
    """
    action_list = []
    count = 0
    j_flag, k_flag, l_flag = -1, -1, -1
    for i in range(TASK, TASK + 1):
        cmd_0 = -10 + i
        j_flag *= -1
        for j in range(21):
            cmd_1 = (-10 + j) * j_flag
            k_flag *= -1
            for k in range(21):
                cmd_2 = (-10 + k) * k_flag
                l_flag *= -1
                for l in range(21):
                    cmd_3 = (-10 + l) * l_flag
                    act_cmd = np.array([cmd_0, cmd_1, cmd_2, cmd_3]) / 10
                    count += 1
                    obs, _, done, _ = Env.step(act_cmd)
                    if done:
                        action_list.append(act_cmd)
                    print(count, act_cmd, done)
    return np.array(action_list)


def self_collision_check_prerecord(all_combinations, sample_size: int, Env, num_dof: int) -> np.array:
    """
    Re-check a set of pre-recorded joint configurations.
    """
    count = 0
    work_space = []
    for comb in all_combinations:
        count += 1
        obs, _, done, _ = Env.step(comb)
        if done:
            work_space.append(comb)
        else:
            print('collision', count)
    return np.array(work_space)


def self_collision_check(sample_size: int, Env) -> np.array:
    """
    Sample joint configurations (for a 4-DOF robot) and record those that are collision-free.
    """
    line_array = np.linspace(-1.0, 1.0, num=sample_size + 1)
    work_space = []
    for m0 in line_array:
        for m1 in line_array:
            for m2 in line_array:
                for m3 in line_array:
                    angle_norm = np.array([m0, m1, m2, m3])
                    obs, _, done, _ = Env.step(angle_norm)
                    if done:
                        print(angle_norm, "recorded")
                        work_space.append(angle_norm)
                    else:
                        print(angle_norm, "no record")
                        break
                    print("------------")
    return np.array(work_space)

