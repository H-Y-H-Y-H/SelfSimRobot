import pybullet as p
import time
import pybullet_data as pd
import gym
import random
import numpy as np
from PIL import Image
import cv2


class FBVSM_Env(gym.Env):
    def __init__(self, width=400, height=400, render_flag=False):

        self.expect_angles = np.array([.0, .0, .0])
        self.width = width
        self.height = height
        self.link_num = 3
        self.force = 1.8
        self.maxVelocity = 1.5
        self.action_space = 50
        self.action_shift = np.asarray([90, 90, 0])
        self.render_flag = render_flag

        """camera parameters"""
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.8, 0, 1.106],
            cameraTargetPosition=[-1, 0, 1.106],
            cameraUpVector=[0, 0, 1])

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=42.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=200)

        self.reset()

    def get_obs(self):
        img = p.getCameraImage(self.width, self.height,
                               self.view_matrix, self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL,
                               shadow=0)
        img = img[2][:, :, :3]
        img = green_black(img)
        # cv2.imshow('Windows', img)
        # cv2.waitKey(1)

        joint_list = []
        for j in range(self.link_num):
            joint_state = p.getJointState(self.robot_id, j)[0]
            joint_list.append(joint_state)

        obs_data = [np.array(joint_list), img]
        return obs_data

    def act(self, angle_array):
        aa_record = angle_array.copy()
        angle_array = angle_array * self.action_space + self.action_shift
        angle_array = (angle_array / 180) * np.pi
        while 1:
            joint_list = []
            for i in range(self.link_num):
                p.setJointMotorControl2(self.robot_id, i, controlMode=p.POSITION_CONTROL, targetPosition=angle_array[i],
                                        force=self.force,
                                        maxVelocity=self.maxVelocity)

                joint_state = p.getJointState(self.robot_id, i)[0]
                joint_list.append(joint_state)

            for _ in range(5):
                p.stepSimulation()

            if abs(joint_list[0] - angle_array[0]) + abs(joint_list[1] - angle_array[1]) + abs(
                    joint_list[2] - angle_array[2]) < 0.0001:
                # print("reached")
                self.expect_angles = aa_record
                break

            if self.render_flag:
                time.sleep(1. / 960.)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pd.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        textureId = p.loadTexture("green.png")
        WallId_front = p.loadURDF("plane.urdf", [-1, 0, 0], p.getQuaternionFromEuler([0, 1.57, 0]))
        p.changeVisualShape(WallId_front, -1, textureUniqueId=textureId)

        startPos = [0, 0, 1]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robot_id = p.loadURDF('arm3dof/urdf/arm3dof.urdf', startPos, startOrientation, useFixedBase=1)

        basePos, baseOrn = p.getBasePositionAndOrientation(self.robot_id)  # Get model position
        basePos_list = [basePos[0], basePos[1], .8]
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20,
                                     cameraTargetPosition=basePos_list)  # fix camera onto model
        angle_array = [np.pi / 2, np.pi / 2, 0]
        for i in range(self.link_num):
            p.setJointMotorControl2(self.robot_id, i, controlMode=p.POSITION_CONTROL, targetPosition=angle_array[i],
                                    force=self.force,
                                    maxVelocity=self.maxVelocity)
        for _ in range(500):
            p.stepSimulation()

        return self.get_obs()

    def step(self, a):
        self.act(a)
        obs = self.get_obs()

        r = 0
        done = False

        return obs, r, done, {}

    def get_traj(self, l_array, step_size=0.1):
        t_angle = np.random.choice(l_array, 3)
        c_angle = self.expect_angles
        # print("-------")
        # print(c_angle, t_angle)
        a_array1 = np.linspace(c_angle[0], t_angle[0], round(abs((t_angle[0] - c_angle[0]) / step_size) + 1))
        a_array2 = np.linspace(c_angle[1], t_angle[1], round(abs((t_angle[1] - c_angle[1]) / step_size) + 1))
        a_array3 = np.linspace(c_angle[2], t_angle[2], round(abs((t_angle[2] - c_angle[2]) / step_size) + 1))
        a_array = {"a1": a_array1, "a2": a_array2, "a3": a_array3}
        return a_array


def green_black(img):
    img = np.array(img)
    t = 60
    # print(img)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x, y, 1] > 100:
                img[x, y] = np.array([255., 255., 255.])
    return img


if __name__ == '__main__':
    RENDER = True

    if RENDER:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)

    env = FBVSM_Env(width=64,
                    height=64,
                    render_flag=RENDER)

    line_array = np.linspace(-1.0, 1.0, num=21)

    # for angle01 in line_array:
    #     for angle02 in line_array:
    #         for angle03 in line_array:
    #             a = [angle01, angle02, angle03]
    #             a = np.asarray(a)
    #             obs, _, _, _ = env.step(a)
    #             print(obs[0] * 180. / np.pi)

    # cur_pos = env.get_obs()
    # target_angle = np.random.choice(line_array, 3)
    # print(cur_pos[0] * 180. / np.pi)

    for i in range(100):
        a_array = env.get_traj(line_array)
        for a1 in a_array['a1']:
            new_a = env.expect_angles.copy()
            new_a[0] = a1
            env.step(new_a)
            print(env.get_obs()[0])
        for a2 in a_array['a2']:
            new_a = env.expect_angles.copy()
            new_a[1] = a2
            env.step(new_a)
            print(env.get_obs()[0])
        for a3 in a_array['a3']:
            new_a = env.expect_angles.copy()
            new_a[2] = a3
            env.step(new_a)
            print(env.get_obs()[0])
