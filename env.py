import pybullet as p
import time
import pybullet_data as pd
import gym
import random
import numpy as np
from PIL import Image
import cv2


class FBVSM_Env(gym.Env):
    def __init__(self, width = 400, height = 400,render_flag = False):


        self.width = width
        self.height = height
        self.link_num = 3
        self.force = 1.8
        self.maxVelocity = 1.5
        self.action_space = 50
        self.action_shift = np.asarray([90,90,0])
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
        # cv2.imshow('Windows', img)
        # cv2.waitKey(1)

        joint_list = []
        for j in range(self.link_num):
            joint_state = p.getJointState(self.robot_id, j)[0]
            joint_list.append(joint_state)


        obs_data = [joint_list, img]
        return obs_data

    def act(self, angle_array):
        angle_array = angle_array * self.action_space + self.action_shift
        angle_array = (angle_array /180) * np.pi
        while 1:
            joint_list = []
            for i in range(self.link_num):
                p.setJointMotorControl2(self.robot_id, i, controlMode=p.POSITION_CONTROL, targetPosition=angle_array[i],
                                        force=self.force,
                                        maxVelocity=self.maxVelocity)

                joint_state = p.getJointState(self.robot_id, i)[0]
                joint_list.append(joint_state)

            p.stepSimulation()

            if abs(joint_list[0] - angle_array[0]) + abs(joint_list[1] - angle_array[1]) + abs(
                    joint_list[2] - angle_array[2]) < 0.0003:
                # print("reached")
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

        return self.get_obs()

    def step(self, a):
        self.act(a)
        obs = self.get_obs()

        r = 0
        done = False

        return obs, r, done, {}


if __name__ == '__main__':
    RENDER = True


    if RENDER:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)

    env = FBVSM_Env(width= 100,
                    height=100,
                    render_flag= RENDER)

    line_array = np.linspace(-1.0, 1.0, num=21)


    for angle01 in line_array:
        for angle02 in line_array:
            for angle03 in line_array:
                a = [angle01,angle02,angle03]
                a = np.asarray(a)
                env.step(a)

    # cur_pos = env.get_obs()
    #
    # for i in range(100):
    #
    #     tar_pos = np.random.uniform(-1,1,size = 3)
    #
    #     a_array = get_traj(cur_pos, tar_pos)
    #
    #     for traj_i in range(a_array):
    #         obs, _,_,_ = env.step(a_array[traj_i])
    #         img = obs[1]
