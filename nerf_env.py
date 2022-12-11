import os

import pybullet as p
import time
import pybullet_data as pd
from func import *
from train_model import VsmModel
import torch


class Camera:
    def __init__(self, static_angles, render_flag=True):
        """
        load a static robot arm
        """
        self.camera_line = None
        self.num_motor = 3
        self.width = 400
        self.height = 400
        self.force = 1.8
        self.maxVelocity = 1.5
        self.action_space = 50
        self.z_offset = 1.106

        self.robot_id = None
        self.re_flag = render_flag
        self.start_angles = static_angles
        self.camera_pos = [0.8, 0, self.z_offset]

        """camera parameters"""
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_pos,
            cameraTargetPosition=[0, 0, self.z_offset],
            cameraUpVector=[0, 0, 1])

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=42.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=1.2)

        self.reset()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pd.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        # textureId = p.loadTexture("green.png")
        # WallId_front = p.loadURDF("plane.urdf", [-1, 0, 0], p.getQuaternionFromEuler([0, 1.57, 0]))
        # p.changeVisualShape(WallId_front, -1, textureUniqueId=textureId)
        # p.changeVisualShape(planeId, -1, textureUniqueId=textureId)

        startPos = [0, 0, 1]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robot_id = p.loadURDF('arm3dof/urdf/arm3dof.urdf', startPos, startOrientation, useFixedBase=1)

        basePos, baseOrn = p.getBasePositionAndOrientation(self.robot_id)  # Get model position
        basePos_list = [basePos[0], basePos[1], .8]
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20,
                                     cameraTargetPosition=basePos_list)  # fix camera onto model
        angle_array = [np.pi / 2, np.pi / 2, 0]

        for i in range(self.num_motor):
            p.setJointMotorControl2(self.robot_id, i, controlMode=p.POSITION_CONTROL, targetPosition=angle_array[i],
                                    force=self.force,
                                    maxVelocity=self.maxVelocity)

        for _ in range(500):
            p.stepSimulation()

    def step(self, action_norm):
        full_matrix = np.dot(rot_Z(action_norm[0] * 360 / 180 * np.pi), rot_Y(action_norm[1] * 90 / 180 * np.pi))
        self.camera_pos = np.dot(full_matrix, np.asarray([0.8, 0, 0, 1]))[:3]
        self.camera_pos[2] += self.z_offset

        self.camera_line = p.addUserDebugLine(self.camera_pos, [0, 0, 1.106], [1, 1, 0])
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_pos,
            cameraTargetPosition=[0, 0, self.z_offset],
            cameraUpVector=[0, 0, 1])
        img = p.getCameraImage(self.width, self.height,
                               self.view_matrix, self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL,
                               shadow=0)

        if self.re_flag:
            for _ in range(100):
                p.stepSimulation()
                time.sleep(1 / 240.)
        else:
            p.stepSimulation()

    def get_img(self):
        pass


def train_nerf_arm(env):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("start", device)

    Lr = 1e-6
    Batch_size = 1  # 128
    b_size = 400
    Num_epoch = 100

    Log_path = "./log_nerf_01/"

    try:
        os.mkdir(Log_path)
    except OSError:
        pass
    Model = VsmModel().to(device)
    Model.input_size = 8

    # todo : check forward box function


if __name__ == "__main__":
    RENDER = True

    if RENDER:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    static_a = np.array([0, 0, 0])
    cam_env = Camera(static_angles=static_a, render_flag=RENDER)
    while 1:
        cam_env.step(np.random.rand(2))
