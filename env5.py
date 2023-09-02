import os

import numpy as np
import pybullet as p
import time
import pybullet_data as pd
import gym
from func import *
import cv2


class FBVSM_Env(gym.Env):
    def __init__(self, show_moving_cam, width=400, height=400, render_flag=False, num_motor=2, max_num_motor=4):

        self.show_moving_cam = show_moving_cam
        self.camera_pos_inverse = None
        self.expect_angles = np.array([.0, .0, .0])
        self.width = width
        self.height = height
        self.force = 1.8
        self.maxVelocity = 1.5
        self.action_space = 90
        self.num_motor = num_motor
        self.max_num_motor = max_num_motor
        self.camera_fov = 42
        #  camera z offset
        self.z_offset = -0.108
        self.render_flag = render_flag
        self.camera_pos = [1, 0, 0]  # previous 0.8 ! May 28,  # 4dof dist=1
        self.camera_line = None
        self.camera_line_m = None
        self.step_id = 0
        self.CAM_POS_X = 1
        self.nf = 0.4 # near and far
        self.full_matrix_inv = 0


        # cube_size = 0.2
        # self.pos_sphere = np.asarray([
        #     [cube_size, cube_size, 0],
        #     [cube_size, -cube_size, 0],
        #     [-cube_size, -cube_size, 0],
        #     [-cube_size, cube_size, 0],
        #     [cube_size, cube_size, cube_size * 2],
        #     [cube_size, -cube_size, cube_size * 2],
        #     [-cube_size, -cube_size, cube_size * 2],
        #     [-cube_size, cube_size, cube_size * 2],
        # ])

        """camera parameters"""
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_pos,
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1])

        #  fov, camera view angle
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=1.0,
            nearVal=0.1,
            farVal=200)

        self.reset()

    def forward_matrix(self, theta, phi):
        full_matrix = np.dot(rot_Z(theta / 180 * np.pi),
                             rot_Y(phi / 180 * np.pi))
        return full_matrix



    def get_obs(self):
        """ self.view_matrix is updating with action"""
        img = p.getCameraImage(self.width, self.height,
                               self.view_matrix, self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL,
                               shadow=0)
        img = img[2][:, :, :3]
        img = green_black(img)
        cv2.imshow('Windows', img)
        self.step_id += 1
        cv2.waitKey(1)

        joint_list = []
        for j in range(self.num_motor):
            joint_state = p.getJointState(self.robot_id, j)[0]
            joint_list.append(joint_state)

        joint_list = np.array(joint_list) / np.pi * 180
        joint_list /= self.action_space

        obs_data = [np.array(joint_list), img]
        return obs_data

    def act(self, action_norm, time_out_step_num=100):
        action_degree = action_norm * self.action_space
        action_rad = action_degree / 180 * np.pi
        reached = True

        if not self.show_moving_cam:
            for moving_times in range(time_out_step_num):
                joint_pos = []
                for i_m in range(self.num_motor):
                    p.setJointMotorControl2(self.robot_id, i_m, controlMode=p.POSITION_CONTROL,
                                            targetPosition=action_rad[i_m],
                                            force=self.force,
                                            maxVelocity=self.maxVelocity)
                    joint_state = p.getJointState(self.robot_id, i_m)[0]
                    joint_pos.append(joint_state)

                if self.num_motor < self.max_num_motor:
                    for i_m in range(self.num_motor, self.max_num_motor):
                        p.setJointMotorControl2(self.robot_id, i_m, controlMode=p.POSITION_CONTROL, targetPosition=0,
                                                force=self.force, maxVelocity=self.maxVelocity)

                joint_pos = np.asarray(joint_pos)

                for _ in range(50):
                    p.stepSimulation()

                # compute dist between target and current:
                joint_error = np.mean((joint_pos - action_rad[:len(joint_pos)]) ** 2)

                if joint_error < 0.0001:
                    break

                elif moving_times == (time_out_step_num-1):
                    reached = False
                    cont_pts1 =p.getContactPoints(self.robot_id, self.robot_id)
                    if cont_pts1 != 0:
                        print("self collision")
                        self.reset()  # if timeout (means self-collision happened) 
                        break
                    else:
                        # unable to reach the target
                        print("MOVING TIME OUT, Please check the act function in the env class")
                        quit()


                if self.render_flag:
                    time.sleep(1. / 960.)

        full_matrix = self.forward_matrix(action_degree[0],action_degree[1])
        self.full_matrix_inv = np.linalg.inv(full_matrix)
        """ inverse of full matrix as the camera view matrix """
        self.camera_pos_inverse = np.dot( self.full_matrix_inv, np.asarray([self.CAM_POS_X, 0, 0, 1]))[:3]
        self.camera_pos_inverse[2] += 0

        """ update view frame """

        move_frame_front = np.dot( self.full_matrix_inv, np.hstack((self.front_view_square, np.ones((4, 1)))).T)[:3]
        move_frame_back = np.dot( self.full_matrix_inv, np.hstack((self.back_view_square, np.ones((5, 1)))).T)[:3]

        move_frame_front = move_frame_front.T
        move_frame_back = move_frame_back.T

        if self.show_moving_cam:
            ##### Move camera with the robot arm ########
            self.camera_pos = np.dot( self.full_matrix_inv, np.asarray([self.CAM_POS_X, 0, 0, 1]))[:3]
            camera_up_vector = np.dot( self.full_matrix_inv, np.asarray([0, 0, 1, 1]))[:3]

            self.camera_pos[2] += 0
            self.view_matrix = p.computeViewMatrix(
                cameraEyePosition=self.camera_pos,
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=camera_up_vector)
            p.removeUserDebugItem(self.camera_line)
            self.camera_line = p.addUserDebugLine(self.camera_pos, [0, 0, 0], [1, 0, 0])

        # ONLY for visualization
        if self.render_flag:
            p.removeUserDebugItem(self.camera_line_inverse)
            self.camera_line_inverse = p.addUserDebugLine(self.camera_pos_inverse, [0, 0, 0], [1, 1, 1])

            for i in range(8):
                p.removeUserDebugItem(self.move_frame_edges_back[i])
                if i in [0, 1, 2, 3]:
                    p.removeUserDebugItem(self.move_frame_edges_front[i])
                    self.move_frame_edges_front[i] = p.addUserDebugLine(move_frame_front[i],
                                                                  move_frame_front[(i + 1) % 4], [1, 1, 1])
                    self.move_frame_edges_back[i] = p.addUserDebugLine(move_frame_back[i], move_frame_back[(i + 1) % 4], [1, 1, 1])

                else:
                    self.move_frame_edges_back[i] = p.addUserDebugLine(move_frame_back[4], move_frame_back[i - 4], [1, 1, 1])

        return reached

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pd.getDataPath())
        planeId = p.loadURDF("plane.urdf",[-1, 0, -1])
        textureId = p.loadTexture("green.png")
        WallId_front = p.loadURDF("plane.urdf", [-1, 0, 0], p.getQuaternionFromEuler([0, 1.57, 0]))
        p.changeVisualShape(WallId_front, -1, textureUniqueId=textureId)
        p.changeVisualShape(planeId, -1, textureUniqueId=textureId)

        startPos = [0, 0, self.z_offset]
        startOrientation = p.getQuaternionFromEuler([0, 0, -np.pi/2])

        self.robot_id = p.loadURDF(
            'DOF4ARM0/urdf/DOF4ARM0.urdf', 
            startPos, 
            startOrientation, 
            flags=p.URDF_USE_SELF_COLLISION,  # no mold penetration
            useFixedBase=1)
        
        basePos, baseOrn = p.getBasePositionAndOrientation(self.robot_id)  # Get model position
        basePos_list = [basePos[0], basePos[1], 0]
        p.resetDebugVisualizerCamera(cameraDistance=self.CAM_POS_X, cameraYaw=75, cameraPitch=-20,
                                     cameraTargetPosition=basePos_list)  # fix camera onto model
        angle_array = [0, 0, 0, 0]  # [np.pi / 2, np.pi / 2, 0, 0]

        for i in range(self.num_motor):
            p.setJointMotorControl2(self.robot_id, i, controlMode=p.POSITION_CONTROL, targetPosition=angle_array[i],
                                    force=self.force,
                                    maxVelocity=self.maxVelocity)
        for _ in range(10):
            p.stepSimulation()

        # visualize camera
        self.camera_line = p.addUserDebugLine(self.camera_pos, [0, 0, 0], [1, 1, 0])

        # visual frame edges
        self.view_edge_mid_len = np.tan(self.camera_fov * np.pi/180 /2) * self.CAM_POS_X # 0.3839
        self.view_edge_front_len = np.tan(self.camera_fov * np.pi/180 /2) * (self.CAM_POS_X - self.nf)
        self.view_edge_back_len = np.tan(self.camera_fov * np.pi/180 /2) * (self.CAM_POS_X+ self.nf)

        # self.mid_view_square = np.array([
        #     [0, self.view_edge_mid_len, self.view_edge_mid_len],
        #     [0, self.view_edge_mid_len, -self.view_edge_mid_len],
        #     [0, -self.view_edge_mid_len, -self.view_edge_mid_len],
        #     [0, -self.view_edge_mid_len, self.view_edge_mid_len],
        # ])

        self.front_view_square = np.array([
            [self.nf, self.view_edge_front_len,  self.view_edge_front_len],
            [self.nf, self.view_edge_front_len, -self.view_edge_front_len],
            [self.nf, -self.view_edge_front_len, -self.view_edge_front_len],
            [self.nf, -self.view_edge_front_len, self.view_edge_front_len],
        ])

        self.back_view_square = np.array([
            [-self.nf, self.view_edge_back_len,  self.view_edge_back_len],
            [-self.nf, self.view_edge_back_len, -self.view_edge_back_len],
            [-self.nf, -self.view_edge_back_len, -self.view_edge_back_len],
            [-self.nf, -self.view_edge_back_len, self.view_edge_back_len],
            [self.CAM_POS_X, 0, 0]
        ])

        self.fixed_frame_edges_back = []
        self.fixed_frame_edges_front=[]
        self.move_frame_edges_back = []
        self.move_frame_edges_front=[]
        # box
        for eid in range(4):

            self.move_frame_edges_front.append(
                p.addUserDebugLine(self.front_view_square[eid], self.front_view_square[(eid + 1) % 4], [1, 1, 1]))
            self.move_frame_edges_back.append(
                p.addUserDebugLine(self.back_view_square[eid], self.back_view_square[(eid + 1) % 4], [1, 1, 1]))

            self.fixed_frame_edges_front.append(
                p.addUserDebugLine(self.front_view_square[eid], self.front_view_square[(eid + 1) % 4], [0, 0, 1]))
            self.fixed_frame_edges_back.append(
                p.addUserDebugLine(self.back_view_square[eid], self.back_view_square[(eid + 1) % 4], [0, 0, 1]))

        # camera to edges
        for eid in range(4):
            self.move_frame_edges_back.append(p.addUserDebugLine(self.camera_pos, self.back_view_square[eid], [1, 1, 0]))
            self.fixed_frame_edges_back.append(p.addUserDebugLine(self.camera_pos, self.back_view_square[eid], [0, 0, 1]))

        # inverse camera line for updating
        self.camera_line_inverse = p.addUserDebugLine(self.camera_pos, [0, 0, 0], [0.0, 0.0, 1.0])

        # visualize sphere
        self.colSphereId_1 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01)

        return self.get_obs()

    def step(self, a):

        done = self.act(a)

        obs = self.get_obs()

        r = 0

        return obs, r, done, {}


def green_black(img):
    img = np.array(img)
    t = 60
    # print(img)

    # Mask image to only select browns
    mask = cv2.inRange(img[...,1], 100, 255)

    # Change image to red where we found brown
    img[mask > 0] = (255, 255, 255)


    # for x in range(img.shape[0]):
    #     for y in range(img.shape[1]):
    #         if img[x, y, 1] > 100:
    #             img[x, y] = np.array([255., 255., 255.])

    return img


def generate_action_list():
    line_array = np.linspace(-1.0, 1.0, num=21)
    t_angle = np.random.choice(line_array, NUM_MOTOR)
    # set target angle:
    # t_angle = [1, -1]
    act_list = []
    for act_i in range(NUM_MOTOR):
        act_list.append(np.linspace(c_angle[act_i], t_angle[act_i],
                                    round(abs((t_angle[act_i] - c_angle[act_i]) / step_size) + 1)))

    return act_list


def self_collision_check(sample_size:int, Env:FBVSM_Env) -> np.array:
    """
    four dof robot config sampling
    sample_size: sampled number for each motor
    Env: robot env
    """
    line_array = np.linspace(-1.0, 1.0, num=sample_size+1)
    work_space = []
    for m0 in line_array:
        for m1 in line_array:
            for m2 in line_array:
                for m3 in line_array:
                    angle_norm = np.array([m0, m1, m2, m3])
                    obs, _, done, _ = Env.step(angle_norm)
                    if done:
                        work_space.append(angle_norm)
                    
    return np.array(work_space)


if __name__ == '__main__':
    RENDER = False
    NUM_MOTOR = 4
    step_size = 0.1

    p.connect(p.GUI) if RENDER else p.connect(p.DIRECT)

    # MOVING CAMERA MODE: MOVING CAMERA - FIXED ROBOT ARM
    # show_moving_cam = True

    # FIXED  CAMERA MODE:  FIXED CAMERA - MOVING ROBOT ARM
    show_moving_cam = False

    env = FBVSM_Env(show_moving_cam,
                    width=300,
                    height=300,
                    render_flag=RENDER,
                    num_motor=NUM_MOTOR)

    obs = env.reset()
    c_angle = obs[0]

    mode = 's'  
    # manual: m
    # or automatic: a
    # or self check: s

    if mode == 'm':
        m_list = []
        m_list.append(p.addUserDebugParameter("motor0: yaw",   -1, 1, 0))
        m_list.append(p.addUserDebugParameter("motor1: pitch", -1, 1, 0))
        m_list.append(p.addUserDebugParameter("motor2: m2",    -1, 1, 0))
        m_list.append(p.addUserDebugParameter("motor3: m3",    -1, 1, 0))

        runTimes = 10000
        for i in range(runTimes):
            for c_id in range(NUM_MOTOR):
                c_angle[c_id] = p.readUserDebugParameter(m_list[c_id])
            obs, _, _, _ = env.step(c_angle)
            print(obs[0])

    elif mode == "a":
        # control the robot to target angles and observe current angles
        act_list = generate_action_list()
        for m_id in range(NUM_MOTOR):
            for single_cmd_value in act_list[m_id]:
                c_angle[m_id] = single_cmd_value
                # print(c_angle.shape)
                obs, _, _, _ = env.step(c_angle)
                print(env.get_obs()[0])

        # print(1, c_angle)
        # env.back_orig()

        for _ in range(1000000):
            p.stepSimulation()
            time.sleep(1 / 240)

    elif mode == "s":
        # get workspace: np.array (nx4)
        WorkSpace = self_collision_check(
            sample_size=2, 
            Env=env)
        print(WorkSpace.shape)
        np.savetxt("workspace.csv", WorkSpace, fmt="%.2f")
