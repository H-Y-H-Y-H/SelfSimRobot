import os
import pybullet as p
import time
import pybullet_data as pd
import gym
from func import *
import cv2


class FBVSM_Env(gym.Env):
    def __init__(self, show_moving_cam, width=400, height=400, render_flag=False, num_motor=2, max_num_motor=3):

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
        #  camera z offset
        self.z_offset = 1.106
        self.action_shift = np.asarray([90, 180, 0, 0])[:num_motor]
        self.render_flag = render_flag
        self.camera_pos = [1, 0, self.z_offset]  # previous 0.8 ! May 28,  # 4dof dist=1
        self.camera_line = None
        self.camera_line_m = None
        self.step_id = 0
        cube_size = 0.2
        self.pos_sphere = np.asarray([
            [cube_size, cube_size, 0],
            [cube_size, -cube_size, 0],
            [-cube_size, -cube_size, 0],
            [-cube_size, cube_size, 0],
            [cube_size, cube_size, cube_size * 2],
            [cube_size, -cube_size, cube_size * 2],
            [-cube_size, -cube_size, cube_size * 2],
            [-cube_size, cube_size, cube_size * 2],
        ])

        """camera parameters"""
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_pos,
            cameraTargetPosition=[0, 0, self.z_offset],
            cameraUpVector=[0, 0, 1])

        #  fov, camera view angle
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=42.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=200)

        self.reset()

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

        joint_list = ((np.array(joint_list) / np.pi * 180) - self.action_shift) / self.action_space

        obs_data = [np.array(joint_list), img]
        return obs_data

    def act(self, action_norm):
        action_norm = action_norm * self.action_space + self.action_shift
        action = (action_norm / 180) * np.pi

        if not self.show_moving_cam:
            for moving_times in range(100):
                joint_pos = []
                for i_m in range(self.num_motor):
                    p.setJointMotorControl2(self.robot_id, i_m, controlMode=p.POSITION_CONTROL,
                                            targetPosition=action[i_m],
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
                joint_error = np.mean((joint_pos - action[:len(joint_pos)]) ** 2)

                if joint_error < 0.0001:
                    break
                # elif moving_times == 99:
                #     print("MOVING TIME OUT, Please check the act function in the env class")
                #     quit()

                if self.render_flag:
                    time.sleep(1. / 960.)

        full_matrix = np.dot(rot_Z(action_norm[0] * self.action_space / 180 * np.pi),
                             rot_Y(action_norm[1] * self.action_space / 180 * np.pi))

        """ inverse of full matrix as the camera view matrix """
        self.camera_pos_inverse = np.dot(np.linalg.inv(full_matrix), np.asarray([0.8, 0, 0, 1]))[:3]
        self.camera_pos_inverse[2] += self.z_offset

        """ update view frame """
        orig_view_square = np.array([
            [0, self.view_edge_len, self.view_edge_len],
            [0, self.view_edge_len, -self.view_edge_len],
            [0, -self.view_edge_len, -self.view_edge_len],
            [0, -self.view_edge_len, self.view_edge_len],
            [0.8, 0, 0]
        ])

        new_view_square = np.dot(np.linalg.inv(full_matrix), np.hstack((orig_view_square, np.ones((5, 1)))).T)[:3]

        new_view_square[2] += self.z_offset
        new_view_square = new_view_square.T

        if self.show_moving_cam:
            ##### Move camera with the robot arm ########
            self.camera_pos = np.dot(np.linalg.inv(full_matrix), np.asarray([0.8, 0, 0, 1]))[:3]
            camera_up_vector = np.dot(np.linalg.inv(full_matrix), np.asarray([0, 0, 1, 1]))[:3]

            self.camera_pos[2] += self.z_offset
            self.view_matrix = p.computeViewMatrix(
                cameraEyePosition=self.camera_pos,
                cameraTargetPosition=[0, 0, self.z_offset],
                cameraUpVector=camera_up_vector)
            p.removeUserDebugItem(self.camera_line)
            self.camera_line = p.addUserDebugLine(self.camera_pos, [0, 0, self.z_offset], [1, 0, 0])

        # ONLY for visualization
        if self.render_flag:
            p.removeUserDebugItem(self.camera_line_inverse)
            self.camera_line_inverse = p.addUserDebugLine(self.camera_pos_inverse, [0, 0, self.z_offset], [1, 1, 1])

            for i in range(8):
                p.removeUserDebugItem(self.move_frame_edges[i])
                if i in [0, 1, 2, 3]:
                    self.move_frame_edges[i] = p.addUserDebugLine(new_view_square[i],
                                                                  new_view_square[(i + 1) % 4], [1, 1, 1])
                else:
                    self.move_frame_edges[i] = p.addUserDebugLine(new_view_square[4], new_view_square[i - 4], [1, 1, 1])

            box_pos = np.dot(
                full_matrix,
                np.hstack((self.pos_sphere, np.ones((8, 1)))).T
            )[:3]
            box_pos[2] += self.z_offset

            box_pos = box_pos.T

            # for i in range(12):
            #     p.removeUserDebugItem(self.cube_line[i])
            #     if i in [0, 1, 2, 4, 5, 6]:
            #         self.cube_line[i] = p.addUserDebugLine(box_pos[i], box_pos[i + 1], [1, 1, 0])
            #     elif i in [3, 7]:
            #         self.cube_line[i] = p.addUserDebugLine(box_pos[i], box_pos[i - 3], [1, 1, 0])
            #     else:
            #         self.cube_line[i] = p.addUserDebugLine(box_pos[i - 8], box_pos[i - 4], [1, 1, 0])

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pd.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        textureId = p.loadTexture("../green.png")
        WallId_front = p.loadURDF("plane.urdf", [-1, 0, 0], p.getQuaternionFromEuler([0, 1.57, 0]))
        p.changeVisualShape(WallId_front, -1, textureUniqueId=textureId)
        p.changeVisualShape(planeId, -1, textureUniqueId=textureId)

        startPos = [0, 0, 1]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robot_id = p.loadURDF('../arm4dof/urdf/arm4dof.urdf', startPos, startOrientation, useFixedBase=1)

        basePos, baseOrn = p.getBasePositionAndOrientation(self.robot_id)  # Get model position
        basePos_list = [basePos[0], basePos[1], .8]
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=75, cameraPitch=-20,
                                     cameraTargetPosition=basePos_list)  # fix camera onto model
        angle_array = [np.pi / 2, np.pi / 2, 0, 0]  # [np.pi / 2, np.pi / 2, 0, 0]

        for i in range(self.num_motor):
            p.setJointMotorControl2(self.robot_id, i, controlMode=p.POSITION_CONTROL, targetPosition=angle_array[i],
                                    force=self.force,
                                    maxVelocity=self.maxVelocity)
        for _ in range(500):
            p.stepSimulation()

        # visualize camera
        self.camera_line = p.addUserDebugLine(self.camera_pos, [0, 0, 1.106], [1, 1, 0])
        # self.camera_line_m = p.addUserDebugLine(self.camera_pos, [0, 0, 1.106], [1, 1, 1])

        # nov 23, visual frame edges
        self.view_edge_len = 0.2
        self.view_square = np.array([
            [0, self.view_edge_len, 1.106 + self.view_edge_len],
            [0, self.view_edge_len, 1.106 - self.view_edge_len],
            [0, -self.view_edge_len, 1.106 - self.view_edge_len],
            [0, -self.view_edge_len, 1.106 + self.view_edge_len]
        ])

        self.frame_edges = []
        self.move_frame_edges = []
        for eid in range(4):
            self.frame_edges.append(
                p.addUserDebugLine(self.view_square[eid], self.view_square[(eid + 1) % 4], [1, 1, 0]))
            self.move_frame_edges.append(
                p.addUserDebugLine(self.view_square[eid], self.view_square[(eid + 1) % 4], [0, 0, 1]))

        for eid in range(4):
            self.frame_edges.append(p.addUserDebugLine(self.camera_pos, self.view_square[eid], [1, 1, 0]))
            self.move_frame_edges.append(p.addUserDebugLine(self.camera_pos, self.view_square[eid], [0, 0, 1]))

        # inverse camera line for updating
        self.camera_line_inverse = p.addUserDebugLine(self.camera_pos, [0, 0, 1.106], [0.0, 0.0, 1.0])

        # visualize sphere
        self.colSphereId_1 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01)

        pos_sphere_ = np.copy(self.pos_sphere)
        pos_sphere_[:, 2] += self.z_offset
        self.cube_line = []
        self.cube_line_copy = []

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
        act_list = []
        for act_i in range(3):
            act_list.append(np.linspace(c_angle[act_i], t_angle[act_i],
                                        round(abs((t_angle[act_i] - c_angle[act_i]) / step_size) + 1)))

        # print("-------")
        # print(c_angle, t_angle)

        return act_list

    def back_orig(self):
        angle_array = [np.pi / 2, np.pi / 2, 0]

        for i in range(self.num_motor):
            p.setJointMotorControl2(self.robot_id, i, controlMode=p.POSITION_CONTROL, targetPosition=angle_array[i],
                                    force=self.force,
                                    maxVelocity=self.maxVelocity)
        for _ in range(500):
            p.stepSimulation()


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
    # line_array = np.linspace(-1.0, 1.0, num=21)
    # t_angle = np.random.choice(line_array, NUM_MOTOR)
    # set target angle:
    t_angle = [1, -1]
    act_list = []
    for act_i in range(NUM_MOTOR):
        act_list.append(np.linspace(c_angle[act_i], t_angle[act_i],
                                    round(abs((t_angle[act_i] - c_angle[act_i]) / step_size) + 1)))

    return act_list


if __name__ == '__main__':
    RENDER = True
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

    mode = 'm'  # manual or automatic

    if mode == 'm':
        m0 = p.addUserDebugParameter("motor0: yaw", -1, 1, 0)
        m1 = p.addUserDebugParameter("motor1: pitch", -1, 1, 0)
        m2 = p.addUserDebugParameter("motor2: m2", -1, 1, 0)
        m3 = p.addUserDebugParameter("motor3: m3", -1, 1, 0)

        runTimes = 10000
        for i in range(runTimes):
            c_angle[0] = p.readUserDebugParameter(m0)
            c_angle[1] = p.readUserDebugParameter(m1)
            c_angle[2] = p.readUserDebugParameter(m2)
            c_angle[3] = p.readUserDebugParameter(m3)
            obs, _, _, _ = env.step(c_angle)
            print(obs[0])

    else:
        # control the robot to  target angles and observe current angles
        act_list = generate_action_list()
        for m_id in range(NUM_MOTOR):
            for single_cmd_value in act_list[m_id]:
                c_angle[m_id] = single_cmd_value
                obs, _, _, _ = env.step(c_angle)
                print(env.get_obs()[0])

        # print(1, c_angle)
        # env.back_orig()

        for _ in range(1000000):
            p.stepSimulation()
            time.sleep(1 / 240)
