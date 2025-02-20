import os
import numpy as np
import pybullet as p
import time
import pybullet_data as pd
import gym
import cv2

# Import helper functions and constants from the predefined file.
from predefined import (
    rot_Z,
    rot_Y,
    green_black,
    generate_action_list,
    NUM_MOTOR,
    TASK
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FBVSM_Env(gym.Env):
    """
    Custom Gym environment that simulates a robot arm in PyBullet.
    """
    def __init__(self,
                 robot_ID: int,
                 width: int = 400,
                 height: int = 400,
                 render_flag: bool = False,
                 num_motor: int = 2,
                 max_num_motor: int = 4,
                 init_angle=None,
                 dark_background: bool = False,
                 rotation_view: bool = False,
                 cam_dist: float = 1.0,
                 object_alpha: float = 1.0):
        if init_angle is None:
            init_angle = [0] * max_num_motor

        self.robot_ID = robot_ID
        self.width = width
        self.height = height
        self.force = 100
        self.maxVelocity = 1.5
        self.action_space = 90
        self.num_motor = num_motor
        self.max_num_motor = max_num_motor
        self.camera_fov = 42
        self.z_offset = -0.108  # camera z offset
        self.render_flag = render_flag
        self.camera_pos = [cam_dist, 0, 0]
        self.camera_line = None
        self.step_id = 0
        self.CAM_POS_X = 1
        self.nf = 0.4  # near plane value
        self.full_matrix_inv = None
        self.PASS_OBS = False
        self.init_angle = np.asarray(init_angle) * np.pi / 2
        self.dark_background = dark_background
        self.rotation_view = rotation_view
        self.start_time = time.time()
        self.object_alpha = object_alpha


        # Initialize the camera matrices.
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_pos,
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1]
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=1.0,
            nearVal=0.1,
            farVal=200
        )

        self.reset()

    def forward_matrix(self, theta: float, phi: float) -> np.ndarray:
        """
        Compute the forward transformation matrix from the given angles.
        """
        full_matrix = np.dot(
            rot_Z(theta / 180 * np.pi),
            rot_Y(phi / 180 * np.pi)
        )
        return full_matrix

    def get_obs(self):
        """
        Capture a camera image and read the robot's joint states.
        """
        img = p.getCameraImage(self.width, self.height,
                               self.view_matrix, self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL,
                               shadow=0)[2]

        img = img[:, :, :3]
        processed_img = green_black(img)
        # Annotate each image with a label.
        annotated_raw = img.copy()

        annotated_processed = processed_img.copy()

        # Concatenate the images horizontally.
        concat_img = np.concatenate((annotated_raw, annotated_processed), axis=1)
        cv2.imshow('Camera View', concat_img)

        self.step_id += 1
        cv2.waitKey(1)

        joint_list = []
        for j in range(self.num_motor):
            joint_state = p.getJointState(self.robot_id, j)[0]
            joint_list.append(joint_state)

        joint_list = np.array(joint_list) / np.pi * 180
        joint_list /= self.action_space

        return [np.array(joint_list), processed_img]

    def act(self, action_norm, time_out_step_num=1000):
        """
        Apply the given action (in normalized units) and step the simulation until
        the desired joint configuration is reached.
        """
        action_degree = action_norm * self.action_space
        action_rad = action_degree / 180 * np.pi
        reached = False

        # Set the camera based on whether rotation_view is enabled.
        if self.rotation_view:
            elapsed_time = time.time() - self.start_time
            camera_yaw = 5 * elapsed_time
            p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraPitch=-30, cameraYaw=camera_yaw,
                                         cameraTargetPosition=[0, 0, 0])
        else:
            p.resetDebugVisualizerCamera(cameraDistance=1, cameraPitch=-30, cameraYaw=135,
                                         cameraTargetPosition=[0, 0, 0])

        # Execute the action by moving the joints.
        for _ in range(time_out_step_num):
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
                    p.setJointMotorControl2(self.robot_id, i_m, controlMode=p.POSITION_CONTROL,
                                            targetPosition=0,
                                            force=self.force,
                                            maxVelocity=self.maxVelocity)

            joint_pos = np.asarray(joint_pos)
            for _ in range(10):
                p.stepSimulation()

            joint_error = np.mean((joint_pos - action_rad[:len(joint_pos)]) ** 2)
            if joint_error < 0.001:
                reached = True
                break

            if not self.dark_background:
                if p.getContactPoints(self.robot_id, self.robot_id) or p.getContactPoints(self.robot_id, self.groundId):
                    print("Self-collision detected! Joint Error:", joint_error)
                    break

        # Compute the inverse camera transformation (for visualization).
        full_matrix = self.forward_matrix(action_degree[0], action_degree[1])
        self.full_matrix_inv = np.linalg.inv(full_matrix)
        self.camera_pos_inverse = np.dot(self.full_matrix_inv, np.asarray([self.CAM_POS_X, 0, 0, 1]))[:3]

        # Visualization: draw the inverse camera line and frame edges.
        move_frame_front = np.dot(self.full_matrix_inv,
                                  np.hstack((self.front_view_square, np.ones((4, 1)))).T)[:3].T
        move_frame_back = np.dot(self.full_matrix_inv,
                                 np.hstack((self.back_view_square, np.ones((5, 1)))).T)[:3].T

        if self.render_flag:
            p.removeUserDebugItem(self.camera_line_inverse)
            self.camera_line_inverse = p.addUserDebugLine(self.camera_pos_inverse, [0, 0, 0], [1, 1, 1])
            for i in range(8):
                p.removeUserDebugItem(self.move_frame_edges_back[i])
                if i < 4:
                    p.removeUserDebugItem(self.move_frame_edges_front[i])
                    self.move_frame_edges_front[i] = p.addUserDebugLine(
                        move_frame_front[i], move_frame_front[(i + 1) % 4], [1, 1, 1])
                    self.move_frame_edges_back[i] = p.addUserDebugLine(
                        move_frame_back[i], move_frame_back[(i + 1) % 4], [1, 1, 1])
                else:
                    self.move_frame_edges_back[i] = p.addUserDebugLine(
                        move_frame_back[4], move_frame_back[i - 4], [1, 1, 1])
        return reached

    def add_obstacles(self, obj_urdf_path, position, orientation):
        self.obstacle_id = p.loadURDF(obj_urdf_path, position, orientation, useFixedBase=1)

    def reset(self):
        """
        Reset the simulation, load the robot, and set up visualization elements.
        """
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pd.getDataPath())

        if self.dark_background:
            textureId = p.loadTexture("RobotArmURDF/white_ground.jpg")
            self.groundId = p.loadURDF("plane.urdf", [0, 0, -0.105])
            p.changeVisualShape(self.groundId, -1, textureUniqueId=textureId)
        else:
            self.groundId = p.loadURDF("plane.urdf", [0, 0, -0.083])
            textureId = p.loadTexture("RobotArmURDF/green.png")
            WallId_front = p.loadURDF("plane.urdf", [-1, 0, 0], p.getQuaternionFromEuler([0, 1.57, 0]))
            p.changeVisualShape(WallId_front, -1, textureUniqueId=textureId)
            p.changeVisualShape(self.groundId, -1, textureUniqueId=textureId)

        startPos = [0, 0, self.z_offset]
        startOrientation = p.getQuaternionFromEuler([0, 0, -np.pi/2])
        if self.robot_ID == 1:
            robo_urdf_path = 'RobotArmURDF/4dof_1st/urdf/4dof_1st.urdf'
        elif self.robot_ID == 2:
            robo_urdf_path = 'RobotArmURDF/4dof_2nd/urdf/4dof_2nd.urdf'
        elif self.robot_ID == 3:
            robo_urdf_path = 'RobotArmURDF/4dof_3rd/urdf/4dof_3rd.urdf'
        else:
            print("The Robot ID index is not correct! Use default robot arm 1")
            robo_urdf_path = 'RobotArmURDF/4dof_1st/urdf/DOF4ARM0.urdf'

        self.robot_id = p.loadURDF(
            robo_urdf_path,
            startPos,
            startOrientation,
            flags=p.URDF_USE_SELF_COLLISION,
            useFixedBase=1
        )

        basePos, _ = p.getBasePositionAndOrientation(self.robot_id)
        basePos_list = [basePos[0], basePos[1], 0]
        p.resetDebugVisualizerCamera(cameraDistance=self.CAM_POS_X, cameraYaw=75, cameraPitch=-20,
                                     cameraTargetPosition=basePos_list)

        for i in range(self.num_motor):
            p.setJointMotorControl2(self.robot_id, i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=self.init_angle[i],
                                    force=self.force,
                                    maxVelocity=self.maxVelocity)
        for _ in range(1000):
            p.stepSimulation()

        num_links = p.getNumJoints(self.robot_id)
        # Update the base (link index -1) and each joint.
        p.changeVisualShape(self.robot_id, -1, rgbaColor=[0, 0, 0, self.object_alpha])
        for i in range(num_links):
            p.changeVisualShape(self.robot_id, i, rgbaColor=[0, 0, 0, self.object_alpha])

        # Optionally update the ground transparency too.
        p.changeVisualShape(self.groundId, -1, rgbaColor=[1, 1, 1, self.object_alpha])

        # Set up camera and frame edge visualization.
        self.camera_line = p.addUserDebugLine(self.camera_pos, [0, 0, 0], [1, 1, 0])
        self.view_edge_mid_len = np.tan(self.camera_fov * np.pi/180 / 2) * self.CAM_POS_X
        self.view_edge_front_len = np.tan(self.camera_fov * np.pi/180 / 2) * (self.CAM_POS_X - self.nf)
        self.view_edge_back_len = np.tan(self.camera_fov * np.pi/180 / 2) * (self.CAM_POS_X + self.nf)

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
        self.fixed_frame_edges_front = []
        self.move_frame_edges_back = []
        self.move_frame_edges_front = []

        for eid in range(4):
            self.move_frame_edges_front.append(
                p.addUserDebugLine(self.front_view_square[eid],
                                   self.front_view_square[(eid + 1) % 4],
                                   [1, 1, 1])
            )
            self.move_frame_edges_back.append(
                p.addUserDebugLine(self.back_view_square[eid],
                                   self.back_view_square[(eid + 1) % 4],
                                   [1, 1, 1])
            )
            self.fixed_frame_edges_front.append(
                p.addUserDebugLine(self.front_view_square[eid],
                                   self.front_view_square[(eid + 1) % 4],
                                   [0, 0, 1])
            )
            self.fixed_frame_edges_back.append(
                p.addUserDebugLine(self.back_view_square[eid],
                                   self.back_view_square[(eid + 1) % 4],
                                   [0, 0, 1])
            )

        for eid in range(4):
            self.move_frame_edges_back.append(
                p.addUserDebugLine(self.camera_pos, self.back_view_square[eid], [1, 1, 0])
            )
            self.fixed_frame_edges_back.append(
                p.addUserDebugLine(self.camera_pos, self.back_view_square[eid], [0, 0, 1])
            )

        self.camera_line_inverse = p.addUserDebugLine(self.camera_pos, [0, 0, 0], [0.0, 0.0, 1.0])
        self.init_obs = self.get_obs()
        return self.init_obs

    def step(self, a):
        """
        Apply an action and return the new observation, reward, done flag, and info.
        """
        done = self.act(a)
        obs = None if self.PASS_OBS else self.get_obs()
        r = 0
        return obs, r, done, {}


if __name__ == '__main__':
    # Set rendering mode and connect.
    RENDER = True
    robot_ID = 1  # 1, 2, or 3 to choose among different robot URDFs
    p.connect(p.GUI) if RENDER else p.connect(p.DIRECT)

    # Initialize the environment.
    env = FBVSM_Env(robot_ID=robot_ID,
                    width=100,
                    height=100,
                    render_flag=RENDER,
                    num_motor=NUM_MOTOR,
                    init_angle=[-np.pi/2, 0, 0, -np.pi/2],
                    object_alpha=1)
    obs = env.reset()
    c_angle = obs[0]

    # Manual mode only: use PyBullet debug sliders to adjust joint commands.
    m_list = [
        p.addUserDebugParameter("motor0: yaw",   -1, 1, 0),
        p.addUserDebugParameter("motor1: pitch", -1, 1, 0),
        p.addUserDebugParameter("motor2: m2",    -1, 1, 0),
        p.addUserDebugParameter("motor3: m3",    -1, 1, 0)
    ]
    runTimes = 10000
    for _ in range(runTimes):
        try:
            for c_id in range(NUM_MOTOR):
                c_angle[c_id] = p.readUserDebugParameter(m_list[c_id])
        except Exception as e:
            print(e)
            continue
        obs, _, _, _ = env.step(c_angle)
        print("Current normalized angles:", np.round(obs[0],3))
