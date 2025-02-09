import pybullet as p
import pybullet_data
import time
import numpy as np

class SimpleRobotArmEnv:
    def __init__(self,
                 urdf_path="4dof_2nd/urdf/4dof_2nd.urdf",
                 use_gui=True):
        """
        Minimal environment that:
          1) Loads a plane
          2) Loads a 4-DOF robot arm URDF
          3) Creates debug sliders so you can set joint angles interactively
        """
        self.use_gui = use_gui
        self.physics_client = p.connect(p.GUI if self.use_gui else p.DIRECT)

        # Add default search path for plane.urdf, etc.
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load plane
        self.plane_id = p.loadURDF("plane.urdf", [0,0,0])
        p.setGravity(0, 0, -9.81)

        # Load the robot (4-DOF). Change the path if needed.
        start_pos = [0, 0, 0]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(urdf_path, start_pos, start_orn, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)

        # Create debug sliders for each joint
        self.debug_sliders = []
        for joint_index in range(self.num_joints):
            slider_id = p.addUserDebugParameter(f"Joint {joint_index}", -3.14, 3.14, 0.0)
            self.debug_sliders.append(slider_id)

    def read_debug_params(self):
        """
        Reads the current slider values from the GUI for each joint.
        Returns a list of target angles (radians).
        """
        target_angles = []
        for slider_id in self.debug_sliders:
            angle = p.readUserDebugParameter(slider_id)
            target_angles.append(angle)
        return target_angles

    def set_joint_angles(self, target_angles, max_velocity=2.0, force=100.0):
        """
        Sets joint angles using POSITION_CONTROL.
        :param target_angles: List of angles in radians.
        :param max_velocity:  Maximum velocity for each joint.
        :param force:         Max force the motors can exert.
        """
        for joint_index, angle in enumerate(target_angles):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                force=force,
                maxVelocity=max_velocity
            )

    def step(self, sleep_time=1./240.):
        """
        Advances the PyBullet simulation by one step.
        Sleeps to match real-time if GUI is on.
        """
        p.stepSimulation()
        if self.use_gui:
            time.sleep(sleep_time)

    def get_joint_positions(self):
        """
        Returns the current joint angles (radians) for all joints.
        """
        joint_positions = []
        for joint_index in range(self.num_joints):
            pos, vel, _, _ = p.getJointState(self.robot_id, joint_index)
            joint_positions.append(pos)
        return joint_positions

    def disconnect(self):
        """ Disconnects from the PyBullet simulation. """
        p.disconnect()

if __name__ == "__main__":
    # Example usage:
    env = SimpleRobotArmEnv(use_gui=True)

    print("Use the on-screen sliders to move each joint. Press Ctrl+C to quit.")
    try:
        while True:
            # 1) Read angles from the debug sliders
            angles = env.read_debug_params()


            ########### TO DO #############
            # updates your angles here with your designed controller and comment above debug sliders.
            # If you need to use this robot for your own research.


            # 2) Command the robot to these angles
            env.set_joint_angles(angles)

            # 3) Step the simulation
            env.step()

            # (Optionally, you can also read back the current angles if desired)
            # current_positions = env.get_joint_positions()
            # print(current_positions)

    except KeyboardInterrupt:
        pass

    env.disconnect()
