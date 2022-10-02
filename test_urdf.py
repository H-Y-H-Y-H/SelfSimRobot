import pybullet as p
import time
import pybullet_data as pd
import numpy as np

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pd.getDataPath())  # optionally
p.setGravity(0, 0, -10)
# planeId = p.loadURDF("plane.urdf")

startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

urdf_path = 'arm3dof/urdf/arm3dof.urdf'
# urdf_path = 'robot_arm/robot_arm1.urdf'

robotid = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=1)
# planeId2 = p.loadURDF("plane.urdf", [-5, 0, 0], p.getQuaternionFromEuler([0, 1.57, 0]))
# planeId3 = p.loadURDF("plane.urdf", [0, 5, 0], p.getQuaternionFromEuler([1.57, 0, 0]))
textureId = p.loadTexture("green.png")
WallId_front = p.loadURDF("plane.urdf",[0,2,0], p.getQuaternionFromEuler([1.57,0,0]))
# WallId_right = p.loadURDF("plane.urdf", [-10, 0, 14], p.getQuaternionFromEuler([1.57, 0, 1.57]))
# WallId_left = p.loadURDF("plane.urdf", [10, 0, 14], p.getQuaternionFromEuler([1.57, 0, -1.57]))


# p.changeVisualShape(planeId, -1, textureUniqueId=textureId)
p.changeVisualShape(WallId_front, -1, textureUniqueId=textureId)
# p.changeVisualShape(WallId_right, -1, textureUniqueId=textureId)
# p.changeVisualShape(WallId_left , -1, textureUniqueId=textureId)

"""camera parameters"""
width = 128
height = 128

view_matrix = p.computeViewMatrix(
    cameraEyePosition=[0, -1, 0.2],
    cameraTargetPosition=[0, 0, 0.2],
    cameraUpVector=[0, 0, 1])

projection_matrix = p.computeProjectionMatrixFOV(
    fov=45.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=10.0)
"""camera parameters"""

img = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

rgbBuffer = img[2]
for i in range(2000):
    p.stepSimulation()
    time.sleep(1. / 240.)

p.disconnect()


