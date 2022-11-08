import numpy as np
import pybullet as p
import time
import pybullet_data
import torch
from PIL import Image
from rays_check import pose_spherical, camera_spherical
from data_collection import angle_sim

width = 300
height = 300
"""camera parameters"""
view_matrix = p.computeViewMatrix(
    # cameraEyePosition=[0.8, 0, 0.606],
    cameraEyePosition=[0.6, -0.3464, 0.4 + 0.606],
    cameraTargetPosition=[0, 0, 0.606],
    cameraUpVector=[0, 0, 1])

projection_matrix = p.computeProjectionMatrixFOV(
    fov=42.0,
    aspect=1.0,
    nearVal=0.1,
    farVal=200)
"""camera parameters"""


def get_img():
    img = p.getCameraImage(width, height, view_matrix, projection_matrix,
                           renderer=p.ER_BULLET_HARDWARE_OPENGL,
                           shadow=0)
    rgbBuffer = img[2][:, :, :3]
    # musk = get_musk(rgbBuffer)
    rgbBuffer[50, 50] = np.array([200, 0, 0])
    img = Image.fromarray(rgbBuffer, 'RGB')
    img.show()
    # img.save(DATA_PATH + "%d.png" % idx)
    return img


def get_ray_bullet():
    o = np.array([0.8, 0, 0.606])
    fov = 42
    near = 0.3  # to camera 0.5
    far = -0.3  # to camera 1.1


def draw_balls():
    p.createMultiBody(0, 0)
    colSphereId_1 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01)
    colSphereId_2 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.02)
    colSphereId_3 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)

    mass = 1
    visualShapeId = -1
    ballOrientation = [0, 0, 0, 1]

    rate = np.tan(21 * np.pi / 180)
    # p.createMultiBody(mass, colSphereId_1, visualShapeId, [0.5, 0, 0.606], ballOrientation)

    # mid
    # p.createMultiBody(mass, colSphereId_2, visualShapeId, [0, 0.8 * rate, 0.606 + 0.8 * rate], ballOrientation)
    # p.createMultiBody(mass, colSphereId_2, visualShapeId, [0, -0.8 * rate, 0.606 + 0.8 * rate], ballOrientation)
    # p.createMultiBody(mass, colSphereId_2, visualShapeId, [0, 0.8 * rate, 0.606 - 0.8 * rate], ballOrientation)
    # p.createMultiBody(mass, colSphereId_2, visualShapeId, [0, -0.8 * rate, 0.606 - 0.8 * rate], ballOrientation)

    # near
    p.createMultiBody(mass, colSphereId_1, visualShapeId, [0.3, 0.5 * rate, 0.606 + 0.5 * rate], ballOrientation)
    p.createMultiBody(mass, colSphereId_1, visualShapeId, [0.3, -0.5 * rate, 0.606 + 0.5 * rate], ballOrientation)
    p.createMultiBody(mass, colSphereId_1, visualShapeId, [0.3, 0.5 * rate, 0.606 - 0.5 * rate], ballOrientation)
    p.createMultiBody(mass, colSphereId_1, visualShapeId, [0.3, -0.5 * rate, 0.606 - 0.5 * rate], ballOrientation)

    # far
    p.createMultiBody(mass, colSphereId_3, visualShapeId, [-0.3, 1.1 * rate, 0.606 + 1.1 * rate], ballOrientation)
    p.createMultiBody(mass, colSphereId_3, visualShapeId, [-0.3, -1.1 * rate, 0.606 + 1.1 * rate], ballOrientation)
    p.createMultiBody(mass, colSphereId_3, visualShapeId, [-0.3, 1.1 * rate, 0.606 - 1.1 * rate], ballOrientation)
    p.createMultiBody(mass, colSphereId_3, visualShapeId, [-0.3, -1.1 * rate, 0.606 - 1.1 * rate], ballOrientation)


def direction_check():
    c2w = pose_spherical(45, -45, 0.8)
    w2c = camera_spherical(45, -45, 0.8)
    print(c2w)
    print(w2c)
    dir_0 = torch.tensor([1., 1., 0]).reshape(3, 1)
    dir_1 = torch.mm(c2w[:3, :3], dir_0)
    print(dir_1)
    return 0


if __name__ == "__main__":
    # direction_check()
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.createCollisionShape(p.GEOM_PLANE)
    draw_balls()
    startPos = [0, 0, 0.5]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])

    urdf_path = 'arm3dof/urdf/arm3dof.urdf'

    robotId = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=1)

    basePos, baseOrn = p.getBasePositionAndOrientation(robotId)  # Get model position
    basePos_list = [basePos[0], basePos[1], 0.3]
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20,
                                 cameraTargetPosition=basePos_list)  # fix camera onto model

    p.addUserDebugLine([0, -0.3, 0], [0, -0.3, 1], [1, 0, 0])
    p.addUserDebugLine([0, 0.3, 0], [0, 0.3, 1], [1, 0, 0])

    ang_list = np.array([90.0, 90.0, .0]) * np.pi / 180
    angle_sim(ang_list, robotId)

    for i in range(10000000):
        p.stepSimulation()
        if i == 100:
            get_img()
        time.sleep(1. / 240.)

    p.disconnect()
