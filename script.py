import pybullet as p
import time
import pybullet_data
import numpy as np
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
# planeId = p.loadURDF("plane.urdf")
startPos = [0,0,1]
startOrientation = p.getQuaternionFromEuler([0,0,0])
startPos = [0, 0, -0.108]
startOrientation = p.getQuaternionFromEuler([0, 0, -np.pi / 2])

boxId= p.loadURDF(
    # 'DOF4ARM0/urdf/DOF4ARM0.urdf',
    '4dof_2nd/urdf/4dof_2nd.urdf',
    startPos,
    startOrientation,
    flags=p.URDF_USE_SELF_COLLISION,  # no mold penetration
    useFixedBase=1)

#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
