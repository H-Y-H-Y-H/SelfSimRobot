# import pybullet as p
# import time
# import pybullet_data
# import numpy as np
# physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# p.setGravity(0,0,-10)
# # planeId = p.loadURDF("plane.urdf")
# startPos = [0,0,1]
# startOrientation = p.getQuaternionFromEuler([0,0,0])
# startPos = [0, 0, -0.108]
# startOrientation = p.getQuaternionFromEuler([0, 0, -np.pi / 2])
#
# boxId= p.loadURDF(
#     # 'DOF4ARM0/urdf/DOF4ARM0.urdf',
#     '4dof_2nd/urdf/4dof_2nd.urdf',
#     startPos,
#     startOrientation,
#     flags=p.URDF_USE_SELF_COLLISION,  # no mold penetration
#     useFixedBase=1)
#
# #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
# for i in range (10000):
#     p.stepSimulation()
#     time.sleep(1./240.)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
# p.disconnect()
import numpy as np
trj = [(-0.5, -0.3, -0.2, -0.2), (-0.4, -0.3, -0.2, -0.2), (-0.30000000000000004, -0.3, -0.2, -0.2), (-0.30000000000000004, -0.19999999999999998, -0.2, -0.2), (-0.30000000000000004, -0.19999999999999998, -0.1, -0.2), (-0.30000000000000004, -0.09999999999999998, -0.1, -0.2), (-0.20000000000000004, -0.09999999999999998, -0.1, -0.2), (-0.20000000000000004, -0.09999999999999998, -0.1, -0.30000000000000004), (-0.20000000000000004, -0.09999999999999998, -0.1, -0.20000000000000004), (-0.20000000000000004, -0.09999999999999998, 0.0, -0.20000000000000004), (-0.10000000000000003, -0.09999999999999998, 0.0, -0.20000000000000004), (-0.10000000000000003, -0.09999999999999998, 0.0, -0.10000000000000003), (-2.7755575615628914e-17, -0.09999999999999998, 0.0, -0.10000000000000003), (-2.7755575615628914e-17, -0.09999999999999998, 0.0, -0.20000000000000004), (0.09999999999999998, -0.09999999999999998, 0.0, -0.20000000000000004), (0.19999999999999998, -0.09999999999999998, 0.0, -0.20000000000000004), (0.19999999999999998, -0.19999999999999998, 0.0, -0.20000000000000004), (0.3, -0.19999999999999998, 0.0, -0.20000000000000004), (0.3, -0.19999999999999998, -0.1, -0.20000000000000004), (0.3, -0.3, -0.1, -0.20000000000000004), (0.3, -0.3, -0.2, -0.20000000000000004), (0.4, -0.3, -0.2, -0.20000000000000004)]
np.savetxt('planning/trajectory/free_collision_planning_1.csv',trj)
