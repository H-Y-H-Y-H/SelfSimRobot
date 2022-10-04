import pybullet as p
import pybullet_data
import numpy as np
import os
import time
import random
from PIL import Image as im


def point_test(check_point, ray_len):

    count_x = 0
    p_list_0 = [
        [check_point[0] + ray_len, check_point[1],           check_point[2]          ],
        [check_point[0] - ray_len, check_point[1],           check_point[2]          ],
        [check_point[0]          , check_point[1] + ray_len, check_point[2]          ],
        [check_point[0]          , check_point[1] - ray_len, check_point[2]          ],
        [check_point[0]          , check_point[1],           check_point[2] + ray_len],
        [check_point[0]          , check_point[1],           check_point[2] - ray_len]
    ]

    p_list_1 = [check_point] * 6

    for i in range(6):
        inside = p.rayTest(p_list_0[i], p_list_1[i])

        if inside[0][-2] == (0.0,0.0,0.0):
            return 0  # outside

    return 1 # inside


def face_sampling(box_len, num_points, filename="facecloud01.csv"):
    """get the point cloud for the robot surface"""
    face_num = num_points * num_points
    start_p = -box_len / 2
    end_p = box_len / 2
    x_p = np.linspace(start_p, end_p, num_points)
    y_p = np.linspace(start_p, end_p, num_points)
    z_p = np.linspace(0, end_p*2, num_points)
    xx, zz = np.meshgrid(x_p, z_p)
    y0 = np.zeros((face_num, 1)) + start_p
    y1 = np.zeros((face_num, 1)) + end_p
    x1 = xx.reshape(face_num, 1)
    z1 = zz.reshape(face_num, 1)

    # front and back
    face_0 = np.concatenate((x1, y0, z1), axis=1)
    face_1 = np.concatenate((x1, y1, z1), axis=1)

    # left and right
    face_2 = np.concatenate((y0, x1, z1), axis=1)
    face_3 = np.concatenate((y1, x1, z1), axis=1)

    # up and down
    face_4 = np.concatenate((x1, z1 + start_p, y0 + end_p), axis=1)
    face_5 = np.concatenate((x1, z1 + start_p, y1 + end_p), axis=1)

    hit_pos = []
    ray_test_1 = p.rayTestBatch(face_0, face_1)
    for t in ray_test_1:
        if t[3] != (0.0,0.0,0.0):
            hit_pos.append(t[3])

    ray_test_2 = p.rayTestBatch(face_1, face_0)
    for t in ray_test_2:
        if t[3] != (0.0,0.0,0.0):
            hit_pos.append(t[3])

    ray_test_3 = p.rayTestBatch(face_2, face_3)
    for t in ray_test_3:
        if t[3] != (0.0,0.0,0.0):
            hit_pos.append(t[3])

    ray_test_4 = p.rayTestBatch(face_3, face_2)
    for t in ray_test_4:
        if t[3] != (0.0,0.0,0.0):
            hit_pos.append(t[3])

    ray_test_5 = p.rayTestBatch(face_4, face_5)
    for t in ray_test_5:
        if t[3] != (0.0,0.0,0.0):
            hit_pos.append(t[3])

    ray_test_6 = p.rayTestBatch(face_5, face_4)
    for t in ray_test_6:
        if t[3] != (0.0,0.0,0.0):
            hit_pos.append(t[3])

    np.savetxt("musk_data/"+filename, hit_pos)


def inside_data_sampling(n, box_size=1,filename="pcloud01.csv"):
    inside_data = []
    while True:
        point = [(random.random()-0.5)*box_size, (random.random()-0.5)*box_size, random.random()*box_size]
        if point_test(point, 1)==1:
            # print(point)
            inside_data.append(point)

        if len(inside_data) == n:
            np.savetxt("musk_data/"+filename, inside_data)
            break

def pixel_sampling(one_step=0.005, steps=80, filename="hh.csv"):
    half_len = one_step * steps
    inside_data = []
    for x in range(steps * 2):
        for y in range(steps * 2):
            for z in range(steps * 2):
                point = [x*one_step-half_len,y*one_step-half_len,z*one_step]
                if point_test(point, 1)==1:
                    # print(point)
                    inside_data.append(point)

    np.savetxt("data_with_para/"+filename, inside_data)

def get_shadow(box_len, num_points, filename="ray_test01.csv"):
    start_p = -box_len / 2
    end_p = box_len / 2

    step_size = box_len / num_points
    x_p = np.linspace(start_p, end_p, num_points).reshape(num_points, 1)
    y0 = np.zeros((num_points, 1)) + start_p
    y1 = np.zeros((num_points, 1)) + end_p

    shadow_matrix = np.zeros((num_points, num_points))

    for i in range(num_points):
        z_p = np.array([box_len - i * step_size] * num_points).reshape(num_points, 1)
        line_0 = np.concatenate((x_p, y0, z_p), axis=1)
        line_1 = np.concatenate((x_p, y1, z_p), axis=1)
        # print(line_0.shape)
        ray_test = p.rayTestBatch(line_0, line_1)
        for x, value in enumerate(ray_test):
            if value[3] != (0.0,0.0,0.0):
                shadow_matrix[i, x] = 1


    np.savetxt("shadow_data/"+filename,shadow_matrix)

    # print(shadow_matrix.astype(np.uint8))
    # shadow_matrix = shadow_matrix * 255
    # musk_data = im.fromarray(shadow_matrix)
    # musk_data.show()

    return shadow_matrix






if __name__ == "__main__":

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")

    startPos = [0,0,1]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    boxId = p.loadURDF("r2d2.urdf",startPos, startOrientation)

    p.setGravity(0, 0, -9.8)
    p.addUserDebugLine([0.5,0.5,0], [0.5,0.5,1], [1,0,0])
    p.addUserDebugLine([0.5,-0.5,0], [0.5,-0.5,1], [1,0,0])
    p.addUserDebugLine([-0.5,-0.5,0], [-0.5,-0.5,1], [1,0,0])
    p.addUserDebugLine([-0.5,0.5,0], [-0.5,0.5,1], [1,0,0])
    for _ in range(1000):
        p.stepSimulation()

    st = time.time()
    # inside_data_sampling(10000)
    # face_sampling(box_len=1, num_points=60)
    get_shadow(box_len=1, num_points=101)
    et = time.time()

    print("Time: ", et-st)
    # print(point_test([0,0,0.3], 1))