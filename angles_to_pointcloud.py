import pybullet as p
import pybullet_data as pd
import numpy as np
import os
import open3d as o3d
from urdfpy import URDF
import copy
import random
import time

num_links = 3
robot = URDF.load('arm3dof/urdf/arm3dof.urdf')
stl_file_path = "arm3dof/meshes/"
mesh_base = o3d.io.read_triangle_mesh(stl_file_path + "base.stl")
mesh_l1 = o3d.io.read_triangle_mesh(stl_file_path + "link1.stl")
mesh_l2 = o3d.io.read_triangle_mesh(stl_file_path + "link2.stl")
mesh_l3 = o3d.io.read_triangle_mesh(stl_file_path + "link3.stl")


def test_orig_matrix():
    mesh_base_c = copy.deepcopy(mesh_base)
    mesh_l1_c = copy.deepcopy(mesh_l1)
    mesh_l2_c = copy.deepcopy(mesh_l2)
    mesh_l3_c = copy.deepcopy(mesh_l3)
    fk = robot.link_fk()
    for i in range(4):
        print(fk[robot.links[i]])

    mesh_l1_c.transform(fk[robot.links[1]])
    mesh_l2_c.transform(fk[robot.links[2]])
    mesh_l3_c.transform(fk[robot.links[3]])

    pc_base = mesh_base_c.sample_points_poisson_disk(1000)
    pc_l1 = mesh_l1_c.sample_points_poisson_disk(1000)
    pc_l2 = mesh_l2_c.sample_points_poisson_disk(1000)
    pc_l3 = mesh_l3_c.sample_points_poisson_disk(1000)

    o3d.visualization.draw_geometries([mesh_base_c, mesh_l1_c, mesh_l2_c, mesh_l3_c])
    o3d.visualization.draw_geometries([pc_base, pc_l1, pc_l2, pc_l3])


def read_matrix_and_visualize(path):
    mesh_base_c = copy.deepcopy(mesh_base)  # .compute_vertex_normals()
    mesh_l1_c = copy.deepcopy(mesh_l1)
    mesh_l2_c = copy.deepcopy(mesh_l2)
    mesh_l3_c = copy.deepcopy(mesh_l3)

    T = []
    for i in range(1, num_links + 1):
        T.append(np.loadtxt(path + "link%d.csv" % i))

    mesh_l1_c.transform(T[0])
    mesh_l2_c.transform(T[1])
    mesh_l3_c.transform(T[2])

    pc_base = mesh_base_c.sample_points_poisson_disk(1000)
    pc_l1 = mesh_l1_c.sample_points_poisson_disk(1000)
    pc_l2 = mesh_l2_c.sample_points_poisson_disk(1000)
    pc_l3 = mesh_l3_c.sample_points_poisson_disk(1000)

    o3d.visualization.draw_geometries([mesh_base_c, mesh_l1_c, mesh_l2_c, mesh_l3_c])
    # o3d.visualization.draw_geometries([pc_base, pc_l1, pc_l2, pc_l3])


def prepare_check():
    mesh_base_c = copy.deepcopy(mesh_base)  # .compute_vertex_normals()
    mesh_l1_c = copy.deepcopy(mesh_l1)
    mesh_l2_c = copy.deepcopy(mesh_l2)
    mesh_l3_c = copy.deepcopy(mesh_l3)

    mesh_base_c = o3d.t.geometry.TriangleMesh.from_legacy(mesh_base_c)
    mesh_l1_c = o3d.t.geometry.TriangleMesh.from_legacy(mesh_l1_c)
    mesh_l2_c = o3d.t.geometry.TriangleMesh.from_legacy(mesh_l2_c)
    mesh_l3_c = o3d.t.geometry.TriangleMesh.from_legacy(mesh_l3_c)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_base_c)
    scene.add_triangles(mesh_l1_c)
    scene.add_triangles(mesh_l2_c)
    scene.add_triangles(mesh_l3_c)

    return scene

def check_occupancy(point_pos, scene):

    query_point = o3d.core.Tensor([point_pos], dtype=o3d.core.Dtype.Float32)
    occupancy = scene.compute_occupancy(query_point)

    return occupancy


if __name__ == "__main__":
    """test orig pos"""
    # test_orig_matrix()

    """check saved data"""
    # for i in range(3):
    #     data_path = "transform_data/data_%d/" % i
    #     read_matrix_and_visualize(path=data_path)

    """check occupancy"""
    check_sen = prepare_check()
    box_size = 0.05
    points = []
    for i in range(1000000):
        points.append([(random.random() - 0.5) * box_size, (random.random() - 0.5) * box_size, random.random() * box_size])
    print('done')

    st = time.time()
    for i in range(len(points)):
        occ = check_occupancy(points[i], check_sen)
        # if occ[0] == 1:
        #     print("1")

    et = time.time()
    print("time used: ", et-st)
