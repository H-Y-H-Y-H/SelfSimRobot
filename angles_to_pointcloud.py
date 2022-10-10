import pybullet as p
import pybullet_data as pd
import numpy as np
import os
import open3d as o3d
from urdfpy import URDF
import copy

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
    mesh_base_c = copy.deepcopy(mesh_base)
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

    # o3d.visualization.draw_geometries([mesh_base_c, mesh_l1_c, mesh_l2_c, mesh_l3_c])
    o3d.visualization.draw_geometries([pc_base, pc_l1, pc_l2, pc_l3])


if __name__ == "__main__":
    # test_orig_matrix()
    for i in range(10):
        data_path = "transform_data/data_%d/" % i
        read_matrix_and_visualize(path=data_path)
