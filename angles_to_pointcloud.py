import pybullet as p
import pybullet_data as pd
import numpy as np
import os
import open3d as o3d
import urdfpy
import copy

if __name__ == "__main__":
    stl_file_path = "arm3dof/meshes/"
    mesh_base = o3d.io.read_triangle_mesh(stl_file_path+"base.stl")
    mesh_l1 = o3d.io.read_triangle_mesh(stl_file_path+"link1.stl")
    mesh_l2 = o3d.io.read_triangle_mesh(stl_file_path + "link2.stl")
    mesh_l3 = o3d.io.read_triangle_mesh(stl_file_path + "link3.stl")

    # T_l1 = np.loadtxt("transform_data/data_01/link0.csv")
    # T_l2 = np.loadtxt("transform_data/data_01/link1.csv")
    # T_l3 = np.loadtxt("transform_data/data_01/link2.csv")

    mesh_base_c = copy.deepcopy(mesh_base)
    mesh_l1_c = copy.deepcopy(mesh_l1)

    mesh_base_c.translate((1.4141863553364E-17, 0.028, 1.94886031123557E-17))
    mesh_l1_c.translate((-4.18213071567017E-09, 0.0313332808054455, - 0.00054042909910519))

    # mesh_l1_c.transform(T_l1)
    # mesh_l2_c.transform(T_l2)
    # mesh_l3_c.transform(T_l3)

    pc_base = mesh_base_c.sample_points_poisson_disk(1000)
    pc_l1 = mesh_l1_c.sample_points_poisson_disk(1000)
    # pc_l2 = mesh_l2_c.sample_points_poisson_disk(1000)
    # pc_l3 = mesh_l3_c.sample_points_poisson_disk(1000)

    # o3d.visualization.draw_geometries([mesh_base, mesh_l1, mesh_l2, mesh_l3])

    # o3d.visualization.draw_geometries([pc_base, pc_l1, pc_l2, pc_l3])
    o3d.visualization.draw_geometries([pc_base, pc_l1])