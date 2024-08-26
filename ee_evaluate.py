import open3d as o3d
import numpy as np
from test_model import query_models
from train import init_models
import torch
from env4 import FBVSM_Env
import pybullet as p
import tqdm

def visualize_ee():
    ee_path = '4dof_2nd/meshes/l4.STL'
    ee = o3d.io.read_triangle_mesh(ee_path)
    ee.compute_vertex_normals()
    ee.paint_uniform_color([0.1, 0.1, 0.7])
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0.06])
    o3d.visualization.draw_geometries([ee, coord])


def load_model_env():
    test_model_pth = 'train_log/%s_id%d_10000(%d)_PE(ee)/best_model/' % ('real', robot_id, 1)
    model, _ = init_models(d_input=(DOF - 2) + 3,
                                   d_filter=128,
                                   output_size=2,
                                   FLAG_PositionalEncoder=True)
    model.load_state_dict(torch.load(test_model_pth + "best_model.pt", map_location=torch.device(device)))
    model = model.to(torch.float64)
    model.eval()

    env = FBVSM_Env(
        show_moving_cam=False,
        robot_ID=robot_id,
        width=width,
        height=height,
        render_flag=RENDER,
        num_motor=DOF,
        dark_background=True,
        init_angle=[-0.5, -0.3, -0.5, -0.2])

    return model, env

def query_simulator(env, angles):
    done = env.act(angles)
    # link_id: -1, 0, 1, 2, 3, 4; base, L1, L2, L3, L4, sphere_link (robot_id = 1)
    link_index = 4
    if not done:
        print("not reached")
        xyz = np.array([0, 0, 0])
    else:
        link_state = p.getLinkState(env.robot_id, link_index)
        xyz = np.array(link_state[4])
    return xyz


def evaluate_ee(ee_model, env, workspace):
    workspace = np.loadtxt(workspace)
    print(workspace.shape)
    record_c = []
    for angles in tqdm.tqdm(workspace):
        degree_angles = angles * action_space
        degree_angles = torch.tensor(degree_angles).to(device)
        xyz_center_m = query_models(degree_angles, model, DOF, mean_ee=True, n_samples = 64)
        xyz_center_m = xyz_center_m.detach().cpu().numpy()

        xyz_center_s = query_simulator(env, angles)
        # combine the two centers
        centers = np.concatenate([xyz_center_m, xyz_center_s]) 

        # print('model-sim: ', centers)
        record_c.append(centers)

    record_c = np.array(record_c)
    np.savetxt('data/action/ee_centers(model-sim).csv', record_c, fmt="%.6f")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train,", device)

    DOF = 4
    RENDER = False
    robot_id = 1
    action_space = 90
    pxs = 100
    height = pxs
    width = pxs
    workspace_path = 'data/action/ee_workspace_10.csv'
    p.connect(p.GUI) if RENDER else p.connect(p.DIRECT)

    model, env = load_model_env()
    evaluate_ee(
        ee_model=model,
        env=env,
        workspace=workspace_path,
    )
 