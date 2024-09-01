# import open3d as o3d
import numpy as np
from test_model import query_models
from train import init_models
import torch
from env4 import FBVSM_Env
import pybullet as p
import tqdm
import matplotlib.pyplot as plt
import time

# def visualize_ee():
#     ee_path = '4dof_2nd/meshes/l4.STL'
#     ee = o3d.io.read_triangle_mesh(ee_path)
#     ee.compute_vertex_normals()
#     ee.paint_uniform_color([0.1, 0.1, 0.7])
#     coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0.06])
#     o3d.visualization.draw_geometries([ee, coord])


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


def evaluate_ee(ee_model, env):
    workspace = np.loadtxt(workspace_path)
    print(workspace.shape)
    record_c = []
    for angles in tqdm.tqdm(workspace): # this takes an hour
        degree_angles = angles * action_space
        degree_angles = torch.tensor(degree_angles).to(device)
        xyz_center_m = query_models(degree_angles, ee_model, DOF, mean_ee=True, n_samples = 64)
        xyz_center_m = xyz_center_m.detach().cpu().numpy()

        xyz_center_s = query_simulator(env, angles)
        # combine the two centers
        centers = np.concatenate([xyz_center_m, xyz_center_s]) 

        # print('model-sim: ', centers)
        record_c.append(centers)

    record_c = np.array(record_c)
    print(record_c.shape)
    np.savetxt(centers_path, record_c, fmt="%.6f")

def plot_ee():
    workspace = np.loadtxt(workspace_path)
    centers = np.loadtxt(centers_path)

    c_m = centers[:, :3] # model prediction
    c_s = centers[:, 3:] # simulator prediction, ground truth

    mse_error = np.linalg.norm(c_m - c_s, axis=1) # shape: (n,)
    print("max error: ", np.max(mse_error))
    # normalized_error = mse_error / np.linalg.norm(c_s, axis=1)
    line_array = np.linspace(-1.0, 1.0, num=11)
    # angle 0, 1, 2, 3; 2 & 3 are the input angles

    """!!! the first row (angle = -1) is unstable, trained with around 200 data in dataset, consider ignore it"""
    ignore_first = False
    if ignore_first:
        start = 1
        end = 11
    else:
        start = 0
        end = 11

    """ee center error VS one angle"""
    ids = [0, 1, 2, 3]
    for angle_id in ids:
        error_1d = []
        for angle_i in line_array:
            mask2_i = np.isclose(workspace[:, angle_id], angle_i, atol=1e-3)
            error_i = mse_error[mask2_i] # shape: (m,), m<=n
            print("angle %d=" %angle_id, np.round(angle_i, 3), "Number of data: ", error_i.shape)
            error_1d.append(np.mean(error_i))

        plt.plot(line_array[start:end], error_1d[start:end])
        plt.xticks(line_array[start:end])
        plt.xlabel('angle %d' % np.round(angle_id, 3))
        plt.ylabel('MSE')
        plt.show()


    """ee center error VS one angle (in the same plot)"""
    ids = [0, 1, 2, 3]

    for angle_id in ids:
        error_1d = []
        for angle_i in line_array:
            mask2_i = np.isclose(workspace[:, angle_id], angle_i, atol=1e-3)
            error_i = mse_error[mask2_i] # shape: (m,), m<=n
            error_1d.append(np.mean(error_i))

        plt.plot(line_array[start:end], error_1d[start:end], label='angle %d' % np.round(angle_id, 3))
        # plt.xlabel('angle %d' % np.round(angle_id, 3))
    plt.legend()
    plt.xticks(line_array[start:end])
    plt.xlabel('angle')
    plt.ylabel('MSE')
    plt.show()


    """ee center error VS two angles"""
    idss = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for angle_ids in idss:
        error_2d = []
        for angle_i in line_array:
            error_i = []
            for angle_j in line_array:
                mask2_i = np.isclose(workspace[:, angle_ids[0]], angle_i, atol=1e-3)
                mask2_j = np.isclose(workspace[:, angle_ids[1]], angle_j, atol=1e-3)
                mask2_ij = np.logical_and(mask2_i, mask2_j)
                error_ij = mse_error[mask2_ij]
                error_i.append(np.mean(error_ij))
            error_2d.append(error_i)
        error_2d = np.array(error_2d)

        fig, ax = plt.subplots()
        # cax = ax.matshow(error_2d[start:end, start:end], cmap='binary')
        # fig.colorbar(cax)
        plt.imshow(error_2d[start:end, start:end], cmap='viridis')
        plt.colorbar(label='MSE')
        plt.xticks(range(len(line_array[start:end])), np.round(line_array[start:end], 2))
        plt.yticks(range(len(line_array[start:end])), np.round(line_array[start:end], 2))
        plt.xlabel('angle %d' % angle_ids[0])
        plt.ylabel('angle %d' % angle_ids[1])
        plt.show()


    """ee center error VS ee abs distance"""
    abs_distance = np.linalg.norm(c_s, axis=1) # shape: (n,)
    # angle1_mask, angle1 != -1
    mask = np.isclose(workspace[:, 1], -1, atol=1e-3)
    inv_mask = np.logical_not(mask)
    abs_distance = abs_distance[inv_mask]
    mse_error = mse_error[inv_mask]
    plt.scatter(abs_distance, mse_error)
    plt.xlabel('ee abs distance')
    plt.ylabel('MSE')
    plt.show()

    """ee center error VS ee axis distance"""
    c_s = c_s[inv_mask]
    x = c_s[:, 0]
    y = c_s[:, 1]
    z = c_s[:, 2] # shape: (n,)
    axiss = {"xy": (x, y), "xz": (x, z), "yz": (y, z)}
    for key, axis in axiss.items():

        X = axis[0]
        Y = axis[1]
        Z = mse_error
        print(np.max(Z), np.min(Z))
        grid_x, grid_y = np.mgrid[min(X):max(X):30j, min(Y):max(Y):30j]
        from scipy.interpolate import griddata
        grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='linear')
        print(grid_z.shape)
        print(np.nanmax(grid_z), np.nanmin(grid_z))
        plt.imshow(grid_z.T, extent=(min(X), max(X), min(Y), max(Y)), origin='lower', cmap='plasma')
        plt.colorbar(label='MSE')
        plt.xlabel('ee position %s' % key[0])
        plt.ylabel('ee position %s' % key[1])
        plt.show()
        
def error_robot_state():
    workspace = np.loadtxt(workspace_path)
    centers = np.loadtxt(centers_path)
    c_m = centers[:, :3] # model prediction
    c_s = centers[:, 3:] # simulator prediction, ground truth
    mse_error = np.linalg.norm(c_m - c_s, axis=1) # shape: (n,)

    sorted_idx = np.argsort(mse_error)
    max_10_idx = sorted_idx[-10:]
    min_10_idx = sorted_idx[:10]
    print("max 10 error: ", mse_error[max_10_idx])
    print("min 10 error: ", mse_error[min_10_idx])

    p.connect(p.GUI)
    env = FBVSM_Env(
        show_moving_cam=False,
        robot_ID=robot_id,
        width=width,
        height=height,
        render_flag=True,
        num_motor=DOF,
        dark_background=True,
        init_angle=[-0.5, -0.3, -0.5, -0.2])
    
    debug_id = None
    for idx in min_10_idx:

        if debug_id is not None:
            p.removeUserDebugItem(debug_id)
        
        angles = workspace[idx]
        print("max error command: ", angles)
        env.act(angles)
        env.get_obs()
        # draw the ee center
        debug_id = p.addUserDebugPoints([c_s[idx]], [[0, 1, 0]], pointSize=20)
        time.sleep(10)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("ee eval: ", device)

    DOF = 4
    RENDER = False
    robot_id = 1
    action_space = 90
    pxs = 100
    height = pxs
    width = pxs
    workspace_path = 'data/action/ee_workspace_10.csv'
    centers_path = 'data/action/ee_centers(model-sim).csv'

    """read workspace, record the ee centers of model and simulator"""
    
    # p.connect(p.GUI) if RENDER else p.connect(p.DIRECT)
    # model, env = load_model_env()
    # evaluate_ee(ee_model=model, env=env)

    """read workspace and centers, plot results"""
    # plot_ee()

    """visualize robot state"""
    error_robot_state()


 