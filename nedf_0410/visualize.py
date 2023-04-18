# combine version April 14

import os
import torch
from train import nerf_forward, get_fixed_camera_rays, init_models
from func import w2c_matrix, c2w_matrix, get_rays
import numpy as np
# from env4 import FBVSM_Env
from env import FBVSM_Env
import pybullet as p
import matplotlib.pyplot as plt

# changed Transparency in urdf, line181, Mar31

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# PATH = "train_log/log_64000data_in6_out1_img100(3)/"
# PATH = "train_log/log_10000data_in7_out1_img100(1)/"

"""global parameters"""
height = 100
width = 100
HYPER_radius_scaler = 4.
dist = 2.
near, far = HYPER_radius_scaler - dist, HYPER_radius_scaler + dist

n_samples_hierarchical = 64
kwargs_sample_stratified = {
    'n_samples': 64,
    'perturb': True,
    'inverse_depth': False
}
kwargs_sample_hierarchical = {
    'perturb': True
}
chunksize = 2 ** 14

Camera_FOV = 42.
camera_angle_x = Camera_FOV * np.pi / 180.
focal = np.asarray(.5 * width / np.tan(.5 * camera_angle_x))
focal = torch.from_numpy(focal.astype('float32')).to(device)

# test_model_pth = "./train_log/log_8000data_in6_out1_img200_crop(2)/"
test_model_pth = "./train_log/test_2dof/"


def scale_points(predict_points):
    scaled_points = predict_points / 5.  # scale /5  0.8
    scaled_points[:, [1, 2]] = scaled_points[:, [2, 1]]
    scaled_points[:, [1, 0]] = scaled_points[:, [0, 1]]
    scaled_points[:, 0] = -scaled_points[:, 0]
    scaled_points[:, 2] += 1.106

    new_points = scaled_points
    return new_points


def load_point_cloud(angle_list: list, debug_points, logger, pc_pth):
    angle_lists = np.asarray([angle_list] * len(logger)) * 90  # -90 to 90
    diff = np.sum(abs(logger - angle_lists), axis=1)
    idx = np.argmin(diff)
    predict_points = np.load(pc_pth + '%04d.npy' % idx)
    # scaled points

    trans_points = scale_points(predict_points)
    # test_points = np.random.rand(100, 3)
    p_rgb = np.ones_like(trans_points)
    p.removeUserDebugItem(debug_points)  # update points every step
    debug_points = p.addUserDebugPoints(trans_points, p_rgb, pointSize=2)

    return debug_points


def interact_env(
        pic_size: int = 100,
        render: bool = True,
        interact: bool = True,
        dof: int = 3,
        logger_pth: str = " "):  # 4dof
    p.connect(p.GUI) if render else p.connect(p.DIRECT)
    logger = np.loadtxt(logger_pth + "logger.csv")

    env = FBVSM_Env(
        show_moving_cam=False,
        width=pic_size,
        height=pic_size,
        render_flag=render,
        num_motor=dof)

    obs = env.reset()

    obs = env.reset()
    c_angle = obs[0]
    debug_points = 0

    if interact:
        # 3 dof
        m0 = p.addUserDebugParameter("motor0: Yaw", -1, 1, 0)
        m1 = p.addUserDebugParameter("motor1: pitch", -1, 1, 0)
        m2 = p.addUserDebugParameter("motor2: m2", -1, 1, 0)

        # m3 = p.addUserDebugParameter("motor3: m3", -1, 1, 0)  # 4dof

        runTimes = 10000
        for i in range(runTimes):
            c_angle[0] = p.readUserDebugParameter(m0)
            c_angle[1] = p.readUserDebugParameter(m1)
            # c_angle[2] = p.readUserDebugParameter(m2)
            # c_angle[3] = p.readUserDebugParameter(m3)  # 4dof
            # print([c_angle[0], c_angle[1], c_angle[2]])
            debug_points = load_point_cloud([c_angle[0], c_angle[1]],
                                            debug_points,
                                            logger, pc_pth=logger_pth + "pc_record/")
            obs, _, _, _ = env.step(c_angle)
            # print(obs[0])


"""
collect point cloud and image data
"""


def test_model(log_pth, angle, model, dof, idx=1, C_POINTS=True):
    # theta, phi, third_angle = angle
    # DOF=4:
    theta, phi = angle

    pose_matrix = w2c_matrix(theta, phi, HYPER_radius_scaler)
    pose_matrix = torch.from_numpy(pose_matrix.astype('float32')).to(device)
    rays_o, rays_d = get_rays(height, width, focal, c2w=pose_matrix)  # apr 14
    # rays_o, rays_d = get_fixed_camera_rays(height, width, focal)
    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])

    count, count_empty = 0, 0

    points_record = np.asarray([])
    points_empty = np.asarray([])
    angle_tensor = torch.from_numpy(np.asarray(angle).astype('float32')).to(device)
    outputs = nerf_forward(rays_o, rays_d,
                           near, far, model, angle_tensor, dof,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           n_samples_hierarchical=n_samples_hierarchical,
                           kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                           chunksize=chunksize)

    # Check for any numerical issues.
    for k, v in outputs.items():
        if torch.isnan(v).any():
            print(f"! [Numerical Alert] {k} contains NaN.")
        if torch.isinf(v).any():
            print(f"! [Numerical Alert] {k} contains Inf.")

    all_points = outputs["query_points"].detach().cpu().numpy().reshape(-1, 3)
    rgb_each_point = outputs["rgb_each_point"].reshape(-1)
    rgb_predicted = outputs['rgb_map']
    np_image = rgb_predicted.reshape([height, width, 1]).detach().cpu().numpy()
    np_image = np.clip(0, 1, np_image)
    np_image_combine = np.dstack((np_image, np_image, np_image))

    plt.imshow(np_image_combine)

    binary_out = torch.relu(rgb_each_point)

    nonempty_idx = torch.nonzero(binary_out).cpu().detach().numpy().reshape(-1)
    empty_idx = torch.nonzero(torch.logical_not(binary_out)).cpu().detach().numpy().reshape(-1)
    select_empty_idx = np.random.choice(empty_idx, size=1000)

    query_xyz = all_points[nonempty_idx]
    # query_rgb = rgb_each_point[query_points]
    # query_rgb = np.sum(rgb_each_point[query_points],1)
    # occupied_point_idx = np.nonzero(query_rgb)
    empty_xyz = all_points[select_empty_idx]

    ax = plt.figure().add_subplot(projection='3d')
    # plt.suptitle('M1: %0.2f,  M2: %0.2f,  M3: %0.2f' % (angle[0], angle[1], angle[2]), fontsize=14)
    target_pose = c2w_matrix(theta, phi, 0.)

    # query_xyz = np.concatenate((query_xyz, np.ones((len(query_xyz), 1))), 1)
    # query_xyz = np.dot(target_pose, query_xyz.T).T[:, :3]

    ax.scatter(
        query_xyz[:, 0],
        -query_xyz[:, 2],
        query_xyz[:, 1],
        # s=1
        # alpha=query_rgb
    )

    ax.scatter(
        empty_xyz[:, 0],
        -empty_xyz[:, 2],
        empty_xyz[:, 1],
        alpha=0.1,
    )
    plt.xlabel('x-axis', fontsize=20)
    plt.ylabel('y-axis', fontsize=20)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    # plt.show()
    os.makedirs(log_pth + '/visual_test', exist_ok=True)
    plt.savefig(log_pth + '/visual_test/%04d.jpg' % idx)
    if C_POINTS:
        os.makedirs(log_pth + '/pc_record', exist_ok=True)
        np.save(log_pth + '/pc_record/%04d.npy' % idx, query_xyz)

    plt.clf()

    return points_record, points_empty


def collect_point_cloud(dof: int = 3, model_pth: str = ""):
    model, optimizer = init_models(d_input=dof + 3,
                                   n_layers=8,
                                   d_filter=128,
                                   output_size=1,
                                   skip=(4,))

    # April 14, 6*256, ...
    model.load_state_dict(torch.load(model_pth + "/best_model/nerf.pt", map_location=torch.device(device)))
    visual_pth = model_pth + "visual/"
    os.makedirs(visual_pth, exist_ok=True)

    model.eval()

    sep = 40
    theta_0_loop = np.linspace(-90., 90, sep, endpoint=False)
    theta_1_loop = np.linspace(-90., 90., sep, endpoint=False)
    theta_2_loop = np.linspace(-90., 90., sep, endpoint=False)

    # dof=4:
    # theta_3_loop = np.linspace(-90., 90., sep, endpoint=False)
    idx_list = []

    C_POINTS = True  # whether collect points.npy, used in test model

    """
    collect images and point clouds and indexes
    """
    for i in range(sep ** dof):
        if dof == 2:
            angle = list([theta_0_loop[i // sep],
                          theta_1_loop[i % sep]])

        if dof == 3:
            angle = list([theta_0_loop[i // (sep ** 2)],
                          theta_1_loop[(i // sep) % sep],
                          theta_2_loop[i % sep]])
        # dof=4:
        # angle = list([theta_0_loop[i // (sep ** 3)],
        #               theta_1_loop[(i // sep ** 2) % sep],
        #               theta_2_loop[(i // sep) % sep],
        #               theta_3_loop[i % sep]])
        idx_list.append(angle)

        p_dense, p_empty = test_model(angle=angle, dof=dof, model=model, log_pth=visual_pth, idx=i)

    np.savetxt(visual_pth + "logger.csv", np.asarray(idx_list), fmt='%i')


if __name__ == "__main__":
    # collect_point_cloud(dof=2, model_pth=test_model_pth)

    interact_env(logger_pth=test_model_pth+"visual/")
