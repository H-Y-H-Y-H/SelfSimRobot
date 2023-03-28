import os

import matplotlib.pyplot as plt
import torch
from train import nerf_forward, get_fixed_camera_rays, init_models
from func import w2c_matrix, c2w_matrix
import numpy as np
from torch import nn

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def pos2img(new_pose, ArmAngle, more_dof=False):
    height, width, focal = 100, 100, 130.25446
    near, far = 2., 6.
    rays_o, rays_d = get_rays(height, width, focal, new_pose)
    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])
    outputs = nerf_forward(rays_o, rays_d,
                           near, far, encode, model,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           n_samples_hierarchical=n_samples_hierarchical,
                           kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                           fine_model=fine_model,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=chunksize,
                           arm_angle=ArmAngle,
                           if_3dof=more_dof)

    rgb_predicted = outputs['rgb_map']
    np_image = rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy()
    return np_image


def angles2pos(theta, phi, arm_angle):
    my_pose = w2c_matrix(theta, phi, 4.)
    my_pose = torch.from_numpy(my_pose.astype('float32')).to(device)
    arm_angle = torch.tensor(arm_angle).to(device)
    image = pos2img(my_pose, arm_angle)
    return image


def make_video():
    import cv2
    size = (100, 100)
    n_frames = 120
    out = cv2.VideoWriter('arm_video_01.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    loop = np.linspace(-180., 180., 120, endpoint=False)
    for i in range(n_frames):
        theta = loop[i % 120]
        phi = (np.sin((i / 120) * 2 * np.pi) * 0.6 - 1.) * 90.
        armAngle = np.sin((i / 120) * 2 * np.pi) * 0.6
        img = angles2pos(theta, phi, armAngle)
        img = (img * 255).astype(np.uint8)
        out.write(img)
    out.release()


def dense_visual_3d(log_pth, count_num=100, draw=True, theta=30, phi=30, idx=1):
    pose_transfer = c2w_matrix(theta, phi, 0.)
    batch_size = 4096

    points_record = np.asarray([])
    dense_record = np.asarray([])
    points_empty = np.asarray([])
    for j in range(10):
        p = np.random.rand(batch_size, 3) * 4. - 2.
        point = encode(torch.from_numpy(p.astype(np.float32))).to(device)
        out = fine_model(point)
        binary_out = torch.relu(out[:, 1])
        dense = 1.0 - torch.exp(-nn.functional.relu(out[:, 1]))

        binary_idx = torch.nonzero(binary_out)
        dense = dense.cpu().detach().numpy()
        binary_idx = binary_idx.cpu().detach().numpy()

        if torch.sum(binary_out) > 0:
            dense_record = np.append(dense_record, dense[binary_idx])
            points_record = np.append(points_record, p[binary_idx].reshape(-1, 3))

        points_empty = np.append(points_empty,
                                 p[torch.nonzero(torch.logical_not(binary_out)).cpu().detach().numpy()].reshape(-1, 3))
    points_record = np.reshape(np.array(points_record), (-1, 3))
    dense_record = np.asarray(dense_record)
    points_empty = np.asarray(points_empty).reshape(-1, 3)

    pr_tran = np.concatenate((points_record, np.ones((len(points_record), 1))), 1)
    pr_tran = np.dot(pose_transfer, pr_tran.T).T[:, :3]
    if draw:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(
            pr_tran[:, 0],
            pr_tran[:, 2],
            pr_tran[:, 1],
            alpha=dense_record,
            # s=0.1
        )
        ax.scatter(
            points_empty[:, 0],
            points_empty[:, 2],
            points_empty[:, 1],
            alpha=0.01,
        )
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        plt.show()
        # os.makedirs(log_pth+'/visual_test',exist_ok= True)
        # plt.savefig(log_pth +'/visual_test/%04d.png' % i)

    return points_record, points_empty


def test_model(log_pth, angle, idx=1):
    theta, phi, third_angle = angle
    # target_pose = c2w_matrix(theta, phi, 0.)
    # target_pose_tensor = torch.from_numpy(target_pose.astype('float32')).to(device)
    rays_o, rays_d = get_fixed_camera_rays(height, width, focal)
    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])

    count, count_empty = 0, 0

    points_record = np.asarray([])
    points_empty = np.asarray([])
    angle_tensor = torch.from_numpy(np.asarray(angle).astype('float32')).to(device)
    outputs = nerf_forward(rays_o, rays_d,
                           near, far, model, angle_tensor, DOF,
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
    plt.suptitle('M1: %0.2f,  M2: %0.2f,  M3: %0.2f' % (angle[0], angle[1], angle[2]), fontsize=14)
    target_pose = c2w_matrix(theta, phi, 0.)

    # query_xyz = np.concatenate((query_xyz, np.ones((len(query_xyz), 1))), 1)
    # query_xyz = np.dot(target_pose, query_xyz.T).T[:, :3]

    ax.scatter(
        query_xyz[:, 0],
        query_xyz[:, 1],
        query_xyz[:, 2],
        s=1
        # alpha=query_rgb
    )

    ax.scatter(
        empty_xyz[:, 0],
        empty_xyz[:, 1],
        empty_xyz[:, 2],
        alpha=0.1,
    )
    plt.xlabel('x-axis', fontsize=20)
    plt.ylabel('y-axis', fontsize=20)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    plt.show()
    # os.makedirs(log_pth + '/visual_test', exist_ok=True)
    # plt.savefig(log_pth + '/visual_test/%04d.png' % idx)
    # plt.clf()

    return points_record, points_empty

def pose_transfer_visualize(points_record, points_empty, theta=30, phi=30):
    pose_transfer = w2c_matrix(theta, phi, 4.)[:3, :3]
    pr_new = np.dot(points_record, pose_transfer)
    pe_new = np.dot(points_empty, pose_transfer)
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(
        pr_new[:, :, 0],
        pr_new[:, :, 2],
        pr_new[:, :, 1],
        # alpha=points_record[:, :, 3]
        alpha=1.
    )
    ax.scatter(
        pe_new[:, :, 0],
        pe_new[:, :, 2],
        pe_new[:, :, 1],
        alpha=0.1
    )
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    plt.show()
    plt.savefig('foo.png')


# def dense_visual_box(theta, phi, more_dof=False):
#     my_pose = w2c_matrix(theta, phi, 4.)
#     my_pose = torch.from_numpy(my_pose.astype('float32')).to(device)
#     height, width, focal = 100, 100, 130.25446
#     near, far = 2., 6.
#     rays_o, rays_d = get_rays(height, width, focal, my_pose)
#     rays_o = rays_o.reshape([-1, 3])
#     rays_d = rays_d.reshape([-1, 3])
#     raw_outputs = nerf_forward(rays_o, rays_d,
#                                near, far, encode, model,
#                                kwargs_sample_stratified=kwargs_sample_stratified,
#                                n_samples_hierarchical=n_samples_hierarchical,
#                                kwargs_sample_hierarchical=kwargs_sample_hierarchical,
#                                fine_model=fine_model,
#                                viewdirs_encoding_fn=encode_viewdirs,
#                                chunksize=chunksize,
#                                if_3dof=more_dof,
#                                only_raw=True)
#
#     ax = plt.figure().add_subplot(projection='3d')
#     ax.view_init(elev=90, azim=-90)
#     for p in raw_outputs:
#         # print(p[-1])
#         p = p.detach().numpy()
#         if p[-1] > 0.:
#             ax.scatter(p[0], p[2], p[1])
#
#     plt.show()
#
#     # print(raw_outputs.shape)


if __name__ == "__main__":

    # test_model_pth = 'train_log/log_1600data/best_model/'
    test_model_pth = 'train_log/log_100data(2)/best_model/'
    DOF = 2
    num_data = 100
    n_samples_hierarchical = 10
    height = 100
    width = 100
    near = 2.
    far = 6.

    data = np.load('data/uniform_data/dof%d_data%d.npz' % (DOF, num_data))
    focal = torch.from_numpy(data['focal'].astype('float32')).to(device)
    print(focal)
    kwargs_sample_stratified = {
        'n_samples': 20,
        'perturb': True,
        'inverse_depth': False
    }
    kwargs_sample_hierarchical = {
        'perturb': True
    }
    chunksize = 2 ** 14

    model, optimizer = init_models(d_input=DOF + 3,
                                   n_layers=8,
                                   d_filter=128)
    model.load_state_dict(torch.load(test_model_pth + "nerf.pt", map_location=torch.device(device)))

    model.eval()

    theta_0_loop = np.linspace(0., 0, 30, endpoint=False)
    theta_1_loop = np.linspace(0., 90., 30, endpoint=False)
    theta_2_loop = np.linspace(0., 0., 30, endpoint=False)

    for i in range(1):
        angle = list([theta_0_loop[i], theta_1_loop[i], theta_2_loop[i]])
        p_dense, p_empty = test_model(angle=angle, log_pth=test_model_pth, idx=i)

        # pose_transfer_visualize(points_record=p_dense, points_empty=p_empty, theta=30, phi=30)
    # dense_visual_box(theta=0., phi=0.)

    # make_video()
