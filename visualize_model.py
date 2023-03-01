import matplotlib.pyplot as plt
import torch
from train_nerf import nerf_forward, get_rays, init_models, prepare_chunks
from func import w2c_matrix, c2w_matrix
import numpy as np
from torch import nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models()
# model.load_state_dict(torch.load("train_log/log01/nerf.pt", map_location=torch.device('cpu')))
fine_model.load_state_dict(
    torch.load("../train_log/log02_100_2dof/best_model/nerf-fine.pt", map_location=torch.device('cpu')))
fine_model.eval()


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


def dense_visual_3d(count_num=100, draw=True, theta=30, phi=30, idx=1):
    pose_transfer = c2w_matrix(theta, phi, 0.)
    count, count_empty = 0, 0

    points_record = []
    dense_record = []
    points_empty = []
    # np.random.seed(10)
    while True:
        p = np.random.rand(1, 3) * 4. - 2.
        point = encode(torch.from_numpy(p.astype(np.float32)))
        out = fine_model(point)
        dense = 1.0 - torch.exp(-nn.functional.relu(out[..., 3]))
        dense = dense.detach().numpy()

        if out[0][-1] > 0.:
            count += 1
            # points_record.append(out.detach().numpy()[0][:3])
            # dense = dense.reshape(1, 1)
            # p_dense = np.concatenate((p, dense), 1)
            # points_record.append(p_dense)
            points_record.append(p)
            dense_record.append(dense)

        else:
            count_empty += 1
            if count_empty < 1000:
                points_empty.append(p)

        if count == count_num:
            break
    points_record = np.array(points_record)
    points_empty = np.array(points_empty)
    print(points_record.shape)
    pr_tran = np.concatenate((points_record.reshape(-1, 3), np.ones((count_num, 1))), 1)
    pr_tran = np.dot(pose_transfer, pr_tran.T).T[:, :3]
    # print(p_tran.shape)
    if draw:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(
            pr_tran[:, 0],
            pr_tran[:, 2],
            pr_tran[:, 1],
            # alpha=points_record[:, :, 3]
            alpha=dense_record,
            # s=0.1
        )
        ax.scatter(
            points_empty[:, :, 0],
            points_empty[:, :, 2],
            points_empty[:, :, 1],
            alpha=0.1,
            # s=0.1,
        )
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()
        plt.savefig('img5/%04d.png' % i)

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


def dense_visual_box(theta, phi, more_dof=False):
    my_pose = w2c_matrix(theta, phi, 4.)
    my_pose = torch.from_numpy(my_pose.astype('float32')).to(device)
    height, width, focal = 100, 100, 130.25446
    near, far = 2., 6.
    rays_o, rays_d = get_rays(height, width, focal, my_pose)
    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])
    raw_outputs = nerf_forward(rays_o, rays_d,
                               near, far, encode, model,
                               kwargs_sample_stratified=kwargs_sample_stratified,
                               n_samples_hierarchical=n_samples_hierarchical,
                               kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                               fine_model=fine_model,
                               viewdirs_encoding_fn=encode_viewdirs,
                               chunksize=chunksize,
                               if_3dof=more_dof,
                               only_raw=True)

    ax = plt.figure().add_subplot(projection='3d')
    ax.view_init(elev=90, azim=-90)
    for p in raw_outputs:
        # print(p[-1])
        p = p.detach().numpy()
        if p[-1] > 0.:
            ax.scatter(p[0], p[2], p[1])

    plt.show()

    # print(raw_outputs.shape)


if __name__ == "__main__":
    # make_video()
    loop = np.linspace(-180., 180., 120, endpoint=False)
    for i in range(120):
        theta = loop[i % 120]
        # phi = (np.sin((i / 120) * 2 * np.pi) * 0.6 - 1.) * 90.
        # phi = -90
        phi = (np.sin((i / 60) * 2 * np.pi) - 1.5) * 30.
        p_dense, p_empty = dense_visual_3d(draw=True, theta=theta, phi=phi, idx=i)
    # pose_transfer_visualize(points_record=p_dense, points_empty=p_empty, theta=30, phi=30)
    # dense_visual_box(theta=0., phi=0.)
