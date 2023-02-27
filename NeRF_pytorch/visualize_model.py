import matplotlib.pyplot as plt
import torch
from train_nerf import nerf_forward, get_rays, init_models, prepare_chunks
from Prepare_func import w2c_matrix
import numpy as np

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
    torch.load("train_log/log02_100_2dof/best_model/nerf-fine.pt", map_location=torch.device('cpu')))
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


def dense_visual_3d(count_num=100, draw=True, theta=0, phi=0):
    pose_transfer = w2c_matrix(theta, phi, 4.)
    count = 0
    points_record = []
    while True:
        p = torch.rand(1, 3) * 6. - 3.
        p = encode(p)
        out = fine_model(p)
        if out[0, -1] > 0.:
            count += 1
            points_record.append(out.detach().numpy()[0][:3])

        if count == count_num:
            break
    points_record = np.array(points_record)
    print(points_record.shape)
    p_tran = np.concatenate((points_record, np.ones((count_num, 1))), 1)
    p_tran = np.dot(np.linalg.inv(pose_transfer), p_tran.T).T[:, :3]
    print(p_tran.shape)
    if draw:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(p_tran[:, 0], p_tran[:, 1], p_tran[:, 2])
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_zlim(-6, 6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


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
    for p in raw_outputs:
        # print(p[-1])
        p = p.detach().numpy()
        if p[-1] > 0.:
            ax.scatter(p[0], p[1], p[2])

    plt.show()

    # print(raw_outputs.shape)


if __name__ == "__main__":
    # make_video()
    # dense_visual_3d(draw=True)
    dense_visual_box(theta=0., phi=0.)
