import matplotlib.pyplot as plt
import torch
from train_nerf import nerf_forward, get_rays, init_models
from Prepare_func import w2c_matrix

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
# model.load_state_dict(torch.load("train_log/nerf.pt", map_location=torch.device('cpu')))
fine_model.load_state_dict(torch.load("train_log/nerf-fine.pt", map_location=torch.device('cpu')))
fine_model.eval()


def pos2img(new_pose, ArmAngle):
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
                           arm_angle=ArmAngle)

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
    import numpy as np
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


def dense_visual_3d():
    pass


if __name__ == "__main__":
    make_video()
