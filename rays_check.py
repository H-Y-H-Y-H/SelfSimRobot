import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def camera_spherical(theta, phi, radius):
    w2c = trans_t(radius)
    w2c = rot_theta(-theta / 180. * np.pi) @ w2c
    w2c = rot_phi(-phi / 180. * np.pi) @ w2c
    w2c = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ w2c
    return w2c


def get_rays(H, W, focal, c2w):
    """nerf-pytorch orig version"""
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - 0.5 * W) / focal, -(j - 0.5 * H) / focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def embed_fn(x):
    L_embed = 8
    rets = [x]
    for i in range(L_embed):
        for fn in [tf.sin, tf.cos]:
            rets.append(fn(2.**i * x))
    return torch.cat(rets, -1)

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, rand=False):
    def batchify(fn, chunk=1024 * 32):
        return lambda inputs: torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    # torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    # raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    # Compute 3D query points
    z_vals = torch.linspace(near, far, N_samples)
    # if rand:
    #     z_vals += torch.random.uniform(list(rays_o.shape[:-1]) + [N_samples]) * (far - near) / N_samples
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Run network
    pts_flat = torch.reshape(pts, [-1, 3])
    pts_flat = embed_fn(pts_flat)
    raw = batchify(network_fn)(pts_flat)
    raw = torch.reshape(raw, list(pts.shape[:-1]) + [4])

    # Compute opacities and colors
    sigma_a = F.relu(raw[..., 3])
    rgb = F.sigmoid(raw[..., :3])

    # Do volume rendering
    # dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1], torch.broadcast_to([1e10], z_vals[..., :1].shape)], -1)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    alpha = 1. - torch.exp(-sigma_a * dists)
    # weights = alpha * tf.math.cumprod(1. - alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(1. - alpha + 1e-10, -1)
    # rgb_map = tf.reduce_sum(weights[..., None] * rgb, -2)
    # depth_map = tf.reduce_sum(weights * z_vals, -1)
    # acc_map = tf.reduce_sum(weights, -1)
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    return rgb_map

def my_rays(H, W, D, c_h=1.106):  # c_h: camera height
    rate = np.tan(21 * np.pi / 180)  # fov = 42
    # co = torch.Tensor([0.8, 0, 0.606])
    #
    #               0.3          0.3        0.5
    #         far -----  object ----- near ----- camera
    #
    near = torch.Tensor([
        [[0.3, 0.5 * rate, c_h + 0.5 * rate], [0.3, -0.5 * rate, c_h + 0.5 * rate]],
        [[0.3, 0.5 * rate, c_h - 0.5 * rate], [0.3, -0.5 * rate, c_h - 0.5 * rate]]
    ])

    far = torch.Tensor([
        [[-0.3, 1.1 * rate, c_h + 1.1 * rate], [-0.3, -1.1 * rate, c_h + 1.1 * rate]],
        [[-0.3, 1.1 * rate, c_h - 1.1 * rate], [-0.3, -1.1 * rate, c_h - 1.1 * rate]]
    ])

    n_y_list = (torch.linspace(near[0, 0, 1], near[0, 1, 1], W + 1) + 0.5 * (near[0, 1, 1] - near[0, 0, 1]) / W)[:-1]
    n_z_list = (torch.linspace(near[0, 0, 2], near[1, 0, 2], H + 1) + 0.5 * (near[1, 0, 2] - near[0, 0, 2]) / H)[:-1]
    f_y_list = (torch.linspace(far[0, 0, 1], far[0, 1, 1], W + 1) + 0.5 * (far[0, 1, 1] - far[0, 0, 1]) / W)[:-1]
    f_z_list = (torch.linspace(far[0, 0, 2], far[1, 0, 2], H + 1) + 0.5 * (far[1, 0, 2] - far[0, 0, 2]) / H)[:-1]

    ny, nz = torch.meshgrid(n_y_list, n_z_list)
    near_face = torch.stack([0.3 * torch.ones_like(ny.t()), ny.t(), nz.t()], -1)

    fy, fz = torch.meshgrid(f_y_list, f_z_list)
    far_face = torch.stack([-0.3 * torch.ones_like(fy.t()), fy.t(), fz.t()], -1)
    D_list = torch.linspace(0, 1, D + 1)[:-1] + .5 * (1 / D)
    box = torch.tensor([])
    for d in D_list:
        one_face = (near_face - far_face) * d + far_face
        one_face = torch.unsqueeze(one_face, dim=0)
        box = torch.cat((one_face, box))

    box = torch.swapaxes(box, 0, 2)
    # box = torch.swapaxes(box, 1, 2)
    return near, far, near_face, far_face, box


if __name__ == "__main__":
    HH, WW = 10, 10
    # camera_angle_x = 42. * np.pi / 180.
    # focal = .5 * W / np.tan(.5 * camera_angle_x)
    # print(H, W, focal)
    #
    # o_list = []
    # d_list = []
    # c2w = pose_spherical(0, -90, 2)
    # print(c2w)
    # o, d = get_rays(H, W, focal, c2w)
    # print(o.shape)
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(0, 0, 0, c='k')
    # ax.scatter3D(
    #     o[:, :, 0],
    #     o[:, :, 1],
    #     o[:, :, 2],
    #     c='r'
    # )
    #
    # ax.scatter3D(
    #     d[:, :, 0],
    #     d[:, :, 1],
    #     d[:, :, 2],
    #     c='g'
    # )
    # ax.set_xlim([-2, 2])
    # ax.set_ylim([-2, 2])
    # ax.set_zlim([-1, 3])

    # plt.show()

    n, f, nf, ff, tf = my_rays(H=HH, W=WW, D=10)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.scatter3D(0, 0, 0.606, c='k')

    ax.scatter3D(
        f[:, :, 0],
        f[:, :, 1],
        f[:, :, 2],
        c='k'
    )

    ax.scatter3D(
        n[:, :, 0],
        n[:, :, 1],
        n[:, :, 2],
        c='k'
    )

    ax.scatter3D(
        nf[:, :, 0],
        nf[:, :, 1],
        nf[:, :, 2],
        c='r'
    )

    ax.scatter3D(
        ff[:, :, 0],
        ff[:, :, 1],
        ff[:, :, 2],
        c='g'
    )

    x_idx = torch.randperm(10)[:5]
    y_idx = torch.randperm(10)[:5]
    print(tf.shape)
    # tf = torch.index_select(tf, 0, x_idx)
    # tf = torch.index_select(tf, 1, y_idx)
    print(tf.shape)

    ax.scatter3D(
        tf[:, 0, :, 0],
        tf[:, 0, :, 1],
        tf[:, 0, :, 2],
        c='orange'
    )
    for i in range(HH):
        for j in range(WW):
            ax.plot3D(
                [nf[i, j, 0], ff[i, j, 0]],
                [nf[i, j, 1], ff[i, j, 1]],
                [nf[i, j, 2], ff[i, j, 2]],
                c='b'
            )
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0.1, 1.1])

    plt.show()
