import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image
import os
from tqdm import trange
import torch
from torch import nn
from typing import Optional, Tuple, List, Union, Callable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# import torch

def rot_X(th):
    matrix = ([
        [1, 0, 0, 0],
        [0, np.cos(th), -np.sin(th), 0],
        [0, np.sin(th), np.cos(th), 0],
        [0, 0, 0, 1]])
    np.asarray(matrix)

    return matrix


def rot_Y(th):
    matrix = ([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])
    np.asarray(matrix)

    return matrix


def rot_Z(th):
    matrix = ([
        [np.cos(th), -np.sin(th), 0, 0],
        [np.sin(th), np.cos(th), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    np.asarray(matrix)

    return matrix

def pts_trans_matrix(theta,phi,no_inverse=False):
    # the coordinates in pybullet, camera is along X axis, but in the pts coordinates, the camera is along z axis

    w2c = transition_matrix("rot_y", -theta / 180. * np.pi)
    w2c = np.dot(transition_matrix("rot_x", phi / 180. * np.pi), w2c)
    if no_inverse == False:
        w2c = np.linalg.inv(w2c)
    return w2c


def rays_np(H, W, D, c_h=1.106):
    """numpy version my_ray"""
    rate = np.tan(21 * np.pi / 180)
    # co = torch.Tensor([0.8, 0, 0.606])
    #
    #               0.3          0.3        0.5
    #         far -----  object ----- near ----- camera
    #
    near = np.array([
        [[0.3, 0.5 * rate, c_h + 0.5 * rate], [0.3, -0.5 * rate, c_h + 0.5 * rate]],
        [[0.3, 0.5 * rate, c_h - 0.5 * rate], [0.3, -0.5 * rate, c_h - 0.5 * rate]]
    ])

    far = np.array([
        [[-0.3, 1.1 * rate, c_h + 1.1 * rate], [-0.3, -1.1 * rate, c_h + 1.1 * rate]],
        [[-0.3, 1.1 * rate, c_h - 1.1 * rate], [-0.3, -1.1 * rate, c_h - 1.1 * rate]]
    ])
    n_y_list = (np.linspace(near[0, 0, 1], near[0, 1, 1], W + 1) + 0.5 * (near[0, 1, 1] - near[0, 0, 1]) / W)[:-1]
    n_z_list = (np.linspace(near[0, 0, 2], near[1, 0, 2], H + 1) + 0.5 * (near[1, 0, 2] - near[0, 0, 2]) / H)[:-1]
    f_y_list = (np.linspace(far[0, 0, 1], far[0, 1, 1], W + 1) + 0.5 * (far[0, 1, 1] - far[0, 0, 1]) / W)[:-1]
    f_z_list = (np.linspace(far[0, 0, 2], far[1, 0, 2], H + 1) + 0.5 * (far[1, 0, 2] - far[0, 0, 2]) / H)[:-1]

    ny, nz = np.meshgrid(n_y_list, n_z_list)
    near_face = np.stack([0.3 * np.ones_like(ny.T), ny.T, nz.T], -1)

    fy, fz = np.meshgrid(f_y_list, f_z_list)
    far_face = np.stack([-0.3 * np.ones_like(fy.T), fy.T, fz.T], -1)
    D_list = np.linspace(0, 1, D + 1)[:-1] + .5 * (1 / D)
    box = []
    for d in D_list:
        one_face = (near_face - far_face) * d + far_face
        box.append(one_face)

    box = np.array(box)
    box = np.swapaxes(box, 0, 2)
    # box = torch.swapaxes(box, 1, 2)
    return near, far, near_face, far_face, box


def transfer_box(vbox, norm_angles, c_h=1.106, forward_flag=False):
    vb_shape = vbox.shape
    flatten_box = vbox.reshape(vb_shape[0] * vb_shape[1] * vb_shape[2], 3)
    flatten_box[:, 2] -= c_h
    full_matrix = np.dot(rot_Z(norm_angles[0] * 360 / 180 * np.pi), rot_Y(norm_angles[1] * 90 / 180 * np.pi))
    if forward_flag:
        # static arm, moving camera
        flatten_new_view_box = np.dot(
            full_matrix,
            np.hstack((flatten_box, np.ones((flatten_box.shape[0], 1)))).T
        )[:3]
    else:
        # static camera, moving arm
        flatten_new_view_box = np.dot(
            np.linalg.inv(full_matrix),
            np.hstack((flatten_box, np.ones((flatten_box.shape[0], 1)))).T
        )[:3]
    flatten_new_view_box[2] += c_h
    flatten_new_view_box = flatten_new_view_box.T
    new_view_box = flatten_new_view_box.reshape(vb_shape[0], vb_shape[1], vb_shape[2], 3)
    return new_view_box, flatten_new_view_box


def get_rays(
        height: int,
        width: int,
        focal_length: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Find origin and direction of rays through every pixel and camera origin.

    # Apply pinhole camera model to gather directions at each pixel
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(focal_length),
        torch.arange(height, dtype=torch.float32).to(focal_length),
        indexing='ij')
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    directions = torch.stack([(i - width * .5) / focal_length,
                              -(j - height * .5) / focal_length,
                              -torch.ones_like(i)
                              ], dim=-1)
    # directions: tan_i, tan_j, -1

    # Apply camera pose to directions
    rays_d = directions
    # rays_d = directions * (nf_size+cam_dist)/cam_dist
    # rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)
    # rays_d = torch.sum(directions[..., None, :] * torch.eye(3), dim=-1)

    # Origin is same for all directions (the optical center)
    rays_o = torch.from_numpy(np.asarray([0,0,1],dtype=np.float32)).expand(directions.shape)

    # Visualization
    # ax = plt.figure().add_subplot(projection='3d')
    # rays_o = rays_o.detach().cpu().numpy().reshape(-1,3)
    # rays_d = rays_d.detach().cpu().numpy().reshape(-1,3)
    # directions = directions.detach().cpu().numpy().reshape(-1, 3)
    # for plt_i in range(len(directions)):
    #     # ax.plot3D([directions[plt_i, 0], 0],
    #     #           [directions[plt_i, 2], 0],
    #     #           [directions[plt_i, 1], 0],c='b')
    #     ax.plot3D([rays_d[plt_i, 0],rays_o[0,0]],
    #               [rays_d[plt_i, 2],rays_o[0,2]],
    #               [rays_d[plt_i, 1],rays_o[0,1]])
    # ax.set_xlabel('X')
    # ax.set_ylabel('z')
    # ax.set_zlabel('y')
    # ax.scatter(rays_o[0][0],rays_o[0][2],rays_o[0][1])
    # plt.show()
    # quit()
    return rays_o, rays_d

def sample_stratified(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        arm_angle: torch.Tensor,
        near: float,
        far: float,
        n_samples: int,
        perturb: Optional[bool] = True,
        inverse_depth: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
  Sample along ray from regularly-spaced bins.
  """

    # Grab samples for space integration along ray
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
    if not inverse_depth:
        # Sample linearly between `near` and `far`
        x_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity)
        x_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    # Draw uniform samples from bins along ray
    if perturb:
        mids = .5 * (x_vals[1:] + x_vals[:-1])
        upper = torch.concat([mids, x_vals[-1:]], dim=-1)
        lower = torch.concat([x_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=x_vals.device)
        x_vals = lower + (upper - lower) * t_rand
    x_vals = x_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

    # Apply scale from `rays_d` and offset from `rays_o` to samples
    # pts: (width, height, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * x_vals[..., :, None]
    pts = pts.view(-1,3)

    pose_matrix = pts_trans_matrix(arm_angle[0].item(),arm_angle[1].item())
    pose_matrix = torch.from_numpy(pose_matrix)

    pose_matrix = pose_matrix.to(pts)
    # Transpose your transformation matrix for correct matrix multiplication
    transformation_matrix = pose_matrix[:3,:3]
    # transformation_matrix = torch.eye(3).to(pts)

    # Apply the transformation
    output_tensor = torch.matmul(pts,transformation_matrix)
    pts = output_tensor.view(x_vals.shape[0], 64, 3)

    # Visualization
    # pts = pts.detach().cpu().numpy().reshape(-1,3)
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('z')
    # ax.set_zlabel('y')
    # ax.scatter(pts[:,0],
    #            pts[:,2],
    #            pts[:,1],
    #            s = 0.1)
    # plt.show()
    # quit()

    return pts, x_vals


def cumprod_exclusive(
        tensor: torch.Tensor
) -> torch.Tensor:
    r"""
    (Courtesy of https://github.com/krrish94/nerf-pytorch)

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    # The last element in each ray(last column) is moved to the first column.
    cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.

    return cumprod


"""
volume rendering
"""


def raw2outputs(
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Convert the raw NeRF output into RGB and other maps.
    """

    # z_vals: size: 2500x64, cropped image 50x50 pixel 64 depth.
    # dists: size 2500x63, the dists between two corresponding points.
    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
    # add one elements for each ray to compensate the size to 64

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 1].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[:, :, 1] + noise) * dists)
    # The larger the dists or the output(density), the closer alpha is to 1.

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., :2])  # [n_rays, n_samples, 3]
    rgb_each_point = weights * torch.sigmoid(raw[..., 1])
    # rgb_each_point = torch.sigmoid(torch.relu(weights)) * raw[..., 3]

    render_img = torch.sum(rgb_each_point, dim=1)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

    # Estimated depth map is predicted distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Disparity map is inverse depth.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))

    # Sum of weights along each ray. In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    # return render_img, depth_map, acc_map, weights, rgb_each_point
    return render_img, rgb_each_point


def raw2dense(
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Convert the raw NeRF output into RGB and other maps.
    """
    # raw = raw[...,0]
    # multiplied_ray = torch.prod(raw,dim=1)*255
    # return multiplied_ray

    # z_vals: size: 2500x64, cropped image 50x50 pixel 64 depth.
    # dists: size 2500x63, the dists between two corresponding points.
    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
    # add one elements for each ray to compensate the size to 64

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 0].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[:, :, 1] + noise) * dists)
    # The larger the dists or the output(density), the closer alpha is to 1.

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., :2])  # [n_rays, n_samples, 3]
    rgb_each_point = weights * torch.sigmoid(raw[..., 1])

    # rgb = raw[..., :2]  # [n_rays, n_samples, 3]
    # rgb_each_point = weights * raw[..., 0]

    render_img = torch.sum(rgb_each_point, dim=1)
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

    # Estimated depth map is predicted distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Disparity map is inverse depth.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))

    # Sum of weights along each ray. In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    # render_img = torch.sigmoid(render_img)

    return render_img, rgb_each_point


def raw2dense_out1(
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Convert the raw NeRF output into RGB and other maps.
    """
    # raw = raw[...,0]
    # multiplied_ray = torch.prod(raw,dim=1)*255
    # return multiplied_ray

    # z_vals: size: 2500x64, cropped image 50x50 pixel 64 depth.
    # dists: size 2500x63, the dists between two corresponding points.
    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    z_vals = z_vals.to(device)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
    dists = torch.cat([dists, dists.max()* torch.ones_like(dists[..., :1])], dim=-1)

    # add one elements for each ray to compensate the size to 64

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1).to(device)

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = torch.tensor(0).to(device)
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 0].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = torch.tensor(1).to(device) - torch.exp(-nn.functional.relu(raw[:, :, 0] + noise) * dists)
    # The smaller the dists or the output(density), the closer alpha is to 1.

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., 0])  # [n_rays, n_samples, 3]
    rgb_each_point = weights * torch.sigmoid(raw[..., 0])

    # rgb = raw[..., :2]  # [n_rays, n_samples, 3]

    render_img = torch.mean(alpha, dim=1)

    return render_img, alpha

    # render_img = torch.sum(rgb_each_point, dim=1)
    # return render_img, rgb_each_point

def sample_pdf(
        bins: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False
) -> torch.Tensor:
    r"""
  Apply inverse transform sampling to a weighted set of points.
  """

    # Normalize weights to get PDF.
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True)  # [n_rays, weights.shape[-1]]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, weights.shape[-1]]
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])  # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)  # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous()  # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True)  # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)  # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                         index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                          index=inds_g)

    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples  # [n_rays, n_samples]


def sample_hierarchical(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False,
        angle: float = 1.,
        more_dof: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
  Apply hierarchical sampling to the rays.
  """

    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples,
                               perturb=perturb)
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF.
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :,
                                                        None]  # [N_rays, N_samples + n_samples, 3]
    if more_dof:
        # for 3dof arm
        add_angle = torch.ones(pts.shape[0], pts.shape[1], 1).to(device) * angle
        pts = torch.cat((pts, add_angle), 2)
        # print(pts.shape)
    return pts, z_vals_combined, new_z_samples


"""
Full Forward Pass
"""


def get_chunks(
        inputs: torch.Tensor,
        chunksize: int = 2 ** 15
) -> List[torch.Tensor]:
    r"""
  Divide an input into chunks.
  """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def prepare_chunks(
        points: torch.Tensor,
        # encoding_function: Callable[[torch.Tensor], torch.Tensor],
        chunksize: int = 2 ** 14
) -> List[torch.Tensor]:
    r"""
  Encode and chunkify points to prepare for NeRF model.
  """
    points = points.reshape((-1, points.shape[-1]))
    # points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points


def prepare_viewdirs_chunks(
        points: torch.Tensor,
        rays_d: torch.Tensor,
        encoding_function: Callable[[torch.Tensor], torch.Tensor],
        chunksize: int = 2 ** 15
) -> List[torch.Tensor]:
    r"""
  Encode and chunkify viewdirs to prepare for NeRF model.
  """
    # Prepare the viewdirs
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs


def nerf_forward(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        model: nn.Module,
        arm_angle: torch.Tensor,
        DOF: int,
        kwargs_sample_stratified: dict = None,
        n_samples_hierarchical: int = 0,
        kwargs_sample_hierarchical: dict = None,
        chunksize: int = 2 ** 15,

        if_3dof: bool = False
) -> dict:
    r"""
    Compute forward pass through model(s).
    """

    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, arm_angle, near, far, **kwargs_sample_stratified)
    # Prepare batches.

    arm_angle = arm_angle / 180 * np.pi
    model_input = torch.cat((query_points, arm_angle[:DOF].repeat(list(query_points.shape[:2]) + [1])), dim=-1)
    # model_input = query_points

    # arm_angle[:DOF] -> use one angle
    # model_input = query_points  # orig version 3 input 2dof, Mar30
    batches = prepare_chunks(model_input, chunksize=chunksize)
    predictions = []
    for batch in batches:
        # print(batch.dtype)
        batch = batch.to(device)
        predictions.append(model(batch))

    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # print(raw.shape)
    # Perform differentiable volume rendering to re-synthesize the RGB image.
    # rgb_map, rgb_each_point = raw2dense(raw, z_vals, rays_d)

    rgb_map, rgb_each_point = raw2dense_out1(raw, z_vals, rays_d)  # out1 raw to dense

    outputs = {
        'rgb_map': rgb_map,
        'rgb_each_point': rgb_each_point,
        'query_points': query_points}

    # Store outputs.
    return outputs


"""make pose"""


# def w2c_matrix(theta, phi, radius):
#     w2c = transition_matrix("tran_z", radius)
#     w2c = np.dot(transition_matrix("rot_y", -theta / 180. * np.pi), w2c)
#     w2c = np.dot(transition_matrix("rot_x", -phi / 180. * np.pi), w2c)
#     return w2c
#
#
# def c2w_matrix(theta, phi, radius):
#     c2w = transition_matrix("tran_z", radius)
#     c2w = np.dot(transition_matrix("rot_x", phi / 180. * np.pi), c2w)
#     c2w = np.dot(transition_matrix("rot_y", theta / 180. * np.pi), c2w)
#     return c2w


def transition_matrix(label, value):
    if label == "rot_x":
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(value), -np.sin(value), 0],
            [0, np.sin(value), np.cos(value), 0],
            [0, 0, 0, 1]])

    if label == "rot_y":
        return np.array([
            [np.cos(value), 0, -np.sin(value), 0],
            [0, 1, 0, 0],
            [np.sin(value), 0, np.cos(value), 0],
            [0, 0, 0, 1]])

    if label == "tran_z":
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, value],
            [0, 0, 0, 1]])

    else:
        return "wrong label"


def plot_3d_visual(x, y, z, if_transform=True):
    if if_transform:
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        z = z.detach().cpu().numpy()

    ax = plt.axes(projection='3d')
    ax.scatter3D(x,
                 y,
                 z, s=1
                 )
    # ax.scatter3D(0,0,0)


if __name__ == "__main__":

    #              -0.4    0      0.4        0.6
    #         far -----  object ----- near ----- camera
    #

    DOF = 2  # the number of motors  # dof4 apr03
    num_data = 20**DOF
    pxs = 100  # collected data pixels

    HEIGHT = pxs
    WIDTH = pxs
    nf_size = 0.4
    cam_dist = 1
    camera_angle_x = 42 * np.pi / 180.
    focal = .5 * WIDTH / np.tan(.5 * camera_angle_x)
    rays_o, rays_d = get_rays(HEIGHT, WIDTH, focal)

    # Visualization
    # ax = plt.figure().add_subplot(projection='3d')
    # rays_o = rays_o.detach().cpu().numpy().reshape(-1,3)
    # rays_d = rays_d.detach().cpu().numpy().reshape(-1,3)
    # directions = directions.detach().cpu().numpy().reshape(-1, 3)
    # for plt_i in range(len(directions)):
    #     ax.plot3D([rays_d[plt_i, 0],rays_o[0,0]],
    #               [rays_d[plt_i, 2],rays_o[0,2]],
    #               [rays_d[plt_i, 1],rays_o[0,1]])
    # print(rays_o)
    # ax.set_xlabel('X')
    # ax.set_ylabel('z')
    # ax.set_zlabel('y')
    # ax.scatter(rays_o[0][0],rays_o[0][2],rays_o[0][1])
    # plt.show()
    # quit()

    data = np.load('data/data_uniform/dof%d_data%d_px%d.npz' % (DOF, num_data, pxs))

    training_angles = torch.from_numpy(data['angles'].astype('float32'))
    training_pose_matrix = torch.from_numpy(data['poses'].astype('float32'))

    idxx = 250
    angle = training_angles[idxx]
    print(angle/90)
    matrix = pts_trans_matrix(angle[0],angle[1])

    pose_matrix = torch.from_numpy(matrix)
    # print(matrix - pose_matrix)
    near, far = cam_dist - nf_size, cam_dist + nf_size
    kwargs_sample_stratified = {
        'n_samples': 64,
        'perturb': True,
        'inverse_depth': False
    }

    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, angle, near, far, **kwargs_sample_stratified)


    # _, _, _, ff, my_box = rays_np(H=6, W=6, D=6)
    #
    # box_shape = my_box.shape
    # # my_box[:, :, :, 2] -= 1.106
    # print(box_shape)
    # # print(my_box)
    # print(ff.shape)
    # print(my_box[:, :, 0, :].shape)
    # new_box, f_new_box = transfer_box(vbox=my_box, norm_angles=[0.75, 0.5], forward_flag=True)
    # print(new_box.shape)
    # print(f_new_box.shape)
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(
    #     my_box[:, :, :, 0],
    #     my_box[:, :, :, 1],
    #     my_box[:, :, :, 2]
    # )
    # ax.scatter3D(
    #     new_box[:, :, :, 0],
    #     new_box[:, :, :, 1],
    #     new_box[:, :, :, 2]
    # )
    # # ax.scatter3D(
    # #     ff[:, :, 0],
    # #     ff[:, :, 1],
    # #     ff[:, :, 2]
    # # )
    # ax.set_xlim([-0.5, 0.5])
    # ax.set_ylim([-0.5, 0.5])
    # ax.set_zlim([1. - 0.5, 1. + 0.5])
    # plt.show()
