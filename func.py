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

def pts_trans_matrix_numpy(theta,phi,no_inverse=False):
    # the coordinates in pybullet, camera is along X axis, but in the pts coordinates, the camera is along z axis

    w2c = transition_matrix("rot_z", -theta / 180. * np.pi)
    w2c = np.dot(transition_matrix("rot_y", -phi / 180. * np.pi), w2c)
    if no_inverse == False:
        w2c = np.linalg.inv(w2c)
    return w2c


def pts_trans_matrix(theta, phi, no_inverse=False):
    # the coordinates in pybullet, camera is along X axis,
    # but in the pts coordinates, the camera is along z axis

    w2c = transition_matrix_torch("rot_z", -theta / 180. * torch.pi)
    w2c = transition_matrix_torch("rot_y", -phi / 180. * torch.pi) @ w2c
    if not no_inverse:
        w2c = torch.inverse(w2c)
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

    directions = torch.stack([(i - width * .5) / focal_length,
                              -(j - height * .5) / focal_length,
                              -torch.ones_like(i)
                              ], dim=-1)
    # directions: tan_i, tan_j, -1

    # Apply camera pose to directions
    rays_d = directions
    rays_o = torch.from_numpy(np.asarray([1,0,0],dtype=np.float32)).expand(directions.shape)

    rays_d_clone = rays_d.clone()
    rays_d[..., 0], rays_d[..., 2] = rays_d_clone[..., 2].clone(), rays_d_clone[..., 0].clone()

    # Origin is same for all directions (the optical center)
    rotation_matrix = torch.tensor([[1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1]])
    rotation_matrix = rotation_matrix[None, None].to(rays_d)

    # Rotate the points
    rays_d = torch.matmul(rays_d, rotation_matrix)
    rays_o = rays_o.reshape(-1,3)
    rays_d = rays_d.reshape(-1,3)
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
    # pts = pts.view(-1,3)

    # pose_matrix = pts_trans_matrix(arm_angle[0].item(),arm_angle[1].item())
    pose_matrix = pts_trans_matrix(arm_angle[0], arm_angle[1])

    # pose_matrix = torch.from_numpy(pose_matrix)

    pose_matrix = pose_matrix.to(pts)
    # Transpose your transformation matrix for correct matrix multiplication
    transformation_matrix = pose_matrix[:3,:3]

    # Apply the transformation
    pts = torch.matmul(pts,transformation_matrix)


    # # Visualization
    # pts = pts.detach().cpu().numpy().reshape(-1,3)
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('y')
    # ax.set_zlabel('Z')
    # pts2 = pts[:32]
    # print(pts2)
    # select_idx = np.random.choice(list(np.arange(len(pts))), size=1000)
    #
    # pts = pts[select_idx]
    # ax.scatter(pts[:,0],
    #            pts[:,1],
    #            pts[:,2],
    #            s = 1)
    # ax.scatter(pts2[:,0],
    #            pts2[:,1],
    #            pts2[:,2],
    #            s = 5,
    #            c = 'r')
    # plt.show()
    # quit()

    return pts, x_vals


# def cumprod_exclusive(
#         tensor: torch.Tensor
# ) -> torch.Tensor:
#     r"""
#     (Courtesy of https://github.com/krrish94/nerf-pytorch)
#
#     Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.
#
#     Args:
#     tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
#       is to be computed.
#     Returns:
#     cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
#       tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
#     """
#
#     # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
#     cumprod = torch.cumprod(tensor, -1)
#     # "Roll" the elements along dimension 'dim' by 1 element.
#     # The last element in each ray(last column) is moved to the first column.
#     cumprod = torch.roll(cumprod, 1, -1)
#     # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
#     cumprod[..., 0] = 1.
#
#     return cumprod


"""
volume rendering
"""
def raw2outputs_OneOutput(
        raw: torch.Tensor,
        x_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    dense = 1.0 - torch.exp(-nn.functional.relu(raw[..., 0]))

    render_img = torch.sum(dense, dim=1)

    return render_img, dense

def raw2outputs_OneOutput_with_dist(
        raw: torch.Tensor,
        x_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    dists = x_vals[..., 1:] - x_vals[..., :-1]

    # add one elements for each ray to compensate the size to 64
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1).to(device)
    rays_d = rays_d.to(device)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    alpha_dense = 1.0 - torch.exp(-nn.functional.relu(raw[..., 0]) * dists)

    render_img = torch.sum(alpha_dense, dim=1)

    return render_img, alpha_dense

def raw2outputs(
        raw: torch.Tensor,
        x_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:

    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 1]))
    rgb_each_point = alpha*raw[..., 0]
    render_img = torch.sum(rgb_each_point, dim=1)

    return render_img, rgb_each_point


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


def prepare_chunks(
        points: torch.Tensor,
        chunksize: int = 2 ** 14
) -> List[torch.Tensor]:

    points = points.reshape((-1, points.shape[-1]))
    points = [points[i:i + chunksize] for i in range(0, points.shape[0], chunksize)]
    return points


def model_forward(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        model: nn.Module,
        arm_angle: torch.Tensor,
        DOF: int,
        chunksize: int = 2 ** 15,
        n_samples: int = 64,
        output_flag: int = 0
) -> dict:

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, arm_angle, near, far, n_samples=n_samples)
    # Prepare batches.

    arm_angle = arm_angle / 180 * np.pi
    if DOF > 2:
        model_input = torch.cat((query_points, arm_angle[2:DOF].repeat(list(query_points.shape[:2]) + [1])), dim=-1)

    # arm_angle[:DOF] -> use one angle
    else:
        model_input = query_points  # orig version 3 input 2dof, Mar30
    batches = prepare_chunks(model_input, chunksize=chunksize)
    predictions = []
    for batch in batches:
        batch = batch.to(device)
        predictions.append(model(batch))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    if output_flag ==0:
        rgb_map, rgb_each_point = raw2outputs(raw, z_vals, rays_d)
    elif output_flag ==1:
        rgb_map, rgb_each_point = raw2outputs_OneOutput(raw, z_vals, rays_d)
    elif output_flag ==2:
        rgb_map, rgb_each_point = raw2outputs_OneOutput_with_dist(raw, z_vals, rays_d)

    outputs = {
        'rgb_map': rgb_map,
        'rgb_each_point': rgb_each_point,
        'query_points': query_points}

    # Store outputs.
    return outputs



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

    if label == "rot_z":
        return np.array([
        [np.cos(value), -np.sin(value), 0, 0],
        [np.sin(value), np.cos(value), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

    else:
        return "wrong label"

def transition_matrix_torch(label, value):
    # Initialize an identity matrix
    matrix = torch.eye(4, dtype=torch.float32)

    if label == "rot_x":
        matrix[1, 1] = torch.cos(value)
        matrix[1, 2] = -torch.sin(value)
        matrix[2, 1] = torch.sin(value)
        matrix[2, 2] = torch.cos(value)

    elif label == "rot_y":
        matrix[0, 0] = torch.cos(value)
        matrix[0, 2] = -torch.sin(value)
        matrix[2, 0] = torch.sin(value)
        matrix[2, 2] = torch.cos(value)

    elif label == "rot_z":
        matrix[0, 0] = torch.cos(value)
        matrix[0, 1] = -torch.sin(value)
        matrix[1, 0] = torch.sin(value)
        matrix[1, 1] = torch.cos(value)

    else:
        raise ValueError("Wrong label")

    return matrix

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
    # rays_o = rays_o.detach().cpu().numpy()
    # rays_d = rays_d.detach().cpu().numpy()
    # rays_d = rays_d[:10]
    # idx = np.random.choice(list(np.arange(len(rays_d))),size = 1000)
    # rays_d = rays_d[idx]
    # rays_o = rays_o[idx]
    # for plt_i in range(len(rays_d)):
    #     ax.plot3D([rays_d[plt_i, 0],0],
    #               [rays_d[plt_i, 1],0],
    #               [rays_d[plt_i, 2],0])
    # print(rays_o)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.scatter(rays_o[0][0],rays_o[0][1],rays_o[0][2])
    # plt.show()
    # quit()

    data = np.load('data/data_uniform/dof%d_data%d_px%d.npz' % (DOF, num_data, pxs))

    training_angles = torch.from_numpy(data['angles'].astype('float32'))
    training_pose_matrix = torch.from_numpy(data['poses'].astype('float32'))

    idxx = 265
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

