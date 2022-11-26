import numpy as np


# import torch

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


def transfer_box(vbox, norm_angles, c_h=1.106):
    vb_shape = vbox.shape
    flatten_box = vbox.reshape(vb_shape[0] * vb_shape[1] * vb_shape[2], 3)
    flatten_box[:, 2] -= c_h
    full_matrix = np.dot(rot_Z(norm_angles[0] * 50 / 180 * np.pi), rot_Y(norm_angles[1] * 50 / 180 * np.pi))

    flatten_new_view_box = np.dot(
        np.linalg.inv(full_matrix),
        np.hstack((flatten_box, np.ones((flatten_box.shape[0], 1)))).T
    )[:3]
    flatten_new_view_box[2] += c_h
    flatten_new_view_box = flatten_new_view_box.T
    new_view_box = flatten_new_view_box.reshape(vb_shape[0], vb_shape[1], vb_shape[2], 3)
    return new_view_box, flatten_new_view_box


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from rays_check import my_rays

    _, _, _, _, my_box = rays_np(H=6, W=6, D=6)
    box_shape = my_box.shape
    # my_box[:, :, :, 2] -= 1.106
    print(box_shape)
    print(my_box)
    new_box, f_new_box = transfer_box(vbox=my_box, norm_angles=[0.5, 0.5])
    print(f_new_box.shape)
    # new_box = my_box.reshape(box_shape[0]*box_shape[1]*box_shape[2], box_shape[-1])
    # print(new_box.shape)
    # new_box = my_box.reshape(box_shape[0], box_shape[1], box_shape[2], box_shape[-1])
    # print(new_box)

    # view_edge_len = 0.3
    # orig_view_box = np.array([
    #     [0, view_edge_len, view_edge_len],
    #     [0, view_edge_len, -view_edge_len],
    #     [0, -view_edge_len, -view_edge_len],
    #     [0, -view_edge_len, view_edge_len],
    #     [0.8, 0, 0]
    # ])
    #
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        my_box[:, :, :, 0],
        my_box[:, :, :, 1],
        my_box[:, :, :, 2]
    )
    ax.scatter3D(
        new_box[:, :, :, 0],
        new_box[:, :, :, 1],
        new_box[:, :, :, 2]
    )
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([1.-0.5, 1.+0.5])
    plt.show()
