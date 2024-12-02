import os
import cv2
import matplotlib.pyplot as plt
import torch
from train import model_forward, init_models
from func import *
import numpy as np
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model( angle_tensor,model,  save_offline_data=False):
    print('angle: ', angle_tensor)
    DOF = len(angle_tensor)
    rays_o, rays_d = get_rays(height, width, focal)
    rays_o = rays_o.reshape([-1, 3]).to(device)
    rays_d = rays_d.reshape([-1, 3]).to(device)

    # angle_tensor = torch.from_numpy(np.asarray(angle).astype('float32')).to(device)
    outputs = model_forward(rays_o, rays_d,
                           near, far, model, angle_tensor, DOF,
                           chunksize=chunksize)

    all_points = outputs["query_points"].detach().cpu().numpy().reshape(-1, 3)
    rgb_each_point = outputs["rgb_each_point"].reshape(-1)
    rgb_predicted = outputs['rgb_map']
    # np_image = rgb_predicted.reshape([height, width, 1]).detach().cpu().numpy()
    # np_image = np.clip(0, 1, np_image)
    # np_image_combine = np.dstack((np_image, np_image, np_image))
    # plt.imshow(np_image_combine)
    # plt.show()

    # binary_out = torch.relu(rgb_predicted)
    rgb_each_point = rgb_each_point.where(rgb_each_point>0.1,torch.tensor(0).to(device))
    nonempty_idx = torch.nonzero(rgb_each_point).cpu().detach().numpy().reshape(-1)

    empty_idx = torch.nonzero(torch.logical_not(rgb_each_point)).cpu().detach().numpy().reshape(-1)
    select_empty_idx = np.random.choice(empty_idx, size=1000)

    query_xyz = all_points[nonempty_idx]
    # query_rgb = rgb_each_point[query_points]
    # query_rgb = np.sum(rgb_each_point[query_points],1)
    # occupied_point_idx = np.nonzero(query_rgb)
    # empty_xyz = all_points[select_empty_idx]

    # ax = plt.figure().add_subplot(projection='3d')

    # title_name = ''
    # for i in range(len(angle)):
    #     title_name+='M%d: %0.2f  '%(i, angle[i])
    # plt.suptitle(title_name, fontsize=14)

    pose_matrix = pts_trans_matrix_numpy(angle_tensor[0].item(),angle_tensor[1].item(),no_inverse=False)
    query_xyz = np.concatenate((query_xyz, np.ones((len(query_xyz), 1))), 1)
    query_xyz = np.dot(pose_matrix, query_xyz.T).T[:, :3]

    # empty_xyz =  np.concatenate((empty_xyz, np.ones((len(empty_xyz), 1))), 1)
    # empty_xyz = np.dot(pose_matrix, empty_xyz.T).T[:, :3]

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.scatter(
    #     occup_points[:, 0],
    #     occup_points[:, 1],
    #     occup_points[:, 2],
    #     # s=1
    #     # alpha=)
    # ax.scatter(
    #     empty_xyz[:, 0],
    #     empty_xyz[:, 1],
    #     empty_xyz[:, 2],
    #     alpha=0.1)
    # # plt.show()
    # os.makedirs(log_pth + '/visual_test', exist_ok=True)
    # plt.savefig(log_pth + '/visual_test/%04d.jpg' % idx)
    # plt.clf()
    # if save_offline_data:
    #     os.makedirs(log_pth + '/pc_record', exist_ok=True)
    #     np.save(log_pth + '/pc_record/%04d.npy' % idx, occup_points)
    return query_xyz

def query_models_separated_outputs( angle, model, DOF, n_samples = 64):

    rays_o, rays_d = get_rays(height, width, focal)
    rays_o = rays_o.reshape([-1, 3]).to(device)
    rays_d = rays_d.reshape([-1, 3]).to(device)

    angle_tensor = angle.to(device)
    rgb_map,query_points, density, visibility = model_forward(rays_o, rays_d,
                           near, far, model, angle_tensor, DOF,
                           chunksize=chunksize,
                            n_samples = n_samples,
                            output_flag=3)

    all_points = query_points.reshape(-1, 3)

    rgb_each_point_density = density.reshape(-1)
    rgb_each_point_visibility = visibility.reshape(-1)

    img_predicted = rgb_map

    pose_matrix_tensor = pts_trans_matrix(angle_tensor[0],angle_tensor[1],no_inverse=False).to(device)
    # pose_matrix_tensor = torch.tensor(pose_matrix_array,dtype=torch.float32).to(device)
    all_points_xyz = torch.cat((all_points, torch.ones((len(all_points),1)).to(device)),dim =1)
    all_points_xyz = torch.tensor(all_points_xyz,dtype=torch.float32)
    all_points_xyz = torch.matmul(pose_matrix_tensor,all_points_xyz.T).T[:,:3]


    mask1 = torch.where(rgb_each_point_density > 0.4, rgb_each_point_density, torch.zeros_like(rgb_each_point_density))
    occ_points_xyz_density = all_points_xyz[mask1.bool()]

    mask2 = torch.where(rgb_each_point_visibility > 0.45, rgb_each_point_visibility, torch.zeros_like(rgb_each_point_visibility))
    occ_points_xyz_visibility = all_points_xyz[mask2.bool() & mask1.bool()]


    return occ_points_xyz_density,occ_points_xyz_visibility

def query_models( angle, model, DOF, mean_ee = False, n_samples = 64):

    rays_o, rays_d = get_rays(height, width, focal)
    rays_o = rays_o.reshape([-1, 3]).to(device)
    rays_d = rays_d.reshape([-1, 3]).to(device)

    angle_tensor = angle.to(device)
    outputs = model_forward(rays_o, rays_d,
                           near, far, model, angle_tensor, DOF,
                           chunksize=chunksize,
                            n_samples = n_samples)

    all_points = outputs["query_points"].reshape(-1, 3)
    rgb_each_point = outputs["rgb_each_point"].reshape(-1)
    img_predicted = outputs['rgb_map']

    pose_matrix_tensor = pts_trans_matrix(angle_tensor[0],angle_tensor[1],no_inverse=False).to(device)
    # pose_matrix_tensor = torch.tensor(pose_matrix_array,dtype=torch.float32).to(device)
    all_points_xyz =torch.cat((all_points, torch.ones((len(all_points),1)).to(device)),dim =1)
    all_points_xyz = torch.tensor(all_points_xyz,dtype=torch.float32)
    all_points_xyz = torch.matmul(pose_matrix_tensor,all_points_xyz.T).T[:,:3]

    # mask = (rgb_each_point > 0.1).float()
    mask = torch.where(rgb_each_point > 0.08, rgb_each_point, torch.zeros_like(rgb_each_point))

    unmasked_occ_points_xyz = all_points_xyz[mask.bool()]

    # Use the mask to compute the weighted sum and mean
    occ_points_xyz = all_points_xyz * mask.unsqueeze(-1)  # Assuming all_points_xyz has shape (N, 3)
    occ_point_center_xyz = occ_points_xyz.sum(dim=0) / mask.sum()

    # rgb_each_point = torch.where(rgb_each_point>0.1,True, False)
    #
    # occ_points_xyz = all_points[rgb_each_point]

    # occ_points_xyz = np.concatenate((occ_points_xyz, np.ones((len(occ_points_xyz), 1))), 1)
    # occ_points_xyz = np.dot(pose_matrix, occ_points_xyz.T).T[:, :3]
    if mean_ee:
        return occ_point_center_xyz
    else:
        return unmasked_occ_points_xyz




def interaction(data_pth, angle_list):
    def call_back_func(x):
        pass

    cv2.namedWindow('interaction')
    cv2.createTrackbar('theta0:', 'interaction', 0, 180, call_back_func)
    cv2.createTrackbar('theta1:', 'interaction', 0, 180, call_back_func)
    cv2.createTrackbar('theta2:', 'interaction', 0, 180, call_back_func)

    while 1:
        theta0 = cv2.getTrackbarPos('theta0:', 'interaction') - 90
        theta1 = cv2.getTrackbarPos('theta1:', 'interaction') - 90
        theta2 = cv2.getTrackbarPos('theta2:', 'interaction') - 90

        target = np.asarray([[theta0, theta1, theta2]] * len(angle_list))

        diff = np.sum(abs(target - angle_list), axis=1)
        idx = np.argmin(diff)

        # idx
        img = cv2.imread(data_pth + '%04d.jpg' % idx)

        cv2.imshow('interaction', img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # destroys all window
    cv2.destroyAllWindows()


n_samples_hierarchical = 64
cam_dist = 1
nf_size = 0.4
near, far = cam_dist - nf_size, cam_dist + nf_size  # real scale dist=1.0
pxs = 100
height = pxs
width = pxs
focal = 130.2545
chunksize = 2 ** 20
kwargs_sample_stratified = {
    'n_samples': 64,
    'perturb': True,
    'inverse_depth': False
}
kwargs_sample_hierarchical = {
    'perturb': True
}

if __name__ == "__main__":

    test_name = 'log_8000data_in6_out1_img100(1)'

    DOF = 3
    test_model_pth = 'train_log/%s/best_model/'%test_name

    visual_pth = 'train_log/%s/'%test_name + 'visual_test01/'
    os.makedirs(visual_pth, exist_ok=True)

    num_data = 20**DOF

    data = np.load('data/data_uniform/dof%d_data%d_px%d.npz' % (DOF, num_data, pxs))

    focal = torch.from_numpy(data['focal'].astype('float32')).to(device)
    print(focal)



    model, optimizer = init_models(d_input=(DOF-2) + 3,  # DOF + 3 -> xyz and angle2 or 3 -> xyz
                                   n_layers=4,
                                   d_filter=128,
                                   skip=(1,2 ),
                                   output_size=2,
                                   )

    # mar29, 8*200, log_1000data_out1_img100
    # 3dof 8*200, 4dof 8*160  apr 14, 64000 160?
    model.load_state_dict(torch.load(test_model_pth + "nerf.pt", map_location=torch.device(device)))

    model.eval()

    sep = 20
    theta_0_loop = np.linspace(-90., 90, sep, endpoint=False)
    theta_1_loop = np.linspace(-90., 90., sep, endpoint=False)
    theta_2_loop = np.linspace(-90., 90., sep, endpoint=False)
    theta_3_loop = np.linspace(-90., 90., sep, endpoint=False)

    # dof=4:
    # theta_3_loop = np.linspace(-90., 90., sep, endpoint=False)
    idx_list = []

    """
    collect images and point clouds and indexes
    """
    for i in range(num_data):  # here 3 or 4

        if DOF == 3:
            angle = list([theta_0_loop[i // (sep ** 2)], theta_1_loop[(i // sep) % sep], theta_2_loop[i % sep]])
        elif DOF == 4:
            angle = list([theta_0_loop[i // (sep ** 3)],
                          theta_1_loop[(i // sep ** 2) % sep],
                          theta_2_loop[(i // sep) % sep],
                          theta_3_loop[i % sep]])
        else:
            # DOF == 2
            angle = list([theta_0_loop[i//sep],
                          theta_1_loop[i%sep]])

        idx_list.append(angle)

        p_dense = test_model(angle,model, save_offline_data = False)

    np.savetxt(visual_pth + "logger.csv", np.asarray(idx_list), fmt='%i')
