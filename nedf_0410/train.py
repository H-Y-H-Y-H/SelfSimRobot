#  from notebook: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
import random
import matplotlib.image
import os

import numpy as np
from tqdm import trange, tqdm
from model import FBV_SM
from func import *


def crop_center(
        img: torch.Tensor,
        frac: float = 0.5
) -> torch.Tensor:
    r"""
  Crop center square from image.
  """
    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))
    return img[h_offset:-h_offset, w_offset:-w_offset]


def init_models(d_input, n_layers, d_filter, skip, pretrained_model_pth=None, lr=5e-5, output_size=2):
    # Models
    model = FBV_SM(d_input=d_input,
                   n_layers=n_layers,
                   d_filter=d_filter,
                   skip=skip,
                   output_size=output_size)

    model.to(device)

    # Pretrained Model
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def train(model, optimizer):
    r"""
    Launch training session for NeRF.
    """
    train_psnrs = []
    val_psnrs = []
    iternums = []
    best_psnr = 0.
    psnr_v_last = 0
    patience = 0
    for i in trange(n_iters):
        model.train()
        for t in range(display_rate):
            # Randomly pick an image as the target.
            if Overfitting_test:
                target_img_idx = OVERFITTING_ID
            else:
                target_img_idx = np.random.randint(train_length)

            target_img = train_data[target_img_idx]["image"].to(device)
            angle = train_data[target_img_idx]["angle"].to(device)
            pose_matrix = train_data[target_img_idx]["matrix"].to(device)

            if center_crop and i < center_crop_iters:
                target_img = crop_center(target_img)
            height, width = target_img.shape[:2]

            rays_o, rays_d = get_rays(height, width, focal, c2w=pose_matrix)
            rays_o = rays_o.reshape([-1, 3])
            rays_d = rays_d.reshape([-1, 3])
            target_img = target_img.reshape([-1])

            # Run one iteration of TinyNeRF and get the rendered RGB image.
            outputs = nerf_forward(rays_o, rays_d,
                                   near, far, model,
                                   kwargs_sample_stratified=kwargs_sample_stratified,
                                   n_samples_hierarchical=n_samples_hierarchical,
                                   kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                                   chunksize=chunksize,
                                   arm_angle=angle,
                                   DOF=DOF)

            # Backprop!
            rgb_predicted = outputs['rgb_map']
            loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Compute mean-squared error between predicted and target images.
            psnr = -10. * torch.log10(loss)
            train_psnrs.append(psnr.item())

        # Evaluate testimg at given display rate.
        model.eval()
        with torch.no_grad():
            valid_epoch_loss = []
            valid_psnr = []
            valid_image = []
            height, width = pxs, pxs

            if Overfitting_test:
                target_img = valid_data[target_img_idx]["image"].to(device)
                angle = valid_data[target_img_idx]["angle"].to(device)
                pose_matrix = valid_data[target_img_idx]["matrix"].to(device)
                rays_o, rays_d = get_rays(height, width, focal, c2w=pose_matrix)

                rays_o = rays_o.reshape([-1, 3])
                rays_d = rays_d.reshape([-1, 3])
                target_img = target_img.reshape([-1])

                outputs = nerf_forward(rays_o, rays_d,
                                       near, far, model,
                                       kwargs_sample_stratified=kwargs_sample_stratified,
                                       n_samples_hierarchical=n_samples_hierarchical,
                                       kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                                       chunksize=chunksize,
                                       arm_angle=angle,
                                       DOF=DOF)

                # Backprop!
                rgb_predicted = outputs['rgb_map']
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
                val_psnr = (-10. * torch.log10(loss)).item()
                valid_epoch_loss = loss.item()
                np_image = rgb_predicted.reshape([height, width, 1]).detach().cpu().numpy()
                np_image = np.clip(0, 1, np_image)
                np_image_combine = np.dstack((np_image, np_image, np_image))
                matplotlib.image.imsave(LOG_PATH + 'image/' + 'overfitting%d.png' % target_img_idx, np_image_combine)

                psnr_v = val_psnr
                val_psnrs.append(psnr_v)
                print("Loss:", valid_epoch_loss, "PSNR: ", psnr_v)

            else:
                for v_i in range(valid_length):
                    angle = valid_data[v_i]["angle"].to(device)
                    img_label = valid_data[v_i]["image"].to(device)
                    pose_matrix = valid_data[v_i]["matrix"].to(device)

                    if center_crop and i < center_crop_iters:
                        img_label = crop_center(img_label)

                    height, width = img_label.shape[:2]
                    rays_o, rays_d = get_rays(height, width, focal, c2w=pose_matrix)
                    # rays_o, rays_d = get_fixed_camera_rays(height, width, focal, distance2camera=4)

                    rays_o = rays_o.reshape([-1, 3])
                    rays_d = rays_d.reshape([-1, 3])
                    outputs = nerf_forward(rays_o, rays_d,
                                           near, far, model,
                                           kwargs_sample_stratified=kwargs_sample_stratified,
                                           n_samples_hierarchical=n_samples_hierarchical,
                                           kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                                           chunksize=chunksize,
                                           arm_angle=angle,
                                           DOF=DOF)

                    rgb_predicted = outputs['rgb_map']
                    # img_label = torch.dstack((testing_img[v_i],testing_img[v_i],testing_img[v_i]))

                    loss = torch.nn.functional.mse_loss(rgb_predicted, img_label.reshape(-1))
                    val_psnr = (-10. * torch.log10(loss)).item()
                    valid_epoch_loss.append(loss.item())
                    valid_psnr.append(val_psnr)
                    np_image = rgb_predicted.reshape([height, width, 1]).detach().cpu().numpy()
                    np_image = np.clip(0, 1, np_image)
                    if v_i < max_pic_save:
                        valid_image.append(np_image)
                psnr_v = np.mean(valid_psnr)
                val_psnrs.append(psnr_v)
                print("Loss:", np.mean(valid_epoch_loss), "PSNR: ", psnr_v)

                # save test image
                np_image_combine = np.hstack(valid_image)
                np_image_combine = np.dstack((np_image_combine, np_image_combine, np_image_combine))

                matplotlib.image.imsave(LOG_PATH + 'image/' + 'latest.png', np_image_combine)
                if Flag_save_image_during_training:
                    matplotlib.image.imsave(LOG_PATH + 'image/' + '%d.png' % i, np_image_combine)

                record_file_train.write(str(psnr) + "\n")
                record_file_val.write(str(psnr_v) + "\n")

                if psnr_v > best_psnr:
                    """record the best image and model"""
                    best_psnr = psnr_v
                    matplotlib.image.imsave(LOG_PATH + 'image/' + 'best.png', np_image_combine)
                    torch.save(model.state_dict(), LOG_PATH + 'best_model/nerf.pt')
                    patience = 0
                else:
                    patience += 1
                # os.makedirs(LOG_PATH + "epoch_%d_model" % i, exist_ok=True)
                # torch.save(model.state_dict(), LOG_PATH + 'epoch_%d_model/nerf.pt' % i)

                if psnr_v < 16.2 and i >= 2000:  # TBD
                    print("restart")
                    return False, train_psnrs, psnr_v

            if patience > Patience_threshold:
                break

            # torch.cuda.empty_cache()    # to save memory
    return True, train_psnrs, val_psnrs


class NerfDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, list_ids, angles):
        """Initialization"""
        self.list_ids = list_ids
        self.angles = angles

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_ids)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_ids[index]
        image = np.loadtxt("data/images/%04d.txt" % ID, dtype=np.float32)  # check 04d
        matrix = np.loadtxt("data/w2c/%04d.txt" % ID, dtype=np.float32)
        angle = self.angles[index]

        sample = {
            "image": torch.from_numpy(image.astype('float32')),
            "matrix": torch.from_numpy(matrix.astype('float32')),
            "angle": torch.from_numpy(angle.astype('float32'))
        }
        return sample


if __name__ == "__main__":

    seed_num = 5
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.manual_seed(seed_num)

    """
    prepare data and parameters
    """
    cam_dist = 4.
    nf_size = 2.
    near, far = cam_dist - nf_size, cam_dist + nf_size  # real scale dist=1.0
    Flag_save_image_during_training = True
    DOF = 3  # the number of motors  # dof4 apr03
    num_data = 8000
    tr = 0.99  # training ratio
    train_length = int(num_data * tr)
    valid_length = num_data - train_length
    pxs = 200  # collected data pixels
    # data = np.load('data/data_uniform/dof%d_data%d_px%d.npz' % (DOF, num_data, pxs))

    Overfitting_test = False
    sample_id = random.sample(range(num_data), num_data)
    angle_data = np.loadtxt("./data/angle.txt", dtype=np.float32)
    train_data = NerfDataset(list_ids=sample_id[:train_length], angles=angle_data)
    valid_data = NerfDataset(list_ids=sample_id[train_length:], angles=angle_data)

    OVERFITTING_ID = 55
    if Overfitting_test:
        valid_img_visual = valid_data[OVERFITTING_ID]["image"]
        valid_img_visual = np.dstack((valid_img_visual, valid_img_visual, valid_img_visual))
    else:
        # valid_amount = int(num_data * (1 - tr))
        max_pic_save = 10
        valid_img_visual = []
        for vimg in range(max_pic_save):
            valid_img_visual.append(crop_center(valid_data[vimg]["image"]))
        valid_img_visual = np.hstack(valid_img_visual)
        valid_img_visual = np.dstack((valid_img_visual, valid_img_visual, valid_img_visual))

    Camera_FOV = 42.
    camera_angle_x = Camera_FOV * np.pi / 180.
    focal = np.asarray(.5 * pxs / np.tan(.5 * camera_angle_x))
    focal = torch.from_numpy(focal.astype('float32')).to(device)

    # Encoders
    """arm dof = 2+3; arm dof=3+3"""

    # Stratified sampling
    n_samples = 64  # Number of spatial samples per ray
    perturb = True  # If set, applies noise to sample positions
    inverse_depth = False  # If set, samples points linearly in inverse depth

    # Hierarchical sampling
    n_samples_hierarchical = 64  # Number of samples per ray
    perturb_hierarchical = False  # If set, applies noise to sample positions

    # Training
    n_iters = 1000
    batch_size = 2 ** 14  # Number of rays per gradient step (power of 2)
    one_image_per_step = True  # One image per gradient step (disables batching)
    chunksize = 2 ** 14  # Modify as needed to fit in GPU memory
    center_crop = True  # Crop the center of image (one_image_per_)   # debug
    center_crop_iters = 1000  # Stop cropping center after this many epochs
    display_rate = 100  # Display test output every X epochs

    # Early Stopping
    warmup_iters = 400  # Number of iterations during warmup phase
    warmup_min_fitness = 10.0  # Min val PSNR to continue training at warmup_iters
    n_restarts = 10  # Number of times to restart if training stalls

    # We bundle the kwargs for various functions to pass all at once.
    kwargs_sample_stratified = {
        'n_samples': n_samples,
        'perturb': perturb,
        'inverse_depth': inverse_depth
    }
    kwargs_sample_hierarchical = {
        'perturb': perturb
    }

    # Run training session(s)
    LOG_PATH = "./train_log/log_%ddata_in6_out1_img%d_crop(2)/" % (num_data, pxs)

    os.makedirs(LOG_PATH + "image/", exist_ok=True)
    os.makedirs(LOG_PATH + "best_model/", exist_ok=True)

    record_file_train = open(LOG_PATH + "log_train.txt", "w")
    record_file_val = open(LOG_PATH + "log_val.txt", "w")
    Patience_threshold = 30  # 20 mar 30

    # Save testing gt image for visualization
    matplotlib.image.imsave(LOG_PATH + 'image/' + 'gt.png', valid_img_visual)

    # pretrained_model_pth = 'train_log/log_1000data/best_model/'

    # DOF = DOF-1
    for _ in range(n_restarts):
        model, optimizer = init_models(d_input=DOF + 3,  # DOF + 3 -> xyz and angle2 or 3 -> xyz
                                       n_layers=6,
                                       d_filter=256,
                                       skip=(3,),
                                       output_size=1)
        print("training started")
        success, train_psnrs, val_psnrs = train(model, optimizer)
        if success and val_psnrs[-1] >= warmup_min_fitness:
            print('Training successful!')
            break

    print(f'Done!')
    record_file_train.close()
    record_file_val.close()

    # torch.save(model.state_dict(), LOG_PATH + 'nerf.pt')
    # torch.save(fine_model.state_dict(), LOG_PATH + 'nerf-fine.pt')
