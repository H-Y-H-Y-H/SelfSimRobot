#  from notebook: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
import random
import torch
from model import FBV_SM, PositionalEncoder
from func import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("train,", device)


def plot_samples(
        z_vals: torch.Tensor,
        z_hierarch: Optional[torch.Tensor] = None,
        ax: Optional[np.ndarray] = None):
    r"""
  Plot stratified and (optional) hierarchical samples.
  """
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, 'b-o')
    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, 'r-o')
    ax.set_ylim([-1, 2])
    ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)
    return ax


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


class EarlyStopping:
    r"""
  Early stopping helper based on fitness criterion.
  """

    def __init__(
            self,
            patience: int = 30,
            margin: float = 1e-4
    ):
        self.best_fitness = 0.0  # In our case PSNR
        self.best_iter = 0
        self.margin = margin
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop

    def __call__(
            self,
            iter: int,
            fitness: float
    ):
        r"""
    Check if criterion for stopping is met.
    """
        if (fitness - self.best_fitness) > self.margin:
            self.best_iter = iter
            self.best_fitness = fitness
        delta = iter - self.best_iter
        stop = delta >= self.patience  # stop training if patience exceeded
        return stop


def init_models(d_input, n_layers, d_filter, skip, pretrained_model_pth=None, lr=5e-5, output_size=2):
    # Models
    model = FBV_SM(d_input=d_input,
                   n_layers=n_layers,
                   d_filter=d_filter,
                   skip=skip,
                   output_size=output_size)

    model.to(device)

    # Pretrained Model
    if pretrained_model_pth != None:
        model.load_state_dict(torch.load(pretrained_model_pth + "nerf.pt", map_location=torch.device(device)))
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
        # Randomly pick an image as the target.
        if Overfitting_test:
            target_img_idx = OVERFITTING_ID
        else:
            target_img_idx = np.random.randint(training_img.shape[0] - 1)

        target_img = training_img[target_img_idx]
        angle = training_angles[target_img_idx]
        pose_matrix = training_pose_matrix[target_img_idx]

        if center_crop and i < center_crop_iters:
            target_img = crop_center(target_img)
        height, width = target_img.shape[:2]
        # print(training_angles[target_img_idx])

        rays_o, rays_d = get_rays(height, width, focal, c2w=pose_matrix)
        # rays_o, rays_d = get_fixed_camera_rays(height, width, focal, distance2camera=4)

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

        # Check for any numerical issues.
        for k, v in outputs.items():
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

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
        if i % display_rate == 0:
            model.eval()
            valid_epoch_loss = []
            valid_psnr = []
            valid_image = []
            height, width = testing_img[0].shape[:2]

            if Overfitting_test:
                target_img = training_img[target_img_idx]
                angle = training_angles[target_img_idx]
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

                loss = torch.nn.functional.mse_loss(rgb_predicted, target_img.reshape(-1))
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
                for v_i in range(valid_amount):
                    angle = testing_angles[v_i]
                    img_label = testing_img[v_i]
                    pose_matrix = testing_pose_matrix[v_i]

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

        # Check PSNR for issues and stop if any are found.
        if i == warmup_iters - 1:
            if psnr_v < warmup_min_fitness:
                print(f'Val PSNR {psnr_v} below warmup_min_fitness {warmup_min_fitness}. Stopping...')
                return False, train_psnrs, psnr_v
        # elif i < warmup_iters:
        #     if warmup_stopper is not None and warmup_stopper(i, psnr):
        #         print(f'Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...')
        #         return False, train_psnrs, psnr_v

        if patience > Patience_threshold:
            break

    return True, train_psnrs, val_psnrs


# def data_loader():


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
    DOF = 2  # the number of motors
    num_data = 1600
    tr = 0.8  # training ratio
    pxs = 100  # collected data pixels
    # data = np.load('data/uniform_data/dof%d_data%d.npz' % (DOF, num_data))
    data = np.load('data/data_May29/dof%d_data%d_px%d.npz' % (DOF, num_data, pxs))
    Overfitting_test = False
    sample_id = random.sample(range(num_data), num_data)
    OVERFITTING_ID = 55
    if Overfitting_test:
        valid_img_visual = data['images'][sample_id[OVERFITTING_ID]]
        valid_img_visual = np.dstack((valid_img_visual, valid_img_visual, valid_img_visual))
    else:
        valid_amount = int(num_data * (1 - tr))
        max_pic_save = 10
        valid_img_visual = []
        for vimg in range(max_pic_save):
            valid_img_visual.append(data['images'][sample_id[int(num_data * tr) + vimg]])
        valid_img_visual = np.hstack(valid_img_visual)
        valid_img_visual = np.dstack((valid_img_visual, valid_img_visual, valid_img_visual))

    # Gather as torch tensors
    focal = torch.from_numpy(data['focal'].astype('float32')).to(device)

    training_img = torch.from_numpy(data['images'][sample_id[:int(num_data * tr)]].astype('float32')).to(device)
    training_angles = torch.from_numpy(data['angles'][sample_id[:int(num_data * tr)]].astype('float32')).to(device)
    training_pose_matrix = torch.from_numpy(data['poses'][sample_id[:int(num_data * tr)]].astype('float32')).to(device)

    testing_img = torch.from_numpy(data['images'][sample_id[int(num_data * tr):]].astype('float32')).to(device)
    testing_angles = torch.from_numpy(data['angles'][sample_id[int(num_data * tr):]].astype('float32')).to(device)
    testing_pose_matrix = torch.from_numpy(data['poses'][sample_id[int(num_data * tr):]].astype('float32')).to(device)

    # Grab rays from sample image
    height, width = training_img.shape[1:3]
    print('IMG (height, width)', (height, width))

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
    n_iters = 100000
    batch_size = 2 ** 14  # Number of rays per gradient step (power of 2)
    one_image_per_step = True  # One image per gradient step (disables batching)
    chunksize = 2 ** 14  # Modify as needed to fit in GPU memory
    center_crop = True  # Crop the center of image (one_image_per_)
    center_crop_iters = 50  # Stop cropping center after this many epochs
    display_rate = 200  # Display test output every X epochs

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
    LOG_PATH = "train_log/log_%ddata_in3_out1_img%d(2)/" % (num_data, pxs)

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
        model, optimizer = init_models(d_input=DOF + 3,  # DOF + 3 -> xyz and angle2
                                       n_layers=8,
                                       d_filter=200,
                                       skip=(4,),
                                       output_size=1)

        # mar30, 2dof, 3input 10 * 128 skip=5 log_1600data_in3_out1_img100
        # mar30, 2dof, 3input 8 * 128 skip=4 log_1600data_in3_out1_img100(2)

        # mar29, 3dof, 6input 8*200, log_1000data_out1_img100  # PSNR 24.06
        # mar29, 3dof, 4input 10*128  # psnr 20

        # 4x64 log_100data; log_100data(1)
        # 8x128 log_100data(2)
        # 6x64
        success, train_psnrs, val_psnrs = train(model, optimizer)
        if success and val_psnrs[-1] >= warmup_min_fitness:
            print('Training successful!')
            break

    print(f'Done!')
    record_file_train.close()
    record_file_val.close()

    # torch.save(model.state_dict(), LOG_PATH + 'nerf.pt')
    # torch.save(fine_model.state_dict(), LOG_PATH + 'nerf-fine.pt')
