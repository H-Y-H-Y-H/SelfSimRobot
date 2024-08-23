#  from notebook: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
import random
from model import FBV_SM, FBV_SM2, PositionalEncoder
from func import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("train,", device)


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


def init_models(d_input, d_filter, pretrained_model_pth=None, lr=5e-4, output_size=2,FLAG_PositionalEncoder = False):

    if FLAG_PositionalEncoder:
        encoder = PositionalEncoder(d_input, n_freqs=10, log_space=True)

        model = FBV_SM(encoder = encoder,
                       d_input=d_input,
                       d_filter=d_filter,
                       output_size=output_size)
    else:
        # Models
        model = FBV_SM(d_input=d_input,
                       d_filter=d_filter,
                       output_size=output_size)
    model.to(device)
    # Pretrained Model
    if pretrained_model_pth != None:
        model.load_state_dict(torch.load(pretrained_model_pth + "best_model.pt", map_location=torch.device(device)))
    # Optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer



def train(model, optimizer):

    loss_v_last = np.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)
    patience = 0
    min_loss = np.inf
    height, width = training_img[0].shape[:2]
    rays_o, rays_d = get_rays(height, width, focal)

    for i in trange(n_iters):
        model.train()

        target_img_idx = np.random.randint(training_img.shape[0] - 1)

        # Pick an image as the target.
        target_img = training_img[target_img_idx]
        angle = training_angles[target_img_idx]

        if center_crop and i < center_crop_iters:
            target_img = crop_center(target_img)
            rays_o_train, rays_d_train = get_rays(int(height*0.5), int(width*0.5), focal)
        else:
            rays_o_train,rays_d_train = rays_o, rays_d

        target_img = target_img.reshape([-1])

        # Run one iteration of TinyNeRF and get the rendered RGB image.
        outputs = model_forward(rays_o_train, rays_d_train,
                               near, far, model,
                               chunksize=chunksize,
                               arm_angle=angle,
                               DOF=DOF,
                               output_flag= different_arch)

        # Backprop!
        rgb_predicted = outputs['rgb_map']
        optimizer.zero_grad()
        target_img = target_img.to(device)
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        loss_train = loss.item()

        # Evaluate testimg at given display rate.
        if i % display_rate == 0:
            model.eval()
            torch.no_grad()
            valid_epoch_loss = []
            valid_psnr = []
            valid_image = []

            for v_i in range(valid_amount):
                angle = testing_angles[v_i]
                img_label = testing_img[v_i]

                # Run one iteration of TinyNeRF and get the rendered RGB image.
                outputs = model_forward(rays_o, rays_d,
                                       near, far, model,
                                       chunksize=chunksize,
                                       arm_angle=angle,
                                       DOF=DOF,
                                     output_flag= different_arch)

                rgb_predicted = outputs['rgb_map']

                img_label_tensor = img_label.reshape(-1).to(device)

                v_loss = torch.nn.functional.mse_loss(rgb_predicted, img_label_tensor)

                valid_epoch_loss.append(v_loss.item())

                np_image = rgb_predicted.reshape([height, width, 1]).detach().cpu().numpy()
                if v_i < max_pic_save:
                    valid_image.append(np_image)
            loss_valid = np.mean(valid_epoch_loss)

            print("Loss:", loss_valid, 'patience', patience)
            scheduler.step(loss_valid)

            # save test image
            np_image_combine = np.hstack(valid_image)
            np_image_combine = np.dstack((np_image_combine, np_image_combine, np_image_combine))
            np_image_combine = np.clip(np_image_combine,0,1)
            matplotlib.image.imsave(LOG_PATH + '/image/' + 'latest.png', np_image_combine)
            if Flag_save_image_during_training:
                matplotlib.image.imsave(LOG_PATH + '/image/' + '%d.png' % i, np_image_combine)

            record_file_train.write(str(loss_train) + "\n")
            record_file_val.write(str(loss_valid) + "\n")
            torch.save(model.state_dict(), LOG_PATH + '/best_model/model_epoch%d.pt'%i)

            if min_loss > loss_valid:
                """record the best image and model"""
                min_loss = loss_valid
                matplotlib.image.imsave(LOG_PATH + '/image/' + 'best.png', np_image_combine)
                torch.save(model.state_dict(), LOG_PATH + '/best_model/best_model.pt')
                patience = 0
            elif loss_valid == loss_v_last:
                print("restart")
                return False
            else:
                patience += 1
            loss_v_last = loss_valid
            # os.makedirs(LOG_PATH + "epoch_%d_model" % i, exist_ok=True)
            # torch.save(model.state_dict(), LOG_PATH + 'epoch_%d_model/nerf.pt' % i)

        if patience > Patience_threshold:
            break

        # torch.cuda.empty_cache()    # to save memory
    return True


if __name__ == "__main__":

    sim_real = 'sim'
    arm_ee = 'arm'
    seed_num = 1
    robotid = 0
    FLAG_PositionalEncoder= True

    # 0:OM, 1:OneOut, 2: OneOut with distance
    different_arch = 0
    print('different_arch',different_arch)

    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    select_data_amount = 8000

    DOF = 4  # the number of motors  # dof4 apr03

    cam_dist = 1.0
    nf_size = 0.4
    near, far = cam_dist - nf_size, cam_dist + nf_size  # real scale dist=1.0
    Flag_save_image_during_training = True

    if FLAG_PositionalEncoder:
        add_name = 'PE'
    else:
        add_name = 'no_PE'

    tr = 0.8  # training ratio
    pxs = 100  # collected data pixels

    # data = np.load('data/%s_data/%s_data_robo%d(%s).npz'%(sim_real,sim_real,robotid,arm_ee))
    # data = np.load('data/%s_data/%s_data_robo%d(%s)_cam%d.npz'%(sim_real,sim_real,robotid,arm_ee,cam_dist*1000))
    data = np.load('data/%s_data/%s_data_robo%d(%s)_cam%d_test.npz'%(sim_real,sim_real,robotid,arm_ee,800)) # 800 test is 1000 ... local data, Jiong
    num_raw_data = len(data["angles"])

    print("DOF, num_data, robot_id, PE",DOF,select_data_amount,robotid,FLAG_PositionalEncoder)
    LOG_PATH = "train_log/%s_id%d_%d(%d)_%s(%s)_cam%d(%d)" % (sim_real,robotid,select_data_amount, seed_num,add_name,arm_ee,cam_dist*1000,seed_num)
    if different_arch != 0:
        LOG_PATH += 'diff_out_%d'%different_arch
    print("Data Loaded!")
    os.makedirs(LOG_PATH + "/image/", exist_ok=True)
    os.makedirs(LOG_PATH + "/best_model/", exist_ok=True)


    sample_id = random.sample(range(num_raw_data), select_data_amount)

    max_pic_save = 6
    start_idx = int(select_data_amount * tr)
    end_idx = start_idx + max_pic_save

    # Select the required images and stack them horizontally
    valid_img_visual = np.hstack(data['images'][sample_id[start_idx:end_idx]])
    valid_angle = data['angles'][sample_id[start_idx:end_idx]]
    np.savetxt(LOG_PATH+'/image/valid_angle.csv',valid_angle)

    # Repeat the stacked image three times along the depth
    valid_img_visual = np.dstack((valid_img_visual, valid_img_visual, valid_img_visual))

    print("Valid Data Loaded!")

    # Gather as torch tensors
    focal = torch.from_numpy(data['focal'].astype('float32'))

    training_img = torch.from_numpy(data['images'][sample_id[:int(select_data_amount * tr)]].astype('float32'))
    training_angles = torch.from_numpy(data['angles'][sample_id[:int(select_data_amount * tr)]].astype('float32'))

    testing_img = torch.from_numpy(data['images'][sample_id[int(select_data_amount * tr):]].astype('float32'))
    testing_angles = torch.from_numpy(data['angles'][sample_id[int(select_data_amount * tr):]].astype('float32'))
    train_amount = len(training_angles)
    valid_amount = len(testing_angles)
    print(valid_amount)

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
    n_iters = 400000
    one_image_per_step = True  # One image per gradient step (disables batching)
    chunksize = 2 ** 20  # Modify as needed to fit in GPU memory
    center_crop = True  # Crop the center of image (one_image_per_)   # debug
    center_crop_iters = 200  # Stop cropping center after this many epochs
    display_rate = 1000 #int(select_data_amount*tr)  # Display test output every X epochs

    # Early Stopping
    warmup_iters = 400  # Number of iterations during warmup phase
    warmup_min_fitness = 10.0  # Min val PSNR to continue training at warmup_iters
    n_restarts = 1000  # Number of times to restart if training stalls

    # We bundle the kwargs for various functions to pass all at once.
    kwargs_sample_stratified = {
        'n_samples': n_samples,
        'perturb': perturb,
        'inverse_depth': inverse_depth
    }
    kwargs_sample_hierarchical = {
        'perturb': perturb
    }


    record_file_train = open(LOG_PATH + "/log_train.txt", "w")
    record_file_val = open(LOG_PATH + "/log_val.txt", "w")
    Patience_threshold = 100

    # Save testing gt image for visualization
    matplotlib.image.imsave(LOG_PATH + '/image/' + 'gt.png', valid_img_visual)

    # pretrained_model_pth = 'train_log/real_train_1_log0928_%ddof_100(0)/best_model/'%num_data
    # pretrained_model_pth = 'train_log/real_id1_10000(1)_PE(arm)/best_model/'

    for _ in range(n_restarts):

        model, optimizer = init_models(d_input=(DOF - 2) + 3,  # DOF + 3 -> xyz and angle2 or 3 -> xyz
                                       d_filter=128,
                                       output_size=2,
                                       lr=5e-4,  # 5e-4
                                       # pretrained_model_pth=pretrained_model_pth,
                                       FLAG_PositionalEncoder = FLAG_PositionalEncoder
                                       )


        success = train(model, optimizer)
        if success:
            print('Training successful!')
            break

    print(f'Done!')
    record_file_train.close()
    record_file_val.close()

