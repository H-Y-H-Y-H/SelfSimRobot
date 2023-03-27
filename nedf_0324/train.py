import os
import random
import torch
from model import NeDF
from tqdm import trange
import numpy as np
import matplotlib.image

# from func import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

"""helper functions"""


def ray_sum():
    # model: xyz to dense
    # ray_sum: dense to pixels' value, 0/1
    pass


def get_ray():
    # pose to o, d
    pass


def sample_points():
    # o, d to x,y,z
    pass


"""training process"""


def train(model, optimizer, n_iter):
    for i in trange(n_iter):
        model.train()
        if Overfitting_test:
            target_img_idx = 0
        else:
            target_img_idx = np.random.randint(training_img.shape[0] - 1)

        target_img = training_img[target_img_idx]
        angle = training_angles[target_img_idx]
        pose_matrix = training_pose_matrix[target_img_idx]

    pass


if __name__ == "__main__":

    seed_num = 6
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.manual_seed(seed_num)

    """
    prepare data and parameters
    """
    near, far = 2., 6.
    Flag_save_image_during_training = True
    DOF = 3  # the number of motors
    num_data = 40
    tr = 0.8  # training ratio
    data = np.load('../data/NeDF_data/dof%d_data%d.npz' % (DOF, num_data))
    Overfitting_test = False
    sample_id = random.sample(range(num_data), num_data)

    if Overfitting_test:
        valid_img_visual = data['images'][sample_id[0]]
        valid_img_visual = np.dstack((valid_img_visual, valid_img_visual, valid_img_visual))
    else:
        valid_amount = int(num_data * (1 - tr))
        max_pic_save = 8
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

    """
    init model
    """
    Model = NeDF(d_input=3,
                 n_layers=6,
                 d_filter=256)
    Model.to(device)

    Learning_rate = 5e-5
    Optimizer = torch.optim.Adam(Model.parameters(), lr=Learning_rate)

    """
    training
    """

    # Run training session(s)
    LOG_PATH = "train_log/log_%ddata/" % num_data

    os.makedirs(LOG_PATH + "image/", exist_ok=True)
    os.makedirs(LOG_PATH + "best_model/", exist_ok=True)

    record_file_train = open(LOG_PATH + "log_train.txt", "w")
    record_file_val = open(LOG_PATH + "log_val.txt", "w")
    Patience_threshold = 20

    # Save testing gt image for visualization
    matplotlib.image.imsave(LOG_PATH + 'image/' + 'gt.png', valid_img_visual)

    # train(model=Model,
    #       optimizer=Optimizer,
    #       n_iter=10000)
