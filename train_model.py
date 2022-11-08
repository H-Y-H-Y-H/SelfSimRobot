import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from rays_check import my_rays
from data_collection import *

n, f, nf, ff, f_box = my_rays(H=400, W=400, D=64)
print("box shape: ", f_box.shape)
img2mse = lambda x, y: torch.mean((x - y) ** 2)
"""
MLP model
"""


class VsmModel(nn.Module):
    def __init__(self):
        super(VsmModel, self).__init__()
        self.ray_length = 128
        self.input_size = 6
        self.output_size = 4

        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return self.fc3(x)


"""
volume rendering
"""


def batchify(fn, chunk=1024 * 64):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def v_render(box_points, network, ang, ray_d):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    z_vals = torch.linspace(0.5, 1.1, 64)  # near 0.5, far 1.1, num=100
    box_flat = torch.reshape(box_points, [-1, 3])
    ang_b = torch.broadcast_to(ang, box_flat.shape)
    input_6 = torch.cat((box_flat, ang_b), 1)
    output_4 = batchify(fn=network)(input_6)
    outputs = torch.reshape(output_4, list(box_points.shape[:-1]) + [output_4.shape[-1]])

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(ray_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(outputs[..., :3])  # [N_rays, N_samples, 3]

    alpha = raw2alpha(outputs[..., 3], dists)  # [N_rays, N_samples]
    # weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * torch.cumprod(1.-alpha + 1e-10, -1)
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    return rgb_map


"""
training 
"""


def train_model(model, batch_size, lr, num_epoch, log_path, id_list):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_list = id_list[:int(data_size * 0.8)]
    test_list = id_list[int(data_size * 0.8):]
    print("train, test, ", train_list.shape, test_list.shape)
    for epoch in range(num_epoch):
        for idx in train_list:
            img = iio.imread(DATA_PATH + "image/%d.png" % idx)
            img = (img / 255.).astype(np.float32)
            img = torch.Tensor(img).to(device)  # .to(device)
            angles = angle_list[idx]
            angles = torch.Tensor(angles).to(device)
            # split 400 to 4 * 100 in x-axis and y-axis
            xs = torch.randperm(400).reshape(4, 100)
            ys = torch.randperm(400).reshape(4, 100)
            one_image_loss = 0
            for x in xs:
                for y in ys:
                    box_p = f_box.clone().to(device)
                    box_p = torch.index_select(box_p, 0, x)
                    box_p = torch.index_select(box_p, 1, y)
                    label = img.clone().to(device)
                    label = torch.index_select(label, 0, x)
                    label = torch.index_select(label, 1, y)

                    d = ff.clone().to(device)
                    d = torch.index_select(d, 0, x)
                    d = torch.index_select(d, 1, y)
                    # 400*400 rays and pixels, split into 100*100 rays and pixels, loop 16 times
                    prediction = v_render(box_p, model, angles, d)

                    optimizer.zero_grad()
                    # print(prediction.shape, label.shape)
                    img_loss = img2mse(prediction, label)
                    img_loss.backward()
                    optimizer.step()
                    one_image_loss += img_loss.item()
                    print("t-loss: ", img_loss.item())

            print("one image loss: ", one_image_loss / (4*4))


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("start", device)

    DATA_PATH = "general_data/data_white/"

    Lr = 1e-6
    Batch_size = 1  # 128
    b_size = 400
    Num_epoch = 100

    Log_path = "./log_01/"

    try:
        os.mkdir(Log_path)
    except OSError:
        pass
    Model = VsmModel().to(device)

    angle_list = np.loadtxt(DATA_PATH + "angles.csv")
    data_size = angle_list.shape[0]
    torch.manual_seed(0)
    random_id_list = torch.randperm(data_size)
    print(random_id_list.shape)
    train_model(model=Model, batch_size=Batch_size, lr=Lr, num_epoch=Num_epoch, log_path=Log_path,
                id_list=random_id_list)
