import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from rays_check import my_rays
from func import rays_np, transfer_box
# from data_collection import *
from env import FBVSM_Env

n, f, nf, ff, f_box = rays_np(H=400, W=400, D=64)
print("box shape: ", f_box.shape)
img2mse = lambda x, y: torch.mean((x - y) ** 2)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("start", device)
"""
MLP model
"""


class VsmModel(nn.Module):
    def __init__(self):
        super(VsmModel, self).__init__()
        self.ray_length = 128
        self.input_size = 5
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

    z_vals = torch.linspace(0.5, 1.1, 64).to(device)  # near 0.5, far 1.1, num=100
    box_flat = torch.reshape(box_points, [-1, 3])
    ang_b = torch.broadcast_to(ang, [box_flat.shape[0], 2])  # TODO may bug here!!
    input_6 = torch.cat((box_flat, ang_b), 1)
    output_4 = batchify(fn=network)(input_6)
    outputs = torch.reshape(output_4, list(box_points.shape[:-1]) + [output_4.shape[-1]])

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(device)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(ray_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(outputs[..., :3])  # [N_rays, N_samples, 3]

    alpha = raw2alpha(outputs[..., 3], dists)  # [N_rays, N_samples]
    # weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * torch.cumprod(1. - alpha + 1e-10, -1)
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    return rgb_map


"""
training 
"""


# def train_model(env, model, batch_size, lr, num_epoch, log_path, id_list):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     train_list = id_list[:int(data_size * 0.8)]
#     test_list = id_list[int(data_size * 0.8):]
#     print("train, test, ", train_list.shape, test_list.shape)
#     for epoch in range(num_epoch):
#
#         for idx in train_list:
#             # img = iio.imread(DATA_PATH + "image/%d.png" % idx)
#             img = (img / 255.).astype(np.float32)
#             img = torch.Tensor(img).to(device)  # .to(device)
#             angles = angle_list[idx]
#             angles = torch.Tensor(angles).to(device)
#             # split 400 to 4 * 100 in x-axis and y-axis
#             xs = torch.randperm(400).reshape(4, 100)
#             ys = torch.randperm(400).reshape(4, 100)
#             one_image_loss = 0
#             for x in xs:
#                 for y in ys:
#                     box_p = f_box.clone().to(device)
#                     box_p = torch.index_select(box_p, 0, x)
#                     box_p = torch.index_select(box_p, 1, y)
#                     label = img.clone().to(device)
#                     label = torch.index_select(label, 0, x)
#                     label = torch.index_select(label, 1, y)
#
#                     d = ff.clone().to(device)
#                     d = torch.index_select(d, 0, x)
#                     d = torch.index_select(d, 1, y)
#                     # 400*400 rays and pixels, split into 100*100 rays and pixels, loop 16 times
#                     prediction = v_render(box_p, model, angles, d)
#
#                     optimizer.zero_grad()
#                     # print(prediction.shape, label.shape)
#                     img_loss = img2mse(prediction, label)
#                     img_loss.backward()
#                     optimizer.step()
#                     one_image_loss += img_loss.item()
#                     print("t-loss: ", img_loss.item())
#
#             print("one image loss: ", one_image_loss / (4 * 4))


"""
online version
"""


def update_model(obs, model, lr, mybox, W=400, H=400):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    a_list, img = obs[0], obs[1]
    # print(img)
    # print(a_list)
    img = (img / 255.).astype(np.float32)
    img = torch.Tensor(img).to(device)
    angles = torch.Tensor(a_list).to(device)
    xs = torch.randperm(W).reshape(4, int(W / 4)).to(device)
    ys = torch.randperm(H).reshape(4, int(H / 4)).to(device)
    one_image_loss = 0
    for x in xs:
        for y in ys:
            f_box_t = torch.Tensor(mybox).to(device)
            box_p = f_box_t.clone().to(device)
            # box_p = f_box.clone().to(device)
            box_p = torch.index_select(box_p, 0, x).to(device)
            box_p = torch.index_select(box_p, 1, y).to(device)
            label = img.clone().to(device)
            label = torch.index_select(label, 0, x).to(device)
            label = torch.index_select(label, 1, y).to(device)

            # d = ff.clone().to(device)
            # d = torch.index_select(d, 0, x).to(device)
            # d = torch.index_select(d, 1, y).to(device)

            d = box_p[:, :, -1, :]
            # 400*400 rays and pixels, split into 100*100 rays and pixels, loop 16 times
            prediction = v_render(box_p, model, angles, d)

            optimizer.zero_grad()
            # print(prediction.shape, label.shape)
            img_loss = img2mse(prediction, label)
            img_loss.backward()
            optimizer.step()
            one_image_loss += img_loss.item()
            print("t-loss: ", img_loss.item())

    print("one image loss: ", one_image_loss / (4 * 4))
    return one_image_loss / (4 * 4)


if __name__ == "__main__":

    DATA_PATH = "general_data/data_white/"

    Lr = 1e-6
    Batch_size = 1  # 128
    b_size = 400
    Num_epoch = 100

    Log_path = "previous_files/log_03/"

    try:
        os.mkdir(Log_path)
    except OSError:
        pass
    Model = VsmModel().to(device)

    # angle_list = np.loadtxt(DATA_PATH + "angles.csv")
    # data_size = angle_list.shape[0]
    torch.manual_seed(0)
    # random_id_list = torch.randperm(data_size)
    # print(random_id_list.shape)
    # train_model(env=env, model=Model, batch_size=Batch_size, lr=Lr, num_epoch=Num_epoch, log_path=Log_path,
    #             id_list=random_id_list)

    import pybullet as p

    RENDER = False
    if RENDER:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    env = FBVSM_Env()
    line_array = np.linspace(-1.0, 1.0, num=21)
    i = 0
    j = 0
    min_loss = 1.
    loss_record = []
    NUM_MOTOR = 2
    step_size = 0.1
    obs = env.reset()
    my_box = f_box
    for _ in range(500):
        t_angle = np.random.choice(line_array, NUM_MOTOR)
        c_angle = obs[0]
        act_list = []

        for act_i in range(NUM_MOTOR):
            act_list.append(np.linspace(c_angle[act_i], t_angle[act_i],
                                        round(abs((t_angle[act_i] - c_angle[act_i]) / step_size) + 1)))

        # update expect_angles and received a_array

        for m_id in range(NUM_MOTOR):
            for single_cmd_value in act_list[m_id]:
                c_angle[m_id] = single_cmd_value
                obs, _, _, _ = env.step(c_angle)
                my_box, _ = transfer_box(my_box, obs[0])  # update box
                im_loss = update_model(obs, Model, Lr, my_box)
                loss_record.append(im_loss)
                if im_loss < min_loss:
                    min_loss = im_loss
                    torch.save(Model.state_dict(), Log_path + 'best_model_MSE.pt')
                    # print("saved")
                # print(env.get_obs()[0])
    # for _ in range(500):
    #     a_array = env.get_traj(line_array)
    #     print("new path ---------------")
    #     j += 1
    #     for a1 in a_array['a1']:
    #         new_a = env.expect_angles.copy()
    #         new_a[0] = a1
    #         obs, _, _, _ = env.step(new_a)
    #         im_loss = update_model(obs, Model, Lr)
    #         loss_record.append(im_loss)
    #         if im_loss < min_loss:
    #             min_loss = im_loss
    #             torch.save(Model.state_dict(), Log_path + 'best_model_MSE.pt')
    #             # print("saved")
    #         i += 1
    #         print("path: ", j, " update times: ", i)
    #
    #     for a2 in a_array['a2']:
    #         new_a = env.expect_angles.copy()
    #         new_a[1] = a2
    #         env.step(new_a)
    #         obs, _, _, _ = env.step(new_a)
    #         im_loss = update_model(obs, Model, Lr)
    #         loss_record.append(im_loss)
    #         if im_loss < min_loss:
    #             min_loss = im_loss
    #             torch.save(Model.state_dict(), Log_path + 'best_model_MSE.pt')
    #             # print("saved")
    #         i += 1
    #         print("path: ", j, " update times: ", i)
    #
    #     for a3 in a_array['a3']:
    #         new_a = env.expect_angles.copy()
    #         new_a[2] = a3
    #         env.step(new_a)
    #         obs, _, _, _ = env.step(new_a)
    #         im_loss = update_model(obs, Model, Lr)
    #         loss_record.append(im_loss)
    #         if im_loss < min_loss:
    #             min_loss = im_loss
    #             torch.save(Model.state_dict(), Log_path + 'best_model_MSE.pt')
    #             # print("saved")
    #         i += 1
    #         print("path: ", j, " update times: ", i)

    np.savetxt(Log_path + "training_MSE.csv", np.asarray(loss_record))
