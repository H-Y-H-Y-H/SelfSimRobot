import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryModel(nn.Module):

    def __init__(self, bsize):
        super(QueryModel, self).__init__()

        self.b_size = bsize
        self.ray_length = 128
        self.input_size = 6
        self.output_size = 1

        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = torch.sigmoid(self.fc3(x))

        return x

    def loss(self, pred, target):
        out_put = torch.tensor([], requires_grad=True)
        for i in range(self.b_size):
            one_ray = pred[i * self.ray_length: i * self.ray_length + self.b_size]
            p = volume_rendering(one_ray)
            out_put = torch.cat((out_put, torch.tensor([p])))
        return torch.mean((out_put - target) ** 2)


def volume_rendering(ray):
    Q = torch.tensor([], requires_grad=True)
    for i in range(len(ray)):
        T = torch.exp(-torch.sum(ray[:i+1]))
        Q = torch.cat((Q, torch.tensor([T * ray[i]])))

    return torch.sum(Q)


if __name__ == "__main__":
    """check rendering function"""
    # import matplotlib.pyplot as plt
    # plot_list = []
    # for i in range(1000):
    #     test_ray = torch.rand(128)
    #     out = volume_rendering(test_ray)
    #     plot_list.append(out.numpy())
    #     print(out)
    #
    # plt.figure()
    # plt.plot(plot_list)
    # plt.ylim([0,1])
    # plt.show()
    # test_ray = torch.rand(128)
    list_len1 = []
    list_len2 = []
    list_len3 = []
    for i in range(128-1):
        test_ray = torch.cat((torch.zeros(128-2-i), torch.ones(1)/2, torch.ones(1), torch.zeros(i)))
        # print(test_ray)
        out = volume_rendering(test_ray)
        print(out)



