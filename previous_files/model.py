import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SequentialVisualSelfModel(nn.Module):

    def __init__(self, in_channels):
        super(SequentialVisualSelfModel, self).__init__()
        # self.act_f = nn.Tanh()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.fc1 = nn.Linear(8192, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 128)

        self.drop = nn.Dropout(p=0.2)

        self.seq_net = nn.LSTM(input_size=256, hidden_size=64*64, num_layers=1)  # 3d output size too big

        self.fc1_action = nn.Linear(3, 32)
        self.fc2_action = nn.Linear(32, 128)

    def image_block(self, x_images):
        out = self.layer1(x_images)  # in: [1，128，128], out: [16，64，64]
        out = self.layer2(out)  # in: [16，64，64], out: [32，64，64]
        out3 = self.layer3(out)  # in: [32，64，64], out3: [32，64，64]
        out4 = self.layer4(torch.add(out3, out))  # in: [32，64，64], out4: [64，32，32]
        out = self.layer5(out4)  # in: [64，32，32], out: [64，32，32]
        out = self.layer6(torch.add(out4, out))  # in: [64，32，32], out: [128，16，16]
        out = self.layer7(out)  # in: [128，16，16], out: [128，8，8]
        out = out.reshape(out.size(0), -1)  # [128, 8, 8] => [8192]

        out = F.relu(self.fc1(out), inplace=True)
        out = self.drop(out)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.drop(out)
        out = F.relu(self.fc3(out), inplace=True)
        out = F.relu(self.fc4(out), inplace=True)
        # out = F.relu(self.fc5(out), inplace=True)

        return out

    def action_block(self, x_action):
        print(x_action.shape)
        out = F.relu(self.fc1_action(x_action), inplace=True)
        out = F.relu(self.fc2_action(out), inplace=True)

        return out

    def forward(self, image_input, action_input):
        image_rep_0 = self.image_block(image_input[0])
        image_rep_1 = self.image_block(image_input[1])
        image_rep_2 = self.image_block(image_input[2])

        action_rep_0 = self.action_block(action_input[0])
        action_rep_1 = self.action_block(action_input[1])
        action_rep_2 = self.action_block(action_input[2])

        combine_0 = torch.cat((image_rep_0, action_rep_0), 1).unsqueeze(0)
        combine_1 = torch.cat((image_rep_1, action_rep_1), 1).unsqueeze(0)
        combine_2 = torch.cat((image_rep_2, action_rep_2), 1).unsqueeze(0)

        seq_input = torch.cat([combine_0, combine_1, combine_2])
        print(seq_input.shape)
        seq_output, _ = self.seq_net(seq_input)

        print(seq_output[-1].shape)
        return seq_output[-1]

    def loss(self, pred, target):
        return torch.mean((pred - target) ** 2)


if __name__ == "__main__":
    """model test"""
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("start", device)

    model = SequentialVisualSelfModel(in_channels=1).to(device)
    batch_size = 10
    step_num = 3
    x_i = torch.cat([torch.randn(batch_size, 1, 128, 128).unsqueeze(0)] * step_num).to(device)
    x_a = torch.cat([torch.randn(batch_size, 3).unsqueeze(0)] * step_num).to(device)

    print(x_i.shape, x_a.shape)
    model.forward(x_i, x_a)

    # model.image_block(x_i)
