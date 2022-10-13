import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import utils, transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class VisualSelfModel(nn.Module):

    def __init__(self):
        super(VisualSelfModel, self).__init__()

        # 59*2, 113*2
        self.input_size = 226
        # 15
        self.output_size = 11

        self.fc1 = nn.Linear(self.input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.output_size)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = F.relu(self.fc1(x),inplace = True)
        x = self.bn2(x)
        x = F.relu(self.fc2(x) ,inplace = True)
        x = torch.sigmoid(self.fc3(x))
        return x

    def loss(self, pred, target):
        return torch.mean((pred - target) ** 2)