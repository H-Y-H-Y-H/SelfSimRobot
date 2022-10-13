import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import matplotlib.pyplot as plt
import torchvision
from model import VisualSelfModel
import numpy as np
import os
import time

# Check GPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("start", device)

DATA_PATH = "/Users/jionglin/Downloads/vsm_data/"

