import random
import torch
from model import NeDF
# from func import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def ray_sum():
    pass


def train(model, optimizer):

    pass


if __name__ == "__main__":
    Model = NeDF(d_input=3,
                 n_layers=6,
                 d_filter=256)
    Model.to(device)
    Learning_rate = 5e-5
    Optimizer = torch.optim.Adam(Model.parameters(), lr=Learning_rate)
    train(model=Model, optimizer=Optimizer)

