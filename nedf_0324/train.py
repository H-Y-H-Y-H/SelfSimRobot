import random
import torch
from model import NeDF
# from func import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


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

