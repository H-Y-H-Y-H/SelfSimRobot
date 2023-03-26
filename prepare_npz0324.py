import os

import numpy as np
import torch

from env import FBVSM_Env
import pybullet as p
import time


def _Data(myenv: FBVSM_Env, dof: int, path: str, sample_num: int):
    # 1. one dof robot arm
    myenv.reset()
    theta_0_list = np.linspace(-1., 1., sample_num, endpoint=False)  # no repeated angles


if __name__ == "__main__":
    pass
