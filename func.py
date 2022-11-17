

import numpy as np

def rot_Y(th):
    matrix = ([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]])
    np.asarray(matrix)

    return matrix

def rot_Z(th):
    matrix = ([
    [np.cos(th), -np.sin(th), 0, 0],
    [np.sin(th), np.cos(th), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])
    np.asarray(matrix)

    return matrix

