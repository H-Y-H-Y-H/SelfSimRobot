import matplotlib.pyplot as plt
import numpy as np
import os

def check_face(box_len=1, num_points=5):
    face_num = num_points * num_points
    start_p = -box_len / 2
    end_p = box_len / 2
    x_p = np.linspace(start_p, end_p, num_points)
    y_p = np.linspace(start_p, end_p, num_points)
    z_p = np.linspace(0, end_p*2, num_points)
    xx, zz = np.meshgrid(x_p, z_p)
    y0 = np.zeros((face_num, 1)) + start_p
    y1 = np.zeros((face_num, 1)) + end_p
    x1 = xx.reshape(face_num, 1)
    z1 = zz.reshape(face_num, 1)
    face_0 = np.concatenate((x1, y0, z1), axis=1)
    face_1 = np.concatenate((x1, y1, z1), axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for f in face_0:
        print(f)
        ax.scatter(f[0],f[1], f[2],s=1,color="black")
    plt.show()


def plot_animation(data_len):
    file_name = "data_with_para/"
    file_list = os.listdir(file_name)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(len(file_list)):
        # data = np.loadtxt("data/arm-pix%d.csv"%i)
        data = np.loadtxt(file_name + file_list[i])
        # for p in data:
        ax.cla()
        ax.scatter(data[:, 0],data[:, 1], data[:, 2],s=1)
        ax.set_xlim([-0.4,0.4])
        ax.set_ylim([-0.4,0.4])
        ax.set_zlim([0,0.8])
        plt.pause(0.5)
    plt.show()

def plot_oneframe(file):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    data = np.loadtxt(file)
    ax.scatter(data[:, 0],data[:, 1], data[:, 2],s=1,color="black")
    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([-0.5,0.5])
    ax.set_zlim([0,1])
    plt.show()


if __name__ == "__main__":
    plot_animation(10)
    # plot_oneframe("data/facecloud01.csv")
    # check_face()