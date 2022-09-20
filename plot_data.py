import matplotlib.pyplot as plt
import numpy as np

def plot_animation(data_len):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(data_len):
        data = np.loadtxt("data/arm%d.csv"%i)
        # for p in data:
        ax.cla()
        ax.scatter(data[:, 0],data[:, 1], data[:, 2],s=5,color="black")
        ax.set_xlim([-0.2,0.2])
        ax.set_ylim([-0.2,0.2])
        ax.set_zlim([0,0.3])
        plt.pause(0.5)
    plt.show()


if __name__ == "__main__":
    plot_animation(20)