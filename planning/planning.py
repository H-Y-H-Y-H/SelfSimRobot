
import matplotlib.pyplot as plt
import numpy as np

def spiral_trj():
    # Number of points
    num_points = 1000

    # Create a 2D array to store the x, y, z positions
    trajectory = np.zeros((num_points, 3))

    # Generate a spiral trajectory
    t = np.linspace(0, 4 * np.pi, num_points)  # t parameter for the spiral
    for i in range(num_points):
        z = t[i] * np.cos(t[i])*0.05 +0.1 # x coordinate
        y = t[i] * np.sin(t[i])*0.05  # y coordinate
        x = t[i]*0.02+0.1  # z coordinate
        trajectory[i] = [x, y, z]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])

    # Show the plot
    plt.show()
    np.savetxt('trajectory/spiral.csv',trajectory)

spiral_trj()


