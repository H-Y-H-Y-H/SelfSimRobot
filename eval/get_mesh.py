import numpy as np
import matplotlib.pyplot as plt

mesh_file = 'sim_robo_1/points/pts1.csv'
points = np.loadtxt(mesh_file)


# Extract the x, y, and z coordinates from the list of points
x_coords = [point[0] for point in points]
y_coords = [point[1] for point in points]
z_coords = [point[2] for point in points]

# Create a new figure
fig = plt.figure()

# Add 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x_coords, y_coords, z_coords, c='r', marker='o',s=10)

# Set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# Determine the global min and max across all dimensions
global_min = min(min(x_coords), min(y_coords), min(z_coords))
global_max = max(max(x_coords), max(y_coords), max(z_coords))

# Set the limits for each axis to ensure uniform intervals
ax.set_xlim([global_min, global_max])
ax.set_ylim([global_min, global_max])
ax.set_zlim([global_min, global_max])

# Show the plot
plt.show()
