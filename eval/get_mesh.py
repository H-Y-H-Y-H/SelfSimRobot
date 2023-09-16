import numpy as np
from skimage.measure import marching_cubes
from stl import mesh

# Sample 3D array
array = np.zeros((50, 50, 50))
array[20:30, 20:30, 20:30] = 1  # A filled cube within the larger cube

# Convert to mesh using marching cubes
vertices, faces, _, _ = marching_cubes(array, level=0.5)

# Export to .obj
with open('output.obj', 'w') as f:
    for v in vertices:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for face in faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

# Export to .stl
my_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, face in enumerate(faces):
    for j in range(3):
        my_mesh.vectors[i][j] = vertices[face[j]]
my_mesh.save('output.stl')
