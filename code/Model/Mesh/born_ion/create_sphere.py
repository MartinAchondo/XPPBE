import trimesh
import numpy as np
import os


mesh = trimesh.creation.uv_sphere(radius=1.0, count=[70, 70])

vertices = mesh.vertices
faces = mesh.faces + 1
vertex_normals = vertices / np.linalg.norm(vertices, axis=1)[:, np.newaxis]

# mesh.show()

file_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(file_path,'born_ion.vert'), 'w') as vertex_file:
    for vertex,normal in zip(vertices,vertex_normals):
        vertex_file.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f} {0} {1} {0}\n")

with open(os.path.join(file_path,'born_ion.face'), 'w') as faces_file:
    for face in faces:
        faces_file.write(f"{face[0]} {face[1]} {face[2]} 1 1 \n")

