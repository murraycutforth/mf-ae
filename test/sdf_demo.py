import numpy as np
import trimesh
import time
import mesh2sdf

trimesh.util.attach_to_log()
mesh = trimesh.creation.uv_sphere(radius=1, count=[100, 100])

r1 = 2.0
r2 = 1.0
r3 = 1.5
mesh.apply_scale([r1, r2, r3])

# normalize mesh
mesh_scale = 1.0
vertices = mesh.vertices
bbmin = vertices.min(0)
bbmax = vertices.max(0)
center = (bbmin + bbmax) * 0.5
scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
vertices = (vertices - center) * scale

t0 = time.time()
size = 100
level = 2 / size
sdf, mesh = mesh2sdf.compute(
    vertices, mesh.faces, size, fix=False, return_mesh=True)
t1 = time.time()

print(f"Time: {t1 - t0:.2f}s")
print(sdf.shape)
print(sdf.min(), sdf.max())
