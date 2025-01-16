import numpy as np
import trimesh

trimesh.util.attach_to_log()
mesh = trimesh.creation.uv_sphere(radius=1, count=[100, 100])

r1 = 2.0
r2 = 1.0
r3 = 1.5
mesh.apply_scale([r1, r2, r3])

mesh.show()



