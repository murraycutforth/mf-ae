import numpy as np
from skimage import measure
import trimesh


def volfrac_to_mesh(tanh: np.ndarray, dx: float):
    """
    Convert volume fraction representation to mesh
    """
    verts, faces, normals, values = measure.marching_cubes(tanh, level=0.5, spacing=(dx, dx, dx))
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True, validate=True)
