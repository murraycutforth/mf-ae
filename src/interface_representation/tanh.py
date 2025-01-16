
import numpy as np
from skimage import measure
import trimesh


def tanh_to_mesh(tanh: np.ndarray, dx: float):
    """
    Convert tanh representation to mesh

    TODO: can we assume that the function is an approimate SDF near the interface?!
    """
    verts, faces, normals, values = measure.marching_cubes(tanh, level=0.5, spacing=(dx, dx, dx))
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True, validate=True)
