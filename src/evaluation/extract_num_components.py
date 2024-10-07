# In this file, we implement functions which extract information from a binary volume to plot evaluation metrics against
# This includes: number of components, surface area to volume ratio (sigma)


import numpy as np
from skimage.measure import label, marching_cubes, mesh_surface_area


def compute_num_components(arr: np.ndarray) -> int:
    # Return total number of connected components

    arr_ = arr.astype(int)
    _, num = label(arr_)
    return num


def compute_surface_area_volume_ratio(arr: np.ndarray) -> float:
    # Compute sigma, large value corresponds to a fine spray
    # Units of [voxel length]^{-1}

    verts, faces, normals, values = marching_cubes(
        arr, level=0.5, spacing=(1, 1, 1), allow_degenerate=False, method='lewiner'
    )

    sa = mesh_surface_area(verts, faces)
    vol = np.sum(arr > 0.5)

    return sa / vol
