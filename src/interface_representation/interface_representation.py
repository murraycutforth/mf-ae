import numpy as np
from scipy.ndimage import distance_transform_edt

import matplotlib.pyplot as plt

from src.interface_representation.utils import InterfaceRepresentationType


def plot_trimesh(mesh):
    # Assuming `mesh` is your trimesh object
    vertices = mesh.vertices
    faces = mesh.faces

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, cmap='viridis', edgecolor='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def check_volfrac_consistency(volfrac: np.ndarray):
    """
    Check that the volume fraction is between 0 and 1
    """
    assert np.all(~np.isnan(volfrac)), "NaN values in volume fraction"
    assert np.all(~np.isinf(volfrac)), "Inf values in volume fraction"
    assert np.min(volfrac) >= 0.0, f"min: {np.min(volfrac)}"
    assert np.max(volfrac) <= 1.0, f"max: {np.max(volfrac)}"


# Deprecated functions below




def diffuse_tanh_to_heaviside(diffuse: np.ndarray, dx: float, interface_width: float):
    """Convert diffuse interface to heaviside representation.
    """
    sdf = np.atanh(2.0 * diffuse - 1) * interface_width
    return sdf_to_volfrac(sdf, dx)


def sdf_to_diffuse_tanh(sdf: np.ndarray, interface_width: float) -> np.ndarray:
    """
    Convert a signed distance function to a diffuse tanh representation
    """
    return 0.5 * (1 + np.tanh(sdf / interface_width))


def heaviside_to_sdf(arr: np.ndarray) -> np.ndarray:
    """
    Convert an array from Heaviside representation to signed distance function
    """
    mask_inside = arr > 1.0 - 1e-6
    mask_outside = arr < 1e-6
    mask_mixed = ~mask_inside & ~mask_outside

    assert np.all(mask_inside | mask_outside | mask_mixed)

    # Zero-th order estimate of SDF in mixed cells
    sdf = np.zeros_like(arr)
    sdf[mask_mixed] = 0.5 - arr[mask_mixed]

    # Set interior region to zero, to compute the exterior SDF (positive)
    arr_outside = arr.copy()
    arr_outside[mask_inside] = 0
    arr_outside[mask_mixed] = 1
    arr_outside[mask_outside] = 1
    sdf_outside = distance_transform_edt(arr_outside, sampling=1.0 / 256)
    sdf[mask_outside] = sdf_outside[mask_outside]

    # Set exterior region to zero, to compute the interior SDF (negative)
    arr_inside = arr.copy()
    arr_inside[mask_outside] = 0
    arr_inside[mask_mixed] = 1
    arr_inside[mask_inside] = 1
    sdf_inside = -distance_transform_edt(arr_inside, sampling=1.0 / 256)
    sdf[mask_inside] = sdf_inside[mask_inside]

    # Check result is finite
    assert np.all(~np.isnan(sdf))
    assert np.all(~np.isinf(sdf))

    return sdf


def convert_arr_from_heaviside(arr: np.ndarray, interface_rep: InterfaceRepresentationType, dx: float) -> np.ndarray:
    """
    Convert an array from Heaviside representation to other representation
    """

    if interface_rep == InterfaceRepresentationType.VOL_FRAC:
        return arr
    elif interface_rep == InterfaceRepresentationType.TANH:
        sdf = heaviside_to_sdf(arr)
        return sdf_to_diffuse_tanh(sdf, interface_width=dx)
    elif interface_rep == InterfaceRepresentationType.SDF:
        return heaviside_to_sdf(arr)
    else:
        raise ValueError


def convert_arr_to_heaviside(arr: np.ndarray, interface_rep: InterfaceRepresentationType, dx: float) -> np.ndarray:
    """
    Convert to Heaviside representation from other representation
    """
    if interface_rep == InterfaceRepresentationType.VOL_FRAC:
        return arr
    elif interface_rep == InterfaceRepresentationType.TANH:
        return diffuse_tanh_to_heaviside(arr, dx=dx, interface_width=dx)
    elif interface_rep == InterfaceRepresentationType.SDF:
        return sdf_to_volfrac(arr, dx=dx)
    else:
        raise ValueError






