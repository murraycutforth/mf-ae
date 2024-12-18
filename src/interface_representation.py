from enum import Enum

import numpy as np
from scipy.ndimage import distance_transform_edt


class InterfaceRepresentationType(Enum):
    """
    Enum class to represent the different types of interface representations
    """
    HEAVISIDE = 1
    DIFFUSE_TANH = 2
    SIGNED_DISTANCE = 3


def sdf_to_diffuse_tanh(sdf: np.ndarray, interface_width: float = 1.0 / 256) -> np.ndarray:
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

    return sdf


def convert_arr_from_heaviside(arr: np.ndarray, interface_rep: InterfaceRepresentationType) -> np.ndarray:
    """
    Convert an array from Heaviside representation to other representation
    """
    dx = 1.0 / 256

    if interface_rep == InterfaceRepresentationType.HEAVISIDE:
        return arr
    elif interface_rep == InterfaceRepresentationType.DIFFUSE_TANH:
        sdf = heaviside_to_sdf(arr)
        return sdf_to_diffuse_tanh(sdf, interface_width=dx)
    elif interface_rep == InterfaceRepresentationType.SIGNED_DISTANCE:
        return heaviside_to_sdf(arr)





