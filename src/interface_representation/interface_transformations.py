"""
In this file, functions for conversion to and from a diffuse to sharp (level set) interface representation are defined
"""
import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.interface_representation.utils import InterfaceRepresentationType


def approximate_sdf_from_diffuse(phi: np.ndarray, epsilon: float):
    assert np.all(phi >= 0) and np.all(phi <= 1), f'Phi values must be in [0, 1], got {phi.min()} and {phi.max()}'
    assert not np.any(np.isnan(phi)), 'NaNs found in input phi'
    assert epsilon > 0, 'Epsilon must be positive'

    # Suppress div by zero warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        psi = - epsilon * np.log(phi / (1.0 - phi))

    dx = 1.0 / len(phi)
    edt_offset = 10 * epsilon

    # Compute distance transform for regions greater than 3 epsilon from interface
    exterior_mask = psi > edt_offset
    exterior_far_sdf = distance_transform_edt(exterior_mask, sampling=dx)

    # Compute distance transform for regions less than 3 epsilon from interface
    interior_mask = psi < -edt_offset
    interior_far_sdf = distance_transform_edt(interior_mask, sampling=dx)

    psi[exterior_mask] = exterior_far_sdf[exterior_mask] + (edt_offset - 0.5 * dx)
    psi[interior_mask] = -interior_far_sdf[interior_mask] - (edt_offset - 0.5 * dx)

    # Check there are no infs remaining
    assert not np.any(np.isinf(psi)), 'Infs found in SDF'
    assert not np.any(np.isnan(psi)), 'NaNs found in SDF'

    return psi


def diffuse_from_sdf(sdf: np.ndarray, epsilon: float):
    phi = 0.5 * (1 + np.tanh(-sdf / (2 * epsilon)))

    return phi


def convert_from_sdf(psi: np.ndarray, representation_type: InterfaceRepresentationType, epsilon: float):
    if representation_type == InterfaceRepresentationType.TANH:
        return diffuse_from_sdf(psi, epsilon)
    elif representation_type == InterfaceRepresentationType.SDF:
        return psi
    else:
        raise ValueError(f'Unsupported representation type: {representation_type}')


def convert_from_tanh(phi: np.ndarray, representation_type: InterfaceRepresentationType, current_epsilon: float, desired_epsilon: float):
    sdf = approximate_sdf_from_diffuse(phi, current_epsilon)

    if representation_type == InterfaceRepresentationType.TANH:
        if desired_epsilon == current_epsilon:
            return phi
        else:
            return diffuse_from_sdf(sdf, desired_epsilon)
    elif representation_type == InterfaceRepresentationType.SDF:
        return sdf
    else:
        raise ValueError(f'Unsupported representation type: {representation_type}')