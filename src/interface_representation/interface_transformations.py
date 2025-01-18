import warnings

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.interface_representation.utils import InterfaceRepresentationType


def approximate_sdf_from_diffuse(phi: np.ndarray, epsilon: float):
    """Approximate formula to convert diffuse interface to signed distance function, invalid far away from
    interface, where values are +/- constant.
    """
    assert np.all(phi >= 0) and np.all(phi <= 1), f'Phi values must be in [0, 1], got {phi.min()} and {phi.max()}'
    assert not np.any(np.isnan(phi)), 'NaNs found in input phi'
    assert epsilon > 0, 'Epsilon must be positive'

    phi = phi.astype(np.float64)  # Necessary to prevent numerical round-off error
    delta = 1e-100  # Avoid log(0) in the next line
    psi = - epsilon * np.log((phi + delta) / ((1.0 - phi) + delta))

    assert not np.any(np.isinf(psi)), 'Infs found in SDF'
    assert not np.any(np.isnan(psi)), 'NaNs found in SDF'
    return psi


def exact_sdf_from_diffuse(phi: np.ndarray, epsilon: float):
    """"Exact signed distance function from diffuse interface representation, meaning that we
    use the Euclidean distance transform to march the SDF outwards from a narrow interfacial region, where
    values are set using the ln(phi/(1-phi)) formula
    """
    assert np.all(phi >= 0) and np.all(phi <= 1), f'Phi values must be in [0, 1], got {phi.min()} and {phi.max()}'
    assert not np.any(np.isnan(phi)), 'NaNs found in input phi'
    assert epsilon > 0, 'Epsilon must be positive'# Suppress div by zero warnings

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
    """
    Convert a signed distance function to a diffuse interface representation.
    Applies to both approximate and exact SDFs.
    """
    phi = 0.5 * (1 + np.tanh(-sdf / (2 * epsilon)))
    return phi


def convert_from_tanh(phi: np.ndarray, representation_type: InterfaceRepresentationType, current_epsilon: float, desired_epsilon: float):
    """
    Convert the interface representation from tanh to another representation type
    """
    sdf = approximate_sdf_from_diffuse(phi, current_epsilon)

    if representation_type == InterfaceRepresentationType.TANH:
        if desired_epsilon == current_epsilon:
            return phi
        else:
            return diffuse_from_sdf(sdf, desired_epsilon)
    elif representation_type == InterfaceRepresentationType.SDF_APPROX:
        return sdf
    elif representation_type == InterfaceRepresentationType.SDF_EXACT:
        return exact_sdf_from_diffuse(phi, current_epsilon)
    else:
        raise ValueError(f'Unsupported representation type: {representation_type}')