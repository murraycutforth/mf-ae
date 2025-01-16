"""
In this file, functions for conversion to and from a diffuse to sharp (level set) interface representation are defined
"""

import numpy as np

from src.interface_representation.utils import InterfaceRepresentationType


def approximate_sdf_from_diffuse(phi: np.ndarray, epsilon: float):
    psi = epsilon * np.log(phi / (1.0 - phi))
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