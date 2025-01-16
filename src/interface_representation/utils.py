from enum import Enum

import numpy as np


class InterfaceRepresentationType(Enum):
    """
    Enum class to represent the different types of interface representations
    """
    TANH = 1
    SDF = 2


def check_tanh_consistency(tanh: np.ndarray):
    """
    Check that the function is between 0 and 1
    """
    assert np.all(~np.isnan(tanh)), "NaN values in volume fraction"
    assert np.all(~np.isinf(tanh)), "Inf values in volume fraction"
    assert np.min(tanh) >= 0, f"min: {np.min(tanh)}"
    assert np.max(tanh) <= 1.0, f"max: {np.max(tanh)}"


def check_sdf_consistency(sdf: np.ndarray, dx: float):
    """
    Check that the SDF is consistent with the grid size
    """
    assert np.all(~np.isnan(sdf)), "NaN values in SDF"
    assert np.all(~np.isinf(sdf)), "Inf values in SDF"
    assert np.min(sdf) >= -np.sqrt(3) * dx * len(sdf), f"min: {np.min(sdf)}"
    assert np.max(sdf) <= np.sqrt(3) * dx * len(sdf), f"max: {np.max(sdf)}"
