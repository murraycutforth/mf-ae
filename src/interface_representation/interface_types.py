from enum import Enum


class InterfaceType(Enum):
    TANH_EPSILON = 1  # Tanh representation with an epsilon parameter
    SIGNED_DISTANCE_EXACT = 2  # Signed distance function
    SIGNED_DISTANCE_APPROXIMATE = 3  # Approximate signed distance function
    HEAVISIDE = 4  # Heaviside function
