import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from src.interface_representation import InterfaceRepresentationType, convert_arr_from_heaviside

logger = logging.getLogger(__name__)


def generate_ellipse(center: np.ndarray, radii: np.ndarray):
    """Heaviside representation of an ellipse-shaped interface in 3D space.
    """
    assert center.shape == (3,), f'Unexpected shape: {center.shape}'
    assert radii.shape == (3,), f'Unexpected shape: {radii.shape}'

    xs = np.linspace(0, 1, 256)
    ys = np.linspace(0, 1, 256)
    zs = np.linspace(0, 1, 256)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    X -= center[0]
    Y -= center[1]
    Z -= center[2]

    X /= radii[0]
    Y /= radii[1]
    Z /= radii[2]

    return (X**2 + Y**2 + Z**2 <= 1).astype(np.float32)


class EllipseDataset(Dataset):
    """Dataset class for synthetic ellipse data, parameterized by center and radius. The volumes are hard-coded to 256^3.
    The interface representation (sharp/diffuse/sdf) is specified on construction.
    """
    def __init__(self,
                 debug: bool = False,
                 num_samples: int = 100,
                 interface_rep: InterfaceRepresentationType = InterfaceRepresentationType.HEAVISIDE):
        super().__init__()

        if debug:
            num_samples = 3

        self.num_samples = num_samples

        centers = np.random.rand(num_samples, 3)
        radii = np.random.rand(num_samples, 3)

        # Interface is represented as a Heaviside function
        self.data = np.array([generate_ellipse(center, radii) for center, radii in zip(centers, radii)])
        self.data = [convert_arr_from_heaviside(d, interface_rep) for d in self.data]

        # Data normalisation
        self.compute_norm_params()
        self.data = [self.normalise_array(d) for d in self.data]

        # Add channel dim and convert to torch tensor
        self.data = [torch.tensor(d, dtype=torch.float32).unsqueeze(0) for d in self.data]

        logger.info(f'Generated {num_samples} samples of ellipse data with interface representation {interface_rep}')
        logger.info(f'Each sample has shape {self.data[0].shape}')
        #logger.info(f'Total memory usage: {sum([d.nbytes for d in self.data]) / 1e6} MB')  # Lassen pytorch < v1.11

    def compute_norm_params(self):
        # Compute params for minmax normalization to range of [0, 1]
        self.min_val = np.min([np.min(d) for d in self.data])
        self.max_val = np.max([np.max(d) for d in self.data])
        self.range = self.max_val - self.min_val

    def normalise_array(self, arr):
        return (arr - self.min_val) / self.range

    def unnormalise_array(self, arr):
        return arr * self.range + self.min_val

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]












