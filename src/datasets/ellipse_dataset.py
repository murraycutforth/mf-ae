import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.draw import ellipsoid
from scipy.ndimage import distance_transform_edt

from src.interface_representation.interface_representation import convert_arr_from_heaviside, convert_arr_to_heaviside
from src.interface_representation.interface_transformations import convert_from_sdf
from src.interface_representation.utils import InterfaceRepresentationType, check_sdf_consistency

logger = logging.getLogger(__name__)


def generate_ellipsoid_vol_fracs(center: np.ndarray, radii: np.ndarray, N: int):
    """Heaviside representation of an ellipse-shaped interface in 3D space.
    """
    assert center.shape == (3,), f'Unexpected shape: {center.shape}'
    assert radii.shape == (3,), f'Unexpected shape: {radii.shape}'
    assert N > 0, f'Invalid volume size: {N}'
    assert np.all(radii > 0) and np.all(radii <= 1), f'Invalid radii: {radii}'
    assert np.all(center >= 0) and np.all(center <= 1), f'Invalid center: {center}'


    xs = np.linspace(0, 1, N)
    ys = np.linspace(0, 1, N)
    zs = np.linspace(0, 1, N)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    X -= center[0]
    Y -= center[1]
    Z -= center[2]

    X /= radii[0]
    Y /= radii[1]
    Z /= radii[2]

    return (X**2 + Y**2 + Z**2 <= 1).astype(np.float32)


def generate_ellipsoid_sdf(center: np.ndarray, radii: np.ndarray, N: int):
    """Signed distance function representation of an ellipse-shaped interface in 3D space.
    """
    assert center.shape == (3,), f'Unexpected shape: {center.shape}'
    assert radii.shape == (3,), f'Unexpected shape: {radii.shape}'
    assert N > 0, f'Invalid volume size: {N}'
    assert np.all(radii > 0) and np.all(radii <= 1), f'Invalid radii: {radii}'
    assert np.all(center >= 0) and np.all(center <= 1), f'Invalid center: {center}'

    vol_frac = generate_ellipsoid_vol_fracs(center, radii, N)
    interior_mask = vol_frac > 0.5
    exterior_mask = vol_frac <= 0.5

    dx = 1.0 / N
    sdf_int = - distance_transform_edt(interior_mask, sampling=dx)
    sdf_ext = distance_transform_edt(exterior_mask, sampling=dx)
    sdf_ext[exterior_mask] -= dx
    sdf = sdf_int + sdf_ext

    check_sdf_consistency(sdf, dx)

    return sdf


class EllipseDataset(Dataset):
    """Dataset class for synthetic ellipse data, parameterized by center and radius.
    The interface representation (diffuse/sdf) is specified on construction.
    """
    def __init__(self,
                 debug: bool = False,
                 num_samples: int = 100,
                 vol_size: int = 64,
                 interface_rep: InterfaceRepresentationType = InterfaceRepresentationType.SDF,
                 epsilon: float = None):
        super().__init__()

        if debug:
            num_samples = 3

        self.num_samples = num_samples
        self.vol_size = vol_size
        self.dx = 1.0 / vol_size
        self.interface_rep = interface_rep
        self.epsilon = epsilon

        centers = np.random.rand(num_samples, 3)
        radii = np.random.uniform(0.2, 0.5, (num_samples, 3))

        # Generate ellipse dataset and convert to desired interface representation
        self.data = np.array([generate_ellipsoid_sdf(center, radii, vol_size) for center, radii in zip(centers, radii)])
        self.data = [convert_from_sdf(d, interface_rep, epsilon) for d in self.data]

        # Data normalisation: minmax normalization to range of [0, 1]
        self.compute_norm_params()
        self.data = [self.normalise_array(d) for d in self.data]

        # Add channel dim and convert to torch tensor
        self.data = [torch.tensor(d, dtype=torch.float32).unsqueeze(0) for d in self.data]

        logger.info(f'Generated {num_samples} samples of ellipse data with interface representation {interface_rep}')
        logger.info(f'Each sample has shape {self.data[0].shape}')

    def compute_norm_params(self):
        # Compute params for minmax normalization to range of [0, 1]
        self.min_val = np.min([np.min(d) for d in self.data])
        self.max_val = np.max([np.max(d) for d in self.data])
        self.range = self.max_val - self.min_val

    def normalise_array(self, arr):
        # Intensity minmax normalization
        return (arr - self.min_val) / self.range

    def unnormalise_array(self, arr):
        # Undo intensity normalisation
        arr = arr * self.range + self.min_val
        return arr

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


#class PatchEllipseDataset(EllipseDataset):
#    """Patch-based dataset class for synthetic ellipse data, parameterized by center and radius. The volumes are hard-coded to 64^3.
#    The interface representation (sharp/diffuse/sdf) is specified on construction.
#    """
#    def __init__(self,
#                 debug: bool = False,
#                 num_samples: int = 100,
#                 patch_size: int = 32,
#                 interface_rep: InterfaceRepresentationType = InterfaceRepresentationType.VOL_FRAC):
#        super().__init__(debug=debug, num_samples=num_samples, interface_rep=interface_rep)
#
#        assert patch_size <= 256, 'Patch size must be less than or equal'
#        self.patch_size = patch_size
#        self.num_patches_per_volume = (self.data[0].numel() // self.patch_size**3)
#        self.patch_data = []
#        self.volume_ids = []
#        np.random.seed(42)  # Ensure reproducibility in random patch selection
#
#        for i in range(len(self.data) * self.num_patches_per_volume):
#            volume_id = i // self.num_patches_per_volume
#            volume = self.data[volume_id]
#            patch = self.extract_patch(volume)
#            self.patch_data.append(patch)
#            self.volume_ids.append(volume_id)
#
#        assert self.num_samples * self.num_patches_per_volume == len(self.patch_data), 'Mismatch in number of patches'
#        logger.info(f'Generated {len(self.patch_data)} patches of size {patch_size}^3 from {len(self.data)} volumes')
#
#    def extract_patch(self, volume):
#        patch_start_inds = np.random.randint(0, 256 - self.patch_size, 3)
#        patch_end_inds = [ind + self.patch_size for ind in patch_start_inds]
#        return volume[:,
#                patch_start_inds[0]:patch_end_inds[0],
#               patch_start_inds[1]:patch_end_inds[1],
#               patch_start_inds[2]:patch_end_inds[2]]
#
#    def __len__(self):
#        return len(self.patch_data)
#
#    def __getitem__(self, idx):
#        return self.patch_data[idx]

















