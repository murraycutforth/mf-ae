import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist

from src.datasets.utils import parallel_unsigned_distances
from src.interface_representation.interface_transformations import diffuse_from_sdf
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

    # Generate a set of points distributed

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


def generate_ellipsoid_surface_point_cloud(center: np.ndarray, radii: np.ndarray, N_phi: int, N_theta: int):
    """Generate a point cloud representation of the surface of an ellipsoid.
    Using spherical polar coordinates, define r=r(theta, phi) as the ellipsoid surface.
    """
    assert center.shape == (3,), f'Unexpected shape: {center.shape}'
    assert radii.shape == (3,), f'Unexpected shape: {radii.shape}'
    assert np.all(radii > 0) and np.all(radii <= 1), f'Invalid radii: {radii}'
    assert np.all(center >= 0) and np.all(center <= 1), f'Invalid center: {center}'
    assert N_phi > 0, f'Invalid discretisation of longitudinal (azimuthal) angle: {N_phi}'
    assert N_theta > 0, f'Invalid discretisation of latitudinal (polar) angle: {N_theta}'

    # Generate spherical coordinates
    phi = np.linspace(0, 2 * np.pi, N_phi)
    theta = np.linspace(0, np.pi, N_theta)
    phi, theta = np.meshgrid(phi, theta)

    # Parametric equations for the ellipsoid surface
    X = radii[0] * np.sin(theta) * np.cos(phi) + center[0]
    Y = radii[1] * np.sin(phi) * np.sin(theta) + center[1]
    Z = radii[2] * np.cos(theta) + center[2]

    # Flatten the arrays to create a point cloud
    points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    return points


def generate_ellipsoid_sdf_from_point_cloud(center: np.ndarray, radii: np.ndarray, N_phi: int, N_theta: int, N_vol: int):
    xs = np.linspace(0, 1, N_vol)
    ys = np.linspace(0, 1, N_vol)
    zs = np.linspace(0, 1, N_vol)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    cell_center_coords = np.stack([X, Y, Z], axis=-1)  # Shape (N_vol, N_vol, N_vol, 3)
    cell_center_coords = cell_center_coords.reshape(-1, 3)

    pc = generate_ellipsoid_surface_point_cloud(center, radii, N_phi, N_theta)  # Shape (N_points, 3)
    ellipsoid_interior_mask = generate_ellipsoid_vol_fracs(center, radii, N_vol).astype(bool)  # Shape (N_vol, N_vol, N_vol)

    # Compute the min distance between each cell center and the ellipsoid surface point cloud
    #pairwise_dist = cdist(cell_center_coords.astype(np.float32), pc.astype(np.float32))  # Shape (N_vol**3, N_points)
    #unsigned_distances = np.min(pairwise_dist, axis=1)
    unsigned_distances = parallel_unsigned_distances(cell_center_coords, pc, 10)

    assert len(unsigned_distances) == N_vol**3, f'Unexpected length: {len(unsigned_distances)}'

    unsigned_distances = unsigned_distances.reshape((N_vol, N_vol, N_vol))

    # Flip sign of interior points
    sdf = unsigned_distances
    sdf[ellipsoid_interior_mask] *= -1

    return sdf





class EllipseDataset(Dataset):
    """Dataset class for synthetic ellipse data, parameterized by center and radius.
    The interface representation (diffuse/sdf) is specified on construction.
    """
    def __init__(self,
                 data_dir: str,
                 split: str,
                 debug: bool = False,
                 ):
        super().__init__()

        self.data_dir = Path(data_dir)

        # Find all .npz filenames in this dir
        self.filenames = list(self.data_dir.glob("*.npz"))
        assert len(self.filenames) > 0, f'No .npz files found in {self.data_dir}'

        # Split the filenames into train, val, test
        np.random.seed(42)
        run_inds = np.arange(len(self.filenames))
        np.random.shuffle(run_inds)

        train_size = int(0.8 * len(run_inds))
        val_size = int(0.2 * len(run_inds))

        logger.info(f'Constructed splits of size (number of runs NOT snapshots): train={train_size}, val={val_size}')

        if split == 'train':
            self.filenames = self.filenames[:train_size]
        elif split == 'val':
            self.filenames = self.filenames[train_size:train_size+val_size]
        elif split == 'test':
            raise NotImplementedError

        if debug:
            self.filenames = self.filenames[:3]

        logger.info(f'Loaded {len(self.filenames)} files for split {split}')
        logger.info(f'First file: {self.filenames[0]}')

        # Load data, add channel dim, convert to pytorch
        self.data = np.array([np.load(f)['phi'] for f in self.filenames])

        # Load data and convert to desired interface representation
        self.data = np.array([np.load(f)['phi'] for f in self.filenames])
        self.data = [torch.tensor(d, dtype=torch.float32).unsqueeze(0) for d in self.data]

        logger.info(f'Generated {len(self.data)} samples of ellipse data')
        logger.info(f'Each sample has shape {self.data[0].shape}')

    def normalise_array(self, arr):
        return arr

    def unnormalise_array(self, arr):
        return arr

    def __len__(self):
        return len(self.data)

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

















