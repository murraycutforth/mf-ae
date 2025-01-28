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


class SpheresDataset(Dataset):
    """Dataset class for synthetic sphere data, parameterized by center, number of spheres, and radius.
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
        self.radius = [np.load(f)['radius'] for f in self.filenames]
        self.n_spheres = [np.load(f)['n_spheres'] for f in self.filenames]

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

