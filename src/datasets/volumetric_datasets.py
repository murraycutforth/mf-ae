import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class VolumeDatasetInMemory(Dataset):
    """Dataset class for loading all volumes in a given directory into memory.

    Notes:
        - Creates a (reproducible) train/val split of the data.
        - Loads all data into memory.
        - Data is loaded from .npz files
    """
    def __init__(self,
                 data_dir: str,
                 split: str,
                 debug: bool = False,
                 data_key: str = 'phi',
                 metadata_keys: list = None,
                 dtype: torch.dtype = torch.float32,
                 max_num_samples: int = None,
                 ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.data_key = data_key
        self.metadata_keys = metadata_keys
        self.data = []
        self.metadata = []

        # Find all .npz filenames in this dir
        self.filenames = list(self.data_dir.glob("*.npz"))
        self.filenames.sort()
        assert len(self.filenames) > 0, f'No .npz files found in {self.data_dir}'

        # Limit the max number of samples considered
        if max_num_samples is not None:
            self.filenames = self.filenames[:max_num_samples]

        # Split the filenames into train, val, test
        train_size = int(0.8 * len(self.filenames))
        val_size = int(0.2 * len(self.filenames))

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
        for f in self.filenames:
            data = np.load(f)
            self.data.append(data[self.data_key])
            if self.metadata_keys:
                self.metadata.append({k: data[k] for k in self.metadata_keys})

        self.data = [torch.tensor(d, dtype=dtype).unsqueeze(0) for d in self.data]

        logger.info(f'Generated {len(self.data)} samples of volumetric data')
        logger.info(f'Each sample has shape {self.data[0].shape}')

    def normalise_array(self, arr):
        return arr

    def unnormalise_array(self, arr):
        return arr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].float()


class PatchVolumeDatasetInMemory(VolumeDatasetInMemory):
    """Patch-based dataset class for compressed volumetric fields.
    Provides torch tensors of shape (1, patch_size, patch_size, patch_size)
    """
    def __init__(self,
                 data_dir: str,
                 split: str,
                 debug: bool = False,
                 data_key: str = 'phi',
                 metadata_keys: list = None,
                 patch_size: int = 32,
                 dtype: torch.dtype = torch.float32,
                 sdf: bool = None,  # TODO: how to get this info in at construction?
                 ):
        super().__init__(data_dir, split, debug, data_key, metadata_keys, dtype)

        self.patch_size = patch_size

        vol_size = self.data[0].shape[-1]
        self.num_patches_per_volume = (vol_size // self.patch_size)**3
        self.patch_start_inds = []
        self.volume_ids = []

        np.random.seed(42)  # Ensure reproducibility in random patch selection

        for i in range(len(self.filenames) * self.num_patches_per_volume):
            patch_start_inds = np.random.randint(0, vol_size - self.patch_size, 3)

            if sdf is not None:
                # Extract this patch, and only keep the patch inds if there is some interior region in the patch
                patch = self.extract_patch(self.data[i // self.num_patches_per_volume], patch_start_inds).squeeze().numpy()
                if sdf:
                    interior_region = patch < 0
                else:
                    interior_region = patch > 0.5

                if not interior_region.any():
                    continue

            self.patch_start_inds.append(patch_start_inds)
            self.volume_ids.append(i // self.num_patches_per_volume)

        logger.info(f'Using {self.num_patches_per_volume} patches per volume')
        logger.info(f'Generated {len(self.patch_start_inds)} patches of size {patch_size}^3 from {len(self.filenames)} volumes')

    def extract_patch(self, volume, patch_start_inds):
        patch_end_inds = [ind + self.patch_size for ind in patch_start_inds]
        return volume[:,
               patch_start_inds[0]:patch_end_inds[0],
               patch_start_inds[1]:patch_end_inds[1],
               patch_start_inds[2]:patch_end_inds[2]]

    def __len__(self):
        return len(self.patch_start_inds)

    def __getitem__(self, idx):
        volume = self.data[self.volume_ids[idx]]
        patch = self.extract_patch(volume, self.patch_start_inds[idx])
        return patch.float()

