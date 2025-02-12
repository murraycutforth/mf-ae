from pathlib import Path
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.interface_representation.utils import InterfaceRepresentationType, check_sdf_consistency

logger = logging.getLogger(__name__)


class PhiDataset(Dataset):
    """Dataset class for loading compressed phi fields.
    Provides torch tensors of shape (1, 256, 256, 256)
    """
    def __init__(self,
                data_dir: str,
                split: str,
                debug: bool = False,
                interface_rep: InterfaceRepresentationType = InterfaceRepresentationType.TANH,
                epsilon: float = 1/256):
        self.data_dir = Path(data_dir)

        # TODO: these attrs are no longer used since refactor, remove? Or do we still need to do any interface transformations?
        self.interface_rep = interface_rep
        self.epsilon = epsilon
        self.epsilon_data = 1/256

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

        # Check size of dataset element
        data = np.load(self.filenames[0])
        phi = data['phi']
        assert phi.shape == (256, 256, 256), f'Unexpected shape: {phi.shape}'

        # Load data and convert to desired interface representation
        self.data = np.array([np.load(f)['phi'].astype(np.float16) for f in self.filenames])
        self.data = [torch.tensor(d, dtype=torch.float16).unsqueeze(0) for d in self.data]

        logger.info(f'Generated {len(self.data)} samples of HIT data with interface representation {interface_rep}')
        logger.info(f'Each sample has shape {self.data[0].shape}')

    def normalise_array(self, arr):
        return arr

    def unnormalise_array(self, arr):
        return arr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].float()


class PatchPhiDataset(PhiDataset):
    """Patch-based dataset class for compressed phi fields.
    Provides torch tensors of shape (1, patch_size, patch_size, patch_size)
    """
    def __init__(self,
                data_dir: str,
                split: str,
                patch_size: int = 64,
                debug: bool = False,
                interface_rep: InterfaceRepresentationType = InterfaceRepresentationType.TANH,
                epsilon: float = 1/256):
        super().__init__(data_dir, split, debug, interface_rep, epsilon)
        self.patch_size = patch_size
        self.num_patches_per_volume = (256 // self.patch_size)**3 // 2  # Overlap of 50%
        self.patch_start_inds = []
        self.volume_ids = []

        np.random.seed(42)  # Ensure reproducibility in random patch selection

        for i in range(len(self.filenames) * self.num_patches_per_volume):
            patch_start_inds = np.random.randint(0, 256 - self.patch_size, 3)
            self.patch_start_inds.append(patch_start_inds)

            volume_id = i // self.num_patches_per_volume
            self.volume_ids.append(volume_id)

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