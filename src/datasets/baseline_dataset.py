from pathlib import Path
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PhiDataset(Dataset):
    """Dataset class for loading compressed phi fields.
    Provides torch tensors of shape (1, 256, 256, 256)
    """
    def __init__(self, data_dir: str, split: str, debug: bool = False):
        # Find all .npz filenames in this dir
        self.data_dir = Path(data_dir)
        self.filenames = list(self.data_dir.glob("*.npz"))
        self.runs = set([f.stem.split('_')[1] for f in self.filenames])

        # Split the filenames into train, val, test
        np.random.seed(42)
        run_inds = np.arange(len(self.runs))
        np.random.shuffle(run_inds)

        train_size = int(0.6 * len(run_inds))
        val_size = int(0.2 * len(run_inds))
        test_size = len(run_inds) - train_size - val_size

        logger.info(f'Constructed splits of size (number of runs NOT snapshots): train={train_size}, val={val_size}, test={test_size}')

        runs_list = list(self.runs)
        runs_list.sort()

        def run_to_filenames(run):
            return [f for f in self.filenames if f.stem.split('_')[1] == run]

        def run_inds_to_filenames(run_inds):
            nested_filenames = [run_to_filenames(runs_list[i]) for i in run_inds]
            return [item for sublist in nested_filenames for item in sublist]

        if split == 'train':
            self.filenames = run_inds_to_filenames(run_inds[:train_size])
        elif split == 'val':
            self.filenames = run_inds_to_filenames(run_inds[train_size:train_size+val_size])
        elif split == 'test':
            self.filenames = run_inds_to_filenames(run_inds[train_size+val_size:])

        if debug:
            self.filenames = self.filenames[:1]

        logger.info(f'Loaded {len(self.filenames)} files for split {split}')

        # Check size of dataset element
        data = np.load(self.filenames[0])
        phi = data['phi']
        assert phi.shape == (256, 256, 256), f'Unexpected shape: {phi.shape}'

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = np.load(self.filenames[idx])
        phi = data['phi']
        phi = torch.tensor(phi, dtype=torch.float32).unsqueeze(0)
        return phi




