"""In this script we apply three approaches to compress the data:
- Extract only the phi field from the arrays and discard other state variables
- Store as compressed npz
"""
import warnings
from pathlib import Path

import numpy as np

from src.paths import local_data_dir


def compress_single_array(filename: Path) -> None:
    """Write out a compressed version of a single array
    """
    data = np.load(filename)
    phi = data[..., 4]

    if phi.max() > 1.:
        warnings.warn(f'phi values outside [0,1] range in {filename}: {phi.min()}, {phi.max()}')

    #assert np.all((phi >= 0.) & (phi <= 1.)), f'phi values outside [0,1] range in {filename}: {phi.min()}, {phi.max()}'
    assert phi.dtype == np.float16

    filename_compressed = filename.with_suffix(".npz")
    np.savez_compressed(filename_compressed, phi=phi)

    print(f'Compressed {filename} to {filename_compressed}')


def main():
    data_dir = local_data_dir()
    filenames = list(data_dir.glob("*.npy"))

    for filename in filenames:
        compress_single_array(filename)


if __name__ == '__main__':
    main()