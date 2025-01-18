"""
The experimental phi data consists of 3D arrays of shape (256, 256, 256) containing the phi field of the HIT dataset.

Here, we convert the phi fields to an approximate signed distance.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.interface_representation.interface_transformations import approximate_sdf_from_diffuse


data_dir = Path('/Volumes/My Passport for Mac/Multiphase-droplet-evolution')


def build_intensity_histogram(filenames):
    # Incrementally build intensity histogram
    bins = np.linspace(-0.2, 1.2, 1000)
    intensity_histogram = np.zeros(len(bins) - 1)

    max_phi = 0
    min_phi = 1
    num_outliers = 0
    name_to_max = {}

    for filename in tqdm(filenames):
        data = np.load(filename)
        phi = data['phi']

        max_phi = max(max_phi, phi.max())
        min_phi = min(min_phi, phi.min())
        num_outliers += int(np.any(phi < 0) or np.any(phi > 1))
        name_to_max[filename] = phi.max()

        intensity_histogram += np.histogram(phi, bins=bins)[0]

    print(f'Min phi: {min_phi}, max phi: {max_phi}, num outliers: {num_outliers}')

    plt.plot(bins[:-1], intensity_histogram)
    plt.yscale('log')
    plt.show()

    # Print the filenames with the highest max phi
    sorted_names = sorted(name_to_max, key=name_to_max.get, reverse=True)
    for name in sorted_names[:100]:
        print(f'{name}: {name_to_max[name]}')


def convert_to_approximate_sdf(filenames):
    outdir = data_dir / 'approximate_sdf'
    outdir.mkdir(exist_ok=True)

    for filename in tqdm(filenames):
        data = np.load(filename)
        phi = data['phi']

        # Convert phi to signed distance
        psi = approximate_sdf_from_diffuse(phi, epsilon=1 / 256)

        # Save the signed distance field
        filename_sdf = outdir / filename.name
        np.savez_compressed(filename_sdf, psi=psi)


def main():
    filenames = list(data_dir.glob("*.npz"))

    print(f'Found {len(filenames)} files to convert')

    build_intensity_histogram(filenames)
    #convert_to_approximate_sdf(filenames)




if __name__ == '__main__':
    main()