"""
The experimental phi data consists of 3D arrays of shape (256, 256, 256) containing the phi field of the HIT dataset.

Here, we convert the phi fields to an approximate signed distance.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.interface_representation.interface_transformations import approximate_sdf_from_diffuse, diffuse_from_sdf, \
    exact_sdf_from_diffuse

data_dir = Path('/Volumes/My Passport for Mac/Multiphase-droplet-evolution')


def plot_many_z_slices(phi, fname):
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))

    for i in range(4):
        for j in range(4):
            z = 16 * (i * 4 + j)
            axs[i, j].imshow(phi[:, :, z], cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title(f'z-ind = {z}')

            # Highlight phi>1 region in red
            axs[i, j].imshow(phi[:, :, z] > 1, cmap='Reds', alpha=0.5)

    plt.suptitle(fname)
    plt.tight_layout()

    plt.show()


def build_intensity_histogram(filenames):
    # Incrementally build intensity histogram
    bins = np.linspace(-0.25, 1.25, 1000)
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

        if phi.max() > 1.1:
            print(f'Filename: {filename}, max phi: {phi.max()}')
            #plot_many_z_slices(phi, filename.stem)

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

        phi = np.clip(phi, 0, 1)

        # Convert phi to signed distance
        psi = approximate_sdf_from_diffuse(phi, epsilon=1 / 256)

        # Save the signed distance field
        filename_sdf = outdir / filename.name
        np.savez_compressed(filename_sdf, phi=psi.astype(np.float32))


def convert_to_exact_sdf(filenames):
    outdir = data_dir / 'exact_sdf'
    outdir.mkdir(exist_ok=True)

    for filename in tqdm(filenames):
        data = np.load(filename)
        phi = data['phi']

        phi = np.clip(phi, 0, 1)

        # Convert phi to signed distance
        psi = exact_sdf_from_diffuse(phi, epsilon=1 / 256)

        # Save the signed distance field
        filename_sdf = outdir / filename.name
        np.savez_compressed(filename_sdf, phi=psi.astype(np.float32))


def convert_to_tanh_smoother(filenames):
    outdir = data_dir / 'tanh_128_smoother'
    outdir.mkdir(exist_ok=True)

    for filename in tqdm(filenames):
        data = np.load(filename)
        phi = data['phi']

        phi = np.clip(phi, 0, 1)

        # Convert phi to signed distance
        psi = approximate_sdf_from_diffuse(phi, epsilon=1 / 256)
        phi = diffuse_from_sdf(psi, epsilon=1 / 128)

        # Save the signed distance field
        filename_sdf = outdir / filename.name
        np.savez_compressed(filename_sdf, phi=phi.astype(np.float32))


def convert_to_tanh_sharper(filenames):
    outdir = data_dir / 'tanh_512_sharper'
    outdir.mkdir(exist_ok=True)

    for filename in tqdm(filenames):
        data = np.load(filename)
        phi = data['phi']

        phi = np.clip(phi, 0, 1)

        # Convert phi to signed distance
        psi = approximate_sdf_from_diffuse(phi, epsilon=1 / 256)
        phi = diffuse_from_sdf(psi, epsilon=1 / 512)

        # Save the signed distance field
        filename_sdf = outdir / filename.name
        np.savez_compressed(filename_sdf, phi=phi.astype(np.float32))


def main():
    filenames = list(data_dir.glob("*.npz"))

    print(f'Found {len(filenames)} files to convert')

    build_intensity_histogram(filenames)
    #convert_to_approximate_sdf(filenames)
    #convert_to_tanh_smoother(filenames)
    #convert_to_tanh_sharper(filenames)
    #convert_to_exact_sdf(filenames)





if __name__ == '__main__':
    main()