from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.datasets.ellipse_dataset import generate_ellipsoid_sdf
from src.interface_representation.interface_transformations import diffuse_from_sdf, approximate_sdf_from_diffuse, \
    convert_from_exact_sdf, exact_sdf_from_diffuse
from src.interface_representation.interface_types import InterfaceType


DATA_DIR = Path('/Volumes/My Passport for Mac/Multiphase-ae/tanh_256/')
OUTDIR = Path('data/patched_hit_experiment')
VOL_SIZE = 64


def get_outdir_name(interface_type, epsilon):
    if interface_type == InterfaceType.TANH_EPSILON:
        return f'{interface_type.name}{epsilon}'
    else:
        return interface_type.name


def get_interfacetype_epsilon_from_outdir(outdir):
    name = outdir.name
    if 'HEAVISIDE' in name:
        return InterfaceType.HEAVISIDE, None
    elif 'SIGNED_DISTANCE_EXACT' in name:
        return InterfaceType.SIGNED_DISTANCE_EXACT, None
    elif 'SIGNED_DISTANCE_APPROXIMATE' in name:
        return InterfaceType.SIGNED_DISTANCE_APPROXIMATE, None
    elif 'TANH_EPSILON' in name:
        epsilon = float(name.split('TANH_EPSILON')[1])
        return InterfaceType.TANH_EPSILON, epsilon
    else:
        raise ValueError(f'Unexpected outdir name: {name}')


def interface_type_generator(interface_types, epsilons):
    for interface_type in interface_types:
        if interface_type == InterfaceType.TANH_EPSILON:
            for epsilon in epsilons:
                outdir = OUTDIR / get_outdir_name(interface_type, epsilon)
                yield (interface_type, epsilon, outdir)
        else:
            outdir = OUTDIR / get_outdir_name(interface_type, None)
            yield (interface_type, None, outdir)


def interior_mask(phi, interface_type):
    if interface_type == InterfaceType.HEAVISIDE or interface_type == InterfaceType.TANH_EPSILON:
        return phi > 0.5
    else:
        return phi < 0.0


def main():
    filenames = list(DATA_DIR.glob("*.npz"))

    print(f'Found {len(filenames)} files to convert')

    interface_types = [
        InterfaceType.SIGNED_DISTANCE_EXACT,
        InterfaceType.SIGNED_DISTANCE_APPROXIMATE,
        InterfaceType.HEAVISIDE,
        InterfaceType.TANH_EPSILON,
    ]

    epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]

    np.random.seed(42)
    num_patches_per_vol = (256 // VOL_SIZE)**3

    for _, _, outdir in interface_type_generator(interface_types, epsilons):
        outdir.mkdir(exist_ok=True, parents=True)

    filename_to_num_patches = defaultdict(int)

    for filename in tqdm(filenames[::-1]):
        data = np.load(filename)
        phi = data['phi']
        phi = np.clip(phi, 0, 1)
        assert phi.shape == (256, 256, 256)

        # Convert phi to signed distance
        sdf_exact = exact_sdf_from_diffuse(phi, epsilon=1 / 256)

        # Create list of patch indices for this volume
        # Each interface type will have the same patches
        patch_start_inds = np.random.randint(0, 256 - VOL_SIZE, (num_patches_per_vol, 3))

        for interface_type, epsilon, outdir in interface_type_generator(interface_types, epsilons):
            phi = convert_from_exact_sdf(sdf_exact, interface_type, epsilon)

            for k in range(num_patches_per_vol):
                patch_start_ind = patch_start_inds[k]
                patch = phi[patch_start_ind[0]:patch_start_ind[0]+VOL_SIZE,
                            patch_start_ind[1]:patch_start_ind[1]+VOL_SIZE,
                            patch_start_ind[2]:patch_start_ind[2]+VOL_SIZE]

                phi_inside = interior_mask(patch, interface_type)

                if np.sum(phi_inside) == 0:
                    continue
                else:
                    filename_to_num_patches[filename] += 1
                    np.savez_compressed(outdir / f'{filename.stem}_patch_{k}.npz', phi=patch.astype(np.float16))

    print(f'Average number of patches per volume: {np.mean(list(filename_to_num_patches.values())) / 9}')  # The 9 is because count each interface type
    print(f'Total number of patches created: {sum(filename_to_num_patches.values()) / 9}')



if __name__ == '__main__':
    main()