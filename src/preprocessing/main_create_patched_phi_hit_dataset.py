
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.datasets.ellipse_dataset import generate_ellipsoid_sdf
from src.interface_representation.interface_transformations import diffuse_from_sdf, approximate_sdf_from_diffuse, \
    convert_from_exact_sdf, exact_sdf_from_diffuse
from src.interface_representation.interface_types import InterfaceType


DATA_DIR = Path('/Volumes/My Passport for Mac/Multiphase-droplet-evolution/tanh_256/')
OUTDIR = Path('data/v6_hit_experiment')


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


def main():
    filenames = list(DATA_DIR.glob("*.npz"))

    print(f'Found {len(filenames)} files to convert')

    interface_types = [
        InterfaceType.SIGNED_DISTANCE_EXACT,
        InterfaceType.SIGNED_DISTANCE_APPROXIMATE,
        InterfaceType.HEAVISIDE,
        InterfaceType.TANH_EPSILON,
    ]

    epsilons = [1/128, 1/64, 1/32, 1/16]

    for _, _, outdir in interface_type_generator(interface_types, epsilons):
        outdir.mkdir(exist_ok=True, parents=True)

    for filename in tqdm(filenames):
        data = np.load(filename)
        phi = data['phi']

        phi = np.clip(phi, 0, 1)

        # Convert phi to signed distance
        sdf_exact = exact_sdf_from_diffuse(phi, epsilon=1 / 256)

        for interface_type, epsilon, outdir in interface_type_generator(interface_types, epsilons):
            phi = convert_from_exact_sdf(sdf_exact, interface_type, epsilon)
            np.savez_compressed(outdir / f'{filename.stem}.npz', phi=phi.astype(np.float16))


if __name__ == '__main__':
    main()