# Create dataset of synthetic ellipsoids with different interface representations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.interface_representation.interface_transformations import convert_from_exact_sdf
from src.interface_representation.interface_types import InterfaceType

NUM_SAMPLES = 2500
VOL_SIZE = 64
OUTDIR = Path('data/spheres')


def sphere_sdf(center: np.ndarray, radius: float, N: int):
    """Signed distance function representation of a sphere-shaped interface in 3D space.
    """
    assert center.shape == (3,), f'Unexpected shape: {center.shape}'
    assert N > 0, f'Invalid volume size: {N}'
    assert radius > 0 and radius <= 1, f'Invalid radius: {radius}'
    assert np.all(center >= 0) and np.all(center <= 1), f'Invalid center: {center}'

    xs = np.linspace(0, 1, N)
    ys = np.linspace(0, 1, N)
    zs = np.linspace(0, 1, N)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    X -= center[0]
    Y -= center[1]
    Z -= center[2]

    return np.sqrt(X**2 + Y**2 + Z**2) - radius


def multi_sphere_sdf(centers: np.ndarray, radius: float, N: int):
    """Signed distance function representation of a multi-sphere-shaped interface in 3D space.
    """
    assert centers.shape == (len(centers), 3), f'Unexpected shape: {centers.shape}'
    assert radius > 0 and radius <= 1, f'Invalid radius: {radius}'
    assert N > 0, f'Invalid volume size: {N}'
    assert np.all(centers >= 0) and np.all(centers <= 1), f'Invalid centers: {centers}'

    phi = np.full((N, N, N), fill_value=1e3, dtype=np.float32)

    for center in centers:
        sdf = sphere_sdf(center, radius, N)
        phi = np.minimum(phi, sdf)

    return phi


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
    interface_types = [
        InterfaceType.SIGNED_DISTANCE_EXACT,
        InterfaceType.SIGNED_DISTANCE_APPROXIMATE,
        InterfaceType.HEAVISIDE,
        InterfaceType.TANH_EPSILON,
    ]

    epsilons = [1/128, 1/64, 1/32, 1/16]

    for _, _, outdir in interface_type_generator(interface_types, epsilons):
        outdir.mkdir(exist_ok=True, parents=True)

    np.random.seed(42)  # Ensure identical validation set between runs
    num_spheres_per_vol = np.random.randint(1, 10, NUM_SAMPLES)

    for i in tqdm(range(NUM_SAMPLES)):
        n_spheres = num_spheres_per_vol[i]
        centers = np.random.rand(n_spheres, 3)
        radius = np.random.uniform(0.05, 0.4)

        sdf_exact = multi_sphere_sdf(centers, radius, VOL_SIZE)

        for interface_type, epsilon, outdir in interface_type_generator(interface_types, epsilons):
            phi = convert_from_exact_sdf(sdf_exact, interface_type, epsilon)
            np.savez_compressed(outdir / f'spheres_{i}.npz', phi=phi, n_spheres=n_spheres, radius=radius)


if __name__ == '__main__':
    main()