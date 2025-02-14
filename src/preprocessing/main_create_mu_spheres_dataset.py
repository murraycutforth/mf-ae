# Create a set of synthetic datasets, parameterised by mu in the log-normal radius distribution.
# In each sample, the number of spheres is chosen so that the total volume fraction is uniform in [0, 0.25], given the
# expected volume of a single sphere.

from pathlib import Path
import logging

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from src.interface_representation.interface_transformations import convert_from_exact_sdf
from src.interface_representation.interface_types import InterfaceType

NUM_SAMPLES_PER_DATASET = 2500
VOL_SIZE = 64
OUTDIR_BASE = Path('data/mu_spheres')


def sphere_sdf(center: np.ndarray, radius: float, N: int):
    """Signed distance function representation of a sphere-shaped interface in 3D space.
    """
    assert center.shape == (3,), f'Unexpected shape: {center.shape}'
    assert N > 0, f'Invalid volume size: {N}'
    assert np.all(center >= 0) and np.all(center <= 1), f'Invalid center: {center}'

    xs = np.linspace(0, 1, N)
    ys = np.linspace(0, 1, N)
    zs = np.linspace(0, 1, N)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    X -= center[0]
    Y -= center[1]
    Z -= center[2]

    return np.sqrt(X**2 + Y**2 + Z**2) - radius


def multi_sphere_sdf(centers: np.ndarray, radii: np.ndarray, N: int):
    """Signed distance function representation of a multi-sphere-shaped interface in 3D space.
    """
    assert centers.shape == (len(centers), 3), f'Unexpected shape: {centers.shape}'
    assert N > 0, f'Invalid volume size: {N}'
    assert np.all(centers >= 0) and np.all(centers <= 1), f'Invalid centers: {centers}'

    phi = np.full((N, N, N), fill_value=1e3, dtype=np.float32)

    for center, radius in zip(centers, radii):
        sdf = sphere_sdf(center, radius, N)
        phi = np.minimum(phi, sdf)

    return phi


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


def get_outdir_name(mu, interface_type, epsilon):
    if interface_type == InterfaceType.TANH_EPSILON:
        return Path(f'spheres_mu_{mu:.2f}') / f'{interface_type.name}{epsilon}'
    else:
        return Path(f'spheres_mu_{mu:.2f}') / interface_type.name


def interface_type_generator(interface_types, epsilons, mu):
    for interface_type in interface_types:
        if interface_type == InterfaceType.TANH_EPSILON:
            for epsilon in epsilons:
                outdir = OUTDIR_BASE / get_outdir_name(mu, interface_type, epsilon)
                yield (interface_type, epsilon, outdir)
        else:
            outdir = OUTDIR_BASE / get_outdir_name(mu, interface_type, None)
            yield (interface_type, None, outdir)


def expected_volume_fraction_per_sphere(mu, sigma):
    exp_r = np.exp(mu + sigma**2 / 2) * 1/VOL_SIZE
    exp_vol = 4/3 * np.pi * exp_r**3
    return exp_vol / 1.0


def main():
    interface_types = [
        InterfaceType.SIGNED_DISTANCE_EXACT,
        InterfaceType.SIGNED_DISTANCE_APPROXIMATE,
        InterfaceType.HEAVISIDE,
        InterfaceType.TANH_EPSILON,
    ]

    epsilons = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]

    mus = [1.0, 2.0, 2.5]
    sigma = 0.5

    # We want to create a dataset for each mu value
    # We configure other params so that the expected volume fraction is sampled uniformly in [0, 0.25] for each vol
    np.random.seed(42)  # Ensure identical validation set between runs
    dx = 1.0 / VOL_SIZE

    for mu in mus:
        expected_f_per_sphere = expected_volume_fraction_per_sphere(mu, sigma)
        max_num_spheres = int(0.25 / expected_f_per_sphere)

        logger.info(f'Creating dataset for mu={mu}, expected volume fraction per sphere={expected_f_per_sphere}, max_num_spheres={max_num_spheres}')

        centers = []
        radii = []
        for i in range(NUM_SAMPLES_PER_DATASET):
            n_spheres = np.random.randint(1, max_num_spheres)
            centers.append(np.random.rand(n_spheres, 3))
            radii.append(np.random.lognormal(mean=mu, sigma=sigma, size=n_spheres) * dx)

        # Create output dirs for each interface type
        for _, _, outdir in interface_type_generator(interface_types, epsilons, mu):
            outdir.mkdir(exist_ok=True, parents=True)

        for i in tqdm(range(NUM_SAMPLES_PER_DATASET)):
            sdf_exact = multi_sphere_sdf(centers[i], radii[i], VOL_SIZE)

            for interface_type, epsilon, outdir in interface_type_generator(interface_types, epsilons, mu):
                phi = convert_from_exact_sdf(sdf_exact, interface_type, epsilon)
                np.savez_compressed(outdir / f'spheres_{i}.npz', phi=phi.astype(np.float16), n_spheres=len(centers[i]), radius=radii[i])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()