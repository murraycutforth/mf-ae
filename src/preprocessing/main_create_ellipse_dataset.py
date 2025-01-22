# Create dataset of synthetic ellipsoids with different interface representations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.datasets.ellipse_dataset import generate_ellipsoid_sdf
from src.interface_representation.interface_transformations import diffuse_from_sdf, approximate_sdf_from_diffuse


NUM_SAMPLES = 5000
VOL_SIZE = 64
OUTDIR = Path('data/ellipsoids')

def main():
    outdir_exact = OUTDIR / 'exact_sdf'
    outdir_exact.mkdir(exist_ok=True, parents=True)
    outdir_approx = OUTDIR / 'approx_sdf'
    outdir_approx.mkdir(exist_ok=True)
    outdir_tanh = OUTDIR / 'tanh'
    outdir_tanh.mkdir(exist_ok=True)
    outdir_tanh_smooth = OUTDIR / 'tanh_smooth'
    outdir_tanh_smooth.mkdir(exist_ok=True)
    outdir_tanh_sharp = OUTDIR / 'tanh_sharp'
    outdir_tanh_sharp.mkdir(exist_ok=True)


    np.random.seed(42)  # Ensure identical validation set between runs
    centers = np.random.rand(NUM_SAMPLES, 3)
    radii = np.random.uniform(0.2, 0.5, (NUM_SAMPLES, 3))


    for i in tqdm(range(NUM_SAMPLES)):
        sdf_exact = generate_ellipsoid_sdf(centers[i], radii[i], VOL_SIZE)
        diffuse = diffuse_from_sdf(sdf_exact, epsilon=1/64)
        diffuse_smooth = diffuse_from_sdf(sdf_exact, epsilon=1/32)
        diffuse_sharp = diffuse_from_sdf(sdf_exact, epsilon=1/128)
        sdf_approx = approximate_sdf_from_diffuse(diffuse, epsilon=1/64)

        # Save to disk
        np.savez_compressed(outdir_exact / f'ellipsoid_{i}.npz', phi=sdf_exact)
        np.savez_compressed(outdir_approx / f'ellipsoid_{i}.npz', phi=sdf_approx)
        np.savez_compressed(outdir_tanh / f'ellipsoid_{i}.npz', phi=diffuse)
        np.savez_compressed(outdir_tanh_smooth / f'ellipsoid_{i}.npz', phi=diffuse_smooth)
        np.savez_compressed(outdir_tanh_sharp / f'ellipsoid_{i}.npz', phi=diffuse_sharp)


if __name__ == '__main__':
    main()