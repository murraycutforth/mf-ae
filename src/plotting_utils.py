import multiprocessing as mp
import os
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional, Tuple, cast
from pathlib import Path

import numpy as np
import skimage.measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import regionprops
from skimage.morphology import label


def write_slice_plot(outpath: Path, data: np.ndarray, vrange=None):
    """Write a plot of orthogonal slices through data
    """
    assert len(data.shape) == 3, "Expected 3D data"

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=200)

    if vrange is None:
        data_min, data_max = data.min(), data.max()
    else:
        data_min, data_max = vrange

    for j in range(3):
        ax = axs[j]
        ax.set_title(f'{j}-slice')
        image = np.take(data, indices=data.shape[j] // 2, axis=j)
        im = ax.imshow(image, cmap="gray", vmin=data_min, vmax=data_max)
        fig.colorbar(im, ax=ax)

    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath)
        plt.close(fig)
    else:
        plt.show()


def read_from_binary_3d(filestr: str, rho: float, dims: Tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = dims
    with open(filestr, "rb") as f:
        f.seek(8)
        file_content = np.fromfile(f, dtype=float)
    if len(file_content) != 5 * nx * ny * nz:
        print("Error: file is wrong shape")
        return np.ones((nx, ny, nz, 5)) * np.nan
    out = np.reshape(file_content, (nx, ny, nz, 5), order="F")
    out[:, :, :, :3] = out[:, :, :, :3] / rho
    return out

def process_and_write_isosurface_plot(inputs):
    #print(f"Processing {restart_file}")
    restart_path, dx, tmpdir, verbose = inputs
    plt.clf()
    restart_id = int(restart_path.stem.replace("Restart_", "").split(".")[0].split("_")[0])
    xs = read_from_binary_3d(str(restart_path), 5, (dx, dx, dx))
    vol = xs[:, :, :, 4]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(0, dx)
    ax.set_ylim(0, dx)
    ax.set_zlim(0, dx)

    if np.any(vol > 0.8):
        verts, faces, normals, values = skimage.measure.marching_cubes(
            vol, 0.8, spacing=(1, 1, 1), allow_degenerate=False, method='lewiner'
        )
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor("k")
        mesh.set_linewidth(0.05)
        mesh.set_alpha(0.9)
        ax.add_collection3d(mesh)
    else:
        pass
    plt.tight_layout()
    outpath = str(tmpdir / Path(f"File_b{restart_id:04d}.png"))
    if verbose:
        print(f"saving {outpath}")
    plt.savefig(outpath)


def write_isosurface_plot(restart_path: Path, dx: float, outdir: Path, verbose: bool) -> None:
    plt.clf()
    restart_id = int(restart_path.stem.split("_")[3])

    if verbose:
        print(f'Restart ID: {restart_id} | Restart Path: {restart_path}')

    xs = np.load(str(restart_path))
    vol = xs[:, :, :, 4]
    outpath = outdir / Path(f"{restart_path.stem}_isosurface.png")

    write_isosurface_plot_from_arr(vol, dx, outpath, 0.5, verbose)


def write_isosurface_plot_from_arr(vol: np.ndarray, dx: float, outname: Path, level: float, verbose: bool) -> None:
    assert len(vol.shape) == 3
    assert vol.shape[0] == vol.shape[1] == vol.shape[2]

    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(0, dx)
    ax.set_ylim(0, dx)
    ax.set_zlim(0, dx)

    if np.any(vol > 0.8):
        verts, faces, normals, values = skimage.measure.marching_cubes(
            vol, level, spacing=(1, 1, 1), allow_degenerate=False, method='lewiner'
        )

        while len(faces) > 500_000:
            faces = faces[::2]

        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor("k")
        mesh.set_linewidth(0.05)
        mesh.set_alpha(0.9)
        ax.add_collection3d(mesh)
    else:
        pass
    plt.tight_layout()

    if verbose:
        print(f"saving {outname}")
    plt.savefig(outname)
    plt.close(fig)


def make_video(directory, tmpdir="~/tmp", dx=256, verbose=False, n_processes: int = 36):
    """Given a directory, looks for all restart files and makes an animation"""
    assert directory != ""
    directory = Path(directory)
    tmpdir = Path(tmpdir)
    assert directory.exists()
    assert tmpdir.exists()

    print(f"Removing tmp files in {tmpdir}")
    for f in tmpdir.rglob("*"):
        if verbose:
            print(f"Removing {f}")
        os.remove(f)

    target_restart_files = list(directory.rglob("*Restart_*.bin"))
    pool = mp.Pool(processes=n_processes)
    inputs = [
        (target_restart_file, dx, tmpdir, verbose) for target_restart_file in target_restart_files
    ]
    results = pool.map(process_and_write_isosurface_plot, inputs)

    cmd_1 = f"cd {str(tmpdir)}; ffmpeg -i File_b%004d.png -r 60 -vf format=yuv420p video.mp4"
    cmd_2 = f"cp {str(tmpdir) / Path('video.mp4')} {directory}"
    if verbose:
        print("running ", cmd_1, cmd_2)
    os.system(cmd_1)
    os.system(cmd_2)
