import mesh2sdf
import numpy as np
import trimesh
from skimage import measure

from src.interface_representation.interface_representation import check_volfrac_consistency
from src.interface_representation.utils import check_tanh_consistency, check_sdf_consistency


def sdf_to_volfrac(sdf: np.ndarray, dx: float):
    """Convert SDF to volume fraction representation.

    Notes:
        - Grid is regular Cartesian with spacing dx, and origin at (0, 0, 0)
        - SDF value in cell i,j,k in grid space corresponds to position (i*dx, j*dx, k*dx) in physical space
        - We want to compute \\int H(sdf(x)) dx over cell volume
        - Currently implemented by an approximate expression, is exact for axis-aligned planar interfaces, and
        converges with O(dx) error for general interfaces
        - OR: we could use exact polygon intersection methods after extracting contour via marching cubes?
        - OR: could use Monte Carlo estimate of integral above?
    """
    check_sdf_consistency(sdf, dx)

    sq32 = np.sqrt(3) / 2
    interior_mask = sdf <= -dx * sq32
    exterior_mask = sdf >= dx * sq32
    mixed_mask = ~interior_mask & ~exterior_mask

    assert np.sum(interior_mask) + np.sum(mixed_mask) + np.sum(exterior_mask) == np.prod(sdf.shape)

    arr = np.zeros_like(sdf)
    arr[interior_mask] = 1.0
    arr[exterior_mask] = 0.0

    mixed_cell_method = _sdf_to_volfrac_mixed_o1_approximation
    #mixed_cell_method = _sdf_to_volfrac_mixed_mc_estimate  # Too slow to be of practical use
    arr[mixed_mask] = mixed_cell_method(sdf, dx, mixed_mask)

    check_volfrac_consistency(arr)
    return arr


def sdf_to_mesh(sdf: np.ndarray, dx: float):
    verts, faces, normals, values = measure.marching_cubes(sdf,
                                                           level=0.0,
                                                           spacing=(dx, dx, dx))
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True, validate=True)


def mesh_to_sdf(mesh: trimesh.Trimesh, dx: float, size: int):
    verts = mesh.vertices
    faces = mesh.faces
    sdf = mesh2sdf.compute(verts, faces, size=size, fix=True, level=2 / size)

    # Plot slices of the SDF
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(sdf[size // 2, :, :])
    plt.colorbar()
    plt.title("SDF Slice")
    plt.show()

    check_sdf_consistency(sdf, dx)
    return sdf


def sdf_to_tanh(sdf: np.ndarray, dx):
    """
    Convert a signed distance function to a diffuse tanh representation of volume fraction
    """
    arr = 0.5 * (1 + np.tanh(-sdf / dx))
    check_tanh_consistency(arr)
    return arr




def _sdf_to_volfrac_mixed_o1_approximation(sdf: np.ndarray, dx: float, mixed_cells: np.ndarray):
    """This expression is correct for a planar interface aligned with one of the grid axes
    """
    arr = ((- sdf[mixed_cells]) / dx) + 0.5
    return np.clip(arr, 0.0, 1.0)


def _sdf_to_volfrac_mixed_mc_estimate(sdf: np.ndarray, dx: float, mixed_cells: np.ndarray):
    """Monte Carlo estimate of volume fraction in mixed cells. NOTE: too slow to be of practical use.
    """
    verts, faces, normals, values = measure.marching_cubes(sdf,
                                                           level=0.0,
                                                           spacing=(dx, dx, dx))

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True, validate=True)

    # Loop over mixed cells, and run a MC estimate in each one
    def _mc_estimate_interior_vol(mesh, dx, i, j, k):
        # Sample points in the cell
        n_points = 1000
        points = np.random.rand(n_points, 3)
        points *= [dx, dx, dx]
        points += [(i - 0.5) * dx, (j-0.5) * dx, (k-0.5) * dx]

        # Check if the points are inside the mesh
        inside = mesh.contains(points)
        return np.sum(inside) / n_points

    arr = np.zeros_like(sdf)
    for i, j, k in zip(*np.where(mixed_cells)):
        arr[i, j, k] = _mc_estimate_interior_vol(mesh, dx, i, j, k)
        print(f"MC estimate for cell ({i}, {j}, {k}): {arr[i, j, k]}")

    return arr[mixed_cells]
