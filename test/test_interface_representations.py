import unittest
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes

from src.interface_representation.sdf import sdf_to_volfrac, sdf_to_tanh, mesh_to_sdf
from src.interface_representation.tanh import tanh_to_mesh
from src.interface_representation.utils import InterfaceRepresentationType

from skimage.draw import ellipsoid

from src.interface_representation.vol_frac import volfrac_to_mesh


def plot_trimesh(mesh):
    # Assuming `mesh` is your trimesh object
    vertices = mesh.vertices
    faces = mesh.faces

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces, cmap='viridis', edgecolor='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


class TestSdfToVolFrac(unittest.TestCase):
    def test_linear_interface_grid_aligned_cell_boundary(self):
        # Manually construct 3D SDF representing a linear interface
        volume_size = 50
        dx = 1 / volume_size
        interface_pos = 0.49  # Interface along a cell boundary

        xs = np.arange(volume_size) * dx  # 0, 0.02, 0.04, ..., 0.98
        ys = np.arange(volume_size) * dx
        zs = np.arange(volume_size) * dx

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        sdf = X - interface_pos  # Linear interface at x = 0.49

        vol_frac = sdf_to_volfrac(sdf, dx)

        true_vol_frac = np.zeros_like(sdf)
        true_vol_frac[sdf >= 0] = 0
        true_vol_frac[sdf < 0] = 1

        np.testing.assert_allclose(vol_frac, true_vol_frac)

        # Repeat for other axes

        sdf = Y - interface_pos
        vol_frac = sdf_to_volfrac(sdf, dx)
        true_vol_frac = np.zeros_like(sdf)
        true_vol_frac[sdf >= 0] = 0
        true_vol_frac[sdf < 0] = 1
        np.testing.assert_allclose(vol_frac, true_vol_frac)

        sdf = Z - interface_pos
        vol_frac = sdf_to_volfrac(sdf, dx)
        true_vol_frac = np.zeros_like(sdf)
        true_vol_frac[sdf >= 0] = 0
        true_vol_frac[sdf < 0] = 1
        np.testing.assert_allclose(vol_frac, true_vol_frac)

    def test_linear_interface_grid_aligned_cell_center(self):

        volume_size = 50
        dx = 1 / volume_size
        interface_pos = 0.50  # Interface along a cell centre

        xs = np.arange(volume_size) * dx  # 0, 0.02, 0.04, ..., 0.98
        ys = np.arange(volume_size) * dx
        zs = np.arange(volume_size) * dx

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        sdf = X - interface_pos  # Linear interface at x = 0.49
        vol_frac = sdf_to_volfrac(sdf, dx)

        true_vol_frac = np.zeros_like(sdf)
        true_vol_frac[sdf >= 0] = 0
        true_vol_frac[sdf < 0] = 1
        true_vol_frac[(-dx / 2 <= sdf) & (sdf < dx / 2)] = 0.5

        np.testing.assert_allclose(vol_frac, true_vol_frac)

        # Repeat for other axes

        interface_pos = 0.50
        sdf = Y - interface_pos
        vol_frac = sdf_to_volfrac(sdf, dx)

        true_vol_frac = np.zeros_like(sdf)
        true_vol_frac[sdf >= 0] = 0
        true_vol_frac[sdf < 0] = 1
        true_vol_frac[(-dx / 2 <= sdf) & (sdf < dx / 2)] = 0.5

        np.testing.assert_allclose(vol_frac, true_vol_frac)

        interface_pos = 0.50
        sdf = Z - interface_pos
        vol_frac = sdf_to_volfrac(sdf, dx)

        true_vol_frac = np.zeros_like(sdf)
        true_vol_frac[sdf >= 0] = 0
        true_vol_frac[sdf < 0] = 1
        true_vol_frac[(-dx / 2 <= sdf) & (sdf < dx / 2)] = 0.5

        np.testing.assert_allclose(vol_frac, true_vol_frac)

    def test_spherical_interface(self):
        dx = 1 / 25
        sdf = ellipsoid(25, 25, 25, levelset=True)
        vol_frac = sdf_to_volfrac(sdf, dx=dx)

        true_vol = 4/3 * np.pi * 1.0**3
        self.assertAlmostEqual(np.sum(vol_frac) * dx**3, true_vol, places=2)

        #verts, faces, normals, values = marching_cubes(sdf, level=0.0)
        #mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True, validate=True)
        #plot_trimesh(mesh)

        # Visualise slices of sdf and vol frac

        plt.figure()
        plt.imshow(sdf[25, :, :])
        plt.colorbar()
        plt.title("SDF Slice")
        plt.show()

        plt.figure()
        plt.imshow(vol_frac[25, :, :])
        plt.colorbar()
        plt.title("Vol Frac Slice")
        plt.show()


class TestSdfToDiffuseTanh(unittest.TestCase):
    def test_linear_interface_grid_aligned_cell_boundary(self):
        volume_size = 50
        dx = 1 / volume_size
        interface_pos = 0.49  # Interface along a cell boundary

        xs = np.arange(volume_size) * dx  # 0, 0.02, 0.04, ..., 0.98
        ys = np.arange(volume_size) * dx
        zs = np.arange(volume_size) * dx

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        sdf = X - interface_pos  # Linear interface at x = 0.49
        tanh = sdf_to_tanh(sdf, dx)

        vol = np.sum(tanh) * dx**3
        true_vol = 0.5
        self.assertAlmostEqual(vol, true_vol, places=2)

        # Visualise slices of sdf and tanh

        #plt.figure()
        #plt.imshow(sdf[:, 25, :])
        #plt.colorbar()
        #plt.title("SDF Slice")
        #plt.show()

        #plt.figure()
        #plt.imshow(tanh[:, 25, :])
        #plt.colorbar()
        #plt.title("Tanh Slice")
        #plt.show()

        sdf = Y - interface_pos
        tanh = sdf_to_tanh(sdf, dx)
        vol = np.sum(tanh) * dx**3
        true_vol = 0.5
        self.assertAlmostEqual(vol, true_vol, places=2)

        #plt.figure()
        #plt.imshow(tanh[25, :, :])
        #plt.colorbar()
        #plt.title("Tanh Slice")
        #plt.show()

    def test_linear_interface_grid_aligned_cell_center(self):
        volume_size = 50
        dx = 1 / volume_size
        interface_pos = 0.50

        xs = np.arange(volume_size) * dx  # 0, 0.02, 0.04, ..., 0.98
        ys = np.arange(volume_size) * dx
        zs = np.arange(volume_size) * dx

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        sdf = X - interface_pos
        tanh = sdf_to_tanh(sdf, dx)

        vol = np.sum(tanh) * dx**3
        true_vol = 0.51
        self.assertAlmostEqual(vol, true_vol, places=2)

        # Visualise slices of sdf and tanh

        #plt.figure()
        #plt.imshow(sdf[:, 25, :])
        #plt.colorbar()
        #plt.title("SDF Slice")
        #plt.show()

        #plt.figure()
        #plt.imshow(tanh[:, 25, :])
        #plt.colorbar()
        #plt.title("Tanh Slice")
        #plt.show()

    def test_spherical_interface(self):
        dx = 1 / 25
        sdf = ellipsoid(25, 25, 25, levelset=True)
        tanh = sdf_to_tanh(sdf, dx=dx)

        true_vol = 4 / 3 * np.pi * 1.0 ** 3
        self.assertAlmostEqual(np.sum(tanh) * dx ** 3, true_vol, places=2)

        # Visualise slices of sdf and tanh

        #plt.figure()
        #plt.imshow(sdf[25, :, :])
        #plt.colorbar()
        #plt.title("SDF Slice")
        #plt.show()

        #plt.figure()
        #plt.imshow(tanh[25, :, :])
        #plt.colorbar()
        #plt.title("Tanh Slice")
        #plt.show()



class TestVolFracToSDF(unittest.TestCase):
    def test_linear_interface_grid_aligned_cell_boundary(self):
        volume_size = 50
        dx = 1 / volume_size
        interface_pos = 0.49

        xs = np.arange(volume_size) * dx  # 0, 0.02, 0.04, ..., 0.98
        ys = np.arange(volume_size) * dx
        zs = np.arange(volume_size) * dx

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        vol_frac = np.zeros_like(X)
        vol_frac[X < interface_pos] = 1.0
        vol_frac[X >= interface_pos] = 0.0

        true_sdf = X - interface_pos

        mesh = volfrac_to_mesh(vol_frac, dx)
        sdf = mesh_to_sdf(mesh, dx, volume_size)

        np.testing.assert_allclose(sdf, true_sdf, atol=1e-2)


class TestTanhToVolFrac(unittest.TestCase):
    def test_linear_interface_grid_aligned_cell_boundary(self):
        volume_size = 50
        dx = 1 / volume_size
        interface_pos = 0.49

        xs = np.arange(volume_size) * dx
        ys = np.arange(volume_size) * dx
        zs = np.arange(volume_size) * dx

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

        sdf = X - interface_pos

        tanh = sdf_to_tanh(sdf, dx)
        mesh = tanh_to_mesh(tanh, dx, volume_size)
        sdf_pred = mesh_to_sdf(mesh, dx, volume_size)
        vol_frac = sdf_to_volfrac(sdf_pred, dx)

        true_vol_frac = np.zeros_like(sdf)
        true_vol_frac[sdf >= 0] = 0
        true_vol_frac[sdf < 0] = 1

        np.testing.assert_allclose(vol_frac, true_vol_frac, atol=1e-2)


class TestTanhToSDF(unittest.TestCase):
    # TODO
    pass





if __name__ == '__main__':
    unittest.main()