import unittest
import numpy as np

from conv_ae_3d.metrics import compute_tanh_heaviside_l1, compute_sdf_heaviside_l1
from src.datasets.ellipse_dataset import generate_ellipsoid_sdf
from src.interface_representation.interface_transformations import convert_from_sdf, InterfaceRepresentationType


class TestSDFHeaviside(unittest.TestCase):
    def setUp(self):
        # Set up 3D sphere SDF
        center = np.array([0.5, 0.5, 0.5])
        radii = np.array([0.3, 0.3, 0.3])
        N = 64
        self.sdf = generate_ellipsoid_sdf(center, radii, N)
        self.dx = 1.0 / N

    def test_perfect_intersection(self):
        sdf_copy = np.copy(self.sdf)
        error = compute_sdf_heaviside_l1(sdf_copy, self.sdf)
        self.assertAlmostEqual(error, 0.0, places=4)

    def test_null_prediction(self):
        sdf_pred = np.ones_like(self.sdf)
        error = compute_sdf_heaviside_l1(sdf_pred, self.sdf)
        sphere_vol = 4/3 * np.pi * 0.3**3
        self.assertAlmostEqual(error * self.dx**3, sphere_vol, places=1)


class TestTanhHeaviside(unittest.TestCase):
    def setUp(self):
        # Set up 3D sphere SDF
        center = np.array([0.5, 0.5, 0.5])
        radii = np.array([0.3, 0.3, 0.3])
        N = 64
        self.sdf = generate_ellipsoid_sdf(center, radii, N)
        self.dx = 1.0 / N
        self.tanh = convert_from_sdf(self.sdf, InterfaceRepresentationType.TANH, self.dx)

    def test_perfect_intersection(self):
        tanh_copy = np.copy(self.tanh)
        error = compute_tanh_heaviside_l1(self.tanh, tanh_copy)
        self.assertAlmostEqual(error, 0.0, places=4)

    def test_null_prediction(self):
        tanh_pred = np.zeros_like(self.tanh)
        error = compute_tanh_heaviside_l1(self.tanh, tanh_pred)
        sphere_vol = 4/3 * np.pi * 0.3**3
        self.assertAlmostEqual(error * self.dx**3, sphere_vol, places=1)





if __name__ == '__main__':
    unittest.main()