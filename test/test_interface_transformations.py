import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.interface_representation.interface_transformations import approximate_sdf_from_diffuse, diffuse_from_sdf


class TestInterfaceTransformations(unittest.TestCase):

    def setUp(self):
        self.epsilon = 0.1
        self.sdf = np.array([[-1, 0, 1], [-0.5, 0, 0.5], [-1, -0.5, 0]])
        self.diffuse = 0.5 * (1 + np.tanh(self.sdf / (2 * self.epsilon)))

    def test_approximate_sdf_from_diffuse(self):
        sdf_approx = approximate_sdf_from_diffuse(self.diffuse, self.epsilon)
        np.testing.assert_allclose(sdf_approx, self.sdf, atol=1e-2)

    def test_diffuse_from_sdf(self):
        diffuse_approx = diffuse_from_sdf(self.sdf, self.epsilon)
        np.testing.assert_allclose(diffuse_approx, self.diffuse, atol=1e-2)

    def test_round_trip_conversion(self):
        sdf_approx = approximate_sdf_from_diffuse(self.diffuse, self.epsilon)
        diffuse_approx = diffuse_from_sdf(sdf_approx, self.epsilon)
        np.testing.assert_allclose(diffuse_approx, self.diffuse, atol=1e-2)

    def test_spheroid_conversion(self):
        # Create a spheroid
        center = np.array([0.5, 0.5, 0.5])
        radii = np.array([0.3, 0.3, 0.5])
        vol_size = 64
        xs = np.linspace(0, 1, vol_size)
        ys = np.linspace(0, 1, vol_size)
        zs = np.linspace(0, 1, vol_size)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        X -= center[0]
        Y -= center[1]
        Z -= center[2]
        X /= radii[0]
        Y /= radii[1]
        Z /= radii[2]
        spheroid_sdf = X**2 + Y**2 + Z**2 - 1

        print(np.min(spheroid_sdf))
        print(np.max(spheroid_sdf))

        # Plot slice
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(spheroid_sdf[vol_size // 2, :, :])
        plt.colorbar()
        plt.title("Spheroid Slice")
        plt.show()

        # Convert to diffuse and back to sdf
        diffuse_spheroid = diffuse_from_sdf(spheroid_sdf, self.epsilon)
        sdf_spheroid = approximate_sdf_from_diffuse(diffuse_spheroid, self.epsilon)
        np.testing.assert_allclose(sdf_spheroid, spheroid_sdf, atol=1e-2)


class Test1DSDF(unittest.TestCase):
    def setUp(self):
        self.epsilon = 0.01
        self.sdf = np.linspace(-0.5, 0.5, 100)
        self.diffuse = 0.5 * (1 + np.tanh(self.sdf / (2 * self.epsilon)))

    def test_approximate_sdf_from_diffuse(self):
        sdf_approx = approximate_sdf_from_diffuse(self.diffuse, self.epsilon)
        np.testing.assert_allclose(sdf_approx, self.sdf, atol=1e-2)

    def test_diffuse_from_sdf(self):
        diffuse_approx = diffuse_from_sdf(self.sdf, self.epsilon)
        np.testing.assert_allclose(diffuse_approx, self.diffuse, atol=1e-2)

    def test_round_trip_conversion(self):
        sdf_approx = approximate_sdf_from_diffuse(self.diffuse, self.epsilon)
        diffuse_approx = diffuse_from_sdf(sdf_approx, self.epsilon)
        np.testing.assert_allclose(diffuse_approx, self.diffuse, atol=1e-2)

    def test_plot_1d_transforms(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.sdf)
        axs[0].set_title("SDF")
        axs[1].plot(self.diffuse)
        axs[1].set_title("Diffuse")
        sdf_approx = approximate_sdf_from_diffuse(self.diffuse, self.epsilon)
        print(sdf_approx)
        axs[0].plot(sdf_approx)
        diffuse_approx = diffuse_from_sdf(sdf_approx, self.epsilon)
        axs[1].plot(diffuse_approx)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    unittest.main()